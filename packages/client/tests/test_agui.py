"""The opt-in AG-UI reader, and the stable chunk/done projection."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, config
from launchdarkly_ai_server.agui import (
    RUN_FINISHED,
    RUN_STARTED,
    TEXT_MESSAGE_CONTENT,
    TEXT_MESSAGE_END,
    TEXT_MESSAGE_START,
    TOOL_CALL_ARGS,
    TOOL_CALL_END,
    TOOL_CALL_RESULT,
    TOOL_CALL_START,
    RunReader,
    StableProjector,
    TurnEmitter,
    stream,
)
from launchdarkly_ai_server.types import InputTokenDetails, JudgeResult, UsageDict

CONTEXT = {"kind": "user", "key": "u1"}


def _client() -> MagicMock:
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    client.variation = AsyncMock(
        return_value={
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "Be helpful.",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
        }
    )
    return client


@pytest.fixture
def mock_ld_client() -> MagicMock:
    client = _client()
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


def _handler(
    events: list[dict[str, Any]],
    *,
    capture_content: bool = False,
) -> ProviderHandler:
    async def fn(cfg, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": "", "usage": {}}

    async def stream_fn(
        cfg, user_input, tool_handlers, variables, history=None
    ) -> AsyncGenerator[dict[str, Any], None]:
        for event in events:
            yield event

    return ProviderHandler(
        fn=fn,
        provides_for=("TestProvider", "messages"),
        stream_fn=stream_fn,
        capture_content=capture_content,
    )


def _types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


class TestTurnEmitter:
    def test_capture_on_includes_args_and_result(self) -> None:
        emitter = TurnEmitter(capture_content=True)
        events: list[dict[str, Any]] = []
        events.extend(emitter.text("Looking up"))
        events.extend(emitter.tool_start("a", "search"))
        events.extend(emitter.tool_args("a", '{"q":'))
        events.extend(emitter.tool_end("a"))
        events.extend(emitter.end_text())
        events.extend(emitter.tool_result("a", {"ok": True}))
        assert _types(events) == [
            TEXT_MESSAGE_START,
            TEXT_MESSAGE_CONTENT,
            TOOL_CALL_START,
            TOOL_CALL_ARGS,
            TOOL_CALL_END,
            TEXT_MESSAGE_END,
            TOOL_CALL_RESULT,
        ]
        assert events[3]["delta"] == '{"q":'
        assert events[2]["parentMessageId"] == events[0]["messageId"]
        assert events[6]["content"] == '{"ok":true}'
        assert events[6]["messageId"] != "a"

    def test_capture_off_keeps_the_lifecycle_without_payloads(self) -> None:
        emitter = TurnEmitter(capture_content=False)
        events: list[dict[str, Any]] = []
        events.extend(emitter.tool_start("a", "search"))
        events.extend(emitter.tool_args("a", '{"q":'))
        events.extend(emitter.tool_end("a", arguments={"q": "x"}))
        events.extend(emitter.tool_result("a", "secret"))
        assert _types(events) == [TOOL_CALL_START, TOOL_CALL_END, TOOL_CALL_RESULT]
        assert events[0]["toolCallName"] == "search"
        assert "parentMessageId" not in events[0]
        assert events[2]["content"] == ""
        assert "delta" not in str(events)

    def test_failure_is_masked_and_a_string_is_not_reencoded(self) -> None:
        emitter = TurnEmitter(capture_content=True)
        events: list[dict[str, Any]] = []
        events.extend(emitter.tool_start("a", "search"))
        events.extend(emitter.tool_result("a", failed=True))
        events.extend(emitter.tool_start("b", "search"))
        events.extend(emitter.tool_result("b", "plain"))
        failed = events[2]
        assert failed["content"] == "An error occurred."
        assert failed["metadata"] == {"launchdarkly": {"toolFailed": True}}
        assert events[5]["content"] == "plain"

    def test_unserializable_result_is_a_failed_result(self) -> None:
        emitter = TurnEmitter(capture_content=True)
        events = emitter.tool_start("a", "search")
        events.extend(emitter.tool_result("a", float("nan")))
        assert events[-1]["content"] == "An error occurred."
        assert events[-1]["metadata"]["launchdarkly"]["toolFailed"] is True

    def test_fail_open_masks_a_call_that_never_finished(self) -> None:
        emitter = TurnEmitter(capture_content=False)
        emitter.tool_start("a", "search")
        events = emitter.fail_open()
        assert [event["type"] for event in events] == [TOOL_CALL_END, TOOL_CALL_RESULT]
        assert events[-1]["content"] == ""
        assert events[-1]["metadata"]["launchdarkly"]["toolFailed"] is True

    def test_tool_end_extends_a_streamed_argument_prefix(self) -> None:
        emitter = TurnEmitter(capture_content=True)
        events = emitter.tool_start("a", "search")
        events.extend(emitter.tool_args("a", '{"q":'))
        events.extend(emitter.tool_end("a", arguments={"q": "x"}))
        deltas = [
            event["delta"] for event in events if event["type"] == "TOOL_CALL_ARGS"
        ]
        assert "".join(deltas) == '{"q":"x"}'

    def test_a_second_start_for_the_same_id_is_dropped(self) -> None:
        emitter = TurnEmitter(capture_content=True)
        assert emitter.tool_start("a", "search")
        assert emitter.tool_start("a", "search") == []

    def test_message_ids_differ_across_runs(self) -> None:
        first = TurnEmitter(capture_content=False).text("a")[0]["messageId"]
        second = TurnEmitter(capture_content=False).text("b")[0]["messageId"]
        assert first != second

    def test_tool_end_extends_a_matching_prefix_only(self) -> None:
        emitter = TurnEmitter(capture_content=True)
        events = emitter.tool_start("a", "search")
        events.extend(emitter.tool_args("a", '{"q":'))
        events.extend(emitter.tool_end("a", arguments={"q": "x"}))
        deltas = [event["delta"] for event in events if event["type"] == TOOL_CALL_ARGS]
        assert "".join(deltas) == '{"q":"x"}'

        spaced = TurnEmitter(capture_content=True)
        spaced.tool_start("a", "search")
        spaced.tool_args("a", '{"q": "')
        closed = spaced.tool_end("a", arguments={"q": "xy"})
        assert not any(event["type"] == TOOL_CALL_ARGS for event in closed)


class TestRunReader:
    def test_legacy_chunks_become_one_text_message(self) -> None:
        reader = RunReader()
        events = [reader.start()]
        events.extend(reader.feed({"type": "chunk", "text": "Hi"}))
        events.extend(reader.feed({"type": "chunk", "text": "!"}))
        events.extend(
            reader.finish(
                {
                    "response": "Hi!",
                    "usage": {"input": 2, "output": 1, "total": 3},
                    "_had_usage": True,
                },
                None,
                provider="OpenAI",
                model="gpt-4",
            )
        )
        assert _types(events) == [
            RUN_STARTED,
            TEXT_MESSAGE_START,
            TEXT_MESSAGE_CONTENT,
            TEXT_MESSAGE_CONTENT,
            TEXT_MESSAGE_END,
            RUN_FINISHED,
        ]
        assert events[0]["protocolVersion"] == "1.0"
        assert events[0]["threadId"] == events[-1]["threadId"]
        assert events[-1]["result"] == "Hi!"
        assert events[-1]["usage"] == [
            {
                "inputTokens": 2,
                "outputTokens": 1,
                "totalTokens": 3,
                "provider": "OpenAI",
                "model": "gpt-4",
            }
        ]
        assert "metadata" not in events[-1]

    def test_forwards_the_subset_and_drops_a_third_type(self) -> None:
        reader = RunReader()
        events = [reader.start()]
        events.extend(reader.feed({"type": TEXT_MESSAGE_START, "messageId": "m1"}))
        events.extend(
            reader.feed(
                {"type": TEXT_MESSAGE_CONTENT, "messageId": "m1", "delta": "Hi"}
            )
        )
        events.extend(reader.feed({"type": "widget", "text": "nope"}))
        events.extend(reader.feed({"type": "chunk", "text": "dropped"}))
        events.extend(
            reader.feed(
                {
                    "type": TOOL_CALL_START,
                    "toolCallId": "a",
                    "toolCallName": "search",
                    "parentMessageId": "m1",
                }
            )
        )
        events.extend(
            reader.feed({"type": TOOL_CALL_ARGS, "toolCallId": "a", "delta": "{}"})
        )
        events.extend(reader.feed({"type": TOOL_CALL_END, "toolCallId": "a"}))
        events.extend(reader.feed({"type": TEXT_MESSAGE_END, "messageId": "m1"}))
        events.extend(
            reader.feed(
                {
                    "type": TOOL_CALL_RESULT,
                    "messageId": "tm1",
                    "toolCallId": "a",
                    "content": "ok",
                }
            )
        )
        events.extend(
            reader.feed(
                {
                    "type": TOOL_CALL_START,
                    "toolCallId": "a",
                    "toolCallName": "search",
                }
            )
        )
        events.extend(
            reader.finish(
                {"response": "Hi", "_had_usage": False},
                None,
                provider="",
                model="",
            )
        )
        assert "widget" not in _types(events)
        assert "chunk" not in _types(events)
        assert "dropped" not in str(events)
        assert _types(events).count(TOOL_CALL_START) == 1
        assert events[-1]["type"] == RUN_FINISHED
        assert "usage" not in events[-1]
        assert events[-1]["result"] == "Hi"

    def test_cache_split_and_judge_metadata(self) -> None:
        reader = RunReader()
        finished = reader.finish(
            {
                "response": "",
                "_had_usage": True,
                "usage": {
                    "input": 5,
                    "output": 1,
                    "total": 6,
                    "input_details": {
                        "uncached": 3,
                        "cache_read": 1,
                        "cache_creation": 1,
                    },
                },
            },
            {
                "quality": JudgeResult(
                    usage=UsageDict(
                        input=1,
                        output=1,
                        total=2,
                        input_details=InputTokenDetails(
                            uncached=1, cache_read=0, cache_creation=0
                        ),
                    ),
                    response="fine",
                    score=1.0,
                )
            },
            provider="",
            model="",
        )
        usage = finished[-1]["usage"][0]
        assert usage["cachedInputTokens"] == 1
        assert usage["cacheWriteInputTokens"] == 1
        assert "provider" not in usage
        assert "result" not in finished[-1]
        assert (
            finished[-1]["metadata"]["launchdarkly"]["judgeResults"]["quality"]["score"]
            == 1.0
        )

    def test_finish_closes_an_open_tool_before_the_run(self) -> None:
        reader = RunReader(thread_id="thread-1", run_id="run-1")
        events = [reader.start()]
        events.extend(
            reader.feed(
                {
                    "type": TOOL_CALL_START,
                    "toolCallId": "a",
                    "toolCallName": "search",
                }
            )
        )
        events.extend(reader.finish({"response": ""}, None, provider="", model=""))
        assert [event["type"] for event in events] == [
            RUN_STARTED,
            TOOL_CALL_START,
            TOOL_CALL_END,
            RUN_FINISHED,
        ]
        assert events[0]["threadId"] == "thread-1"
        assert events[-1]["runId"] == "run-1"

    def test_a_result_without_an_end_closes_the_call_first(self) -> None:
        reader = RunReader()
        reader.feed(
            {"type": TOOL_CALL_START, "toolCallId": "a", "toolCallName": "search"}
        )
        events = reader.feed({"type": TOOL_CALL_RESULT, "toolCallId": "a"})
        assert [event["type"] for event in events] == [
            TOOL_CALL_END,
            TOOL_CALL_RESULT,
        ]
        assert events[-1]["content"] == ""


class TestStableProjector:
    def test_text_content_is_a_chunk_and_a_third_type_is_not_done(self) -> None:
        projector = StableProjector()
        assert projector.feed({"type": "widget"}) == []
        assert projector.done is None
        assert projector.feed(
            {"type": TEXT_MESSAGE_CONTENT, "messageId": "m", "delta": "Hi"}
        ) == [{"type": "chunk", "text": "Hi"}]
        assert projector.feed({"type": "chunk", "text": "nope"}) == []
        assert (
            projector.feed(
                {"type": TOOL_CALL_START, "toolCallId": "a", "toolCallName": "search"}
            )
            == []
        )
        done = {"type": "done", "response": "Hi", "usage": {"input": 1}}
        assert projector.feed(done) == []
        assert projector.done is done


class TestConfigAndReader:
    async def test_config_stream_stays_chunk_then_done(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = _handler(
            [
                {"type": TEXT_MESSAGE_START, "messageId": "m1"},
                {"type": TEXT_MESSAGE_CONTENT, "messageId": "m1", "delta": "Hi"},
                {
                    "type": TOOL_CALL_START,
                    "toolCallId": "a",
                    "toolCallName": "search",
                    "parentMessageId": "m1",
                },
                {"type": TOOL_CALL_ARGS, "toolCallId": "a", "delta": '{"q":"x"}'},
                {"type": TOOL_CALL_END, "toolCallId": "a"},
                {"type": TEXT_MESSAGE_END, "messageId": "m1"},
                {
                    "type": TOOL_CALL_RESULT,
                    "messageId": "tm",
                    "toolCallId": "a",
                    "content": "secret",
                },
                {"type": "widget", "name": "nope"},
                {
                    "type": "done",
                    "output": "Hi",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            ]
        )
        events = [
            event
            async for event in config(key="flag", handler=handler).stream("q", CONTEXT)
        ]
        assert events == [
            {"type": "chunk", "text": "Hi"},
            {
                "type": "done",
                "response": "Hi",
                "usage": {"input": 3, "output": 1, "total": 4},
                "judge_results": None,
            },
        ]

    async def test_reader_brackets_the_subset(self, mock_ld_client: MagicMock) -> None:
        handler = _handler(
            [
                {"type": "chunk", "text": "Hi"},
                {
                    "type": "done",
                    "output": "Hi",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            ]
        )
        events = [
            event
            async for event in stream(config(key="flag", handler=handler), "q", CONTEXT)
        ]
        assert _types(events) == [
            RUN_STARTED,
            TEXT_MESSAGE_START,
            TEXT_MESSAGE_CONTENT,
            TEXT_MESSAGE_END,
            RUN_FINISHED,
        ]
        assert events[-1]["result"] == "Hi"
        assert events[-1]["usage"][0]["provider"] == "TestProvider"
        assert events[-1]["usage"][0]["model"] == "gpt-4"
        assert events[-1]["usage"][0]["inputTokens"] == 3

    async def test_a_raised_turn_has_no_run_finished(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def fn(cfg, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
            return {"output": "", "usage": {}}

        async def stream_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "chunk", "text": "Hi"}
            raise RuntimeError("stream-fail")
            yield  # pragma: no cover

        handler = ProviderHandler(
            fn=fn,
            provides_for=("TestProvider", "messages"),
            stream_fn=stream_fn,
        )
        seen: list[str] = []
        with pytest.raises(RuntimeError, match="stream-fail"):
            async for event in stream(
                config(key="flag", handler=handler), "q", CONTEXT
            ):
                seen.append(event["type"])
        assert seen == [RUN_STARTED, TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT]
        assert RUN_FINISHED not in seen

    async def test_judge_results_land_on_run_finished(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = _handler(
            [
                {"type": "chunk", "text": "Hi"},
                {
                    "type": "done",
                    "output": "Hi",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ]
        )
        judged = {
            "quality": JudgeResult(
                usage=UsageDict(input=1, output=1, total=2),
                response="fine",
                score=0.5,
            )
        }
        with patch(
            "launchdarkly_ai_server.judges.run_judges",
            new=AsyncMock(return_value=judged),
        ):
            events = [
                event
                async for event in stream(
                    config(key="flag", handler=handler), "q", CONTEXT
                )
            ]
        assert (
            events[-1]["metadata"]["launchdarkly"]["judgeResults"]["quality"]["score"]
            == 0.5
        )

    async def test_closing_early_does_not_finish_the_run(
        self, mock_ld_client: MagicMock
    ) -> None:
        closed = False

        async def fn_unused(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:
            return {"output": "", "usage": {}}

        async def stream_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> AsyncGenerator[dict[str, Any], None]:
            nonlocal closed
            try:
                for event in [
                    {"type": "chunk", "text": "Hi"},
                    {"type": "chunk", "text": " there"},
                    {
                        "type": "done",
                        "output": "Hi there",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]:
                    yield event
            finally:
                closed = True

        handler = ProviderHandler(
            fn=fn_unused,
            provides_for=("TestProvider", "messages"),
            stream_fn=stream_fn,
            capture_content=False,
        )

        gen = stream(config(key="flag", handler=handler), "q", CONTEXT)
        first = await gen.__anext__()
        second = await gen.__anext__()
        await gen.aclose()
        assert first["type"] == RUN_STARTED
        assert second["type"] == TEXT_MESSAGE_START
        assert closed is True
