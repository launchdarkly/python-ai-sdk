"""OpenCode handler: synthetic session events, stable stream, experimental reader."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, SDKOptions, config, configure
from launchdarkly_ai_server.experimental.opencode import (
    OpenCodeSessionError,
    create_opencode_handler,
)
from launchdarkly_ai_server.experimental.rich_stream import stream as rich_stream

CONTEXT = {"kind": "user", "key": "user-1"}

VARIATION = {
    "model": {"name": "test-model"},
    "provider": {"name": "Acme"},
    "instructions": "Be brief.",
    "_ldMeta": {
        "enabled": True,
        "variationKey": "v1",
        "version": 1,
        "mode": "agent",
    },
}


def bus(kind: str, **properties: Any) -> dict[str, Any]:
    return {"type": kind, "properties": properties}


def text_delta(message_id: str, part_id: str, delta: str) -> dict[str, Any]:
    return bus(
        "message.part.delta",
        sessionID="ses",
        messageID=message_id,
        partID=part_id,
        field="text",
        delta=delta,
    )


def tool_part(
    call_id: str,
    status: str,
    *,
    message_id: str = "m1",
    name: str = "bash",
    raw_input: Any = None,
    output: Any = None,
    part_id: str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"status": status}
    if raw_input is not None:
        state["input"] = raw_input
    if output is not None:
        state["output"] = output
    return bus(
        "message.part.updated",
        part={
            "id": part_id or f"part-{call_id}",
            "messageID": message_id,
            "type": "tool",
            "callID": call_id,
            "tool": name,
            "state": state,
        },
    )


def assistant_completed(
    message_id: str,
    *,
    tokens: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "id": message_id,
        "role": "assistant",
        "time": {"created": 1, "completed": 2},
    }
    if tokens is not None:
        info["tokens"] = tokens
    return bus("message.updated", info=info)


def source(events: list[dict[str, Any]]) -> Any:
    async def _events() -> AsyncIterator[dict[str, Any]]:
        for event in events:
            yield event

    return _events


async def collect(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built = create_opencode_handler(capture_content=True, events=source(events))
    gen = await built.stream({}, "hi", None, None)
    return [event async for event in gen]


def types_of(events: list[dict[str, Any]]) -> list[str]:
    return [str(event["type"]) for event in events]


@pytest.fixture(autouse=True)
def _reset_sdk() -> Iterator[None]:
    lifecycle_module._reset_for_testing()
    yield
    lifecycle_module._reset_for_testing()


@pytest.fixture
def mock_ld_client() -> Iterator[MagicMock]:
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    client.variation = AsyncMock(return_value=VARIATION)
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


class TestSurface:
    def test_provides_for_is_any_provider_in_agent_mode(self) -> None:
        handler = create_opencode_handler(events=source([]))
        assert handler.provides_for == ("*", "agent")

    def test_names_stay_off_the_package_root(self) -> None:
        import launchdarkly_ai_server as package
        from launchdarkly_ai_server.experimental import opencode

        assert set(opencode.__all__) == {
            "OpenCodeSessionError",
            "create_opencode_handler",
        }
        public = {name for name in dir(opencode) if not name.startswith("_")}
        assert public == set(opencode.__all__)
        for name in opencode.__all__:
            assert name not in package.__all__
            assert not hasattr(package, name)


class TestTextAndTools:
    async def test_text_and_a_successful_tool(self) -> None:
        events = await collect(
            [
                text_delta("m1", "p1", "Looking up"),
                tool_part("c1", "running", raw_input={"command": "ls"}),
                tool_part("c1", "running", raw_input={"command": "ls"}),
                tool_part(
                    "c1", "completed", raw_input={"command": "ls"}, output="file.txt"
                ),
                tool_part(
                    "c1", "completed", raw_input={"command": "ls"}, output="file.txt"
                ),
                assistant_completed("m1"),
            ],
        )
        assert types_of(events) == [
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "TOOL_CALL_RESULT",
            "TEXT_MESSAGE_END",
            "done",
        ]
        assert events[0]["messageId"] == "m1"
        assert events[1] == {
            "type": "TEXT_MESSAGE_CONTENT",
            "messageId": "m1",
            "delta": "Looking up",
        }
        assert events[2]["toolCallId"] == "c1"
        assert events[2]["toolCallName"] == "bash"
        assert events[2]["parentMessageId"] == "m1"
        assert events[3] == {
            "type": "TOOL_CALL_ARGS",
            "toolCallId": "c1",
            "delta": '{"command":"ls"}',
        }
        assert "toolCallName" not in events[3]
        assert events[5]["content"] == "file.txt"
        assert events[5]["messageId"] != "c1"
        assert "metadata" not in events[5]
        assert events[-1]["output"] == "Looking up"

    async def test_a_failed_tool_is_masked(self) -> None:
        events = await collect(
            [
                tool_part(
                    "c1",
                    "error",
                    raw_input={"command": "rm"},
                    output="disk on fire",
                )
            ],
        )
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == "An error occurred."
        assert result["metadata"] == {"launchdarkly": {"toolFailed": True}}
        assert "disk on fire" not in str(result)
        assert "parentMessageId" not in events[0]

    async def test_a_string_argument_is_not_encoded_again(self) -> None:
        events = await collect(
            [tool_part("c1", "running", raw_input="ls")],
        )
        args = next(event for event in events if event["type"] == "TOOL_CALL_ARGS")
        assert args["delta"] == "ls"

    async def test_nan_result_is_a_failed_tool(self) -> None:
        events = await collect(
            [tool_part("c1", "completed", raw_input={}, output=float("nan"))],
        )
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        assert result["metadata"] == {"launchdarkly": {"toolFailed": True}}
        assert result["content"] == "An error occurred."

    async def test_part_id_is_the_call_id_when_call_id_is_absent(self) -> None:
        part = tool_part("ignored", "running", raw_input={"q": 1}, part_id="part-9")
        del part["properties"]["part"]["callID"]
        events = await collect([part])
        assert events[0]["toolCallId"] == "part-9"

    async def test_interleaved_tools_keep_arrival_order(self) -> None:
        events = await collect(
            [
                tool_part("a", "running", raw_input={"q": "a"}, name="search"),
                tool_part("b", "running", raw_input={"q": "b"}, name="weather"),
                tool_part("b", "completed", raw_input={"q": "b"}, output="sun"),
                tool_part("a", "completed", raw_input={"q": "a"}, output="rows"),
            ],
        )
        results = [event for event in events if event["type"] == "TOOL_CALL_RESULT"]
        assert [event["toolCallId"] for event in results] == ["b", "a"]
        assert [event["content"] for event in results] == ["sun", "rows"]

    async def test_completed_assistant_usage_is_summed_once(self) -> None:
        tokens = {"input": 3, "output": 4, "cache": {"read": 1, "write": 2}}
        events = await collect(
            [
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1", tokens=tokens),
                assistant_completed("m1", tokens=tokens),
            ],
        )
        assert events[-1]["usage"] == {
            "input_tokens": 3,
            "output_tokens": 4,
            "cache_read_input_tokens": 1,
            "cache_creation_input_tokens": 2,
        }

    async def test_text_after_a_tool_is_a_new_message(self) -> None:
        events = await collect(
            [
                text_delta("m1", "p1", "A"),
                tool_part("c1", "running", raw_input={"q": 1}),
                tool_part("c1", "completed", raw_input={"q": 1}, output="ok"),
                assistant_completed("m1"),
                text_delta("m2", "p2", "B"),
                assistant_completed("m2"),
            ],
        )
        starts = [event for event in events if event["type"] == "TEXT_MESSAGE_START"]
        assert [event["messageId"] for event in starts] == ["m1", "m2"]
        assert events[-1]["output"] == "AB"

    async def test_a_text_snapshot_does_not_repeat_deltas(self) -> None:
        events = await collect(
            [
                text_delta("m1", "p1", "Hi"),
                bus(
                    "message.part.updated",
                    part={
                        "id": "p1",
                        "messageID": "m1",
                        "type": "text",
                        "text": "Hi",
                    },
                ),
                assistant_completed("m1"),
            ],
        )
        contents = [
            event for event in events if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert contents == [
            {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Hi"}
        ]

    async def test_session_error_raises_without_the_provider_body(self) -> None:
        handler = create_opencode_handler(
            capture_content=True,
            events=source(
                [
                    text_delta("m1", "p1", "Hi"),
                    {
                        "type": "session.error",
                        "properties": {"error": {"data": {"message": "secret-body"}}},
                    },
                ]
            ),
        )
        seen: list[str] = []
        with pytest.raises(OpenCodeSessionError, match="session failed") as raised:
            async for event in await handler.stream({}, "hi", None, None):
                seen.append(event["type"])
        assert "secret-body" not in str(raised.value)
        assert "done" not in seen

    async def test_pending_tool_state_emits_nothing(self) -> None:
        events = await collect(
            [tool_part("c1", "pending", raw_input={}, output=None)],
        )
        assert types_of(events) == ["done"]


class TestCapture:
    async def test_capture_off_omits_argument_and_result_text(self) -> None:
        handler = create_opencode_handler(
            capture_content=False,
            events=source(
                [
                    tool_part(
                        "c1",
                        "completed",
                        raw_input={"command": "ls"},
                        output="file.txt",
                    )
                ]
            ),
        )
        events = [event async for event in await handler.stream({}, None, None, None)]
        assert "TOOL_CALL_ARGS" not in types_of(events)
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == ""
        assert "metadata" not in result

    async def test_explicit_false_overrides_configure_true(self) -> None:
        configure(options=SDKOptions(capture_content=True))
        try:
            handler = create_opencode_handler(
                capture_content=False,
                events=source(
                    [tool_part("c1", "running", raw_input={"command": "ls"})]
                ),
            )
            events = [
                event async for event in await handler.stream({}, None, None, None)
            ]
            assert "TOOL_CALL_ARGS" not in types_of(events)
        finally:
            lifecycle_module._reset_for_testing()

    async def test_omitted_capture_reads_configure_at_invocation(self) -> None:
        configure(options=SDKOptions(capture_content=True))
        try:
            handler = create_opencode_handler(
                events=source(
                    [tool_part("c1", "running", raw_input={"command": "ls"})]
                ),
            )
            events = [
                event async for event in await handler.stream({}, None, None, None)
            ]
            args = next(event for event in events if event["type"] == "TOOL_CALL_ARGS")
            assert args["delta"] == '{"command":"ls"}'
        finally:
            lifecycle_module._reset_for_testing()


class TestStableStream:
    async def test_tool_events_do_not_close_the_stable_stream(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = create_opencode_handler(
            capture_content=True,
            events=source(
                [
                    text_delta("m1", "p1", "Looking up"),
                    tool_part("c1", "running", raw_input={"command": "ls"}),
                    tool_part(
                        "c1",
                        "completed",
                        raw_input={"command": "ls"},
                        output="file.txt",
                    ),
                    assistant_completed("m1"),
                    text_delta("m2", "p2", "Done."),
                    assistant_completed("m2"),
                ]
            ),
        )
        seen = [
            event
            async for event in config(key="flag", handler=handler).stream("q", CONTEXT)
        ]
        assert [event["type"] for event in seen] == ["chunk", "chunk", "done"]
        assert [event["text"] for event in seen if event["type"] == "chunk"] == [
            "Looking up",
            "Done.",
        ]
        done = seen[-1]
        assert done["response"] == "Looking upDone."
        assert "toolCallId" not in done
        assert done["usage"]["input"] == 0

    async def test_session_error_does_not_yield_stable_done(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = create_opencode_handler(
            events=source(
                [
                    text_delta("m1", "p1", "Hi"),
                    {"type": "session.error", "properties": {"error": {"name": "X"}}},
                ]
            ),
        )
        seen: list[dict[str, Any]] = []
        with pytest.raises(OpenCodeSessionError):
            async for event in config(key="flag", handler=handler).stream("q", CONTEXT):
                seen.append(event)
        assert [event["type"] for event in seen] == ["chunk"]
        assert seen[0]["text"] == "Hi"

    async def test_a_chunk_handler_tool_event_is_not_the_done_event(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def invoke(
            config: Any,
            user_input: str | None,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> dict[str, Any]:
            return {"output": "Hi", "usage": {"input_tokens": 1, "output_tokens": 1}}

        async def stream_fn(
            config: Any,
            user_input: str | None,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> AsyncGenerator[dict[str, Any], None]:
            yield {
                "type": "TOOL_CALL_START",
                "toolCallId": "c1",
                "toolCallName": "bash",
            }
            yield {"type": "chunk", "text": "Hi"}
            yield {
                "type": "done",
                "output": "Hi",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        handler = ProviderHandler(
            fn=invoke, provides_for=("Acme", "agent"), stream_fn=stream_fn
        )
        seen = [
            event
            async for event in config(key="flag", handler=handler).stream("q", CONTEXT)
        ]
        assert [event["type"] for event in seen] == ["chunk", "done"]
        assert seen[-1]["response"] == "Hi"
        assert "toolCallId" not in seen[-1]


class TestRichReader:
    async def test_rich_reader_forwards_text_and_tools(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = create_opencode_handler(
            capture_content=True,
            events=source(
                [
                    text_delta("m1", "p1", "Looking up"),
                    tool_part("c1", "running", raw_input={"command": "ls"}),
                    tool_part(
                        "c1",
                        "error",
                        raw_input={"command": "ls"},
                        output="nope",
                    ),
                    assistant_completed(
                        "m1",
                        tokens={
                            "input": 3,
                            "output": 4,
                            "cache": {"read": 1, "write": 2},
                        },
                    ),
                ]
            ),
        )
        instance = config(key="flag", handler=handler)
        seen = [event async for event in rich_stream(instance, "q", CONTEXT)]
        kinds = [event["type"] for event in seen]
        assert kinds[0] == "RUN_STARTED"
        assert kinds[-1] == "RUN_FINISHED"
        assert seen[0]["threadId"] == seen[-1]["threadId"]
        assert seen[0]["runId"] == seen[-1]["runId"]
        assert seen[0]["protocolVersion"] == "1.0"
        assert "TEXT_MESSAGE_CONTENT" in kinds
        assert "TOOL_CALL_RESULT" in kinds
        result = next(event for event in seen if event["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == "An error occurred."
        assert result["metadata"] == {"launchdarkly": {"toolFailed": True}}
        finished = seen[-1]
        assert finished["result"] == "Looking up"
        assert finished["usage"] == [
            {
                "inputTokens": 6,
                "outputTokens": 4,
                "totalTokens": 10,
                "cachedInputTokens": 1,
                "cacheWriteInputTokens": 2,
            }
        ]
        assert "metadata" not in finished

    async def test_session_error_has_no_run_finished(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = create_opencode_handler(
            events=source(
                [
                    {"type": "session.error", "properties": {"error": {"name": "X"}}},
                ]
            ),
        )
        seen: list[str] = []
        with pytest.raises(OpenCodeSessionError):
            async for event in rich_stream(
                config(key="flag", handler=handler), "q", CONTEXT
            ):
                seen.append(event["type"])
        assert seen == ["RUN_STARTED"]
        assert "RUN_FINISHED" not in seen

    async def test_legacy_chunks_are_one_text_message(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def invoke(
            config: Any,
            user_input: str | None,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> dict[str, Any]:
            return {"output": "Hi"}

        async def stream_fn(
            config: Any,
            user_input: str | None,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "chunk", "text": "Hi"}
            yield {"type": "TOOL_CALL_END", "toolCallId": "c1"}
            yield {"type": "done", "output": "Hi"}

        handler = ProviderHandler(
            fn=invoke, provides_for=("Acme", "agent"), stream_fn=stream_fn
        )
        seen = [
            event
            async for event in rich_stream(
                config(key="flag", handler=handler), "q", CONTEXT
            )
        ]
        assert [event["type"] for event in seen] == [
            "RUN_STARTED",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TOOL_CALL_END",
            "TEXT_MESSAGE_END",
            "RUN_FINISHED",
        ]
        assert seen[2]["delta"] == "Hi"
        assert seen[-1]["result"] == "Hi"
        assert "usage" not in seen[-1]
