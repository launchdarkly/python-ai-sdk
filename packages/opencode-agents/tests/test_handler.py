"""OpenCode handler: synthetic session events, stable chunk then done stream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_opencode_agents import (
    OpenCodeSessionError,
    create_opencode_handler,
    opencode_agents,
)
from launchdarkly_ai_server import SDKOptions, config, configure

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


def text_delta(
    message_id: str,
    part_id: str | None,
    delta: str,
    *,
    session_id: str = "ses",
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "sessionID": session_id,
        "messageID": message_id,
        "field": "text",
        "delta": delta,
    }
    if part_id is not None:
        properties["partID"] = part_id
    return bus("message.part.delta", **properties)


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


def assistant_open(message_id: str) -> dict[str, Any]:
    return bus(
        "message.updated",
        info={"id": message_id, "role": "assistant", "time": {"created": 1}},
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
    async def _events(
        config: Any,
        user_input: str | None,
        variables: Any,
        history: Any,
        session_id: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        del config, user_input, variables, history, session_id
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

    def test_package_exports_the_handler(self) -> None:
        import launchdarkly_ai_opencode_agents as package

        assert set(package.__all__) == {
            "OpenCodeSessionError",
            "create_opencode_handler",
            "opencode_agents",
        }
        for name in package.__all__:
            assert hasattr(package, name)

    def test_capture_content_follows_configure(self) -> None:
        configure(options=SDKOptions(capture_content=True))
        omitted = create_opencode_handler(events=source([]))
        explicit = create_opencode_handler(capture_content=False, events=source([]))
        assert omitted.capture_content is True
        assert explicit.capture_content is False


class TestText:
    async def test_text_around_a_tool_is_chunks_then_done(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Looking up"),
                tool_part("c1", "running", raw_input={"command": "ls"}),
                tool_part("c1", "running", raw_input={"command": "ls"}),
                tool_part(
                    "c1", "completed", raw_input={"command": "ls"}, output="file.txt"
                ),
                assistant_completed("m1"),
                assistant_open("m2"),
                text_delta("m2", "p2", "Done."),
                assistant_completed("m2"),
            ],
        )
        assert types_of(events) == ["chunk", "chunk", "done"]
        assert [event["text"] for event in events if event["type"] == "chunk"] == [
            "Looking up",
            "Done.",
        ]
        assert events[-1]["output"] == "Looking upDone."
        assert "file.txt" not in str(events)
        assert "toolCallId" not in str(events)

    async def test_a_failed_tool_does_not_leak_into_the_stream(self) -> None:
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
        assert types_of(events) == ["done"]
        assert events[-1]["output"] == ""
        assert "disk on fire" not in str(events)

    async def test_pending_tool_state_emits_nothing(self) -> None:
        events = await collect(
            [tool_part("c1", "pending", raw_input={}, output=None)],
        )
        assert types_of(events) == ["done"]

    async def test_completed_assistant_usage_is_summed_once(self) -> None:
        tokens = {"input": 3, "output": 4, "cache": {"read": 1, "write": 2}}
        events = await collect(
            [
                assistant_open("m1"),
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

    async def test_a_text_snapshot_does_not_repeat_deltas(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
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
        chunks = [event for event in events if event["type"] == "chunk"]
        assert chunks == [{"type": "chunk", "text": "Hi"}]
        assert events[-1]["output"] == "Hi"

    async def test_a_snapshot_then_later_deltas_append(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                bus(
                    "message.part.updated",
                    part={"id": "p1", "messageID": "m1", "type": "text", "text": "He"},
                ),
                text_delta("m1", "p1", "llo"),
                bus(
                    "message.part.updated",
                    part={
                        "id": "p1",
                        "messageID": "m1",
                        "type": "text",
                        "text": "Hello",
                    },
                ),
                assistant_completed("m1"),
            ],
        )
        chunks = [event["text"] for event in events if event["type"] == "chunk"]
        assert chunks == ["He", "llo"]
        assert events[-1]["output"] == "Hello"

    async def test_user_text_is_not_the_assistant_reply(self) -> None:
        events = await collect(
            [
                bus(
                    "message.part.updated",
                    part={
                        "id": "pu",
                        "messageID": "u1",
                        "type": "text",
                        "text": "secret prompt",
                    },
                ),
                bus(
                    "message.updated",
                    info={
                        "id": "u1",
                        "role": "user",
                        "time": {"created": 1, "completed": 1},
                    },
                ),
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ],
        )
        assert events[-1]["output"] == "Hi"
        assert "secret prompt" not in str(events)

    async def test_session_error_raises_without_the_provider_body(self) -> None:
        handler = create_opencode_handler(
            capture_content=True,
            events=source(
                [
                    assistant_open("m1"),
                    text_delta("m1", "p1", "Hi"),
                    {
                        "type": "session.error",
                        "properties": {
                            "error": {
                                "name": "ProviderAuthError",
                                "data": {"message": "secret-body"},
                            }
                        },
                    },
                ]
            ),
        )
        seen: list[str] = []
        with pytest.raises(OpenCodeSessionError, match="session failed") as raised:
            async for event in await handler.stream({}, "hi", None, None):
                seen.append(event["type"])
        assert "ProviderAuthError" in str(raised.value)
        assert "secret-body" not in str(raised.value)
        assert "done" not in seen


class TestStableStream:
    async def test_a_tool_does_not_close_the_stream(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = create_opencode_handler(
            events=source(
                [
                    assistant_open("m1"),
                    text_delta("m1", "p1", "Looking up"),
                    tool_part("c1", "running", raw_input={"command": "ls"}),
                    tool_part(
                        "c1",
                        "completed",
                        raw_input={"command": "ls"},
                        output="file.txt",
                    ),
                    assistant_completed("m1"),
                    assistant_open("m2"),
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
                    assistant_open("m1"),
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


class TestInvocation:
    def test_events_is_required(self) -> None:
        with pytest.raises(TypeError):
            create_opencode_handler()  # type: ignore[call-arg]

    async def test_events_receives_the_prompt_and_the_session(self) -> None:
        seen: dict[str, Any] = {}

        async def events(
            config: Any,
            user_input: str | None,
            variables: Any,
            history: Any,
            session_id: str | None,
        ) -> AsyncIterator[dict[str, Any]]:
            seen["instructions"] = config.get("instructions")
            seen["user_input"] = user_input
            seen["variables"] = variables
            seen["history"] = history
            seen["session_id"] = session_id
            yield assistant_open("m1")
            yield text_delta("m1", "p1", "Hi")
            yield assistant_completed("m1")

        handler = create_opencode_handler(events=events, session_id="ses")
        events_out = [
            event
            async for event in await handler.stream(
                {"instructions": "Be brief."},
                "Hello",
                None,
                {"name": "Ada"},
                [{"role": "user", "content": "earlier"}],
            )
        ]
        assert seen == {
            "instructions": "Be brief.",
            "user_input": "Hello",
            "variables": {"name": "Ada"},
            "history": [{"role": "user", "content": "earlier"}],
            "session_id": "ses",
        }
        assert events_out[-1]["output"] == "Hi"

    async def test_other_sessions_are_ignored_and_idle_ends_the_run(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi", session_id="ses"),
                text_delta("m9", "p9", " OTHER", session_id="other"),
                {
                    "type": "session.error",
                    "properties": {
                        "sessionID": "other",
                        "error": {"name": "OtherError", "data": {"message": "nope"}},
                    },
                },
                text_delta("m1", "p1", "!", session_id="ses"),
                bus("session.idle", sessionID="ses"),
                text_delta("m1", "p1", "AFTER", session_id="ses"),
            ],
        )
        assert events[-1]["output"] == "Hi!"
        assert "OTHER" not in events[-1]["output"]
        assert "AFTER" not in events[-1]["output"]
        assert "nope" not in str(events)

    async def test_nested_session_ids_do_not_lock_the_run(self) -> None:
        events = await collect(
            [
                bus(
                    "message.updated",
                    info={
                        "id": "m1",
                        "role": "assistant",
                        "sessionID": "ses",
                        "time": {"created": 1},
                    },
                ),
                text_delta("m1", "p1", "Hi", session_id="ses"),
                bus(
                    "message.part.updated",
                    part={
                        "id": "px",
                        "messageID": "mx",
                        "sessionID": "other",
                        "type": "text",
                        "text": "FOREIGN",
                    },
                ),
                bus(
                    "message.updated",
                    info={
                        "id": "mx",
                        "role": "assistant",
                        "sessionID": "other",
                        "time": {"created": 1, "completed": 2},
                        "tokens": {"input": 9, "output": 9},
                    },
                ),
                text_delta("m1", "p1", "!", session_id="ses"),
                bus("session.idle", sessionID="ses"),
                text_delta("m1", "p1", "AFTER", session_id="ses"),
            ],
        )
        assert events[-1]["output"] == "Hi!"
        assert "FOREIGN" not in events[-1]["output"]
        assert "AFTER" not in events[-1]["output"]
        assert "usage" not in events[-1]

    async def test_a_delta_without_part_id_is_not_repeated_by_its_snapshot(
        self,
    ) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", None, "Hi"),
                bus(
                    "message.part.updated",
                    part={"id": "p1", "messageID": "m1", "type": "text", "text": "Hi"},
                ),
                assistant_completed("m1"),
            ],
        )
        chunks = [event["text"] for event in events if event["type"] == "chunk"]
        assert chunks == ["Hi"]
        assert events[-1]["output"] == "Hi"

    async def test_skip_judges_runs_events_once(
        self, mock_ld_client: MagicMock
    ) -> None:
        judged = {
            **VARIATION,
            "judgeConfiguration": {
                "judges": [{"key": "judge-flag", "samplingRate": 1}]
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=judged)
        calls: list[str | None] = []

        async def events(
            config: Any,
            user_input: str | None,
            variables: Any,
            history: Any,
            session_id: str | None,
        ) -> AsyncIterator[dict[str, Any]]:
            del config, variables, history, session_id
            calls.append(user_input)
            yield assistant_open("m1")
            yield text_delta("m1", "p1", "Hi")
            yield assistant_completed("m1")

        handler = create_opencode_handler(events=events, session_id="ses")
        skipped = [
            event
            async for event in config(
                key="flag", handler=handler, skip_judges=True
            ).stream("q", CONTEXT)
        ]
        assert [event["type"] for event in skipped] == ["chunk", "done"]
        assert calls == ["q"]

        calls.clear()
        await _drain(config(key="flag", handler=handler).stream("q", CONTEXT))
        assert calls[0] == "q"
        assert len(calls) == 2
        assert calls[1] != "q"

    async def test_the_wrapper_forwards_variables_history_and_session(
        self, mock_ld_client: MagicMock
    ) -> None:
        seen: dict[str, Any] = {}

        async def events(
            config: Any,
            user_input: str | None,
            variables: Any,
            history: Any,
            session_id: str | None,
        ) -> AsyncIterator[dict[str, Any]]:
            seen["user_input"] = user_input
            seen["name"] = (
                variables.get("name") if isinstance(variables, dict) else None
            )
            seen["history"] = history
            seen["session_id"] = session_id
            seen["instructions"] = (
                config.get("instructions") if isinstance(config, dict) else None
            )
            yield assistant_open("m1")
            yield text_delta("m1", "p1", "Hi")
            yield assistant_completed("m1")

        result = await opencode_agents(
            "flag",
            "Hello",
            CONTEXT,
            events=events,
            session_id="ses",
            variables={"name": "Ada"},
            history=[{"role": "user", "content": "earlier"}],
            skip_judges=True,
        )
        assert result.response == "Hi"
        assert seen["user_input"] == "Hello"
        assert seen["name"] == "Ada"
        assert seen["history"] == [{"role": "user", "content": "earlier"}]
        assert seen["session_id"] == "ses"
        assert seen["instructions"] == "Be brief."

    async def test_a_run_opens_an_invoke_agent_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from launchdarkly_ai_opencode_agents import spans as span_module

        class _Span:
            def __init__(self, name: str, context: Any = None) -> None:
                self.name = name
                self.context = context
                self.attributes: dict[str, Any] = {}
                self.events: list[str] = []

            def set_attribute(self, key: str, value: Any) -> None:
                self.attributes[key] = value

            def add_event(
                self, name: str, attributes: dict[str, Any] | None = None
            ) -> None:
                self.events.append(name)

            def set_status(self, code: Any, description: str | None = None) -> None:
                return None

            def record_exception(self, exc: BaseException) -> None:
                return None

            def end(self) -> None:
                return None

        class _Tracer:
            def __init__(self) -> None:
                self.spans: list[_Span] = []

            def get_tracer(self, name: str) -> _Tracer:
                return self

            def start_span(self, name: str, context: Any = None) -> _Span:
                span = _Span(name, context)
                self.spans.append(span)
                return span

            def set_span_in_context(self, span: _Span) -> Any:
                return span

        tracer = _Tracer()
        monkeypatch.setattr(span_module, "trace", tracer)
        handler = create_opencode_handler(
            events=source(
                [
                    assistant_open("m1"),
                    text_delta("m1", "p1", "Hi"),
                    assistant_completed("m1"),
                ]
            )
        )
        seen = [event async for event in await handler.stream({}, "hi", None, None)]
        assert seen[-1]["output"] == "Hi"
        assert tracer.spans[0].name == "invoke_agent"
        assert tracer.spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"
        assert tracer.spans[0].attributes["launchdarkly.operation.type"] == "gen_ai"


async def _drain(stream: Any) -> None:
    async for _event in stream:
        pass
