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
        assert [
            event["delta"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ] == [
            "Looking up",
            "Done.",
        ]
        start = next(event for event in events if event["type"] == "TOOL_CALL_START")
        assert start["toolCallId"] == "c1"
        assert start["toolCallName"] == "bash"
        assert start["parentMessageId"]
        args = next(event for event in events if event["type"] == "TOOL_CALL_ARGS")
        assert args["delta"] == '{"command":"ls"}'
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == "file.txt"
        assert events[-1]["output"] == "Looking upDone."
        assert types_of(events).count("TOOL_CALL_START") == 1

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
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == "An error occurred."
        assert result["metadata"] == {"launchdarkly": {"toolFailed": True}}
        assert events[-1]["type"] == "done"
        assert events[-1]["output"] == ""
        assert "disk on fire" not in str(events)

    async def test_a_tool_before_the_role_waits_for_the_assistant_message(self) -> None:
        events = await collect(
            [
                tool_part("c1", "running", raw_input={"command": "ls"}),
                text_delta("m1", "p1", "Looking"),
                assistant_open("m1"),
                tool_part("c1", "completed", output="file.txt"),
                assistant_completed("m1"),
            ]
        )
        start = next(event for event in events if event["type"] == "TOOL_CALL_START")
        text = next(event for event in events if event["type"] == "TEXT_MESSAGE_START")
        assert start["parentMessageId"] == text["messageId"]
        assert events.index(text) < events.index(start)

    async def test_a_late_completion_does_not_split_the_open_message(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "A"),
                assistant_open("m2"),
                text_delta("m2", "p2", "B1"),
                text_delta("m3", "p3", "stale"),
                tool_part(
                    "c3",
                    "completed",
                    message_id="m3",
                    raw_input={"command": "old"},
                    output="old.txt",
                ),
                assistant_completed("m3"),
                assistant_completed("m1"),
                text_delta("m2", "p4", "B2"),
                assistant_completed("m2"),
            ]
        )
        texts = [
            event["delta"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert texts == ["A", "B1", "B2"]
        message_ids = [
            event["messageId"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert message_ids[1] == message_ids[2]
        assert message_ids[0] != message_ids[1]
        assert "stale" not in texts
        assert not any(
            event.get("toolCallId") == "c3"
            for event in events
            if event["type"].startswith("TOOL_CALL")
        )

    async def test_capture_off_hides_tool_content(self) -> None:
        built = create_opencode_handler(
            capture_content=False,
            events=source(
                [
                    assistant_open("m1"),
                    tool_part("c1", "running", raw_input={"command": "ls"}),
                    tool_part("c1", "completed", output="file.txt"),
                    assistant_completed("m1"),
                ]
            ),
        )
        gen = await built.stream({}, "hi", None, None)
        events = [event async for event in gen]
        assert any(event["type"] == "TOOL_CALL_START" for event in events)
        assert not any(event["type"] == "TOOL_CALL_ARGS" for event in events)
        rendered = str(events)
        assert "ls" not in rendered
        assert "file.txt" not in rendered

    async def test_an_unlabeled_tool_does_not_inherit_an_open_message(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                tool_part(
                    "c1", "running", message_id="m2", raw_input={"command": "ls"}
                ),
            ]
        )
        text = next(event for event in events if event["type"] == "TEXT_MESSAGE_START")
        start = next(event for event in events if event["type"] == "TOOL_CALL_START")
        assert "parentMessageId" not in start
        ended = next(event for event in events if event["type"] == "TEXT_MESSAGE_END")
        assert events.index(ended) < events.index(start)
        assert ended["messageId"] == text["messageId"]

    async def test_a_user_message_tool_is_not_emitted(self) -> None:
        events = await collect(
            [
                tool_part(
                    "c1", "running", message_id="u1", raw_input={"command": "ls"}
                ),
                bus(
                    "message.updated",
                    info={"id": "u1", "role": "user", "time": {"created": 1}},
                ),
            ]
        )
        assert not any(event["type"].startswith("TOOL_CALL") for event in events)

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
        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert deltas == ["Hi"]
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
        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert deltas == ["He", "llo"]
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
