"""OpenCode handler: synthetic session events, stable chunk then done stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import aclosing
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_opencode_agents import (
    OpenCodeSessionError,
    create_opencode_handler,
    opencode_agents,
)
from launchdarkly_ai_opencode_agents.server import default_model
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


class Script:
    """One fake `opencode serve` over ``httpx.MockTransport``."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.sessions: list[str] = []
        self.messages: list[dict[str, Any]] = []
        self.aborts: list[str] = []

    def transport(self) -> httpx.MockTransport:
        script = self

        async def handle(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/event" and request.method == "GET":
                body = "".join(
                    f"data: {json.dumps(event)}\n\n" for event in script.events
                )
                return httpx.Response(200, text=body)
            if path == "/session" and request.method == "POST":
                session_id = (
                    "ses" if not script.sessions else f"ses-{len(script.sessions) + 1}"
                )
                script.sessions.append(session_id)
                return httpx.Response(200, json={"id": session_id})
            if path.endswith("/message") and request.method == "POST":
                script.messages.append(json.loads(request.content.decode()))
                return httpx.Response(200, json={})
            if path.endswith("/abort") and request.method == "POST":
                script.aborts.append(path)
                return httpx.Response(200, json={})
            return httpx.Response(404)

        return httpx.MockTransport(handle)


def playing(events: list[dict[str, Any]], **kwargs: Any) -> tuple[Any, Script]:
    script = Script(events)
    client = httpx.AsyncClient(transport=script.transport(), base_url="http://opencode")
    handler = create_opencode_handler(
        base_url="http://opencode", client=client, **kwargs
    )
    return handler, script


async def collect(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built, _script = playing(events, capture_content=True)
    gen = await built.stream(
        {
            "instructions": "Be brief.",
            "model": {"name": "m"},
            "provider": {"name": "Acme"},
        },
        "hi",
        None,
        None,
    )
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
    def test_model_names_map_onto_opencode_providers(self) -> None:
        assert default_model(
            {"provider": {"name": "Bedrock"}, "model": {"name": "claude"}}
        ) == ("amazon-bedrock", "claude")
        assert default_model(
            {"provider": {"name": "Gemini"}, "model": {"name": "gemini-2"}}
        ) == ("google", "gemini-2")
        assert default_model(
            {
                "provider": {"name": "Anthropic"},
                "model": {"name": "amazon-bedrock/claude"},
            }
        ) == ("amazon-bedrock", "claude")

    def test_provides_for_is_any_provider_in_agent_mode(self) -> None:
        handler, _script = playing([])
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
        omitted, _script = playing([])
        explicit, _explicit_script = playing([], capture_content=False)
        assert omitted.capture_content is True
        assert explicit.capture_content is False


class TestText:
    async def test_text_around_a_tool_keeps_text_and_the_call(self) -> None:
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

    async def test_a_failed_tool_masks_the_provider_body(self) -> None:
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

    async def test_a_tool_for_another_message_does_not_attach_to_open_text(
        self,
    ) -> None:
        events = await collect(
            [
                assistant_open("m2"),
                assistant_open("m1"),
                text_delta("m1", "p1", "A"),
                tool_part(
                    "c2", "running", message_id="m2", raw_input={"command": "ls"}
                ),
                assistant_completed("m1"),
            ]
        )
        text = next(event for event in events if event["type"] == "TEXT_MESSAGE_START")
        start = next(event for event in events if event["type"] == "TOOL_CALL_START")
        ended = next(event for event in events if event["type"] == "TEXT_MESSAGE_END")
        assert "parentMessageId" not in start
        assert events.index(ended) < events.index(start)
        assert ended["messageId"] == text["messageId"]

    async def test_a_tool_that_finishes_after_the_text_is_not_a_failure(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                tool_part("c1", "running", raw_input={"command": "ls"}),
                assistant_completed("m1"),
                tool_part("c1", "completed", output="file.txt"),
            ]
        )
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == "file.txt"
        assert "metadata" not in result
        assert events[-1]["output"] == "Hi"

    async def test_a_tool_waits_until_assistant_text_is_open(self) -> None:
        events = await collect(
            [
                tool_part("c1", "running", raw_input={"command": "ls"}),
                assistant_open("m1"),
                text_delta("m1", "p1", "Looking"),
                tool_part("c1", "completed", output="file.txt"),
                assistant_completed("m1"),
            ]
        )
        text = next(event for event in events if event["type"] == "TEXT_MESSAGE_START")
        start = next(event for event in events if event["type"] == "TOOL_CALL_START")
        assert events.index(text) < events.index(start)
        assert start["parentMessageId"] == text["messageId"]
        ended = next(event for event in events if event["type"] == "TEXT_MESSAGE_END")
        assert ended["messageId"] == text["messageId"]

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
        assert texts == ["A", "B1", "B2", "stale"]
        message_ids = [
            event["messageId"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert message_ids[1] == message_ids[2]
        assert message_ids[0] != message_ids[1]
        assert message_ids[3] != message_ids[1]
        open_end = next(
            event
            for event in events
            if event["type"] == "TEXT_MESSAGE_END"
            and event["messageId"] == message_ids[1]
        )
        stale = next(
            event
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT" and event["delta"] == "stale"
        )
        assert events.index(open_end) < events.index(stale)
        start = next(
            event
            for event in events
            if event["type"] == "TOOL_CALL_START" and event["toolCallId"] == "c3"
        )
        result = next(
            event
            for event in events
            if event["type"] == "TOOL_CALL_RESULT" and event["toolCallId"] == "c3"
        )
        assert result["content"] == "old.txt"
        assert "metadata" not in result
        assert start.get("parentMessageId") == message_ids[3]
        assert events.index(open_end) < events.index(start)
        assert events[-1]["output"] == "AB1B2stale"

    async def test_late_text_does_not_split_the_open_message(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "A"),
                assistant_open("m2"),
                text_delta("m2", "p2", "B1"),
                bus(
                    "message.part.updated",
                    part={"id": "p1", "messageID": "m1", "type": "text", "text": "A"},
                ),
                text_delta("m1", "p1", "X"),
                text_delta("m2", "p2", "B2"),
                assistant_completed("m2"),
            ]
        )
        b_ids = [
            event["messageId"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
            and event["delta"] in ("B1", "B2")
        ]
        assert b_ids == [b_ids[0], b_ids[0]]
        late = next(
            event
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT" and event["delta"] == "X"
        )
        assert late["messageId"] != b_ids[0]
        assert events[-1]["output"] == "AB1B2X"

    async def test_a_later_snapshot_does_not_repeat_held_text(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "A"),
                assistant_open("m2"),
                text_delta("m2", "p2", "B1"),
                text_delta("m1", "p1", "X"),
                assistant_completed("m2"),
                bus(
                    "message.part.updated",
                    part={"id": "p1", "messageID": "m1", "type": "text", "text": "AX"},
                ),
                assistant_completed("m1"),
            ]
        )
        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert deltas == ["A", "B1", "X"]
        assert events[-1]["output"] == "AB1X"

    async def test_unfinished_assistant_text_is_emitted_at_session_end(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "A"),
                assistant_open("m2"),
                text_delta("m2", "p2", "B1"),
                text_delta("m1", "p1", " later"),
            ]
        )
        b = next(
            event
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT" and event["delta"] == "B1"
        )
        later = next(
            event
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT" and event["delta"] == " later"
        )
        ended = next(
            event
            for event in events
            if event["type"] == "TEXT_MESSAGE_END"
            and event["messageId"] == b["messageId"]
        )
        assert later["messageId"] != b["messageId"]
        assert events.index(ended) < events.index(later)
        assert events[-1]["output"] == "AB1 later"

    async def test_a_completed_message_is_emitted_when_nothing_is_open(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m2", "p2", "Kept"),
                assistant_completed("m2"),
                text_delta("m1", "p1", "After"),
                assistant_completed("m1"),
            ]
        )
        contents = [
            event for event in events if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert [event["delta"] for event in contents] == ["Kept", "After"]
        assert contents[0]["messageId"] != contents[1]["messageId"]
        ends = [
            event["messageId"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_END"
        ]
        assert ends == [contents[0]["messageId"], contents[1]["messageId"]]
        assert events[-1]["output"] == "KeptAfter"

    async def test_capture_off_hides_tool_content(self) -> None:
        built, _script = playing(
            [
                assistant_open("m1"),
                tool_part("c1", "running", raw_input={"command": "ls"}),
                tool_part("c1", "completed", output="file.txt"),
                assistant_completed("m1"),
            ],
            capture_content=False,
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
        handler, _script = playing(
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
            ],
            capture_content=True,
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
        handler, _script = playing(
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
        handler, script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                {"type": "session.error", "properties": {"error": {"name": "X"}}},
            ]
        )
        seen: list[dict[str, Any]] = []
        with pytest.raises(OpenCodeSessionError):
            async for event in config(key="flag", handler=handler).stream("q", CONTEXT):
                seen.append(event)
        assert [event["type"] for event in seen] == ["chunk"]
        assert seen[0]["text"] == "Hi"
        assert script.aborts


class TestInvocation:
    def test_base_url_is_required(self) -> None:
        with pytest.raises(TypeError):
            create_opencode_handler()  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="base_url"):
            create_opencode_handler(base_url="")
        with pytest.raises(TypeError, match="base_url"):
            opencode_agents("flag", "hi", CONTEXT)

    async def test_the_first_session_on_the_stream_is_not_adopted(self) -> None:
        events = await collect(
            [
                bus(
                    "message.updated",
                    info={
                        "id": "other",
                        "role": "assistant",
                        "sessionID": "other",
                        "time": {"created": 1},
                    },
                ),
                text_delta("other", "po", "their answer", session_id="other"),
                assistant_completed("other"),
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
                assistant_completed("m1"),
            ]
        )
        assert events[-1]["output"] == "Hi"
        assert "their answer" not in str(events)

    async def test_events_receives_the_prompt_and_the_session(self) -> None:
        handler, script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )
        events_out = [
            event
            async for event in await handler.stream(
                {
                    "instructions": "Be brief.",
                    "model": {"name": "claude"},
                    "provider": {"name": "Anthropic"},
                },
                "Hello",
                None,
                {"name": "Ada"},
                [{"role": "user", "content": "earlier"}],
            )
        ]
        posted = script.messages[0]
        assert posted["system"] == "Be brief."
        assert posted["parts"] == [
            {"type": "text", "text": "User: earlier\n\nUser: Hello"}
        ]
        assert posted["model"] == {"providerID": "anthropic", "modelID": "claude"}
        assert script.sessions == ["ses"]
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
        contents = [
            event["delta"]
            for event in events
            if event["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert contents == ["Hi"]
        assert events[-1]["output"] == "Hi"

    async def test_a_judge_opens_its_own_session(
        self, mock_ld_client: MagicMock
    ) -> None:
        judged = {
            **VARIATION,
            "judgeConfiguration": {
                "judges": [{"key": "judge-flag", "samplingRate": 1}]
            },
        }
        handler, script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )

        async def variation(key: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            if key == "judge-flag":
                return {
                    **VARIATION,
                    "instructions": (
                        "Grade {{response_to_evaluate}}. History: {{message_history}}"
                    ),
                }
            return judged

        mock_ld_client.variation = variation
        await _drain(config(key="flag", handler=handler).stream("q", CONTEXT))
        assert len(script.sessions) == 2
        assert script.sessions[0] != script.sessions[1]
        prompts = [message["parts"][0]["text"] for message in script.messages]
        systems = [message.get("system", "") for message in script.messages]
        assert prompts[0] == "q"
        assert "{{response_to_evaluate}}" not in systems[1]
        assert "{{message_history}}" not in systems[1]
        assert "Hi" in systems[1]

    async def test_the_wrapper_posts_history_on_a_fresh_session(
        self, mock_ld_client: MagicMock
    ) -> None:
        script = Script(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )
        client = httpx.AsyncClient(
            transport=script.transport(), base_url="http://opencode"
        )
        result = await opencode_agents(
            "flag",
            "Hello",
            CONTEXT,
            base_url="http://opencode",
            client=client,
            history=[{"role": "user", "content": "earlier"}],
        )
        assert result.response == "Hi"
        assert script.messages[0]["system"] == "Be brief."
        assert "User: earlier" in script.messages[0]["parts"][0]["text"]
        assert "User: Hello" in script.messages[0]["parts"][0]["text"]

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
        handler, _script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )
        seen = [event async for event in await handler.stream({}, "hi", None, None)]
        assert seen[-1]["output"] == "Hi"
        assert tracer.spans[0].name == "invoke_agent"
        assert tracer.spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"
        assert tracer.spans[0].attributes["launchdarkly.operation.type"] == "gen_ai"

    async def test_block_array_history_reaches_the_span(
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
        handler, _script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ],
            capture_content=True,
        )
        history = [
            {"role": "system", "content": "do not record"},
            {"role": "user", "content": "earlier"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see "},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.test/a.png",
                        },
                    },
                    {"type": "text", "text": "this"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "noted"}],
            },
        ]
        seen = [
            event async for event in await handler.stream({}, "", None, None, history)
        ]
        assert seen[-1]["output"] == "Hi"
        recorded = json.loads(tracer.spans[0].attributes["gen_ai.input.messages"])
        assert recorded == [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "earlier"}],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "content": "see this"}],
            },
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "noted"}],
            },
        ]
        assert (
            "do not record" not in tracer.spans[0].attributes["gen_ai.input.messages"]
        )
        assert "gen_ai.prompt.3.role" not in tracer.spans[0].attributes

    async def test_a_failed_message_post_does_not_wait_for_the_bus(self) -> None:
        script = _OpenBus(message_status=400)
        client = httpx.AsyncClient(
            transport=script.transport(), base_url="http://opencode"
        )
        handler = create_opencode_handler(base_url="http://opencode", client=client)

        async def _run() -> None:
            async for _event in await handler.stream(
                {
                    "instructions": "Be brief.",
                    "model": {"name": "m"},
                    "provider": {"name": "Acme"},
                },
                "hi",
                None,
                None,
            ):
                return

        from launchdarkly_ai_opencode_agents.server import OpenCodeHTTPError

        with pytest.raises(OpenCodeHTTPError, match="400"):
            await asyncio.wait_for(_run(), timeout=2)
        assert script.aborts

    async def test_a_cancelled_sse_read_is_awaited_before_the_stream_closes(
        self,
    ) -> None:
        bus = _CloseAfterRead()
        client = httpx.AsyncClient(
            transport=bus.transport(), base_url="http://opencode"
        )
        handler = create_opencode_handler(base_url="http://opencode", client=client)

        async def _run() -> None:
            async for _event in await handler.stream(
                {
                    "instructions": "Be brief.",
                    "model": {"name": "m"},
                    "provider": {"name": "Acme"},
                },
                "hi",
                None,
                None,
            ):
                return

        from launchdarkly_ai_opencode_agents.server import OpenCodeHTTPError

        with pytest.raises(OpenCodeHTTPError, match="400"):
            await asyncio.wait_for(_run(), timeout=2)
        assert bus.read_finished.is_set()
        assert bus.closed
        assert not bus.closed_before_read_finished

    async def test_breaking_aborts_without_waiting_for_gc(self) -> None:
        script = _OpenBus()
        client = httpx.AsyncClient(
            transport=script.transport(), base_url="http://opencode"
        )
        handler = create_opencode_handler(base_url="http://opencode", client=client)

        async def _run() -> None:
            async with aclosing(
                await handler.stream(
                    {
                        "instructions": "Be brief.",
                        "model": {"name": "m"},
                        "provider": {"name": "Acme"},
                    },
                    "hi",
                    None,
                    None,
                )
            ) as gen:
                async for event in gen:
                    if event["type"] == "TEXT_MESSAGE_CONTENT":
                        break

        await asyncio.wait_for(_run(), timeout=2)
        assert script.aborts

    async def test_a_late_completion_survives_the_post_returning(self) -> None:
        script = _OpenBus(delay_tail=0.2)
        client = httpx.AsyncClient(
            transport=script.transport(), base_url="http://opencode"
        )
        handler = create_opencode_handler(
            base_url="http://opencode", client=client, capture_content=True
        )
        seen = [
            event
            async for event in await handler.stream(
                {
                    "instructions": "Be brief.",
                    "model": {"name": "m"},
                    "provider": {"name": "Acme"},
                },
                "hi",
                None,
                None,
            )
        ]
        assert seen[-1]["output"] == "Hi!"
        assert seen[-1]["usage"]["output_tokens"] == 4

    async def test_an_idle_without_a_session_id_does_not_end_the_run(self) -> None:
        events = await collect(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                bus("session.idle"),
                text_delta("m1", "p1", "!", session_id="ses"),
                bus("session.idle", sessionID="ses"),
            ]
        )
        assert events[-1]["output"] == "Hi!"


class _CloseAfterRead(httpx.AsyncByteStream):
    """``/event`` stays open. The message POST fails while the next read is blocked."""

    def __init__(self) -> None:
        self.second_read = asyncio.Event()
        self.read_finished = asyncio.Event()
        self.closed = False
        self.closed_before_read_finished = False

    def transport(self) -> httpx.MockTransport:
        stream = self

        async def handle(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/event" and request.method == "GET":
                return httpx.Response(200, stream=stream)
            if path == "/session" and request.method == "POST":
                return httpx.Response(200, json={"id": "ses"})
            if path.endswith("/message") and request.method == "POST":
                await stream.second_read.wait()
                return httpx.Response(400, json={})
            if path.endswith("/abort") and request.method == "POST":
                return httpx.Response(200, json={})
            return httpx.Response(404)

        return httpx.MockTransport(handle)

    async def __aiter__(self) -> Any:
        yield b":\n\n"
        self.second_read.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.read_finished.set()
            raise

    async def aclose(self) -> None:
        if not self.read_finished.is_set():
            self.closed_before_read_finished = True
        self.closed = True


class _ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: Any) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> Any:
        async for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _OpenBus:
    """An `/event` stream that stays open, unlike a finite ``Script`` body."""

    def __init__(self, *, message_status: int = 200, delay_tail: float = 0) -> None:
        self.message_status = message_status
        self.delay_tail = delay_tail
        self.sessions: list[str] = []
        self.messages: list[dict[str, Any]] = []
        self.aborts: list[str] = []
        self.release = asyncio.Event()

    def transport(self) -> httpx.MockTransport:
        script = self

        async def handle(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/event" and request.method == "GET":

                async def chunks() -> Any:
                    # A rejected message sends nothing. Events here would let
                    # the loop notice the 400 without racing the POST.
                    head = (
                        []
                        if script.message_status >= 400
                        else [
                            assistant_open("m1"),
                            text_delta("m1", "p1", "Hi"),
                        ]
                    )
                    for event in head:
                        yield f"data: {json.dumps(event)}\n\n".encode()
                    if script.delay_tail:
                        await asyncio.sleep(script.delay_tail)
                        tail = [
                            text_delta("m1", "p1", "!", session_id="ses"),
                            assistant_completed("m1", tokens={"input": 1, "output": 4}),
                            bus("session.idle", sessionID="ses"),
                        ]
                        for event in tail:
                            yield f"data: {json.dumps(event)}\n\n".encode()
                        return
                    await script.release.wait()

                return httpx.Response(200, stream=_ByteStream(chunks()))
            if path == "/session" and request.method == "POST":
                script.sessions.append("ses")
                return httpx.Response(200, json={"id": "ses"})
            if path.endswith("/message") and request.method == "POST":
                script.messages.append(json.loads(request.content.decode()))
                if script.message_status >= 400:
                    return httpx.Response(script.message_status, json={})
                if script.delay_tail:
                    return httpx.Response(200, json={})
                await script.release.wait()
                return httpx.Response(200, json={})
            if path.endswith("/abort") and request.method == "POST":
                script.aborts.append(path)
                script.release.set()
                return httpx.Response(200, json={})
            return httpx.Response(404)

        return httpx.MockTransport(handle)


async def _drain(stream: Any) -> None:
    async for _event in stream:
        pass
