"""OpenCode handler: synthetic session events, stable chunk then done stream."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import aclosing
from pathlib import Path
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
from launchdarkly_ai_opencode_agents.process import _is_healthy
from launchdarkly_ai_opencode_agents.server import default_model
from launchdarkly_ai_server import SDKOptions, config, configure
from tests.forwarding_spec import (
    OPENCODE_AGENTS,
    candidate_keys,
    probe_forwarded_keys,
)
from tests.never_forwarded import NEVER_FORWARDED_BAG, find_leaks

_FAKE = Path(__file__).with_name("fake_opencode.py")

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
    """A fake ``opencode`` executable. The handler starts and stops it."""

    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        mode: str = "script",
        message_status: int = 200,
        delay_tail: float = 0,
        delay_ready: float = 0,
        exit_stderr: str | None = None,
    ) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="oc-fake-"))
        self.cwd = self.root / "work"
        self.cwd.mkdir()
        self.config = self.root / "opencode.json"
        self.config.write_text("{}", encoding="utf-8")
        self.state = self.root / "state"
        self.state.mkdir()
        self.spec_path = self.root / "spec.json"
        self.spec_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "events": events,
                    "message_status": message_status,
                    "delay_tail": delay_tail,
                    "delay_ready": delay_ready,
                    "exit_stderr": exit_stderr,
                }
            ),
            encoding="utf-8",
        )
        _scripts.append(self)

    def handler_args(self) -> dict[str, Any]:
        return {
            "cwd": self.cwd,
            "config": self.config,
            "env": {
                "LD_OPENCODE_FAKE_DIR": str(self.state),
                "LD_OPENCODE_FAKE_SPEC": str(self.spec_path),
            },
            "command": [sys.executable, str(_FAKE)],
        }

    def _records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(
            self.state.glob("*.json"), key=lambda item: item.stat().st_mtime
        ):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return records

    @property
    def sessions(self) -> list[str]:
        found: list[str] = []
        for record in self._records():
            found.extend(record.get("sessions") or [])
        return found

    @property
    def messages(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for record in self._records():
            found.extend(record.get("messages") or [])
        return found

    @property
    def aborts(self) -> list[str]:
        found: list[str] = []
        for record in self._records():
            found.extend(record.get("aborts") or [])
        return found

    @property
    def pids(self) -> list[int]:
        return [int(record["pid"]) for record in self._records() if "pid" in record]

    @property
    def requests(self) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        for record in self._records():
            found.extend(record.get("requests") or [])
        return found

    @property
    def session_bodies(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for record in self._records():
            found.extend(record.get("session_bodies") or [])
        return found

    @property
    def mcp_calls(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for record in self._records():
            found.extend(record.get("mcp_calls") or [])
        return found

    @property
    def mcp_disconnects(self) -> list[str]:
        found: list[str] = []
        for record in self._records():
            found.extend(record.get("mcp_disconnects") or [])
        return found


_scripts: list[Script] = []


def playing(events: list[dict[str, Any]], **kwargs: Any) -> tuple[Any, Script]:
    script = Script(events)
    handler = create_opencode_handler(**script.handler_args(), **kwargs)
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
    for script in _scripts:
        for pid in script.pids:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        shutil.rmtree(script.root, ignore_errors=True)
    _scripts.clear()


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
            "OpenCodeApplyPatch",
            "OpenCodeBash",
            "OpenCodeEdit",
            "OpenCodeGlob",
            "OpenCodeGrep",
            "OpenCodeRead",
            "OpenCodeSessionError",
            "OpenCodeSkill",
            "OpenCodeTask",
            "OpenCodeTodoWrite",
            "OpenCodeWebFetch",
            "OpenCodeWebSearch",
            "OpenCodeWrite",
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
    def test_directory_config_and_env_are_required(self) -> None:
        with pytest.raises(TypeError):
            create_opencode_handler()  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="working directory"):
            create_opencode_handler(cwd="", config="/tmp/opencode.json", env={})
        with pytest.raises(TypeError, match="config file"):
            create_opencode_handler(cwd="/tmp", config="", env={})
        with pytest.raises(TypeError, match="working directory"):
            opencode_agents("flag", "hi", CONTEXT)
        with pytest.raises(TypeError, match="base_url"):
            create_opencode_handler(
                cwd="/tmp",
                config="/tmp/opencode.json",
                env={},
                base_url="http://127.0.0.1:9",  # type: ignore[call-arg]
            )
        with pytest.raises(TypeError):
            create_opencode_handler(
                cwd="/tmp",
                config="/tmp/opencode.json",
                env={},
                client=object(),  # type: ignore[call-arg]
            )
        with pytest.raises(TypeError, match="URL"):
            create_opencode_handler(
                cwd="/tmp",
                config="/tmp/opencode.json",
                env={},
                command="http://127.0.0.1:9",
            )

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

    async def test_agent_instructions_render_and_history_does_not(self) -> None:
        handler, script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )
        assert handler.provides_for == ("*", "agent")
        events_out = [
            event
            async for event in await handler.stream(
                {
                    "instructions": "Hello {{caller_email}}",
                    "model": {"name": "claude"},
                    "provider": {"name": "Anthropic"},
                },
                "next {{caller_email}}",
                None,
                {"caller_email": "ada@example.com"},
                [{"role": "user", "content": "earlier {{caller_email}}"}],
            )
        ]
        posted = script.messages[0]
        assert posted["system"] == "Hello ada@example.com"
        assert posted["parts"] == [
            {
                "type": "text",
                "text": "User: earlier {{caller_email}}\n\nUser: next {{caller_email}}",
            }
        ]
        assert events_out[-1]["output"] == "Hi"

    async def test_templates_render_in_instructions_and_config_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from launchdarkly_ai_opencode_agents import spans as span_module

        class _Span:
            def __init__(self, name: str, context: Any = None) -> None:
                self.name = name
                self.attributes: dict[str, Any] = {}

            def set_attribute(self, key: str, value: Any) -> None:
                self.attributes[key] = value

            def add_event(
                self, name: str, attributes: dict[str, Any] | None = None
            ) -> None:
                return None

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
        handler, script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ],
            capture_content=True,
        )
        config = {
            "instructions": "Hello {{user.name}} {{missing}}",
            "messages": [
                {"role": "user", "content": "Assist {{caller_email}}"},
                {"role": "system", "content": "skip {{caller_email}}"},
            ],
            "model": {"name": "claude"},
            "provider": {"name": "Anthropic"},
        }
        variables = {"user": {"name": "Ada"}, "caller_email": "ada@example.com"}
        await _drain(
            await handler.stream(
                config,
                "ask {{caller_email}}",
                None,
                variables,
                [{"role": "user", "content": "earlier {{caller_email}}"}],
            )
        )
        posted = script.messages[0]
        assert posted["system"] == "Hello Ada {{missing}}"
        assert posted["parts"][0]["text"] == (
            "User: Assist ada@example.com\n\n"
            "User: earlier {{caller_email}}\n\n"
            "User: ask {{caller_email}}"
        )
        recorded = tracer.spans[0].attributes["gen_ai.system_instructions"]
        assert "Hello Ada {{missing}}" in recorded
        assert "{{user.name}}" not in recorded

        bare, bare_script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )
        await _drain(
            await bare.stream(
                {
                    "instructions": "Hello {{caller_email}}",
                    "model": {"name": "claude"},
                    "provider": {"name": "Acme"},
                },
                "hi",
                None,
                None,
            )
        )
        assert bare_script.messages[0]["system"] == "Hello {{caller_email}}"

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
        assert len(set(script.pids)) == 2
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
        result = await opencode_agents(
            "flag",
            "Hello",
            CONTEXT,
            history=[{"role": "user", "content": "earlier"}],
            **script.handler_args(),
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
        script = Script([], mode="open", message_status=400)
        handler = create_opencode_handler(**script.handler_args())

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
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The loopback socket cannot tell a cancelled read from the close that
        # unblocks it. Record the client: ``_next_line`` must finish before
        # GET /event is closed.
        import httpx

        from launchdarkly_ai_opencode_agents import server as server_mod
        from launchdarkly_ai_opencode_agents.server import OpenCodeHTTPError

        bus = Script([], mode="close_after_read")
        handler = create_opencode_handler(**bus.handler_args())
        order: list[str] = []
        real_next_line = server_mod._next_line

        async def _recording_next_line(lines: Any) -> str | None:
            try:
                return await real_next_line(lines)
            finally:
                order.append("read_finished")

        monkeypatch.setattr(server_mod, "_next_line", _recording_next_line)
        real_aclose = httpx.Response.aclose

        async def _recording_aclose(response: httpx.Response) -> None:
            request = response.request
            if request.method == "GET" and request.url.path == "/event":
                order.append("response_closed")
            await real_aclose(response)

        monkeypatch.setattr(httpx.Response, "aclose", _recording_aclose)

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

        with pytest.raises(OpenCodeHTTPError, match="400"):
            await asyncio.wait_for(_run(), timeout=2)
        close_at = order.index("response_closed")
        assert any(name == "read_finished" for name in order)
        assert all(
            index < close_at
            for index, name in enumerate(order)
            if name == "read_finished"
        )

    async def test_breaking_aborts_without_waiting_for_gc(self) -> None:
        script = Script([], mode="open")
        handler = create_opencode_handler(**script.handler_args())

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
        script = Script([], mode="open", delay_tail=0.2)
        handler = create_opencode_handler(**script.handler_args(), capture_content=True)
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


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class TestProcess:
    async def test_a_fake_opencode_is_started_spoken_to_and_stopped(self) -> None:
        handler, script = playing(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
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
        assert seen[-1]["output"] == "Hi"
        records = script._records()
        assert len(records) == 1
        argv = records[0]["argv"]
        assert argv[argv.index("serve") + 1] == "--port"
        assert argv[argv.index("--hostname") + 1] == "127.0.0.1"
        assert "--print-logs" not in argv
        assert "--log-level" not in argv
        assert records[0]["cwd"] == str(script.cwd.resolve())
        assert records[0]["config"] == str(script.config.resolve())
        paths = [item["path"] for item in script.requests]
        assert "/event" in paths
        assert "/session" in paths
        assert any(path.endswith("/message") for path in paths)
        assert script.pids
        assert all(not _alive(pid) for pid in script.pids)

    async def test_closing_the_stream_stops_the_process(self) -> None:
        script = Script([], mode="open")
        handler = create_opencode_handler(**script.handler_args())

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

        await asyncio.wait_for(_run(), timeout=5)
        assert script.aborts
        assert script.pids
        assert all(not _alive(pid) for pid in script.pids)

    async def test_the_process_is_dead_when_the_context_returns(self) -> None:
        script = Script([], mode="open")
        handler = create_opencode_handler(**script.handler_args())
        with pytest.raises(TimeoutError):
            async with handler:
                async with asyncio.timeout(0.25):
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
                                await asyncio.sleep(2)
        assert script.pids
        assert all(not _alive(pid) for pid in script.pids)

    async def test_boot_is_outside_the_caller_timeout(self) -> None:
        script = Script(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ],
            delay_ready=0.6,
        )
        handler = create_opencode_handler(**script.handler_args())
        async with handler:
            async with asyncio.timeout(0.2):
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
        assert seen[-1]["output"] == "Hi"

    async def test_two_concurrent_contexts_leave_no_process(self) -> None:
        script = Script([])
        handler = create_opencode_handler(**script.handler_args())

        async def user() -> None:
            async with handler:
                await asyncio.sleep(0.1)

        await asyncio.gather(user(), user())
        assert script.pids
        assert len(script.pids) == 1
        assert all(not _alive(pid) for pid in script.pids)

    async def test_a_generated_password_replaces_one_from_the_environment(
        self,
    ) -> None:
        script = Script(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )
        args = script.handler_args()
        args["env"] = {**args["env"], "OPENCODE_SERVER_PASSWORD": "inherited-secret"}
        handler = create_opencode_handler(**args)
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
        assert seen[-1]["output"] == "Hi"
        records = script._records()
        assert len(records) == 1
        password = records[0]["password"]
        assert isinstance(password, str) and password
        assert password != "inherited-secret"

    async def test_an_inherited_username_is_replaced(self) -> None:
        script = Script(
            [
                assistant_open("m1"),
                text_delta("m1", "p1", "Hi"),
                assistant_completed("m1"),
            ]
        )
        args = script.handler_args()
        args["env"] = {**args["env"], "OPENCODE_SERVER_USERNAME": "someone-else"}
        handler = create_opencode_handler(**args)
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
        assert seen[-1]["output"] == "Hi"
        records = script._records()
        assert len(records) == 1
        assert records[0]["username"] == "opencode"

    async def test_a_startup_failure_includes_the_stderr_tail(self) -> None:
        prefix = "HEAD-MARKER"
        tail = "TAIL-MARKER"
        script = Script([], exit_stderr=prefix + ("x" * 4000) + tail)
        handler = create_opencode_handler(**script.handler_args())
        with pytest.raises(RuntimeError) as caught:
            async with handler:
                pass
        message = str(caught.value)
        assert "exited during startup" in message
        assert tail in message
        assert prefix not in message

    def test_health_requires_a_healthy_body(self) -> None:
        assert _is_healthy(httpx.Response(200, json={"healthy": True, "version": "1"}))
        assert not _is_healthy(httpx.Response(404, json={"healthy": True}))
        assert not _is_healthy(httpx.Response(200, json={}))
        assert not _is_healthy(httpx.Response(200, json={"healthy": False}))
        assert not _is_healthy(httpx.Response(200, content=b"ok"))


async def _drain(stream: Any) -> None:
    async for _event in stream:
        pass


#: Every field of OpenCode's ``PromptInput`` (the message route's body), in both spellings.
_PROMPT_INPUT_FIELDS = frozenset(
    {
        "sessionID",
        "session_id",
        "messageID",
        "message_id",
        "model",
        "agent",
        "noReply",
        "no_reply",
        "tools",
        "format",
        "system",
        "variant",
        "parts",
    }
)


class TestForwardsNoModelParameters:
    """Probes ``invoke`` and ``stream`` one key at a time: no ``model.parameters`` key changes the
    message body the handler posts, on either path."""

    @staticmethod
    def _config(parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "instructions": "Be brief.",
            "model": {"name": "m", "parameters": parameters},
            "provider": {"name": "Acme"},
        }

    @staticmethod
    def _events() -> list[dict[str, Any]]:
        return [
            assistant_open("m1"),
            text_delta("m1", "p1", "Hi"),
            assistant_completed("m1"),
        ]

    async def test_invoke(self) -> None:
        async def call(parameters: dict[str, Any]) -> object:
            handler, script = playing(self._events())
            await handler(self._config(parameters), "q", None, None)
            return script.messages[0]

        candidates = candidate_keys(_PROMPT_INPUT_FIELDS, OPENCODE_AGENTS)
        assert await probe_forwarded_keys(candidates, call) == OPENCODE_AGENTS

    async def test_stream(self) -> None:
        async def call(parameters: dict[str, Any]) -> object:
            handler, script = playing(self._events())
            async for _event in await handler.stream(
                self._config(parameters), "q", None, None
            ):
                pass
            return script.messages[0]

        candidates = candidate_keys(_PROMPT_INPUT_FIELDS, OPENCODE_AGENTS)
        assert await probe_forwarded_keys(candidates, call) == OPENCODE_AGENTS

    async def test_never_forwarded_keys_do_not_reach_the_body(self) -> None:
        handler, script = playing(self._events())
        await handler(self._config(dict(NEVER_FORWARDED_BAG)), "q", None, None)
        assert not find_leaks(script.messages[0])
