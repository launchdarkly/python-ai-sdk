"""The OpenCode handler honors AI Config tools (Option C of the tool-contract design).

A variation's ``tools``, filtered to the keys the caller also passed in
``tool_handlers``, is the catalog OpenCode is allowed to show the model for
one call. A ``NativeTool`` is allowed by name. A plain callable is served
from this call's own loopback MCP bridge, registered on the running
``opencode serve`` with ``POST /mcp`` and torn down with
``POST /mcp/{name}/disconnect``.

``fake_opencode.py`` never actually calls back into a registered MCP server:
it only records the ``POST /mcp`` body and plays back a scripted event list.
These tests verify the handler's side of that contract (the ruleset sent,
the registration body, the teardown call, and the spans and tracking a
scripted tool part produces) against the fake. The bridge's own MCP
protocol behavior is tested directly, with no fake opencode involved, in
``test_tool_bridge.py``; nothing here has been run against a real OpenCode
MCP client.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from opentelemetry.trace import StatusCode as SpanStatusCode

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_opencode_agents import (
    OpenCodeApplyPatch,
    OpenCodeBash,
    OpenCodeEdit,
    OpenCodeSkill,
    OpenCodeWebFetch,
    OpenCodeWrite,
    create_opencode_handler,
)
from launchdarkly_ai_opencode_agents.handler import _build_tool_ruleset, _plan_tools
from launchdarkly_ai_opencode_agents.server import OpenCodeHTTPError
from launchdarkly_ai_server import NativeTool, wrap_tool_handlers

CONTEXT = {"kind": "user", "key": "user-1"}
_TRACK_DATA: dict[str, Any] = {
    "runId": "run-1",
    "configKey": "cfg",
    "variationKey": "v1",
    "version": 1,
    "modelName": "m",
    "providerName": "p",
}


def _wrapped(handlers: dict[str, Any]) -> dict[str, Any]:
    """``tool_handlers`` reach a handler already wrapped by ``tracking.py`` --
    a ``NativeTool`` becomes a tracking stub carrying ``NATIVE_TOOL_KEY``, and
    a plain callable becomes an async wrapper. Calling the handler's
    ``.stream()`` directly, as these tests do, bypasses that layer, so each
    test wraps its own map the same way ``execute_and_stream`` would.
    """
    return wrap_tool_handlers(handlers, CONTEXT, _TRACK_DATA)


# Duplicated from test_handler.py rather than imported: this directory has no
# `__init__.py`, pytest collects it with `--import-mode=importlib`, and a
# root-level `tests/` package already exists for the repo's own shared test
# utilities, so `tests.test_handler` does not resolve to this package's
# `test_handler.py`. `fake_opencode.py` is only ever invoked as a subprocess
# (never imported), so it is reused as-is, by path.

_FAKE = Path(__file__).with_name("fake_opencode.py")

VARIATION: dict[str, Any] = {
    "model": {"name": "test-model"},
    "provider": {"name": "Acme"},
    "instructions": "Be brief.",
    "_ldMeta": {"enabled": True, "variationKey": "v1", "version": 1, "mode": "agent"},
}


def bus(kind: str, **properties: Any) -> dict[str, Any]:
    return {"type": kind, "properties": properties}


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
    message_id: str, *, tokens: dict[str, Any] | None = None
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
        mcp_status: str = "connected",
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
            json.dumps({"mode": mode, "events": events, "mcp_status": mcp_status}),
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
    def pids(self) -> list[int]:
        return [int(record["pid"]) for record in self._records() if "pid" in record]

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


async def _drain(stream: Any) -> None:
    async for _event in stream:
        pass


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


_CONFIG_WITH_TOOLS: dict[str, Any] = {
    "instructions": "Be brief.",
    "model": {"name": "m"},
    "provider": {"name": "Acme"},
    "tools": {
        "run_shell": {
            "name": "run_shell",
            "type": "function",
            "description": "Run a shell command",
        },
        "fetch_url": {
            "name": "fetch_url",
            "type": "function",
            "description": "Fetch a URL",
        },
        "get_weather": {
            "name": "get_weather",
            "type": "function",
            "description": "Look up the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
        "unused": {
            "name": "unused",
            "type": "function",
            "description": "No handler is registered for this key",
        },
    },
}


class TestPlanTools:
    """``_plan_tools`` and ``_build_tool_ruleset`` as pure functions."""

    def test_offered_is_the_intersection_of_config_tools_and_tool_handlers(
        self,
    ) -> None:
        plan = _plan_tools(
            _CONFIG_WITH_TOOLS,
            _wrapped(
                {
                    "run_shell": OpenCodeBash,
                    # Not a declared config tool: must not be offered.
                    "extra": OpenCodeWebFetch,
                }
            ),
        )
        assert plan.offered_names == ["bash"]
        assert plan.callables == {}

    def test_a_config_tool_with_no_handler_is_not_offered(self) -> None:
        plan = _plan_tools(_CONFIG_WITH_TOOLS, _wrapped({"run_shell": OpenCodeBash}))
        assert "unused" not in plan.tool_display_names.values()
        assert "fetch_url" not in plan.callables
        assert "get_weather" not in plan.callables

    def test_native_tool_is_offered_by_its_provider_name(self) -> None:
        plan = _plan_tools(_CONFIG_WITH_TOOLS, _wrapped({"run_shell": OpenCodeBash}))
        assert plan.offered_names == ["bash"]
        assert plan.tool_display_names["bash"] == "run_shell"
        assert "bash" in plan.native_stubs

    def test_a_custom_named_native_tool_keeps_its_own_provider_name(self) -> None:
        custom = NativeTool("observability_query_logs")
        plan = _plan_tools(
            {**_CONFIG_WITH_TOOLS, "tools": {"query_logs": {"name": "query_logs"}}},
            _wrapped({"query_logs": custom}),
        )
        assert plan.offered_names == ["observability_query_logs"]
        assert plan.tool_display_names["observability_query_logs"] == "query_logs"

    def test_a_plain_callable_becomes_a_pending_bridge_entry(self) -> None:
        async def get_weather(args: dict[str, Any]) -> str:
            return f"sunny in {args['city']}"

        plan = _plan_tools(_CONFIG_WITH_TOOLS, _wrapped({"get_weather": get_weather}))
        assert (
            plan.offered_names == []
        )  # not named yet: the bridge has no name until it starts
        assert set(plan.callables) == {"get_weather"}
        entry = plan.callables["get_weather"]
        assert entry.description == "Look up the weather"
        assert entry.parameters["properties"]["city"]["type"] == "string"

    def test_no_tools_at_all_plans_nothing(self) -> None:
        plan = _plan_tools(_CONFIG_WITH_TOOLS, None)
        assert plan.offered_names == []
        assert plan.callables == {}
        plan2 = _plan_tools(
            {"instructions": "hi"}, _wrapped({"run_shell": OpenCodeBash})
        )
        assert plan2.offered_names == []

    def test_ruleset_always_denies_before_allowing(self) -> None:
        ruleset = _build_tool_ruleset(["bash", "ld_abc_get_weather"])
        assert ruleset[0] == {"permission": "*", "action": "deny", "pattern": "*"}
        assert ruleset[1:] == [
            {"permission": "bash", "action": "allow", "pattern": "*"},
            {"permission": "ld_abc_get_weather", "action": "allow", "pattern": "*"},
        ]

    def test_an_empty_offered_set_still_denies_everything(self) -> None:
        assert _build_tool_ruleset([]) == [
            {"permission": "*", "action": "deny", "pattern": "*"}
        ]

    def test_write_and_apply_patch_are_the_edit_permission(self) -> None:
        assert _build_tool_ruleset(["write"]) == [
            {"permission": "*", "action": "deny", "pattern": "*"},
            {"permission": "edit", "action": "allow", "pattern": "*"},
        ]
        # All three tool ids are one grant, sent once.
        assert _build_tool_ruleset(["edit", "write", "apply_patch"]) == [
            {"permission": "*", "action": "deny", "pattern": "*"},
            {"permission": "edit", "action": "allow", "pattern": "*"},
        ]


class TestSessionPermission:
    """What the handler actually sends OpenCode, through the fake server."""

    async def test_no_tools_sends_the_deny_all_ruleset(self) -> None:
        handler, script = playing([assistant_open("m1"), assistant_completed("m1")])
        await _drain(await handler.stream({}, "hi", None, None))
        assert script.session_bodies[-1]["permission"] == [
            {"permission": "*", "action": "deny", "pattern": "*"}
        ]

    async def test_a_native_tool_is_allowed_by_its_provider_name(self) -> None:
        handler, script = playing([assistant_open("m1"), assistant_completed("m1")])
        await _drain(
            await handler.stream(
                _CONFIG_WITH_TOOLS, "hi", _wrapped({"run_shell": OpenCodeBash}), None
            )
        )
        permission = script.session_bodies[-1]["permission"]
        assert permission[0] == {"permission": "*", "action": "deny", "pattern": "*"}
        assert {"permission": "bash", "action": "allow", "pattern": "*"} in permission
        # Only the offered tool is allowed -- not every config tool.
        assert len(permission) == 2

    async def test_offering_only_write_sends_the_edit_permission(self) -> None:
        handler, script = playing([assistant_open("m1"), assistant_completed("m1")])
        config = {
            "instructions": "Be brief.",
            "model": {"name": "m"},
            "provider": {"name": "Acme"},
            "tools": {"save": {"name": "save", "type": "function"}},
        }
        await _drain(
            await handler.stream(config, "hi", _wrapped({"save": OpenCodeWrite}), None)
        )
        permission = script.session_bodies[-1]["permission"]
        assert permission == [
            {"permission": "*", "action": "deny", "pattern": "*"},
            {"permission": "edit", "action": "allow", "pattern": "*"},
        ]

    @pytest.mark.parametrize("sentinel", [OpenCodeEdit, OpenCodeApplyPatch])
    async def test_offering_only_edit_or_apply_patch_sends_the_edit_permission(
        self, sentinel: Any
    ) -> None:
        handler, script = playing([assistant_open("m1"), assistant_completed("m1")])
        config = {
            "instructions": "Be brief.",
            "model": {"name": "m"},
            "provider": {"name": "Acme"},
            "tools": {"change": {"name": "change", "type": "function"}},
        }
        await _drain(
            await handler.stream(config, "hi", _wrapped({"change": sentinel}), None)
        )
        assert script.session_bodies[-1]["permission"] == [
            {"permission": "*", "action": "deny", "pattern": "*"},
            {"permission": "edit", "action": "allow", "pattern": "*"},
        ]

    async def test_skill_is_denied_unless_it_is_offered(self) -> None:
        handler, script = playing([assistant_open("m1"), assistant_completed("m1")])
        config = {
            "instructions": "Be brief.",
            "model": {"name": "m"},
            "provider": {"name": "Acme"},
            "tools": {
                "run_shell": {"name": "run_shell", "type": "function"},
                "load_skill": {"name": "load_skill", "type": "function"},
            },
        }
        await _drain(
            await handler.stream(
                config, "hi", _wrapped({"run_shell": OpenCodeBash}), None
            )
        )
        denied = script.session_bodies[-1]["permission"]
        assert {"permission": "skill", "action": "allow", "pattern": "*"} not in denied
        assert denied == [
            {"permission": "*", "action": "deny", "pattern": "*"},
            {"permission": "bash", "action": "allow", "pattern": "*"},
        ]

        await _drain(
            await handler.stream(
                config, "hi", _wrapped({"load_skill": OpenCodeSkill}), None
            )
        )
        assert script.session_bodies[-1]["permission"] == [
            {"permission": "*", "action": "deny", "pattern": "*"},
            {"permission": "skill", "action": "allow", "pattern": "*"},
        ]

    async def test_a_bridge_that_does_not_connect_fails_the_call(self) -> None:
        async def get_weather(args: dict[str, Any]) -> str:
            return f"sunny in {args['city']}"

        script = Script(
            [assistant_open("m1"), assistant_completed("m1")], mcp_status="failed"
        )
        handler = create_opencode_handler(**script.handler_args())
        with pytest.raises(OpenCodeHTTPError, match="its status is 'failed'"):
            await _drain(
                await handler.stream(
                    _CONFIG_WITH_TOOLS,
                    "hi",
                    _wrapped({"get_weather": get_weather}),
                    None,
                )
            )

    async def test_a_callable_tool_registers_an_mcp_bridge_then_disconnects(
        self,
    ) -> None:
        async def get_weather(args: dict[str, Any]) -> str:
            return f"sunny in {args['city']}"

        handler, script = playing([assistant_open("m1"), assistant_completed("m1")])
        await _drain(
            await handler.stream(
                _CONFIG_WITH_TOOLS, "hi", _wrapped({"get_weather": get_weather}), None
            )
        )
        assert len(script.mcp_calls) == 1
        call = script.mcp_calls[0]
        name = call["name"]
        assert name.startswith("ld_")
        assert call["config"]["type"] == "remote"
        assert call["config"]["url"].startswith("http://127.0.0.1:")
        assert call["config"]["url"].endswith(f"/mcp/{name.removeprefix('ld_')}")
        assert call["config"]["headers"]["Authorization"].startswith("Bearer ")
        assert call["config"]["oauth"] is False
        assert call["config"]["timeout"] >= 600_000

        permission = script.session_bodies[-1]["permission"]
        allowed = {
            rule["permission"] for rule in permission if rule["action"] == "allow"
        }
        assert allowed == {f"{name}_get_weather"}

        assert script.mcp_disconnects == [name]

    async def test_native_and_callable_tools_can_be_offered_together(self) -> None:
        async def get_weather(args: dict[str, Any]) -> str:
            return "sunny"

        handler, script = playing([assistant_open("m1"), assistant_completed("m1")])
        await _drain(
            await handler.stream(
                _CONFIG_WITH_TOOLS,
                "hi",
                _wrapped({"run_shell": OpenCodeBash, "get_weather": get_weather}),
                None,
            )
        )
        permission = script.session_bodies[-1]["permission"]
        allow_names = {
            rule["permission"] for rule in permission if rule["action"] == "allow"
        }
        assert "bash" in allow_names
        assert any(name.endswith("_get_weather") for name in allow_names)
        assert len(script.mcp_calls) == 1

    async def test_two_calls_on_one_shared_process_get_different_bridge_names(
        self,
    ) -> None:
        async def get_weather(args: dict[str, Any]) -> str:
            return "sunny"

        script = Script([assistant_open("m1"), assistant_completed("m1")])
        from launchdarkly_ai_opencode_agents import create_opencode_handler

        handler = create_opencode_handler(**script.handler_args())
        async with handler:
            await _drain(
                await handler.stream(
                    _CONFIG_WITH_TOOLS,
                    "hi",
                    _wrapped({"get_weather": get_weather}),
                    None,
                )
            )
            await _drain(
                await handler.stream(
                    _CONFIG_WITH_TOOLS,
                    "hi",
                    _wrapped({"get_weather": get_weather}),
                    None,
                )
            )
        names = [call["name"] for call in script.mcp_calls]
        assert len(names) == 2
        assert names[0] != names[1]
        assert script.mcp_disconnects == names


class TestToolBridgeEndToEnd:
    """Drives a registered bridge exactly as the fake session recorded it.

    The bridge is registered with ``POST /mcp`` before the session is even
    created, so pulling just the first yielded AG-UI event (the tool call
    the scripted session reports) is enough to know registration already
    happened and the bridge is still alive: the handler's ``finally`` that
    stops it has not run yet, because the generator is suspended mid-stream.
    """

    async def test_the_registered_bridge_answers_a_real_tools_call(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def get_weather(args: dict[str, Any]) -> str:
            return f"sunny in {args['city']}"

        handler, script = playing(
            [
                assistant_open("m1"),
                tool_part("c1", "running", name="bash", raw_input={"command": "ls"}),
            ]
        )
        agen = (
            await handler.stream(
                _CONFIG_WITH_TOOLS, "hi", _wrapped({"get_weather": get_weather}), None
            )
        ).__aiter__()
        first = await agen.__anext__()
        assert first["type"] == "TOOL_CALL_START"
        call = script.mcp_calls[0]
        async with httpx.AsyncClient() as http:
            response = await http.post(
                call["config"]["url"],
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    # The bridge's own tool name is the bare AI Config key, not
                    # the `<server>_<tool>` name OpenCode composes for the
                    # model: that composition is applied only to what the
                    # model sees, never sent back in `tools/call`.
                    "params": {"name": "get_weather", "arguments": {"city": "Oslo"}},
                },
                headers=call["config"]["headers"],
            )
        result = response.json()["result"]
        assert result["isError"] is False, result
        assert result["content"] == [{"type": "text", "text": "sunny in Oslo"}]
        await agen.aclose()


class TestToolSpans:
    """``chat`` and ``execute_tool`` spans, opened from the same bus events as the AG-UI stream."""

    @staticmethod
    def _tracer(monkeypatch: pytest.MonkeyPatch):
        from launchdarkly_ai_opencode_agents import spans as span_module

        class _Span:
            def __init__(self, name: str, context: Any = None) -> None:
                self.name = name
                self.context = context
                self.attributes: dict[str, Any] = {}
                self.status: Any = None
                self.status_description: str | None = None
                self.exceptions: list[BaseException] = []
                self.ended = 0

            def set_attribute(self, key: str, value: Any) -> None:
                self.attributes[key] = value

            def add_event(
                self, name: str, attributes: dict[str, Any] | None = None
            ) -> None:
                return None

            def set_status(self, code: Any, description: str | None = None) -> None:
                self.status = code
                self.status_description = description

            def record_exception(self, exc: BaseException) -> None:
                self.exceptions.append(exc)

            def end(self, end_time: int | None = None) -> None:
                self.ended += 1

        class _Tracer:
            def __init__(self) -> None:
                self.spans: list[_Span] = []

            def get_tracer(self, name: str) -> _Tracer:
                return self

            def start_span(
                self, name: str, context: Any = None, start_time: int | None = None
            ) -> _Span:
                span = _Span(name, context)
                self.spans.append(span)
                return span

            def set_span_in_context(self, span: _Span) -> Any:
                return ("context-of", span)

        tracer = _Tracer()
        monkeypatch.setattr(span_module, "trace", tracer)
        return tracer

    async def test_a_native_tool_call_opens_execute_tool_under_the_config_key(
        self, monkeypatch: pytest.MonkeyPatch, mock_ld_client: MagicMock
    ) -> None:
        tracer = self._tracer(monkeypatch)
        handler, _script = playing(
            [
                assistant_open("m1"),
                tool_part("c1", "running", name="bash", raw_input={"command": "ls"}),
                tool_part(
                    "c1",
                    "completed",
                    name="bash",
                    raw_input={"command": "ls"},
                    output="a.txt",
                ),
                assistant_completed("m1"),
            ]
        )
        await _drain(
            await handler.stream(
                _CONFIG_WITH_TOOLS, "hi", _wrapped({"run_shell": OpenCodeBash}), None
            )
        )
        tool_spans = [s for s in tracer.spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 1
        span = tool_spans[0]
        # Displayed under the AI Config key, not OpenCode's own tool id.
        assert span.name == "execute_tool run_shell"
        assert span.attributes["gen_ai.tool.name"] == "run_shell"
        assert span.attributes["gen_ai.tool.call.id"] == "c1"
        assert span.ended == 1
        assert span.status == SpanStatusCode.OK
        # The NativeTool's own tracking stub fired exactly once.
        tool_call_events = [
            call
            for call in mock_ld_client.track.call_args_list
            if call.args[0] == "$ld:ai:tool_call"
        ]
        assert len(tool_call_events) == 1
        # `toolKey` is the AI Config key `wrap_tool_handlers` was given, not
        # the provider's own tool name.
        assert tool_call_events[0].args[2]["toolKey"] == "run_shell"

    async def test_a_failed_tool_call_fails_its_span(
        self, monkeypatch: pytest.MonkeyPatch, mock_ld_client: MagicMock
    ) -> None:
        tracer = self._tracer(monkeypatch)
        handler, _script = playing(
            [
                assistant_open("m1"),
                tool_part("c1", "running", name="bash", raw_input={"command": "rm"}),
                tool_part("c1", "error", name="bash", raw_input={"command": "rm"}),
                assistant_completed("m1"),
            ]
        )
        await _drain(
            await handler.stream(
                _CONFIG_WITH_TOOLS, "hi", _wrapped({"run_shell": OpenCodeBash}), None
            )
        )
        tool_spans = [s for s in tracer.spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 1
        assert len(tool_spans[0].exceptions) == 1

    async def test_a_chat_span_opens_and_closes_per_assistant_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracer = self._tracer(monkeypatch)
        handler, _script = playing(
            [
                assistant_open("m1"),
                assistant_completed("m1", tokens={"input": 10, "output": 5}),
                assistant_open("m2"),
                assistant_completed("m2", tokens={"input": 3, "output": 1}),
            ]
        )
        await _drain(await handler.stream({}, "hi", None, None))
        chat_spans = [s for s in tracer.spans if s.name.startswith("chat ")]
        assert len(chat_spans) == 2
        assert chat_spans[0].attributes["gen_ai.usage.input_tokens"] == 10
        assert chat_spans[0].attributes["gen_ai.usage.output_tokens"] == 5
        assert chat_spans[1].attributes["gen_ai.usage.input_tokens"] == 3
        assert all(span.ended == 1 for span in chat_spans)

    async def test_a_second_completion_does_not_open_another_chat_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracer = self._tracer(monkeypatch)
        handler, _script = playing(
            [
                assistant_open("m1"),
                assistant_completed("m1", tokens={"input": 10, "output": 5}),
                assistant_completed("m1", tokens={"input": 10, "output": 5}),
            ]
        )
        await _drain(await handler.stream({}, "hi", None, None))
        chat_spans = [s for s in tracer.spans if s.name.startswith("chat ")]
        assert len(chat_spans) == 1
        assert chat_spans[0].ended == 1

    async def test_a_tokenless_second_completion_does_not_open_another_chat_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracer = self._tracer(monkeypatch)
        handler, _script = playing(
            [
                assistant_open("m1"),
                assistant_completed("m1"),
                assistant_completed("m1", tokens={"input": 4, "output": 2}),
            ]
        )
        seen = [event async for event in await handler.stream({}, "hi", None, None)]
        chat_spans = [s for s in tracer.spans if s.name.startswith("chat ")]
        assert len(chat_spans) == 1
        assert chat_spans[0].ended == 1
        # The closed span is not rewritten. Tokens on the later completion
        # still count once, on the run.
        assert seen[-1]["usage"] == {"input_tokens": 4, "output_tokens": 2}

    async def test_cancelling_the_stream_marks_open_spans_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracer = self._tracer(monkeypatch)
        script = Script([], mode="open")
        handler = create_opencode_handler(**script.handler_args())

        async def consume() -> None:
            async for _event in await handler.stream({}, "hi", None, None):
                pass

        task = asyncio.create_task(consume())
        for _ in range(50):
            if any(span.name.startswith("chat ") for span in tracer.spans):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("chat span never opened")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        chat_spans = [s for s in tracer.spans if s.name.startswith("chat ")]
        assert len(chat_spans) == 1
        assert chat_spans[0].attributes.get("launchdarkly.run.cancelled") is True
        assert "launchdarkly.stream.abandoned" not in chat_spans[0].attributes
        roots = [s for s in tracer.spans if s.name == "invoke_agent"]
        assert roots[0].attributes.get("launchdarkly.run.cancelled") is True
        assert "launchdarkly.stream.abandoned" not in roots[0].attributes

    async def test_abandoning_the_stream_marks_open_spans_abandoned_not_failed(
        self, monkeypatch: pytest.MonkeyPatch, mock_ld_client: MagicMock
    ) -> None:
        tracer = self._tracer(monkeypatch)
        handler, _script = playing(
            [
                assistant_open("m1"),
                tool_part("c1", "running", name="bash", raw_input={"command": "sleep"}),
                # No completion: the stream is abandoned while this call is open.
            ],
        )
        agen = (
            await handler.stream(
                _CONFIG_WITH_TOOLS, "hi", _wrapped({"run_shell": OpenCodeBash}), None
            )
        ).__aiter__()
        await agen.__anext__()
        await agen.aclose()
        tool_spans = [s for s in tracer.spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 1
        span = tool_spans[0]
        assert span.ended == 1
        assert span.attributes.get("launchdarkly.stream.abandoned") is True
        assert span.exceptions == []
