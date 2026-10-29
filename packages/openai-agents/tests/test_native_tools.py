"""A NativeTool is never sent to the model as a function tool."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from launchdarkly_ai_server import NATIVE_TOOL_KEY, NativeTool

# The warn-once set is process-wide, so each test uses its own key.
PREFIX = "native-web-openai-agents"


def _config_tools(key: str) -> dict[str, Any]:
    return {
        "search": {"description": "d", "parameters": {"type": "object"}},
        key: {"description": "d", "parameters": {"type": "object"}},
    }


def _stub(native: NativeTool) -> Any:
    """The tracking stub ``wrap_tool_handlers`` makes from a NativeTool."""

    def stub() -> None:
        return None

    setattr(stub, NATIVE_TOOL_KEY, native)
    return stub


def _assert_skipped(
    key: str, build: Any, names: Any, caplog: pytest.LogCaptureFixture
) -> None:
    native = NativeTool("web_search")
    tools = _config_tools(key)
    with caplog.at_level(logging.WARNING):
        first = build(tools, {"search": lambda *a, **k: "x", key: _stub(native)})
        second = build(tools, {"search": lambda *a, **k: "x", key: native})
    assert names(first) == ["search"]
    assert names(second) == ["search"]
    assert sum(f'"{key}"' in r.getMessage() for r in caplog.records) == 1


def test_handler_never_sends_a_native_tool(caplog: pytest.LogCaptureFixture) -> None:
    from launchdarkly_ai_openai_agents.handler import _build_agent_tools

    _assert_skipped(
        PREFIX + "handler",
        lambda c, h: _build_agent_tools(c, h),
        lambda r: [t.name for t in r],
        caplog,
    )


def test_native_graph_never_sends_a_native_tool(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agents

    from launchdarkly_ai_openai_agents.native_graph import _build_node_tools

    # The adapter calls ``agents.tool(...)`` as a decorator factory; stub it as the other tests do.
    monkeypatch.setattr(
        agents,
        "tool",
        lambda **kw: lambda fn: SimpleNamespace(name=kw["name"]),
        raising=False,
    )

    _assert_skipped(
        PREFIX + "native_graph",
        lambda c, h: _build_node_tools(SimpleNamespace(config={"tools": c}), h),
        lambda r: [t.name for t in r],
        caplog,
    )
