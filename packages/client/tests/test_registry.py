"""
Tests for §3.6 Registry and compose, §3.7 resolve_handlers and resolve_tools.
Reference: TESTING.md §3.6–3.7
"""

import logging
from unittest.mock import AsyncMock

import pytest

from launchdarkly_ai_server import (
    ProviderHandler,
    Registry,
    compose,
    resolve_handlers,
    resolve_tools,
)


def _make_handler(
    provides_for: tuple[str, str] | None = ("Test", "messages"),
) -> ProviderHandler:
    async def fn(config, user_input, tool_handlers, variables):  # type: ignore[override]
        return {"output": "ok"}

    return ProviderHandler(fn=fn, provides_for=provides_for)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §3.6 Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_initially_empty(self) -> None:
        r = Registry()
        assert r.handlers == []
        assert r.tools == {}

    def test_constructor_accepts_initial_config(self) -> None:
        h = _make_handler()
        tool_fn = AsyncMock()
        r = Registry(handlers=[h], tools={"t": tool_fn})
        assert len(r.handlers) == 1
        assert "t" in r.tools

    def test_register_appends_handlers(self) -> None:
        r = Registry()
        h = _make_handler(("Provider1", "messages"))
        r.register(handlers=[h])
        assert len(r.handlers) == 1

    def test_duplicate_handler_warns_and_replaces(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        r = Registry()
        h1 = _make_handler(("Provider1", "messages"))
        h2 = _make_handler(("Provider1", "messages"))
        with caplog.at_level(logging.WARNING):
            r.register(handlers=[h1])
            r.register(handlers=[h2])
        assert len(r.handlers) == 1
        assert r.handlers[0] is h2
        assert any("already registered" in m for m in caplog.messages)

    def test_handler_without_provides_for_always_appended(self) -> None:
        r = Registry()
        h1 = _make_handler(None)
        h2 = _make_handler(None)
        r.register(handlers=[h1])
        r.register(handlers=[h2])
        assert len(r.handlers) == 2

    def test_register_appends_tools(self) -> None:
        r = Registry()
        r.register(tools={"my_tool": AsyncMock()})
        assert "my_tool" in r.tools

    def test_duplicate_tool_warns_and_replaces(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        r = Registry()
        fn1, fn2 = AsyncMock(), AsyncMock()
        with caplog.at_level(logging.WARNING):
            r.register(tools={"t": fn1})
            r.register(tools={"t": fn2})
        assert r.tools["t"] is fn2
        assert any("already registered" in m for m in caplog.messages)

    def test_register_is_additive(self) -> None:
        r = Registry()
        r.register(handlers=[_make_handler(("A", "messages"))])
        r.register(tools={"x": AsyncMock()})
        r.register(handlers=[_make_handler(("B", "agent"))])
        assert len(r.handlers) == 2
        assert "x" in r.tools


# ---------------------------------------------------------------------------
# §3.6 compose
# ---------------------------------------------------------------------------


class TestCompose:
    def test_returns_new_registry(self) -> None:
        a, b = Registry(), Registry()
        result = compose(a, b)
        assert result is not a
        assert result is not b

    def test_b_overrides_a_on_handler_conflict(self) -> None:
        h_a = _make_handler(("P", "messages"))
        h_b = _make_handler(("P", "messages"))
        a = Registry(handlers=[h_a])
        b = Registry(handlers=[h_b])
        result = compose(a, b)
        assert len(result.handlers) == 1
        assert result.handlers[0] is h_b

    def test_b_overrides_a_on_tool_conflict(self) -> None:
        fn_a, fn_b = AsyncMock(), AsyncMock()
        a = Registry(tools={"t": fn_a})
        b = Registry(tools={"t": fn_b})
        result = compose(a, b)
        assert result.tools["t"] is fn_b

    def test_non_conflicting_entries_are_merged(self) -> None:
        h_a = _make_handler(("A", "messages"))
        h_b = _make_handler(("B", "agent"))
        fn_a, fn_b = AsyncMock(), AsyncMock()
        a = Registry(handlers=[h_a], tools={"tool_a": fn_a})
        b = Registry(handlers=[h_b], tools={"tool_b": fn_b})
        result = compose(a, b)
        assert len(result.handlers) == 2
        assert "tool_a" in result.tools
        assert "tool_b" in result.tools


# ---------------------------------------------------------------------------
# §3.7 resolve_handlers
# ---------------------------------------------------------------------------


class TestResolveHandlers:
    def test_no_registry_local_handlers_present(self) -> None:
        local = [_make_handler()]
        result = resolve_handlers(None, local)
        assert result == local

    def test_registry_present_no_local_handlers(self) -> None:
        h = _make_handler()
        r = Registry(handlers=[h])
        result = resolve_handlers(r, None)
        assert result == [h]

    def test_both_present_local_precedes_registry(self) -> None:
        local_h = _make_handler(("Local", "messages"))
        reg_h = _make_handler(("Reg", "messages"))
        r = Registry(handlers=[reg_h])
        result = resolve_handlers(r, [local_h])
        assert result is not None
        assert result[0] is local_h
        assert result[1] is reg_h

    def test_registry_present_but_empty_no_local(self) -> None:
        r = Registry()
        result = resolve_handlers(r, None)
        assert result is None


# ---------------------------------------------------------------------------
# §3.7 resolve_tools
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_no_registry(self) -> None:
        local = {"t": AsyncMock()}
        result = resolve_tools(None, local)
        assert result == local

    def test_registry_present_no_local_tools(self) -> None:
        fn = AsyncMock()
        r = Registry(tools={"t": fn})
        result = resolve_tools(r, None)
        assert result == {"t": fn}

    def test_both_present_local_wins(self) -> None:
        fn_reg = AsyncMock()
        fn_local = AsyncMock()
        r = Registry(tools={"t": fn_reg})
        result = resolve_tools(r, {"t": fn_local})
        assert result is not None
        assert result["t"] is fn_local

    def test_registry_present_but_empty_no_local(self) -> None:
        r = Registry()
        result = resolve_tools(r, None)
        assert result is None
