"""
Tests for §2.1 graph convenience wrapper (claude_graph).
Reference: TESTING.md §2.1
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage

from launchdarkly_ai_claude_agents.graph import claude_graph


class TestClaudeGraph:
    def test_graph_called_with_correct_key(self) -> None:
        with patch("launchdarkly_ai_claude_agents.graph._graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("my-flag-key")
            mock_graph.assert_called_once()
            assert mock_graph.call_args[0][0] == "my-flag-key"

    def test_handlers_pre_populated(self) -> None:
        with patch("launchdarkly_ai_claude_agents.graph._graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("key")
            kw = mock_graph.call_args[1]
            handlers = kw.get("handlers", [])
            assert len(handlers) == 1

    def test_user_supplied_options_forwarded(self) -> None:
        ctx = {"kind": "user", "key": "u1"}
        with patch("launchdarkly_ai_claude_agents.graph._graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("key", context=ctx)
            kw = mock_graph.call_args[1]
            assert kw.get("context") == ctx

    def test_user_cannot_override_handlers(self) -> None:
        with patch("launchdarkly_ai_claude_agents.graph._graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("key", extra_kwarg=42)
            kw = mock_graph.call_args[1]
            # The graph wrapper always injects its own handler; no user overwrite allowed
            assert "handlers" in kw
            assert len(kw["handlers"]) == 1


class TestClaudeGraphHostOptions:
    """``cwd`` and ``setting_sources`` build the bound handler and never reach ``graph()``."""

    @staticmethod
    async def _bound_query_options(monkeypatch: pytest.MonkeyPatch, **host: Any) -> Any:
        import launchdarkly_ai_claude_agents.handler as handler_mod

        captured: list[Any] = []

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            captured.append(kwargs["options"])
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s",
                usage={"input_tokens": 1, "output_tokens": 1},
                result="done",
            )

        monkeypatch.setattr(handler_mod, "query", _query)
        with patch("launchdarkly_ai_claude_agents.graph._graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("key", **host)
        graph_kwargs = mock_graph.call_args[1]
        assert "cwd" not in graph_kwargs
        assert "setting_sources" not in graph_kwargs
        (handler,) = graph_kwargs["handlers"]
        await handler(
            {
                "model": {"name": "claude-opus-4-5"},
                "provider": {"name": "Anthropic"},
                "instructions": "x",
            },
            "q",
            {},
        )
        assert len(captured) == 1
        return captured[0]

    async def test_binds_a_handler_with_no_host_settings_and_no_cwd_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opts = await self._bound_query_options(monkeypatch)
        assert opts.setting_sources == []
        assert opts.cwd is None

    async def test_forwards_cwd_and_setting_sources_to_the_bound_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opts = await self._bound_query_options(
            monkeypatch, cwd="/srv/app", setting_sources=["user"]
        )
        assert opts.cwd == "/srv/app"
        assert opts.setting_sources == ["user"]
