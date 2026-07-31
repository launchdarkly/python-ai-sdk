"""
Tests for §2.1 graph convenience wrapper (claude_graph).
Reference: TESTING.md §2.1
"""

from unittest.mock import MagicMock, patch

from launchdarkly_ai_claude_agents.graph import claude_graph


class TestClaudeGraph:
    def test_graph_called_with_correct_key(self) -> None:
        with patch("launchdarkly_ai_claude_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("my-flag-key")
            mock_graph.assert_called_once()
            assert mock_graph.call_args[0][0] == "my-flag-key"

    def test_handlers_pre_populated(self) -> None:
        with patch("launchdarkly_ai_claude_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("key")
            kw = mock_graph.call_args[1]
            handlers = kw.get("handlers", [])
            assert len(handlers) == 1

    def test_user_supplied_options_forwarded(self) -> None:
        ctx = {"kind": "user", "key": "u1"}
        with patch("launchdarkly_ai_claude_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("key", context=ctx)
            kw = mock_graph.call_args[1]
            assert kw.get("context") == ctx

    def test_user_cannot_override_handlers(self) -> None:
        with patch("launchdarkly_ai_claude_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            claude_graph("key", extra_kwarg=42)
            kw = mock_graph.call_args[1]
            # The graph wrapper always injects its own handler; no user overwrite allowed
            assert "handlers" in kw
            assert len(kw["handlers"]) == 1
