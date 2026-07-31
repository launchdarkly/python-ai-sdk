"""
Tests for §2.1 graph convenience wrapper (openai_graph) and §2.x.4 full coverage.
Reference: TESTING.md §2.1, §2.x.4
"""

from unittest.mock import MagicMock, patch

from launchdarkly_ai_openai_agents.graph import openai_graph


class TestOpenAIGraph:
    def test_graph_called_with_correct_key(self) -> None:
        with patch("launchdarkly_ai_openai_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            openai_graph("my-flag-key")
            mock_graph.assert_called_once()
            assert mock_graph.call_args[0][0] == "my-flag-key"

    def test_handlers_pre_populated_with_openai_agent_handler(self) -> None:
        with patch("launchdarkly_ai_openai_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            openai_graph("key")
            kw = mock_graph.call_args[1]
            handlers = kw.get("handlers", [])
            assert len(handlers) == 1

    def test_user_supplied_options_forwarded(self) -> None:
        ctx = {"kind": "user", "key": "u1"}
        with patch("launchdarkly_ai_openai_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            openai_graph("key", context=ctx)
            kw = mock_graph.call_args[1]
            assert kw.get("context") == ctx

    def test_user_cannot_override_handlers(self) -> None:
        with patch("launchdarkly_ai_openai_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            openai_graph("key", extra=1)
            kw = mock_graph.call_args[1]
            assert "handlers" in kw
            assert len(kw["handlers"]) == 1
