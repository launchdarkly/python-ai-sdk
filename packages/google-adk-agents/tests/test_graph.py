"""Graph convenience wrapper. Reference: TESTING.md §2.1 and §2.x.4."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from launchdarkly_ai_google_adk_agents.graph import google_adk_graph


class TestGoogleAdkGraph:
    def test_forwards_the_flag_key(self) -> None:
        with patch("launchdarkly_ai_google_adk_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            google_adk_graph("my-flag")
            assert mock_graph.call_args.args[0] == "my-flag"

    def test_prewires_one_wildcard_agent_handler(self) -> None:
        with patch("launchdarkly_ai_google_adk_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            google_adk_graph("my-flag")
            handlers = mock_graph.call_args.kwargs["handlers"]
            assert len(handlers) == 1
            assert handlers[0].provides_for == ("*", "agent")

    def test_caller_cannot_replace_handlers(self) -> None:
        with patch("launchdarkly_ai_google_adk_agents.graph.graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            google_adk_graph(
                "my-flag", handlers=["nope"], tool_handlers={"lookup": lambda: None}
            )
            kwargs = mock_graph.call_args.kwargs
            assert len(kwargs["handlers"]) == 1
            assert kwargs["handlers"][0].provides_for == ("*", "agent")
            assert "lookup" in kwargs["tool_handlers"]

    def test_vertex_and_model_options_reach_the_handler(self) -> None:
        model = object()
        with (
            patch("launchdarkly_ai_google_adk_agents.graph.graph") as mock_graph,
            patch(
                "launchdarkly_ai_google_adk_agents.graph.create_google_adk_agents_handler"
            ) as factory,
        ):
            factory.return_value = MagicMock(provides_for=("*", "agent"))
            mock_graph.return_value = MagicMock()
            google_adk_graph(
                "my-flag",
                use_vertexai=True,
                project="p",
                location="us-central1",
                model=model,
                capture_content=True,
            )
            factory.assert_called_once_with(
                use_vertexai=True,
                project="p",
                location="us-central1",
                model=model,
                capture_content=True,
            )
