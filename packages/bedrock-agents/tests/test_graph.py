"""Contract tests for the Bedrock Strands graph convenience wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from launchdarkly_ai_bedrock_agents.graph import bedrock_graph


class TestBedrockGraph:
    def test_passes_key_and_options_to_graph(self) -> None:
        tools = {"lookup": MagicMock()}
        registry = MagicMock()

        with patch("launchdarkly_ai_bedrock_agents.graph.graph") as graph_mock:
            expected = MagicMock()
            graph_mock.return_value = expected
            actual = bedrock_graph(
                "bedrock-graph", tool_handlers=tools, registry=registry
            )

        assert actual is expected
        graph_mock.assert_called_once()
        assert graph_mock.call_args.args[0] == "bedrock-graph"
        assert graph_mock.call_args.kwargs["tool_handlers"] is tools
        assert graph_mock.call_args.kwargs["registry"] is registry

    def test_prebinds_exactly_one_bedrock_agent_handler(self) -> None:
        with patch("launchdarkly_ai_bedrock_agents.graph.graph") as graph_mock:
            bedrock_graph("bedrock-graph")

        handlers = graph_mock.call_args.kwargs["handlers"]
        assert len(handlers) == 1
        assert handlers[0].provides_for == ("Bedrock", "agent")

    def test_caller_cannot_override_prebound_handler(self) -> None:
        wrong_handler = MagicMock()

        with patch("launchdarkly_ai_bedrock_agents.graph.graph") as graph_mock:
            bedrock_graph("bedrock-graph", handlers=[wrong_handler])

        handlers = graph_mock.call_args.kwargs["handlers"]
        assert len(handlers) == 1
        assert handlers != [wrong_handler]
