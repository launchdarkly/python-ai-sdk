"""Graph convenience wrapper for langchain-agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_langchain_agents.handler import create_langchain_agents_handler
from launchdarkly_ai_server import graph, report_usage, within_sdk


def langchain_graph(key: str, llm: Any = None, **options: Any) -> Any:
    """
    Runs an agent graph with the LangChain agent handler pre-bound.

    Equivalent to ``graph(key, handlers=[create_langchain_agents_handler(llm)], **options)``.
    Use the base ``graph()`` directly for multi-provider graphs.
    """
    report_usage("langchain-agents.langchainGraph")
    with within_sdk():
        return graph(key, handlers=[create_langchain_agents_handler(llm)], **options)
