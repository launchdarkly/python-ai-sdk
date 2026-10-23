"""Graph convenience wrapper for langchain-agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_langchain_agents.handler import _create_langchain_agents_handler
from launchdarkly_ai_server import _graph, report_usage

from ._version import PACKAGE_NAME, __version__


def langchain_graph(key: str, llm: Any = None, **options: Any) -> Any:
    """
    Runs an agent graph with the LangChain agent handler pre-bound.

    Equivalent to ``graph(key, handlers=[create_langchain_agents_handler(llm)], **options)``.
    Use the base ``graph()`` directly for multi-provider graphs.
    """
    report_usage("langchain-agents.langchainGraph", PACKAGE_NAME, __version__)
    return _graph(key, handlers=[_create_langchain_agents_handler(llm)], **options)
