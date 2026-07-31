"""Graph convenience wrapper for openai-agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_openai_agents.handler import create_openai_agent_handler
from launchdarkly_ai_server import graph


def openai_graph(key: str, **options: Any) -> Any:
    """
    Runs an agent graph with the OpenAI agent handler pre-bound.

    Equivalent to ``graph(key, handlers=[create_openai_agent_handler()], **options)``.
    Use the base ``graph()`` directly for multi-provider graphs.
    """
    return graph(key, handlers=[create_openai_agent_handler()], **options)
