"""Graph convenience wrapper for claude-agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_claude_agents.handler import create_claude_agents_handler
from launchdarkly_ai_server import graph


def claude_graph(key: str, **options: Any) -> Any:
    """
    Runs an agent graph with the Claude agent handler pre-bound.

    Equivalent to ``graph(key, handlers=[create_claude_agents_handler()], **options)``.
    Use the base ``graph()`` directly for multi-provider graphs.
    """
    return graph(key, handlers=[create_claude_agents_handler()], **options)
