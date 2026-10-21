"""Graph convenience wrapper for claude-agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_claude_agents.handler import _create_claude_agents_handler
from launchdarkly_ai_server import _graph, report_usage

from ._version import PACKAGE_NAME, __version__


def claude_graph(key: str, **options: Any) -> Any:
    """
    Runs an agent graph with the Claude agent handler pre-bound.

    Equivalent to ``graph(key, handlers=[create_claude_agents_handler()], **options)``.
    Use the base ``graph()`` directly for multi-provider graphs.
    """
    report_usage("claude-agents.claudeGraph", PACKAGE_NAME, __version__)
    return _graph(key, handlers=[_create_claude_agents_handler()], **options)
