"""Graph convenience wrapper for claude-agents."""

from __future__ import annotations

import os
from typing import Any

from claude_agent_sdk import SettingSource

from launchdarkly_ai_claude_agents.handler import _create_claude_agents_handler
from launchdarkly_ai_server import _graph, report_usage

from ._version import PACKAGE_NAME, __version__


def claude_graph(
    key: str,
    *,
    cwd: str | os.PathLike[str] | None = None,
    setting_sources: list[SettingSource] | None = None,
    **options: Any,
) -> Any:
    """
    Runs an agent graph with the Claude agent handler pre-bound.

    Equivalent to ``graph(key, handlers=[create_claude_agents_handler(cwd=cwd,
    setting_sources=setting_sources)], **options)``. *cwd* and *setting_sources* build the
    handler, with the same defaults, and are not passed on to ``graph()``.
    Use the base ``graph()`` directly for multi-provider graphs.
    """
    report_usage("claude-agents.claudeGraph", PACKAGE_NAME, __version__)
    handler = _create_claude_agents_handler(cwd=cwd, setting_sources=setting_sources)
    return _graph(key, handlers=[handler], **options)
