"""Graph convenience wrapper for Bedrock agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import _graph, report_usage

from ._version import PACKAGE_NAME, __version__
from .handler import _create_bedrock_agents_handler


def bedrock_graph(key: str, **options: Any) -> Any:
    """Create a generic graph with exactly one Bedrock agent handler."""
    report_usage("bedrock-agents.bedrockGraph", PACKAGE_NAME, __version__)
    options.pop("handlers", None)
    return _graph(key, handlers=[_create_bedrock_agents_handler()], **options)
