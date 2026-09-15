"""Graph convenience wrapper for Bedrock agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import graph

from .handler import create_bedrock_agents_handler


def bedrock_graph(key: str, **options: Any) -> Any:
    """Create a generic graph with exactly one Bedrock agent handler."""
    options.pop("handlers", None)
    return graph(key, handlers=[create_bedrock_agents_handler()], **options)
