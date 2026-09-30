from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import graph

from .handler import ModelSource, create_vercel_agents_handler


def vercel_graph(
    key: str,
    model: ModelSource | None = None,
    *,
    capture_content: bool = False,
    **options: Any,
) -> Any:
    """Create a LaunchDarkly graph with one wildcard Vercel handler."""
    options.pop("handlers", None)
    handler = create_vercel_agents_handler(model=model, capture_content=capture_content)
    return graph(key, handlers=[handler], **options)
