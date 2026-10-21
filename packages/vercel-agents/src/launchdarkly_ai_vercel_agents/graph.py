from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import _graph, report_usage

from ._version import PACKAGE_NAME, __version__
from .handler import ModelSource, _create_vercel_agents_handler


def vercel_graph(
    key: str,
    model: ModelSource | None = None,
    *,
    capture_content: bool | None = None,
    **options: Any,
) -> Any:
    """Create a LaunchDarkly graph with one wildcard Vercel handler."""
    report_usage("vercel-agents.vercelGraph", PACKAGE_NAME, __version__)
    options.pop("handlers", None)
    handler = _create_vercel_agents_handler(
        model=model, capture_content=capture_content
    )
    return _graph(key, handlers=[handler], **options)
