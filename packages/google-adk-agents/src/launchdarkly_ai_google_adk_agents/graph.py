"""Graph convenience wrapper for the Google ADK agents handler."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import graph

from .handler import create_google_adk_agents_handler


def google_adk_graph(
    key: str,
    *,
    use_vertexai: bool = False,
    project: str | None = None,
    location: str | None = None,
    model: Any = None,
    capture_content: bool = False,
    **options: Any,
) -> Any:
    """Runs an agent graph with one wildcard ADK handler pre-bound.

    ``handlers`` passed by the caller is ignored. Auth and model options go to the
    handler factory, not to ``graph()``.
    """
    options.pop("handlers", None)
    return graph(
        key,
        handlers=[
            create_google_adk_agents_handler(
                use_vertexai=use_vertexai,
                project=project,
                location=location,
                model=model,
                capture_content=capture_content,
            )
        ],
        **options,
    )
