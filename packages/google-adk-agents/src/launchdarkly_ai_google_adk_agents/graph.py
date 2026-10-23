"""Graph convenience wrapper for the Google ADK agents handler."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import _graph, report_usage

from ._version import PACKAGE_NAME, __version__
from .handler import _create_google_adk_agents_handler


def google_adk_graph(
    key: str,
    *,
    api_key: str | None = None,
    use_vertexai: bool = False,
    project: str | None = None,
    location: str | None = None,
    model: Any = None,
    capture_content: bool | None = None,
    **options: Any,
) -> Any:
    """Runs an agent graph with one wildcard ADK handler pre-bound.

    ``handlers`` passed by the caller is ignored. Auth and model options go to the
    handler factory, not to ``graph()``.
    """
    report_usage("google-adk-agents.googleAdkGraph", PACKAGE_NAME, __version__)
    options.pop("handlers", None)
    return _graph(
        key,
        handlers=[
            _create_google_adk_agents_handler(
                api_key=api_key,
                use_vertexai=use_vertexai,
                project=project,
                location=location,
                model=model,
                capture_content=capture_content,
            )
        ],
        **options,
    )
