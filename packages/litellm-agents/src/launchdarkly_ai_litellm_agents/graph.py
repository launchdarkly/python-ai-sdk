"""Graph convenience wrapper for LiteLLM-backed Agents SDK runs."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import _graph, report_usage

from ._version import PACKAGE_NAME, __version__
from .handler import ModelFactory, _create_litellm_agents_handler


def litellm_graph(
    key: str,
    *,
    model_factory: ModelFactory | None = None,
    capture_content: bool | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **options: Any,
) -> Any:
    report_usage("litellm-agents.litellmGraph", PACKAGE_NAME, __version__)
    return _graph(
        key,
        handlers=[
            _create_litellm_agents_handler(
                model_factory=model_factory,
                capture_content=capture_content,
                base_url=base_url,
                api_key=api_key,
            )
        ],
        **options,
    )
