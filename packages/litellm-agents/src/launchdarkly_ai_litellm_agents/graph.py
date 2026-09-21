"""Graph convenience wrapper for LiteLLM-backed Agents SDK runs."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import graph

from .handler import ModelFactory, create_litellm_agents_handler


def litellm_graph(
    key: str,
    *,
    model_factory: ModelFactory | None = None,
    capture_content: bool = False,
    **options: Any,
) -> Any:
    return graph(
        key,
        handlers=[
            create_litellm_agents_handler(
                model_factory=model_factory, capture_content=capture_content
            )
        ],
        **options,
    )
