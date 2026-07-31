"""
Example: graph() — model-driven agent graph execution.

Usage (via main.py):
    python main.py graph <flag-key> "<user input>"
"""

from __future__ import annotations

import json

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, write_output
from launchdarkly_ai_server import global_registry, graph


async def run(key: str, user_input: str) -> None:
    response = await graph(
        key,
        registry=global_registry,
    ).invoke(user_input, new_context(), {"user_id": "user-123"})

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
