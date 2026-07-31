"""
Example: to_claude_agents() — framework-native graph runner via the Claude Agent SDK.

Resolves a LaunchDarkly agent graph flag and executes it using Claude's native
multi-agent primitives instead of the SDK's model-driven router.

Usage (via main.py):
    python main.py native-graph <flag-key> "<user input>"
"""

from __future__ import annotations

import json

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, write_output
from launchdarkly_ai_claude_agents import to_claude_agents
from launchdarkly_ai_server import global_registry, resolve_graph


async def run(key: str, user_input: str) -> None:
    context = new_context()

    response = await to_claude_agents(
        resolve_graph(key, context=context, registry=global_registry),
        {"context": context},
    ).invoke(user_input, {"user_id": "user-123"})

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
