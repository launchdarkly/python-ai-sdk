"""
Example: to_lang_graph() — framework-native graph runner via the LangGraph adapter.

Resolves a LaunchDarkly agent graph flag and executes it using LangGraph's
StateGraph primitives instead of the SDK's model-driven router.  This example
specifically exercises the annotation-resolution path that unit tests mock away —
`add_messages` must be a module-level import in `native_graph.py`, otherwise
LangGraph raises `NameError` when constructing `StateGraph(WorkflowState)`.

Usage (via main.py):
    python main.py native-graph-langchain <flag-key> "<user input>"
"""

from __future__ import annotations

import json

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, write_output
from launchdarkly_ai_langchain_agents import to_lang_graph
from launchdarkly_ai_server import global_registry, resolve_graph


async def run(key: str, user_input: str) -> None:
    context = new_context()

    response = await to_lang_graph(
        resolve_graph(key, context=context, registry=global_registry),
        {"context": context},
    ).invoke(user_input, {"user_id": "user-123"})

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
