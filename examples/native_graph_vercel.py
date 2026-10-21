"""Framework-native graph runner using Vercel ``ai.Agent`` instances."""

from __future__ import annotations

import json

from examples.tools import get_preferences, web_search
from examples.utils import new_context, write_output
from launchdarkly_ai_server import resolve_graph
from launchdarkly_ai_vercel_agents import to_vercel_agents


async def run(key: str, user_input: str) -> None:
    context = new_context()
    response = await to_vercel_agents(
        resolve_graph(key, context=context),
        {
            "context": context,
            "tool_handlers": {
                "user-preferences-lookup": get_preferences,
                "web-search-tool": web_search,
            },
        },
    ).invoke(user_input, {"user_id": "user-123"})

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
