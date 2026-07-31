"""
Example: openai_agents() — OpenAI agents handler.

Usage (via main.py):
    python main.py openai-agents <flag-key> "<user input>"
"""

from __future__ import annotations

import json

from examples.tools import (
    fetch_launchdarkly_documentation,
    get_preferences,
    search_ld_documentation,
)
from examples.utils import new_context, write_output
from launchdarkly_ai_openai_agents import openai_agents


async def run(key: str, user_input: str) -> None:
    response = await openai_agents(
        key,
        user_input,
        new_context(),
        tool_handlers={
            "get-user-preferences": get_preferences,
            "search-ld-documentation": search_ld_documentation,
            "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
        },
    )

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
