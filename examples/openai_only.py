"""
Example: OpenAI agents handler via the openai_agents() convenience function.

Usage (via main.py):
    python main.py openai-only <flag-key> "<user input>"
"""

from __future__ import annotations

import json

from examples.tools import (
    fetch_launchdarkly_documentation,
    get_preferences,
    search_ld_documentation,
    web_search,
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
            "fetch-ld-documentation": fetch_launchdarkly_documentation,
            "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
            "web-search": web_search,
            "web-search-tool": web_search,
        },
    )

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
