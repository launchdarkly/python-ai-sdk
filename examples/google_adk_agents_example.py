"""
Example: google_adk_agents() — Google ADK agents handler (wildcard provider).

Gemini Developer API is the default (`GOOGLE_API_KEY`, `GEMINI_API_KEY`, or
`GOOGLE_GENAI_API_KEY`). Vertex is opt-in: pass `use_vertexai=True` plus project
and location, or set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

Usage (via main.py):
    python main.py google-adk-agents <flag-key> "<user input>"
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
from launchdarkly_ai_google_adk_agents import google_adk_agents


async def run(key: str, user_input: str) -> None:
    response = await google_adk_agents(
        key,
        user_input,
        new_context(),
        tool_handlers={
            "get-user-preferences": get_preferences,
            "search-ld-documentation": search_ld_documentation,
            "fetch-ld-documentation": fetch_launchdarkly_documentation,
            "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
            "web-search": web_search,
        },
    )

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
