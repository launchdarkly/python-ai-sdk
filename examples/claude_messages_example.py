"""
Example: claude_messages() — Anthropic Claude messages handler.

Usage (via main.py):
    python main.py claude-messages <flag-key> "<user input>"
"""

from __future__ import annotations

import json

from examples.tools import (
    fetch_launchdarkly_documentation,
    get_preferences,
    search_ld_documentation,
)
from examples.utils import new_context, write_output
from launchdarkly_ai_claude_messages import claude_messages


async def run(key: str, user_input: str) -> None:
    response = await claude_messages(
        key,
        user_input,
        new_context(),
        tool_handlers={
            "get-user-preferences": get_preferences,
            "search-ld-documentation": search_ld_documentation,
            "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
        },
        variables={"user_input": user_input},
    )

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
