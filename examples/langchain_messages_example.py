"""
Example: langchain_messages() — LangChain messages handler.

Usage (via main.py):
    python main.py langchain-messages <flag-key> "<user input>"
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
from launchdarkly_ai_langchain_messages import langchain_messages


async def run(key: str, user_input: str) -> None:
    response = await langchain_messages(
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
        variables={"user_input": user_input},
    )

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
