"""
Example: config() with LangChain handlers (messages + agents).
Uses both LangChain handlers directly — no routing, no global registry.
The wildcard provider ('*') allows either handler to serve any provider's flag.

Usage (via main.py):
    python main.py langchain <flag-key> "<user input>"
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
from launchdarkly_ai_langchain_agents import create_langchain_agents_handler
from launchdarkly_ai_langchain_messages import create_langchain_messages_handler
from launchdarkly_ai_server import config


async def run(key: str, user_input: str) -> None:
    response = await config(
        key=key,
        handler=[
            create_langchain_messages_handler(),
            create_langchain_agents_handler(),
        ],
        tool_handlers={
            "get-user-preferences": get_preferences,
            "search-ld-documentation": search_ld_documentation,
            "fetch-ld-documentation": fetch_launchdarkly_documentation,
            "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
            "web-search": web_search,
        },
    ).invoke(user_input, new_context(), variables={"user_input": user_input})

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
