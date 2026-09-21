"""
Example: config() with LiteLLM handlers for messages and agents.

Python calls LiteLLM in-process, so provider-qualified models and aliases from
the evaluated AI Config are routed using the installed LiteLLM configuration.

Usage (via main.py):
    python main.py litellm <flag-key> "<user input>"
"""

from __future__ import annotations

from examples.tools import (
    fetch_launchdarkly_documentation,
    get_preferences,
    search_ld_documentation,
    web_search,
)
from examples.utils import new_context, write_output
from launchdarkly_ai_litellm_agents import create_litellm_agents_handler
from launchdarkly_ai_litellm_messages import create_litellm_messages_handler
from launchdarkly_ai_server import config


async def run(key: str, user_input: str) -> None:
    response = await config(
        key=key,
        handler=[
            create_litellm_messages_handler(),
            create_litellm_agents_handler(),
        ],
        tool_handlers={
            "get-user-preferences": get_preferences,
            "search-ld-documentation": search_ld_documentation,
            "fetch-ld-documentation": fetch_launchdarkly_documentation,
            "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
            "web-search": web_search,
        },
    ).invoke(user_input, new_context(), variables={"user_input": user_input})

    write_output(response)
