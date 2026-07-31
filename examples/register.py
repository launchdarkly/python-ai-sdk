"""
Populates the global registry with all available handlers and shared tools.

This module is imported for its side-effect by the agent, streaming, and
graph examples.  Handler construction is deferred to _register() so that
provider SDKs (anthropic / openai) are only instantiated when the module
is actually executed — not at import time.
"""

from __future__ import annotations

from examples.tools import (
    fetch_launchdarkly_documentation,
    get_preferences,
    search_ld_documentation,
    web_search,
)
from launchdarkly_ai_claude_agents import ClaudeWebSearch, create_claude_agents_handler
from launchdarkly_ai_claude_messages import create_claude_messages_handler
from launchdarkly_ai_openai_agents import create_openai_agent_handler
from launchdarkly_ai_openai_messages import create_openai_messages_handler
from launchdarkly_ai_server import global_registry

global_registry.register(
    handlers=[
        create_openai_messages_handler(),
        create_openai_agent_handler(),
        create_claude_agents_handler(),
        create_claude_messages_handler(),
    ],
    tools={
        # LD documentation agent tools
        "web-search": ClaudeWebSearch,
        "get-user-preferences": get_preferences,
        "search-ld-documentation": search_ld_documentation,
        "fetch-ld-documentation": fetch_launchdarkly_documentation,
        "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
        # Travel graph tools
        "user-preferences-lookup": get_preferences,
        "web-search-tool": web_search,
    },
)
