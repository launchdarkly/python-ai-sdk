"""
Example: langchain_messages() against a Claude model with extended thinking turned on.

With thinking on, Anthropic returns ``content`` as a list of blocks — a ``thinking`` block followed
by a ``text`` block — instead of a plain string. A handler that only reads string content reports
an empty response for these runs while the tokens are still spent, so this example fails loudly
when no text comes back.

Anthropic omits thinking from the turn that follows a tool result, so give this a prompt the model
can answer on its own — a run that goes through the tool loop ends on a plain string and never
reaches the block-shaped content this exercises.

Usage (via main.py):
    python main.py langchain-thinking <flag-key> "Reason it out yourself without any tools: what is 17 times 23?"
"""

from __future__ import annotations

import json
from typing import Any

from examples.tools import (
    fetch_launchdarkly_documentation,
    get_preferences,
    search_ld_documentation,
)
from examples.utils import new_context, write_output
from launchdarkly_ai_langchain_messages import create_langchain_messages_handler
from launchdarkly_ai_server import config

# Anthropic requires max_tokens to exceed the thinking budget.
_THINKING_BUDGET_TOKENS = 1024
_MAX_TOKENS = 4096


async def run(key: str, user_input: str) -> None:
    from langchain_anthropic import ChatAnthropic

    def build_model(ai_config: Any) -> Any:
        model = ai_config.get("model") or {}
        raw = model.get("parameters")
        parameters: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        return ChatAnthropic(
            timeout=None,
            stop=None,
            **parameters,
            model_name=str(model.get("name") or "claude-sonnet-4-5"),
            thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET_TOKENS},
            max_tokens_to_sample=_MAX_TOKENS,
        )

    response = await config(
        key=key,
        handler=create_langchain_messages_handler(llm=build_model),
        tool_handlers={
            "get-user-preferences": get_preferences,
            "search-ld-documentation": search_ld_documentation,
            "fetch-launchdarkly-documentation": fetch_launchdarkly_documentation,
        },
    ).invoke(user_input, new_context(), variables={"user_input": user_input})

    if not (response.response or "").strip():
        raise RuntimeError(
            "Model returned no text. A thinking-enabled model returns content as a list of "
            "blocks, and the handler dropped it."
        )

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
