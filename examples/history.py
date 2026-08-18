"""
Example: config().invoke() with conversation history.

Demonstrates passing a generated conversation history to invoke() so that the
model sees prior turns when generating its response.  The default prompt
explicitly references earlier turns so the response should mention LaunchDarkly
and feature flags — a simple smoke-check that history was ingested.

Usage (via main.py):
    python main.py history <flag-key> "<user input>"
"""

from __future__ import annotations

import json
import re
import sys

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, write_output
from launchdarkly_ai_server import config, conversation_id, global_registry

HISTORY = [
    {
        "role": "user",
        "content": "What is LaunchDarkly?",
    },
    {
        "role": "assistant",
        "content": "LaunchDarkly is a feature management platform that enables teams to safely deploy, manage, and measure the impact of feature flags and software releases.",
    },
    {
        "role": "user",
        "content": "How does it help with AI features specifically?",
    },
    {
        "role": "assistant",
        "content": "LaunchDarkly provides an AI SDK that allows you to manage AI model configurations, prompts, and parameters through feature flags, enabling safe experimentation and rollout of AI-powered features.",
    },
]

HISTORY_PROMPT = (
    "Based on what you told me about LaunchDarkly and its AI SDK, "
    "what are the key benefits of using feature flags for AI rollouts? "
    "Reference our earlier discussion."
)


async def run(key: str, user_input: str) -> None:
    prompt = user_input or HISTORY_PROMPT
    with conversation_id("history-example"):
        response = await config(
            key=key,
            registry=global_registry,
        ).invoke(prompt, new_context(), variables=None, history=HISTORY)

    text = str(
        response.get("response", "")
        if isinstance(response, dict)
        else getattr(response, "response", "")
    )
    references_history = bool(
        re.search(
            r"launchdarkly|feature flag|feature management|ai sdk", text, re.IGNORECASE
        )
        and len(text) > 20
    )

    tag = "REFERENCED" if references_history else "DID NOT reference"
    print(f"[history-check] Model {tag} prior conversation history", file=sys.stderr)
    if not references_history:
        print(
            "[history-check] WARNING: Response may not reflect conversation history. Inspect output manually.",
            file=sys.stderr,
        )

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
