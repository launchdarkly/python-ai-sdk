"""
Example: config().stream() — tokens are printed as they arrive,
then the final usage + judge results are logged when the stream ends.

Also the end-to-end check for call-time conversation binding: the generator is built inside the
``conversation_id`` block and iterated *outside* it, which is what a chat app does when it hands
the stream to a transport. An async generator body does not run until the first ``__anext__``, so
before ``stream()`` bound at call time this produced spans with no ``gen_ai.conversation.id`` at
all — silently. Every span of this run should carry the id printed below.

Usage (via main.py):
    python main.py streaming <flag-key> "<user input>"
"""

from __future__ import annotations

import sys

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, new_conversation_id
from launchdarkly_ai_server import config, conversation_id, global_registry


async def run(key: str, user_input: str) -> None:
    conversation = new_conversation_id("streaming-example")
    print(f"[conversation] {conversation}", file=sys.stderr)

    with conversation_id(conversation):
        stream = config(
            key=key,
            registry=global_registry,
        ).stream(user_input, new_context())

    async for event in stream:
        if event["type"] == "chunk":
            sys.stdout.write(event.get("text", ""))
            sys.stdout.flush()
        else:
            # Final event — full response + normalised usage
            sys.stdout.write("\n\n")
            print("Usage:", json_pretty(event.get("usage")))
            if event.get("judgeResults"):
                print("Judge results:", json_pretty(event["judgeResults"]))


def json_pretty(obj: object) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)
