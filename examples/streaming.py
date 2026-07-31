"""
Example: config().stream() — tokens are printed as they arrive,
then the final usage + judge results are logged when the stream ends.

Usage (via main.py):
    python main.py streaming <flag-key> "<user input>"
"""

from __future__ import annotations

import sys

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context
from launchdarkly_ai_server import config, global_registry


async def run(key: str, user_input: str) -> None:
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
