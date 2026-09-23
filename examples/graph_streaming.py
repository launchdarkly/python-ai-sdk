"""
Example: ``graph().stream()`` — the streaming counterpart to ``examples/graph_example.py``.

Model text goes to stdout; node boundaries go to stderr so stdout stays a clean transcript.

Like ``examples/streaming.py``, the generator is built inside ``conversation_id`` and iterated
*outside* it. That is the shape a server produces when it hands a stream to a transport, and it
is what exercises call-time binding: an async generator body does not run until the first
``__anext__``, so both the conversation id and the ``ld.ai.graph`` span's OTel parent have to be
captured when ``stream()`` is called, not when iteration starts.

Writes no JSON output file, same as the single-config streaming example.

Usage (via main.py):
    python main.py graph-streaming <flag-key> "<user input>"
"""

from __future__ import annotations

import json
import sys

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, new_conversation_id
from launchdarkly_ai_server import conversation_id, global_registry, graph


async def run(key: str, user_input: str) -> None:
    conversation = new_conversation_id("graph-streaming-example")
    print(f"[conversation] {conversation}", file=sys.stderr)

    with conversation_id(conversation):
        stream = graph(
            key,
            registry=global_registry,
        ).stream(user_input, new_context(), {"user_id": "user-123"})

    async for event in stream:
        if event["type"] == "chunk":
            sys.stdout.write(event.get("text", ""))
            sys.stdout.flush()
        elif event["type"] == "node_start":
            print(f"\n[node_start] {event['nodeKey']}", file=sys.stderr)
        elif event["type"] == "node_done":
            print(
                f"\n[node_done] {event['nodeKey']} usage={json.dumps(event.get('usage'))}",
                file=sys.stderr,
            )
        elif event["type"] == "handoff":
            print(
                f"[handoff] {event['sourceKey']} -> {event['targetKey']}",
                file=sys.stderr,
            )
        else:
            # Final event — usage aggregated across nodes, plus graph judge results when configured.
            sys.stdout.write("\n\n")
            print("Usage:", json.dumps(event.get("usage"), indent=2, default=str))
            if event.get("judgeResults"):
                print(
                    "Judge results:",
                    json.dumps(event["judgeResults"], indent=2, default=str),
                )
            sys.stdout.write("\n")
