"""
Example: ``launchdarkly_ai_server.agui.stream`` — one AI Config turn as AG-UI events.

``config().stream()`` still yields ``chunk`` then ``done``. This reader is the
opt-in entry: ``RUN_STARTED``, the text and tool events the handler yields,
then ``RUN_FINISHED``. The SDK still runs the tool loop. The stream only
reports it.

Text deltas go to stdout. Run and tool events go to stderr, the same split as
``examples/graph_streaming.py``. The generator is built inside
``conversation_id`` and iterated outside it, the same shape as
``examples/streaming.py``.

Not registered in ``main.py``. ``run`` takes the same ``(flag key, user input)``
pair as the other examples.
"""

from __future__ import annotations

import json
import sys
from uuid import uuid4

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, new_conversation_id
from launchdarkly_ai_server import config, conversation_id, global_registry
from launchdarkly_ai_server.agui import stream


async def run(key: str, user_input: str) -> None:
    conversation = new_conversation_id("agui-stream-example")
    thread_id = f"thread-{uuid4().hex[:8]}"
    run_id = f"run-{uuid4().hex[:8]}"
    print(f"[conversation] {conversation}", file=sys.stderr)
    print(f"[thread] {thread_id} [run] {run_id}", file=sys.stderr)

    with conversation_id(conversation):
        events = stream(
            config(key=key, registry=global_registry),
            user_input,
            new_context(),
            thread_id=thread_id,
            run_id=run_id,
        )

    async for event in events:
        kind = event["type"]
        if kind == "TEXT_MESSAGE_CONTENT":
            sys.stdout.write(event.get("delta") or "")
            sys.stdout.flush()
        elif kind == "RUN_STARTED":
            print(
                f"[RUN_STARTED] thread={event['threadId']} run={event['runId']}",
                file=sys.stderr,
            )
        elif kind == "TOOL_CALL_START":
            print(
                f"[tool] start {event['toolCallName']} id={event['toolCallId']}",
                file=sys.stderr,
            )
        elif kind == "TOOL_CALL_ARGS":
            print(
                f"[tool] args id={event['toolCallId']} {event.get('delta') or ''}",
                file=sys.stderr,
            )
        elif kind == "TOOL_CALL_END":
            print(f"[tool] end id={event['toolCallId']}", file=sys.stderr)
        elif kind == "TOOL_CALL_RESULT":
            failed = (
                (event.get("metadata") or {}).get("launchdarkly", {}).get("toolFailed")
            )
            print(
                f"[tool] result id={event['toolCallId']} failed={bool(failed)}",
                file=sys.stderr,
            )
        elif kind == "RUN_FINISHED":
            sys.stdout.write("\n\n")
            print(
                "Usage:",
                json.dumps(event.get("usage"), indent=2, default=str),
            )
            judges = (
                (event.get("metadata") or {})
                .get("launchdarkly", {})
                .get("judgeResults")
            )
            if judges:
                print("Judge results:", json.dumps(judges, indent=2, default=str))
            sys.stdout.write("\n")
