"""
Example: a multi-turn conversation grouped under one ``gen_ai.conversation.id``, with inline
judge evaluation on every turn.

This is the end-to-end check for O11Y-1888. Run it, then open the printed conversation id in
LaunchDarkly's Conversations view and confirm:

  1. One conversation, three turns — not three conversations. Every span of every turn carries
     the same id: root, ``chat``, ``execute_tool``, and the judge's own ``invoke_agent``.
  2. Each turn shows a score badge, sourced from the ``gen_ai.evaluation.result`` span event on
     the judge span.
  3. No judge reasoning anywhere in the telemetry. The score and the judge's config key are
     exported; the explanation is not, because it is model prose about the user's conversation
     and content attributes require ``capture_content``. The reasoning IS printed below, straight
     from ``judge_results`` — that is the caller's copy, and it is unaffected.

The flag key must point at an AI Config with a ``judge_configuration``, otherwise there are no
judge turns to look at.

Usage (via main.py):
    python main.py conversation <flag-key> "<opening message>"
"""

from __future__ import annotations

import sys
from typing import Any

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, new_conversation_id
from launchdarkly_ai_server import config, conversation_id, global_registry

FOLLOW_UPS = [
    "Can you give me a concrete example of that?",
    "What is the most common mistake teams make with it?",
]


async def run(key: str, user_input: str) -> None:
    conversation = new_conversation_id("conversation-example")
    ctx = new_context()
    history: list[dict[str, Any]] = []

    print(f"[conversation] {conversation}", file=sys.stderr)

    turns = [user_input or "What is a feature flag?", *FOLLOW_UPS]

    for index, prompt in enumerate(turns, start=1):
        # One binding per turn, same id every time — that is what makes them one conversation
        # rather than three. Re-binding per turn is the realistic shape: each turn is usually a
        # separate inbound request that looks the id up from its own thread/session.
        with conversation_id(conversation):
            response = await config(
                key=key,
                registry=global_registry,
            ).invoke(prompt, ctx, None, history)

        text = (
            response.response
            if isinstance(response.response, str)
            else str(response.response)
        )
        print(f"\n─── turn {index} ───\n> {prompt}\n{text}")

        judge_results = response.judge_results or {}
        for judge_key, result in judge_results.items():
            # `response` here is the judge's reasoning. It reaches the caller and is deliberately
            # absent from the span — see the module docstring.
            score = getattr(result, "score", None)
            reasoning = getattr(result, "response", None)
            print(
                f"[judge] {judge_key} score={score} reasoning={reasoning}",
                file=sys.stderr,
            )
        if not judge_results:
            print(
                "[judge] no judges ran — does this AI Config have a judge_configuration?",
                file=sys.stderr,
            )

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": text})

    print(
        f"\n[conversation] done — open {conversation} in the Conversations view",
        file=sys.stderr,
    )
