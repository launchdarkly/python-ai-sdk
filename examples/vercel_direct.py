"""Inject a constructed OpenAI model so the Vercel adapter skips AI Gateway.

Usage (via main.py):
    python main.py vercel-direct launch-darkly-documentation-summarizer-messages-openai "What is the LaunchDarkly AI SDK?"
"""

from __future__ import annotations

import json

import ai

from examples.utils import new_context, write_output
from launchdarkly_ai_server import AiConfigRep
from launchdarkly_ai_vercel_messages import vercel_messages


def openai_model(config: AiConfigRep) -> ai.Model:
    return ai.Model(id=str(config["model"]["name"]), provider=ai.get_provider("openai"))


async def run(key: str, user_input: str) -> None:
    response = await vercel_messages(
        key,
        user_input,
        new_context(),
        model=openai_model,
        variables={"user_input": user_input},
    )
    print(json.dumps(response, indent=2, default=str))
    write_output(response)
