"""
Example: graph().invoke() with multimodal conversation history.

Passes a `history` list containing an image content block to a graph flag. Only
the root node receives the history; downstream nodes see it through the normal
node-to-node data passing. The image is a generated solid red square, so the
model naming the colour is the signal that the image actually reached the
provider.

Usage (via main.py):
    python main.py graph-history <graph-flag-key> "<user input>"
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, solid_color_png_base64, write_output
from launchdarkly_ai_server import global_registry, graph

IMAGE_BLOCK = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": solid_color_png_base64((255, 0, 0)),
    },
}

COLOR_QUESTION = (
    "What colour is the square in the image I shared? Answer with just the colour name."
)

# Two supported shapes: history that carries only context (the user turn arrives
# as user_input), and history that already ends with the user turn (user_input
# is empty).
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "image-in-history + question as user_input",
        "history": [{"role": "user", "content": [IMAGE_BLOCK]}],
        "user_input": COLOR_QUESTION,
    },
    {
        "name": "history ends with the user turn, empty user_input",
        "history": [
            {"role": "user", "content": "I am going to share an image with you."},
            {"role": "assistant", "content": "Sure — go ahead and share it."},
            {
                "role": "user",
                "content": [IMAGE_BLOCK, {"type": "text", "text": COLOR_QUESTION}],
            },
        ],
        "user_input": "",
    },
]


async def run(key: str, user_input: str) -> None:
    failures: list[str] = []

    for scenario in SCENARIOS:
        response = await graph(key, registry=global_registry).invoke(
            user_input or scenario["user_input"],
            new_context(),
            {"user_id": "user-123"},
            history=scenario["history"],
        )

        text = str(
            response.get("response", "")
            if isinstance(response, dict)
            else getattr(response, "response", "")
        )
        saw_color = bool(re.search(r"\bred\b", text, re.IGNORECASE))

        tag = "SAW" if saw_color else "DID NOT see"
        print(
            f"[graph-history-check] {scenario['name']}: model {tag} the image from history",
            file=sys.stderr,
        )
        if not saw_color:
            failures.append(scenario["name"])
            print(
                f"[graph-history-check] response was: {text[:300]}",
                file=sys.stderr,
            )

        print(json.dumps(response, indent=2, default=str))
        write_output(response)

    if failures:
        raise RuntimeError(
            "graph() did not forward history to the root node for: "
            + ", ".join(failures)
            + ". Before the history feature lands this is the expected result."
        )
