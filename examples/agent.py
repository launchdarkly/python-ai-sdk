"""
Example: config() — selects handler via a LaunchDarkly flag.

Usage (via main.py):
    python main.py agent <flag-key> "<user input>"
"""

from __future__ import annotations

import json

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, write_output
from launchdarkly_ai_server import config, global_registry


async def run(key: str, user_input: str) -> None:
    response = await config(
        key=key,
        registry=global_registry,
    ).invoke(user_input, new_context())

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
