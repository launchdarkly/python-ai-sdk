"""
Example: enable conversation content for every handler at process startup.

The handlers in the global registry are built with no ``capture_content``
argument. They read the process default when they run. ``configure`` is
separate from ``init_client``: call it once, before the first AI invocation.
An explicit ``True`` or ``False`` on a factory would override this default.

Run from the repository root:

    python -m examples.capture_content <flag-key> "<user input>"
"""

from __future__ import annotations

import json

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, write_output
from launchdarkly_ai_server import (
    SDKOptions,
    config,
    configure,
    global_registry,
    init_client,
    shutdown,
)

_DEFAULT_KEY = "launch-darkly-documentation-summarizer"
_DEFAULT_USER_INPUT = "What is the LaunchDarkly AI SDK?"


async def run(key: str, user_input: str) -> None:
    response = await config(
        key=key,
        registry=global_registry,
    ).invoke(user_input, new_context())

    print(json.dumps(response, indent=2, default=str))
    write_output(response)


async def _main(key: str, user_input: str) -> None:
    configure(options=SDKOptions(capture_content=True))
    await init_client()
    try:
        await run(key, user_input)
    finally:
        await shutdown()


if __name__ == "__main__":
    import asyncio
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    key = sys.argv[1].strip() if len(sys.argv) > 1 else _DEFAULT_KEY
    user_input = sys.argv[2].strip() if len(sys.argv) > 2 else _DEFAULT_USER_INPUT
    asyncio.run(_main(key, user_input))
