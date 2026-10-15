"""
Example: enable conversation content for every handler at process startup.

The handlers in the global registry are built with no ``capture_content``
argument. They read the process default when they run. ``configure`` is
separate from ``init_client``: ``main.py`` initializes the client, and this
example calls ``configure`` once before the first AI invocation. An explicit
``True`` or ``False`` on a factory would override this default.

Usage (via main.py):
    python main.py capture-content <flag-key> "<user input>"
"""

from __future__ import annotations

import json

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_context, write_output
from launchdarkly_ai_server import SDKOptions, config, configure, global_registry


async def run(key: str, user_input: str) -> None:
    configure(options=SDKOptions(capture_content=True))
    response = await config(
        key=key,
        registry=global_registry,
    ).invoke(user_input, new_context())

    print(json.dumps(response, indent=2, default=str))
    write_output(response)
