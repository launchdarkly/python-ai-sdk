"""
Cleanup hooks that ``shutdown()`` runs.

This is how core reaches experimental state without importing it: an
experimental module registers its own cleanup when it loads, and ``shutdown()``
runs whatever is registered. Core never imports, names or calls experimental
code (TESTING.md §0.3), so a hook's *name* comes from the module that
registers it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

# The lifecycle logger: this warning came from shutdown() before the hooks existed.
logger = logging.getLogger("launchdarkly_ai_server.lifecycle")

_hooks: dict[str, Callable[[], None]] = {}


def register_shutdown_hook(name: str, hook: Callable[[], None]) -> None:
    """
    Registers *hook* to run on every ``shutdown()``, including before a client
    exists. Keyed by *name*, so registering the same feature again replaces the
    earlier hook rather than adding another.
    """
    _hooks[name] = hook


def run_shutdown_hooks(*, raise_errors: bool = False) -> None:
    """
    Runs every registered hook. Experimental code must not break a core call, so
    a hook that raises is logged and the rest still run. *raise_errors* is for
    the test reset, where a failed cleanup should fail the test instead.
    """
    for name, hook in list(_hooks.items()):
        if raise_errors:
            hook()
            continue
        try:
            hook()
        except Exception:
            logger.warning("Could not clear the %s state", name, exc_info=True)
