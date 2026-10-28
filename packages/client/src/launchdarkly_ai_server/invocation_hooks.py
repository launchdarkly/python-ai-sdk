"""
Hooks that ``config().invoke()`` and ``.stream()`` run before the handler.

This is how a feature reacts to an invocation without core importing it: a
feature module registers a hook, and core runs whatever is registered. Core
never imports, names or calls a feature module, so a hook's *name* comes from
the module that registers it. Same shape as ``shutdown_hooks``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .types import LDContext, ProviderHandler

logger = logging.getLogger("launchdarkly_ai_server.client")


@dataclass(frozen=True)
class InvocationEvent:
    """What a hook sees: one ``config()`` call, after routing and before the handler runs."""

    config_key: str
    value: dict[str, Any]
    reason: Any
    context: LDContext
    user_input: str | None
    history: list[dict[str, Any]] | None
    variables: dict[str, Any] | None
    handlers: list[ProviderHandler]
    run_id: str
    variation_key: str


_hooks: dict[str, Callable[[InvocationEvent], None]] = {}


def register_invocation_hook(
    name: str, hook: Callable[[InvocationEvent], None] | None
) -> None:
    """
    Registers *hook* under *name*, replacing any earlier hook with that name.
    Passing ``None`` removes it.
    """
    if hook is None:
        _hooks.pop(name, None)
    else:
        _hooks[name] = hook


def has_invocation_hooks() -> bool:
    return bool(_hooks)


def run_invocation_hooks(event: InvocationEvent) -> None:
    """Runs every hook. A hook that raises is logged and must not break the call."""
    for name, hook in list(_hooks.items()):
        try:
            hook(event)
        except Exception:
            logger.warning("The %s invocation hook failed", name, exc_info=True)
