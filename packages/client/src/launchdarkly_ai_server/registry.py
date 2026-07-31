from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .types import NativeTool, ProviderHandler

logger = logging.getLogger(__name__)


class Registry:
    """
    Manages handlers and tools for ``routed_model`` and ``graph``.
    """

    def __init__(
        self,
        *,
        handlers: list[ProviderHandler] | None = None,
        tools: dict[str, Callable[..., Any] | NativeTool] | None = None,
    ) -> None:
        self._handlers: list[ProviderHandler] = list(handlers) if handlers else []
        self._tools: dict[str, Callable[..., Any] | NativeTool] = (
            dict(tools) if tools else {}
        )

    @property
    def handlers(self) -> list[ProviderHandler]:
        return list(self._handlers)

    @property
    def tools(self) -> dict[str, Callable[..., Any] | NativeTool]:
        return dict(self._tools)

    def register(
        self,
        *,
        handlers: list[ProviderHandler] | None = None,
        tools: dict[str, Callable[..., Any] | NativeTool] | None = None,
    ) -> None:
        if handlers:
            for handler in handlers:
                if handler.provides_for is not None:
                    # Check for duplicate by providesFor key
                    key = handler.provides_for
                    existing_idx = next(
                        (
                            i
                            for i, h in enumerate(self._handlers)
                            if h.provides_for == key
                        ),
                        None,
                    )
                    if existing_idx is not None:
                        logger.warning(
                            "Handler for %s already registered; replacing with new handler.",
                            key,
                        )
                        self._handlers[existing_idx] = handler
                    else:
                        self._handlers.append(handler)
                else:
                    # No providesFor — always append, never deduplicate
                    self._handlers.append(handler)

        if tools:
            for name, fn in tools.items():
                if name in self._tools:
                    logger.warning(
                        "Tool '%s' already registered; replacing with new tool.", name
                    )
                self._tools[name] = fn


def compose(a: Registry, b: Registry) -> Registry:
    """
    Returns a new ``Registry`` that merges *a* and *b*. When both *a* and *b*
    have handlers or tools with the same key, *b* wins.
    """
    result = Registry()
    # Seed with a's entries
    for handler in a.handlers:
        result._handlers.append(handler)
    result._tools.update(a.tools)

    # Overlay b (b wins on conflict)
    result.register(handlers=b.handlers, tools=b.tools)
    return result


def resolve_handlers(
    registry: Registry | None,
    local_handlers: list[ProviderHandler] | None,
) -> list[ProviderHandler] | None:
    """
    Merges registry handlers with locally-supplied handlers.
    Local handlers precede registry handlers so ``select_handler`` finds
    the local one first on a conflict.
    """
    reg_handlers = registry.handlers if registry is not None else []
    if local_handlers and reg_handlers:
        return list(local_handlers) + reg_handlers
    if local_handlers:
        return local_handlers
    if reg_handlers:
        return reg_handlers
    return None


def resolve_tools(
    registry: Registry | None,
    local_tools: dict[str, Callable[..., Any] | NativeTool] | None,
) -> dict[str, Callable[..., Any] | NativeTool] | None:
    """
    Merges registry tools with locally-supplied tools. Local keys win.
    """
    reg_tools = registry.tools if registry is not None else {}
    if local_tools and reg_tools:
        merged = dict(reg_tools)
        merged.update(local_tools)
        return merged
    if local_tools:
        return local_tools
    if reg_tools:
        return reg_tools
    return None


global_registry = Registry()
