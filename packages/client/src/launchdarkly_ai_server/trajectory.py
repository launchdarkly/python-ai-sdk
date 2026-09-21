"""Tool-call trajectory capture, shared by both judge paths.

Handlers return only ``{output, usage}`` and record tool traffic onto spans, so
the calls made on the way to an output were unavailable to a judge. Both paths
therefore record them here, by wrapping the caller's tool implementations
before the handler is invoked -- which works with every handler package without
changing any of them, since a handler still looks a tool up by key and calls
it.

The trajectory reaches a judge through ``message_history``, built by
:func:`judge_scoring.build_message_history` so both paths show the same shape.

Two rules the rest of this module exists to keep: the recorder never changes
what a tool does or how it is called, and a recorder belongs to exactly one
invocation or row, since both run concurrently against one shared tool map.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .types import NativeTool

#: A trajectory goes into a judge prompt, so an agent looping over a large
#: result set would otherwise spend the judge's context window on a tail it
#: never reads. Calls past the limit still execute; they are only counted.
MAX_RECORDED_TOOL_CALLS = 50

#: Bounds a single tool that returns a whole document, for the same reason.
MAX_RECORDED_VALUE_CHARS = 2000

_TRUNCATION_SUFFIX = "… (truncated)"

#: Synthetic routing tools ``graph.route`` injects on a multi-edge node. Not
#: tools the agent was given, and absent from the config a per-node judge is
#: scored against -- showing them would let a judge grade a handoff as tool
#: use. ``wrap_tool_handlers`` skips them too.
HANDOFF_TOOL_PREFIX = "__handoff_"

ToolImplementation = Callable[..., Any] | NativeTool


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call, with how it turned out.

    ``result`` and ``error`` are mutually exclusive; both are ``None`` while
    the call is still in flight.
    """

    name: str
    arguments: Any = None
    result: Any = None
    error: str | None = None


class TrajectoryRecorder:
    """Records one unit's tool calls, in the order the calls were started.

    A slot is reserved on call and filled in on completion, so concurrent tool
    calls keep their start order instead of their return order.
    """

    def __init__(self, limit: int = MAX_RECORDED_TOOL_CALLS) -> None:
        self._limit = limit
        self._invocations: list[ToolInvocation] = []
        self._omitted = 0
        self._observable: list[str] = []

    @property
    def invocations(self) -> list[ToolInvocation]:
        """The recorded calls, oldest first."""
        return list(self._invocations)

    @property
    def omitted(self) -> int:
        """How many calls executed past the recording limit."""
        return self._omitted

    @property
    def observable_tools(self) -> list[str]:
        """Keys of the tools this recorder can observe being called."""
        return list(self._observable)

    def wrap(
        self,
        tool_handlers: Mapping[str, ToolImplementation],
        *,
        exposed: Collection[str] | None = None,
    ) -> dict[str, ToolImplementation]:
        """Return ``tool_handlers`` with each callable recording into this unit.

        Keys are preserved exactly; a handler resolves a tool by the key the
        model named.

        ``exposed`` is the tool keys the config actually offered the model.
        Pass it when the implementation map can be wider -- online,
        ``config()`` merges a ``Registry``'s tools in, and describing those
        would let a judge penalise an agent for ignoring a tool it never had.
        ``None`` means every callable is in scope, which is the offline case:
        the runner resolves the config's tools from this same map.
        """
        wrapped: dict[str, ToolImplementation] = {}
        # Rebuilt, so re-wrapping a map cannot list a tool twice.
        self._observable = []
        for name, implementation in tool_handlers.items():
            if (
                isinstance(implementation, NativeTool)
                or not callable(implementation)
                or name.startswith(HANDOFF_TOOL_PREFIX)
                or (exposed is not None and name not in exposed)
            ):
                # Provider-executed, invalid, synthetic routing, or never
                # offered to the model: nothing to observe, or nothing to grade
                # the agent on. Passed through untouched so the handler treats
                # it exactly as it would have.
                wrapped[name] = implementation
                continue
            self._observable.append(name)
            wrapped[name] = self._record(name, implementation)
        return wrapped

    def _record(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``original`` without changing how it is called.

        A sync tool stays sync: offline these are passed straight to the
        handler, and a caller's own handler may call a sync tool directly and
        use the value. A blanket async wrapper handed it a coroutine object
        instead. (Online, ``wrap_tool_handlers`` wraps this again and is always
        async, so a handler awaits there as it always has.)
        """
        if asyncio.iscoroutinefunction(original):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                slot = self._reserve(name, _call_arguments(args, kwargs))
                return await self._await_and_record(slot, original(*args, **kwargs))

            return async_wrapper

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            slot = self._reserve(name, _call_arguments(args, kwargs))
            try:
                result = original(*args, **kwargs)
            except Exception as error:
                self._complete(slot, error=f"{error}")
                raise
            if inspect.isawaitable(result):
                # Record on completion, so a pending coroutine's repr never
                # reaches a judge. Never awaited means never recorded, which is
                # accurate: the call did not complete.
                return self._await_and_record(slot, result)
            self._complete(slot, result=result)
            return result

        return sync_wrapper

    async def _await_and_record(self, slot: int | None, awaitable: Any) -> Any:
        try:
            result = await awaitable
        except Exception as error:
            self._complete(slot, error=f"{error}")
            raise
        self._complete(slot, result=result)
        return result

    def _reserve(self, name: str, arguments: Any) -> int | None:
        if len(self._invocations) >= self._limit:
            self._omitted += 1
            return None
        self._invocations.append(ToolInvocation(name=name, arguments=arguments))
        return len(self._invocations) - 1

    def _complete(
        self, slot: int | None, *, result: Any = None, error: str | None = None
    ) -> None:
        if slot is None:
            return
        self._invocations[slot] = replace(
            self._invocations[slot], result=result, error=error
        )


def row_fields(recorder: TrajectoryRecorder) -> dict[str, Any]:
    """The trajectory keys a generated-row record carries.

    Paired with :func:`render_row_trajectory` so one module owns both halves:
    renaming a key without its reader would render every trajectory empty,
    which reads exactly like an agent that called no tools.
    """
    return {
        "tool_calls": recorder.invocations,
        "tool_calls_omitted": recorder.omitted,
        "observable_tools": recorder.observable_tools,
    }


def render_row_trajectory(row_result: Mapping[str, Any]) -> str:
    """Render the trajectory carried by a generated-row record."""
    return render_trajectory(
        row_result.get("tool_calls") or [],
        observable_tools=row_result.get("observable_tools") or [],
        omitted=int(row_result.get("tool_calls_omitted") or 0),
    )


def render_trajectory(
    invocations: list[ToolInvocation],
    *,
    observable_tools: list[str] | None = None,
    omitted: int = 0,
) -> str:
    """Render a trajectory as the text a judge reads.

    ``""`` when there was nothing observable, so the caller adds no block at
    all. A unit that *had* tools and called none says so explicitly instead:
    "called nothing" is the finding a tool-selection judge most needs, and an
    omitted block would read as a unit with no tools.
    """
    available = list(observable_tools or [])
    if not available and not invocations:
        return ""

    lines: list[str] = []
    if available:
        lines.append(f"Tools available: {', '.join(available)}")
    if not invocations:
        lines.append("No tool calls were made while producing the response.")
        return "\n".join(lines)

    lines.append("Tool calls made while producing the response, in order:")
    for position, invocation in enumerate(invocations, start=1):
        lines.append(f"{position}. {invocation.name}")
        lines.append(f"   arguments: {_render_value(invocation.arguments)}")
        if invocation.error is not None:
            lines.append(f"   error: {_render_value(invocation.error)}")
        else:
            lines.append(f"   result: {_render_value(invocation.result)}")
    if omitted > 0:
        lines.append(f"({omitted} further tool call(s) were made but not recorded.)")
    return "\n".join(lines)


def _call_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Normalize how a handler passed a tool its arguments.

    Every handler here calls a tool with the model's argument bag as one
    positional mapping, so that shape is preserved verbatim; anything else is
    recorded structurally rather than guessed at.
    """
    if len(args) == 1 and not kwargs:
        return args[0]
    if kwargs and not args:
        return dict(kwargs)
    if not args and not kwargs:
        return None
    return {"args": list(args), "kwargs": dict(kwargs)}


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate(value)
    try:
        # ensure_ascii=False: a model reads this. The default would show the
        # judge "caf\u00e9" instead of "café". Keys stay sorted for
        # deterministic output.
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        rendered = str(value)
    return _truncate(rendered)


def _truncate(text: str) -> str:
    if len(text) <= MAX_RECORDED_VALUE_CHARS:
        return text
    return text[:MAX_RECORDED_VALUE_CHARS] + _TRUNCATION_SUFFIX
