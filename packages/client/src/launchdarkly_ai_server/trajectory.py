"""Tool-call trajectory capture, shared by both judge paths.

A judge can only grade what it is shown. Handler packages record tool traffic
onto OpenTelemetry spans and return only ``{output, usage}``, so by the time a
criterion ran, the calls a row made on its way to that output were gone --
which made "did the agent call the right tools, in the right order, with the
right arguments?" an unaskable question of an SDK-run evaluation, even though
the evaluation had just run the agent that answered it.

Both judge paths therefore record the trajectory themselves, by wrapping the
caller's tool implementations before handing them to the handler: once per row
in the offline evaluations runner, and once per invocation in
``tracking.execute_and_track``. Wrapping is what makes this work with every
handler package without changing any of them: a handler looks a tool up by its
key and calls it, exactly as before.

The recorded trajectory reaches a judge through ``message_history``, built by
:func:`judge_scoring.build_message_history` -- one function for both paths, so
an online judge and an offline one are shown the same shape.

Three properties are load-bearing.

**The recorder observes; it never intervenes.** A wrapped tool returns what the
original returned and raises what the original raised. A row whose trajectory
hits :data:`MAX_RECORDED_TOOL_CALLS` still executes every remaining call --
truncation drops the *record*, never the work, because an evaluation that
changed the agent's behavior would no longer be evaluating the agent.

**A recorder belongs to one invocation.** The offline runner generates rows
concurrently against one shared tool map, so a single shared recorder would
splice one row's calls into another row's trajectory and hand the judge a
conversation that never happened. The same holds for concurrent online
invocations, which is why ``execute_and_track`` builds its own per call.

**Only observable tools are described.** A ``NativeTool`` is executed inside the
provider, so no local wrapper ever sees it and its calls cannot appear in the
trajectory. Such a tool is therefore left out of the rendered "tools available"
line as well: naming a tool whose use is invisible would let a judge conclude
the model ignored a tool it may well have called.

Online, ``tracking.wrap_tool_handlers`` does turn a ``NativeTool`` into a
callable tracking stub, so a native call *is* locally observable there -- but it
is still skipped, and deliberately. The stub returns nothing, so recording it
would show a judge a tool call with an empty result while the provider's real
result stayed invisible. Recording is therefore composed *inside* that wrapper,
on the original map, so both paths see natives identically.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .types import NativeTool

#: How many tool calls one row's trajectory records. A trajectory is
#: interpolated into a judge prompt, so an agent that loops over a large tool
#: result set would otherwise spend the judge's context window -- and its
#: budget -- on the tail of a trajectory the judge stopped reading. Calls past
#: the limit still execute and are reported as a count.
MAX_RECORDED_TOOL_CALLS = 50

#: How many characters one rendered argument bag or tool result contributes.
#: Bounds a single tool that returns a whole document, for the same reason.
MAX_RECORDED_VALUE_CHARS = 2000

_TRUNCATION_SUFFIX = "… (truncated)"

ToolImplementation = Callable[..., Any] | NativeTool


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call made while generating a row, with how it turned out.

    ``result`` and ``error`` are mutually exclusive: a call that raised has no
    result, and a call that returned has no error. Both are ``None`` on a call
    that is still in flight, which is only observable from inside the wrapper.
    """

    name: str
    arguments: Any = None
    result: Any = None
    error: str | None = None


class TrajectoryRecorder:
    """Records one row's tool calls, in the order the calls were started.

    A slot is reserved when a call starts and filled in when it finishes, so
    tools a handler runs concurrently keep their start order rather than being
    reordered by which of them returned first.
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
        """Keys of the tools this recorder can actually observe being called."""
        return list(self._observable)

    def wrap(
        self, tool_handlers: Mapping[str, ToolImplementation]
    ) -> dict[str, ToolImplementation]:
        """Return ``tool_handlers`` with each callable recording into this row.

        Keys are preserved exactly: a handler resolves a tool by the key the
        model named, so renaming one here would break the lookup.
        """
        wrapped: dict[str, ToolImplementation] = {}
        # Rebuilt rather than appended to, so re-wrapping a map does not report
        # the same tool as available twice.
        self._observable = []
        for name, implementation in tool_handlers.items():
            if isinstance(implementation, NativeTool) or not callable(implementation):
                # Provider-executed, or already invalid and reported as such by
                # tool resolution. Either way there is nothing local to observe,
                # so pass the value through rather than replacing it with a
                # wrapper the handler would treat differently.
                wrapped[name] = implementation
                continue
            self._observable.append(name)
            wrapped[name] = self._record(name, implementation)
        return wrapped

    def _record(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            slot = self._reserve(name, _call_arguments(args, kwargs))
            try:
                result = original(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as error:
                self._complete(slot, error=f"{error}")
                raise
            self._complete(slot, result=result)
            return result

        return wrapper

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

    Paired with :func:`render_row_trajectory` so one module owns both halves of
    the record's shape: a key renamed here without its reader being updated
    would silently render every row's trajectory as empty, which reads exactly
    like an agent that called no tools.
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
    """Render a row's trajectory as the text a judge reads.

    Returns ``""`` when there was nothing observable to report, so the caller
    can skip the block entirely rather than telling a judge about tools in a
    run that had none.

    The empty trajectory of a row that *did* have tools is reported explicitly:
    "this agent called nothing" is the finding a judge grading tool selection
    most needs, and an omitted block would read as a run without tools.
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

    Every handler package in this SDK calls a tool with the model's argument
    bag as one positional mapping, so that is the shape worth preserving
    verbatim; the rest are recorded structurally rather than guessed at.
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
        # ensure_ascii=False, because this string is read by a model. The
        # default escapes every non-ASCII character, so a tool that returned
        # "café" or "東京" reached the judge as "caf\u00e9" / "\u6771\u4eac" --
        # noise that the judge then has to grade a tool result through, and a
        # gratuitous difference from what any other SDK would show for the
        # same call. Key order stays sorted so one language's own output is
        # deterministic; matching another language's byte-for-byte is not the
        # goal, but showing the judge the actual characters is.
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
