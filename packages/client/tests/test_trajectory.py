from __future__ import annotations

import asyncio
from typing import Any

import pytest

from launchdarkly_ai_server.trajectory import (
    MAX_RECORDED_VALUE_CHARS,
    ToolInvocation,
    TrajectoryRecorder,
    render_trajectory,
)
from launchdarkly_ai_server.types import NativeTool


def test_wrapped_sync_tool_stays_sync() -> None:
    """A blanket async wrapper would hand a caller's handler a coroutine
    object where it used to get the tool's value.
    """

    def lookup(args: dict[str, Any]) -> str:
        return f"order {args['id']}"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})

    assert not asyncio.iscoroutinefunction(wrapped["lookup"])
    assert wrapped["lookup"]({"id": "A1"}) == "order A1"
    assert recorder.invocations == [
        ToolInvocation(name="lookup", arguments='{"id":"A1"}', result="order A1")
    ]


@pytest.mark.asyncio
async def test_wrapped_async_tool_is_awaited() -> None:
    async def lookup(args: dict[str, Any]) -> str:
        await asyncio.sleep(0)
        return f"order {args['id']}"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})

    assert asyncio.iscoroutinefunction(wrapped["lookup"])
    assert await wrapped["lookup"]({"id": "A1"}) == "order A1"
    assert recorder.invocations[0].result == "order A1"


def test_wrapped_tool_reraises_and_records_the_failure() -> None:
    """A tool that failed must still fail its caller."""

    def refund(args: dict[str, Any]) -> str:
        raise RuntimeError("gateway timeout")

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"refund": refund})

    with pytest.raises(RuntimeError, match="gateway timeout"):
        wrapped["refund"]({"id": "A1"})

    assert recorder.invocations == [
        ToolInvocation(name="refund", arguments='{"id":"A1"}', error="gateway timeout")
    ]


@pytest.mark.asyncio
async def test_concurrent_calls_keep_their_start_order() -> None:
    """A judge asked whether search came before refund is reading a sequence,
    so completion order would answer a different question.
    """
    started: dict[str, asyncio.Event] = {"slow": asyncio.Event()}

    async def slow(args: dict[str, Any]) -> str:
        started["slow"].set()
        await asyncio.sleep(0.02)
        return "slow done"

    async def fast(args: dict[str, Any]) -> str:
        await started["slow"].wait()
        return "fast done"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"slow": slow, "fast": fast})

    slow_task = asyncio.create_task(wrapped["slow"]({}))
    await started["slow"].wait()
    await wrapped["fast"]({})
    await slow_task

    assert [invocation.name for invocation in recorder.invocations] == ["slow", "fast"]


def test_calls_past_the_limit_still_execute_but_are_only_counted() -> None:
    calls: list[int] = []

    def append(args: dict[str, Any]) -> str:
        calls.append(args["n"])
        return "ok"

    recorder = TrajectoryRecorder(limit=2)
    wrapped = recorder.wrap({"append": append})
    for n in range(5):
        wrapped["append"]({"n": n})

    # Every call ran: truncation bounds the record, not the behaviour.
    assert calls == [0, 1, 2, 3, 4]
    assert len(recorder.invocations) == 2
    assert recorder.omitted == 3


def test_native_tools_pass_through_unwrapped_and_undescribed() -> None:
    """Invisible, so not advertised either: listing it would let a judge
    conclude the model ignored a tool it may well have used.
    """
    native = NativeTool("WebSearch")
    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"web_search": native, "lookup": lambda args: "ok"})

    assert wrapped["web_search"] is native
    assert recorder.observable_tools == ["lookup"]


def test_keyword_arguments_are_recorded() -> None:
    def lookup(**kwargs: Any) -> str:
        return "ok"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})
    wrapped["lookup"](id="A1")

    assert recorder.invocations[0].arguments == '{"id":"A1"}'


def test_render_lists_available_tools_calls_arguments_and_results() -> None:
    rendered = render_trajectory(
        [
            ToolInvocation(name="lookup", arguments={"id": "A1"}, result="shipped"),
            ToolInvocation(
                name="refund", arguments={"id": "A1"}, error="gateway timeout"
            ),
        ],
        observable_tools=["lookup", "refund"],
    )

    assert rendered == (
        "Tools available: lookup, refund\n"
        "Tool calls made while producing the response, in order:\n"
        "1. lookup\n"
        '   arguments: {"id":"A1"}\n'
        "   result: shipped\n"
        "2. refund\n"
        '   arguments: {"id":"A1"}\n'
        "   error: gateway timeout"
    )


def test_render_reports_an_empty_trajectory_when_tools_were_available() -> None:
    """ "Called nothing" is the finding a tool-selection judge most needs."""
    rendered = render_trajectory([], observable_tools=["lookup"])

    assert rendered == (
        "Tools available: lookup\nNo tool calls were made while producing the response."
    )


def test_render_is_empty_when_there_was_nothing_observable() -> None:
    assert render_trajectory([], observable_tools=[]) == ""


def test_render_reports_omitted_calls() -> None:
    rendered = render_trajectory(
        [ToolInvocation(name="lookup", arguments=None, result="ok")],
        observable_tools=["lookup"],
        omitted=3,
    )

    assert "(3 further tool call(s) were made but not recorded.)" in rendered


def test_render_truncates_an_oversized_value() -> None:
    """The rendered value, suffix included, must not exceed the cap -- a
    trajectory goes into a judge prompt, so appending the suffix on top would
    quietly break the budget it exists to enforce.
    """
    rendered = render_trajectory(
        [ToolInvocation(name="fetch", arguments={}, result="x" * 5000)],
        observable_tools=["fetch"],
    )

    line = next(line for line in rendered.splitlines() if line.startswith("   result:"))
    value = line.removeprefix("   result: ")
    assert len(value) == MAX_RECORDED_VALUE_CHARS
    assert value.endswith("… (truncated)")


def test_render_serializes_unserializable_values_without_raising() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "<opaque>"

    rendered = render_trajectory(
        [ToolInvocation(name="fetch", arguments={"k": Opaque()}, result=Opaque())],
        observable_tools=["fetch"],
    )

    assert "<opaque>" in rendered


def test_render_falls_back_to_a_placeholder_when_str_itself_raises() -> None:
    """Rendering runs after the tool call already succeeded, so a value whose
    own __str__ raises must not turn a successful call into a failed one.
    """

    class Unrenderable:
        def __str__(self) -> str:
            raise RuntimeError("closed")

    rendered = render_trajectory(
        [ToolInvocation(name="fetch", arguments={}, result=Unrenderable())],
        observable_tools=["fetch"],
    )

    assert "   result: <unrenderable Unrenderable>" in rendered


def test_render_does_not_escape_non_ascii() -> None:
    """A judge reads this string, so it must see the actual characters.

    json.dumps escapes non-ASCII by default, which reached the judge as
    ``caf\\u00e9`` -- noise it then has to grade a tool result through.
    """
    rendered = render_trajectory(
        [
            ToolInvocation(
                name="lookup",
                arguments={"city": "café", "place": "東京"},
                result={"note": "naïve"},
            )
        ],
        observable_tools=["lookup"],
    )

    assert '   arguments: {"city":"café","place":"東京"}' in rendered
    assert '   result: {"note":"naïve"}' in rendered
    assert "\\u" not in rendered


def test_render_leaves_a_non_ascii_string_result_alone() -> None:
    """A string result was never JSON-encoded, so it never escaped. Pins that
    both paths agree now.
    """
    rendered = render_trajectory(
        [ToolInvocation(name="lookup", arguments={}, result="café 東京")],
        observable_tools=["lookup"],
    )

    assert "   result: café 東京" in rendered


def test_a_sync_tool_returning_an_awaitable_records_on_completion() -> None:
    """Records on completion, so a judge never reads a pending coroutine's
    repr. Never awaited means never recorded -- the call did not complete.
    """

    async def inner() -> str:
        return "shipped"

    def lookup(args: dict[str, Any]) -> Any:
        return inner()

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})
    pending = wrapped["lookup"]({"id": "A1"})

    assert recorder.invocations[0].result is None
    assert (
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(pending)
        == "shipped"
    )
    assert recorder.invocations[0].result == "shipped"


def test_handoff_tools_are_not_recorded_or_described() -> None:
    """Not tools the agent was given, and absent from the config a per-node
    judge is scored against. wrap_tool_handlers excludes them too.
    """
    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap(
        {
            "lookup": lambda args: "shipped",
            "__handoff_billing": lambda args: "Handoff to billing recorded",
        }
    )

    assert recorder.observable_tools == ["lookup"]
    # Still passed through, so routing keeps working.
    assert "__handoff_billing" in wrapped
    wrapped["__handoff_billing"]({})
    assert [i.name for i in recorder.invocations] == []


def test_only_tools_the_config_exposed_are_recorded_or_described() -> None:
    """The map can be wider than what the model was offered; describing the
    extras would let a judge penalise an agent for ignoring them.
    """
    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap(
        {"lookup": lambda args: "shipped", "refund": lambda args: "refunded"},
        exposed={"lookup"},
    )

    assert recorder.observable_tools == ["lookup"]
    wrapped["refund"]({})
    assert [i.name for i in recorder.invocations] == []
    wrapped["lookup"]({})
    assert [i.name for i in recorder.invocations] == ["lookup"]


def test_recorded_values_are_bounded_immediately_not_at_render_time() -> None:
    """An evaluation holds many rows' recorded calls in memory until scoring.
    Retaining each raw, unbounded return value until then multiplies memory by
    row count and call count; the recorder must hold bounded text instead.
    """

    def fetch(args: dict[str, Any]) -> str:
        return "x" * (MAX_RECORDED_VALUE_CHARS * 5)

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"fetch": fetch})
    wrapped["fetch"]({})

    stored = recorder.invocations[0].result
    assert isinstance(stored, str)
    assert len(stored) <= MAX_RECORDED_VALUE_CHARS


def test_recorded_arguments_are_unaffected_by_the_tool_mutating_them() -> None:
    """A judge grades what the model sent, not what the tool did with it
    afterward -- the recorded call must describe the call boundary, not the
    tool's side effects.
    """

    def pop_id(args: dict[str, Any]) -> str:
        order_id = args.pop("id")
        return f"order {order_id}"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": pop_id})

    assert wrapped["lookup"]({"id": "A1"}) == "order A1"
    assert recorder.invocations[0].arguments == '{"id":"A1"}'


def test_recorded_result_is_unaffected_by_the_handler_mutating_it() -> None:
    """A judge grades what the tool actually returned, not what the calling
    handler did with that value afterward while preparing its own answer.
    """

    def lookup(args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "pending"}

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})

    result = wrapped["lookup"]({})
    result["status"] = "shipped"  # the handler mutates the value it got back

    assert recorder.invocations[0].result == '{"status":"pending"}'


def test_no_exposed_set_means_every_callable_is_in_scope() -> None:
    """The offline case: the runner resolves the config's tools from this same
    map, so no filter is needed.
    """
    recorder = TrajectoryRecorder()
    recorder.wrap({"lookup": lambda args: "a", "refund": lambda args: "b"})

    assert recorder.observable_tools == ["lookup", "refund"]
