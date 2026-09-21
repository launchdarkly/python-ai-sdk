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
    """A sync tool is still called, and still returns, synchronously.

    Wrapping everything as a coroutine function would hand a caller's own
    handler a coroutine object where it used to get the tool's value.
    """

    def lookup(args: dict[str, Any]) -> str:
        return f"order {args['id']}"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})

    assert not asyncio.iscoroutinefunction(wrapped["lookup"])
    assert wrapped["lookup"]({"id": "A1"}) == "order A1"
    assert recorder.invocations == [
        ToolInvocation(name="lookup", arguments={"id": "A1"}, result="order A1")
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
    """The recorder observes; a tool that failed must still fail its caller.

    Swallowing the exception here would turn a broken tool into a silent one and
    let the agent's error handling go unevaluated.
    """

    def refund(args: dict[str, Any]) -> str:
        raise RuntimeError("gateway timeout")

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"refund": refund})

    with pytest.raises(RuntimeError, match="gateway timeout"):
        wrapped["refund"]({"id": "A1"})

    assert recorder.invocations == [
        ToolInvocation(name="refund", arguments={"id": "A1"}, error="gateway timeout")
    ]


@pytest.mark.asyncio
async def test_concurrent_calls_keep_their_start_order() -> None:
    """Order is call order, not completion order.

    A judge asked whether the agent called `search` before `refund` is reading a
    sequence, so a trajectory reordered by which tool happened to return first
    would answer a different question than the one asked.
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

    # Every call ran: truncation bounds the record, never the agent's behavior.
    assert calls == [0, 1, 2, 3, 4]
    assert len(recorder.invocations) == 2
    assert recorder.omitted == 3


def test_native_tools_pass_through_unwrapped_and_undescribed() -> None:
    """A provider-executed tool is invisible, so it is not advertised either.

    Listing it as available while never being able to show a call to it would
    let a judge conclude the model ignored a tool it may well have used.
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

    assert recorder.invocations[0].arguments == {"id": "A1"}


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
    rendered = render_trajectory(
        [ToolInvocation(name="fetch", arguments={}, result="x" * 5000)],
        observable_tools=["fetch"],
    )

    assert f"   result: {'x' * MAX_RECORDED_VALUE_CHARS}… (truncated)" in rendered


def test_render_serializes_unserializable_values_without_raising() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "<opaque>"

    rendered = render_trajectory(
        [ToolInvocation(name="fetch", arguments={"k": Opaque()}, result=Opaque())],
        observable_tools=["fetch"],
    )

    assert "<opaque>" in rendered


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
    """A string result is passed through, not JSON-encoded, so it never was
    escaped -- this pins that the two paths agree now that the JSON one does
    not escape either.
    """
    rendered = render_trajectory(
        [ToolInvocation(name="lookup", arguments={}, result="café 東京")],
        observable_tools=["lookup"],
    )

    assert "   result: café 東京" in rendered


def test_a_sync_tool_returning_an_awaitable_records_on_completion() -> None:
    """A sync callable can still hand back an awaitable.

    The record completes when someone awaits it, so a judge never reads the
    repr of a pending coroutine. A caller who never awaits records nothing,
    which is accurate: the call never completed.
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
    """Synthetic routing tools are not tools the agent was given.

    graph.route injects them onto a multi-edge node and a per-node judge is
    scored against the node's *original* config, which does not list them --
    so showing them invites the judge to grade a handoff as tool use. §3.8's
    tracking wrapper excludes them for the same reason.
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
    """The implementation map can be wider than what the model was offered.

    Online, config() merges a Registry's tools into the map it hands the
    handler while the flag variation decides what the model sees. Describing
    the whole registry made a judge penalise an agent for ignoring tools it
    was never offered.
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


def test_no_exposed_set_means_every_callable_is_in_scope() -> None:
    """The offline case: the runner resolves the config's tools from the same
    map, so the two agree by construction and no filter is needed.
    """
    recorder = TrajectoryRecorder()
    recorder.wrap({"lookup": lambda args: "a", "refund": lambda args: "b"})

    assert recorder.observable_tools == ["lookup", "refund"]
