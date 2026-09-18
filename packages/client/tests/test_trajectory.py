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


@pytest.mark.asyncio
async def test_wrapped_tool_returns_what_the_original_returned() -> None:
    def lookup(args: dict[str, Any]) -> str:
        return f"order {args['id']}"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})

    assert await wrapped["lookup"]({"id": "A1"}) == "order A1"
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

    assert await wrapped["lookup"]({"id": "A1"}) == "order A1"
    assert recorder.invocations[0].result == "order A1"


@pytest.mark.asyncio
async def test_wrapped_tool_reraises_and_records_the_failure() -> None:
    """The recorder observes; a tool that failed must still fail its caller.

    Swallowing the exception here would turn a broken tool into a silent one and
    let the agent's error handling go unevaluated.
    """

    def refund(args: dict[str, Any]) -> str:
        raise RuntimeError("gateway timeout")

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"refund": refund})

    with pytest.raises(RuntimeError, match="gateway timeout"):
        await wrapped["refund"]({"id": "A1"})

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


@pytest.mark.asyncio
async def test_calls_past_the_limit_still_execute_but_are_only_counted() -> None:
    calls: list[int] = []

    def append(args: dict[str, Any]) -> str:
        calls.append(args["n"])
        return "ok"

    recorder = TrajectoryRecorder(limit=2)
    wrapped = recorder.wrap({"append": append})
    for n in range(5):
        await wrapped["append"]({"n": n})

    # Every call ran: truncation bounds the record, never the agent's behavior.
    assert calls == [0, 1, 2, 3, 4]
    assert len(recorder.invocations) == 2
    assert recorder.omitted == 3


@pytest.mark.asyncio
async def test_native_tools_pass_through_unwrapped_and_undescribed() -> None:
    """A provider-executed tool is invisible, so it is not advertised either.

    Listing it as available while never being able to show a call to it would
    let a judge conclude the model ignored a tool it may well have used.
    """
    native = NativeTool("WebSearch")
    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"web_search": native, "lookup": lambda args: "ok"})

    assert wrapped["web_search"] is native
    assert recorder.observable_tools == ["lookup"]


@pytest.mark.asyncio
async def test_keyword_arguments_are_recorded() -> None:
    def lookup(**kwargs: Any) -> str:
        return "ok"

    recorder = TrajectoryRecorder()
    wrapped = recorder.wrap({"lookup": lookup})
    await wrapped["lookup"](id="A1")

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
