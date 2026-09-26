"""One message_history for every judge path.

The three paths -- online inline, online deferred, and offline evaluations --
each joined their own, and disagreed. These tests hold them to
``build_message_history``.
"""

from __future__ import annotations

import pickle
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import JudgeTask, ProviderHandler, run_judge, run_judges
from launchdarkly_ai_server.judge_scoring import (
    FORMATTING_INSTRUCTIONS,
    build_message_history,
)
from launchdarkly_ai_server.tracking import execute_and_track
from launchdarkly_ai_server.trajectory import TrajectoryRecorder, render_trajectory

CONTEXT = {"kind": "user", "key": "u1"}

JUDGE_CONFIG = {
    "model": {"name": "gpt-4"},
    "provider": {"name": "TestProvider"},
    "instructions": "Judge it",
}


@pytest.fixture
def mock_ld_client() -> Any:
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    client.variation = AsyncMock(return_value=None)
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


def capturing_judge_handler(seen: list[dict[str, Any]]) -> ProviderHandler:
    async def fn(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        seen.append(dict(variables or {}))
        return {
            "output": '{"score": 1, "reasoning": "ok"}',
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    return ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]


def judged_config() -> dict[str, Any]:
    return {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "hi",
        "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1}]},
    }


@pytest.fixture
def judge_variation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract_variation(key: str, context: Any) -> dict[str, Any]:
        return {
            "config": dict(JUDGE_CONFIG),
            "meta": {"variationKey": "v", "version": 1},
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.lifecycle.extract_variation", fake_extract_variation
    )


# ─── the builder itself ──────────────────────────────────────────────────────


def test_builder_orders_the_conversation_and_appends_the_format_block() -> None:
    history = build_message_history(user_input="Q", trajectory="T", output="A")

    assert history == f"Q\n\nT\n\nA\n\n{FORMATTING_INSTRUCTIONS}"


def test_builder_skips_empty_parts() -> None:
    """Keeps a judge authored before trajectories existed scoring unchanged."""
    assert build_message_history(user_input="Q", trajectory="", output="A") == (
        f"Q\n\nA\n\n{FORMATTING_INSTRUCTIONS}"
    )
    assert build_message_history(output="A") == f"A\n\n{FORMATTING_INSTRUCTIONS}"


def test_builder_always_carries_the_format_block() -> None:
    """Judges from the AI Library's templates read the JSON shape from here."""
    assert FORMATTING_INSTRUCTIONS in build_message_history()


# ─── online: inline ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inline_judge_is_shown_the_trajectory(
    mock_ld_client: Any, judge_variation: None
) -> None:
    seen: list[dict[str, Any]] = []
    await run_judges(
        config=judged_config(),
        user_context=CONTEXT,
        handler=capturing_judge_handler(seen),
        user_input="Where is order A1?",
        llm_response="It shipped.",
        base_track_data={},
        trajectory="Tools available: lookup\n1. lookup\n   result: shipped",
    )

    history = seen[0]["message_history"]
    assert "1. lookup" in history
    assert (
        history.index("Where is order A1?")
        < history.index("1. lookup")
        < history.index("It shipped.")
    )


@pytest.mark.asyncio
async def test_inline_judge_without_a_trajectory_is_unchanged(
    mock_ld_client: Any, judge_variation: None
) -> None:
    seen: list[dict[str, Any]] = []
    await run_judges(
        config=judged_config(),
        user_context=CONTEXT,
        handler=capturing_judge_handler(seen),
        user_input="Q",
        llm_response="A",
        base_track_data={},
    )

    assert seen[0]["message_history"] == build_message_history(
        user_input="Q", output="A"
    )


# ─── online: deferred ────────────────────────────────────────────────────────


def deferred_task(**overrides: Any) -> JudgeTask:
    fields: dict[str, Any] = {
        "config_key": "judge-1",
        "judge_config": dict(JUDGE_CONFIG),
        "judge_meta": {"variationKey": "v", "version": 1},
        "actual_output": "It shipped.",
        "user_context": CONTEXT,
        "judge_provider": "TestProvider",
        "judge_mode": "messages",
        "collapse_messages": False,
        "parent_track_data": {},
    }
    fields.update(overrides)
    return JudgeTask(**fields)


@pytest.mark.asyncio
async def test_deferred_judge_is_shown_the_input_and_the_trajectory(
    mock_ld_client: Any,
) -> None:
    """This path carried neither before, grading a response in isolation."""
    seen: list[dict[str, Any]] = []
    task = deferred_task(
        user_input="Where is order A1?",
        trajectory="Tools available: lookup\n1. lookup\n   result: shipped",
    )

    await run_judge(task, [capturing_judge_handler(seen)])

    history = seen[0]["message_history"]
    assert (
        history.index("Where is order A1?")
        < history.index("1. lookup")
        < history.index("It shipped.")
    )


@pytest.mark.asyncio
async def test_deferred_and_inline_agree_on_the_same_row(
    mock_ld_client: Any, judge_variation: None
) -> None:
    """The point of the shared builder: same inputs, identical history."""
    inline_seen: list[dict[str, Any]] = []
    deferred_seen: list[dict[str, Any]] = []
    trajectory = "Tools available: lookup\n1. lookup\n   result: shipped"

    await run_judges(
        config=judged_config(),
        user_context=CONTEXT,
        handler=capturing_judge_handler(inline_seen),
        user_input="Where is order A1?",
        llm_response="It shipped.",
        base_track_data={},
        trajectory=trajectory,
    )
    await run_judge(
        deferred_task(user_input="Where is order A1?", trajectory=trajectory),
        [capturing_judge_handler(deferred_seen)],
    )

    assert inline_seen[0]["message_history"] == deferred_seen[0]["message_history"]


def test_judge_task_stays_picklable_with_the_new_fields() -> None:
    """JudgeTask crosses a thread boundary, so it must stay primitives."""
    task = deferred_task(user_input="Q", trajectory="T")

    assert pickle.loads(pickle.dumps(task)).trajectory == "T"


# ─── online: capture ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_and_track_records_the_trajectory(mock_ld_client: Any) -> None:
    def lookup(args: dict[str, Any]) -> str:
        return f"order {args['id']} shipped"

    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        # Awaited: wrap_tool_handlers wraps the recorder's wrapper and is
        # always async, so an online handler awaits as it always has. Offline
        # there is no such wrapper -- see test_trajectory.py.
        await tool_handlers["lookup"]({"id": "A1"})
        return {"output": "It shipped.", "usage": {}}

    result = await execute_and_track(
        config_key="c",
        config={
            "model": {"name": "m"},
            "provider": {"name": "TestProvider"},
            # Only tools the config offers are described.
            "tools": {"lookup": {"description": "", "parameters": {}}},
        },
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="Where is order A1?",
        tool_handlers={"lookup": lookup},
    )

    assert "Tools available: lookup" in result["trajectory"]
    assert '1. lookup\n   arguments: {"id":"A1"}' in result["trajectory"]
    assert "result: order A1 shipped" in result["trajectory"]


@pytest.mark.asyncio
async def test_a_native_tool_is_not_recorded_online_either(
    mock_ld_client: Any,
) -> None:
    """The stub wrap_tool_handlers substitutes for a native tool is callable,
    so the call *is* observable online -- but it returns nothing, so recording
    it would show a judge a call with an empty result. Both paths skip it.
    """
    from launchdarkly_ai_server import NativeTool

    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        # Not awaited: the native stub is sync, unlike the async wrapper a
        # real callable gets. Pre-existing asymmetry.
        tool_handlers["web_search"]({"q": "x"})
        return {"output": "done", "usage": {}}

    result = await execute_and_track(
        config_key="c",
        config={"model": {"name": "m"}, "provider": {"name": "TestProvider"}},
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="q",
        tool_handlers={"web_search": NativeTool("WebSearch")},
    )

    assert result["trajectory"] == ""


@pytest.mark.asyncio
async def test_a_native_tool_declared_in_the_config_is_not_listed_as_available(
    mock_ld_client: Any,
) -> None:
    """A NativeTool's calls can never be recorded locally, so listing it under
    "Tools available" would read as "the model had this and did not use it"
    even when the provider ran it -- the same false negative a config with no
    tools at all avoids by rendering no trajectory block.
    """
    from launchdarkly_ai_server import NativeTool

    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        tool_handlers["web_search"]({"q": "x"})
        return {"output": "done", "usage": {}}

    result = await execute_and_track(
        config_key="c",
        config={
            "model": {"name": "m"},
            "provider": {"name": "TestProvider"},
            "tools": {"web_search": {"description": "", "parameters": {}}},
        },
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="q",
        tool_handlers={"web_search": NativeTool("WebSearch")},
    )

    assert result["trajectory"] == ""


@pytest.mark.asyncio
async def test_tool_tracking_still_fires_under_the_recorder(
    mock_ld_client: Any,
) -> None:
    """Recording composes inside wrap_tool_handlers, which must still fire."""

    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        await tool_handlers["lookup"]({"id": "A1"})
        return {"output": "ok", "usage": {}}

    await execute_and_track(
        config_key="c",
        config={
            "model": {"name": "m"},
            "provider": {"name": "TestProvider"},
            "tools": {"lookup": {"description": "", "parameters": {}}},
        },
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="q",
        tool_handlers={"lookup": lambda args: "shipped"},
    )

    tool_events = [
        call.args
        for call in mock_ld_client.track.call_args_list
        if call.args[0] == "$ld:ai:tool_call"
    ]
    assert len(tool_events) == 1
    assert tool_events[0][2]["toolKey"] == "lookup"


@pytest.mark.asyncio
async def test_a_run_with_no_tools_reports_no_trajectory(mock_ld_client: Any) -> None:
    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        return {"output": "ok", "usage": {}}

    result = await execute_and_track(
        config_key="c",
        config={"model": {"name": "m"}, "provider": {"name": "TestProvider"}},
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="q",
    )

    assert result["trajectory"] == ""


def test_the_recorder_renders_identically_for_both_paths() -> None:
    """Both paths call render_trajectory, so one fixture pins the shape."""
    recorder = TrajectoryRecorder()
    recorder.wrap({"lookup": lambda args: "shipped"})

    assert render_trajectory(
        recorder.invocations,
        observable_tools=recorder.observable_tools,
        omitted=recorder.omitted,
    ) == (
        "Tools available: lookup\nNo tool calls were made while producing the response."
    )


@pytest.mark.asyncio
async def test_a_registry_tool_the_config_omits_is_not_described(
    mock_ld_client: Any,
) -> None:
    """Describing a registry tool the variation omits would let a judge
    penalise an agent for ignoring a tool it never had.
    """

    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        await tool_handlers["lookup"]({"id": "A1"})
        return {"output": "ok", "usage": {}}

    result = await execute_and_track(
        config_key="c",
        config={
            "model": {"name": "m"},
            "provider": {"name": "TestProvider"},
            # The variation offers one tool; the map carries two.
            "tools": {"lookup": {"description": "", "parameters": {}}},
        },
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="q",
        tool_handlers={
            "lookup": lambda args: "shipped",
            "registry_only": lambda args: "never offered",
        },
    )

    assert "Tools available: lookup" in result["trajectory"]
    assert "registry_only" not in result["trajectory"]


@pytest.mark.asyncio
async def test_a_config_tool_with_no_implementation_is_still_available(
    mock_ld_client: Any,
) -> None:
    """The config's own tool catalog determines what the model was offered,
    independent of whether a local implementation was registered for it -- a
    handler package can build the provider's tool list straight from the
    config. Omitting an unimplemented tool from "Tools available" would let a
    judge read an incomplete catalog instead of one the agent chose not to use.
    """

    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        return {"output": "I do not know.", "usage": {}}

    result = await execute_and_track(
        config_key="c",
        config={
            "model": {"name": "m"},
            "provider": {"name": "TestProvider"},
            "tools": {"lookup_order": {"description": "", "parameters": {}}},
        },
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="q",
        tool_handlers=None,
    )

    assert (
        "Tools available: lookup_order\n"
        "No tool calls were made while producing the response."
    ) in result["trajectory"]


@pytest.mark.asyncio
async def test_a_handoff_tool_is_not_described_online(mock_ld_client: Any) -> None:
    """A per-node judge is scored against the node's original config, which
    lists no handoffs -- so a handoff must not read as tool use.
    """

    async def handler(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        # Handoff tools stay sync so routing can record the choice with a bare
        # call, so this must not be awaited either.
        tool_handlers["__handoff_billing"]({})
        return {"output": "ok", "usage": {}}

    result = await execute_and_track(
        config_key="c",
        config={
            "model": {"name": "m"},
            "provider": {"name": "TestProvider"},
            "tools": {"__handoff_billing": {"description": "", "parameters": {}}},
        },
        meta={"variationKey": "v", "version": 1},
        user_context=CONTEXT,
        handler=handler,  # type: ignore[arg-type]
        user_input="q",
        tool_handlers={"__handoff_billing": lambda args: "Handoff recorded"},
    )

    assert result["trajectory"] == ""
