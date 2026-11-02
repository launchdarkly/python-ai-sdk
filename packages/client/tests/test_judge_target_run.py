"""A judge's spans point at the run it graded.

A judge runs after the graded handler has ended its span, so its spans start a
trace of their own. Every span of a judge's run carries the run id, config key
and variation key of the run it graded, so the two traces can be joined exactly
instead of by timestamp.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, graph, run_judges
from launchdarkly_ai_server.conversation import ConversationIdSpanProcessor
from launchdarkly_ai_server.judges import (
    LD_JUDGE_TARGET_CONFIG_KEY,
    LD_JUDGE_TARGET_RUN_ID,
    LD_JUDGE_TARGET_VARIATION_KEY,
    run_judge,
)
from launchdarkly_ai_server.types import JudgeTask

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(ConversationIdSpanProcessor())
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
_tracer = _provider.get_tracer("@launchdarkly/ai-claude-messages")
_foreign_tracer = _provider.get_tracer("opentelemetry.instrumentation.httpx")

TARGET_KEYS = (
    LD_JUDGE_TARGET_RUN_ID,
    LD_JUDGE_TARGET_CONFIG_KEY,
    LD_JUDGE_TARGET_VARIATION_KEY,
)

CONTEXT = {"kind": "user", "key": "alexis"}

GRADED_RUN = {
    "runId": "graded-run-id",
    "configKey": "pr-reviewer",
    "variationKey": "opus",
    "version": 3,
    "modelName": "claude-opus-5-5",
    "providerName": "Anthropic",
}

JUDGE_CONFIG = {
    "model": {"name": "claude-sonnet-5"},
    "provider": {"name": "Anthropic"},
    "instructions": "You are a judge.",
    "evaluationMetricKey": "$ld:ai:judge:no-nitpicks",
}
JUDGE_META = {
    "enabled": True,
    "variationKey": "default",
    "version": 1,
    "mode": "messages",
}

JUDGE_OUTPUT = {
    "output": '{"score": 1, "reasoning": "ok"}',
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _open_judge_spans(label: str = "judge") -> None:
    """Stands in for a judge handler's spans.

    Opens the judge's ``invoke_agent`` and a ``chat`` span under it on an SDK
    tracer, after the graded run is over. ``label`` lets a test tell apart the
    spans of different judges in one trace.
    """
    with _tracer.start_as_current_span(
        "invoke_agent", attributes={"test.label": label}
    ):
        _tracer.start_span("chat", attributes={"test.label": label}).end()


async def _judge_fn(
    config: Any,
    user_input: Any,
    tool_handlers: Any,
    variables: Any,
    history: Any = None,
) -> dict[str, Any]:
    _open_judge_spans()
    return JUDGE_OUTPUT


HANDLER = ProviderHandler(fn=_judge_fn, provides_for=("Anthropic", "messages"))  # type: ignore[arg-type]


def _finished() -> list[ReadableSpan]:
    return list(_exporter.get_finished_spans())


@pytest.fixture(autouse=True)
def _reset_exporter() -> Iterator[None]:
    _exporter.clear()
    yield
    _exporter.clear()


@pytest.fixture
def mock_ld_client() -> Iterator[MagicMock]:
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    client.variation = AsyncMock(return_value={**JUDGE_CONFIG, "_ldMeta": JUDGE_META})
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


async def _run_inline(
    handler: ProviderHandler = HANDLER,
    base_track_data: dict[str, Any] | None = None,
) -> None:
    with patch.object(random, "random", return_value=0.0):
        await run_judges(
            config={
                **JUDGE_CONFIG,
                "judgeConfiguration": {
                    "judges": [{"key": "no-nitpicks", "samplingRate": 1.0}]
                },
            },
            user_context=CONTEXT,
            handler=handler,
            user_input=None,
            llm_response="Verdict: approve",
            base_track_data=dict(
                GRADED_RUN if base_track_data is None else base_track_data
            ),
        )


def _expect_every_span_points_at_the_graded_run() -> None:
    spans = _finished()
    assert sorted(span.name for span in spans) == ["chat", "invoke_agent"]
    for span in spans:
        attributes = span.attributes or {}
        assert attributes[LD_JUDGE_TARGET_RUN_ID] == "graded-run-id"
        assert attributes[LD_JUDGE_TARGET_CONFIG_KEY] == "pr-reviewer"
        assert attributes[LD_JUDGE_TARGET_VARIATION_KEY] == "opus"


class TestJudgeSpansPointAtTheRunTheyGraded:
    async def test_stamps_every_span_of_an_inline_judge_with_the_graded_run(
        self, mock_ld_client: MagicMock
    ) -> None:
        await _run_inline()
        _expect_every_span_points_at_the_graded_run()

    async def test_stamps_every_span_of_a_deferred_judge_with_the_graded_run_from_the_task(
        self, mock_ld_client: MagicMock
    ) -> None:
        task = JudgeTask(
            config_key="no-nitpicks",
            judge_config=dict(JUDGE_CONFIG),
            judge_meta=dict(JUDGE_META),  # type: ignore[arg-type]
            actual_output="Verdict: approve",
            user_context=CONTEXT,
            judge_provider="Anthropic",
            judge_mode="messages",
            collapse_messages=False,
            parent_track_data=dict(GRADED_RUN),
        )
        await run_judge(task, [HANDLER])
        _expect_every_span_points_at_the_graded_run()

    def test_leaves_spans_outside_a_judge_run_alone(self) -> None:
        _tracer.start_span("invoke_agent").end()
        [span] = _finished()
        attributes = span.attributes or {}
        assert not [key for key in TARGET_KEYS if key in attributes]

    async def test_leaves_spans_from_other_tracers_alone(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
            _open_judge_spans()
            _foreign_tracer.start_span("POST").end()
            return JUDGE_OUTPUT

        await _run_inline(
            ProviderHandler(fn=fn, provides_for=("Anthropic", "messages"))
        )  # type: ignore[arg-type]
        [foreign] = [span for span in _finished() if span.name == "POST"]
        attributes = foreign.attributes or {}
        assert not [key for key in TARGET_KEYS if key in attributes]

    async def test_writes_the_attributes_with_content_capture_off(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = ProviderHandler(
            fn=_judge_fn,  # type: ignore[arg-type]
            provides_for=("Anthropic", "messages"),
            capture_content=False,
        )
        await _run_inline(handler)
        _expect_every_span_points_at_the_graded_run()

    async def test_omits_an_empty_value_and_writes_the_others(
        self, mock_ld_client: MagicMock
    ) -> None:
        await _run_inline(base_track_data={**GRADED_RUN, "variationKey": ""})
        spans = _finished()
        assert sorted(span.name for span in spans) == ["chat", "invoke_agent"]
        for span in spans:
            attributes = span.attributes or {}
            assert LD_JUDGE_TARGET_VARIATION_KEY not in attributes
            assert attributes[LD_JUDGE_TARGET_RUN_ID] == "graded-run-id"
            assert attributes[LD_JUDGE_TARGET_CONFIG_KEY] == "pr-reviewer"

    async def test_keeps_a_value_the_span_already_has(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
            _open_judge_spans()
            _tracer.start_span("own", attributes={LD_JUDGE_TARGET_RUN_ID: "own"}).end()
            return JUDGE_OUTPUT

        await _run_inline(
            ProviderHandler(fn=fn, provides_for=("Anthropic", "messages"))
        )  # type: ignore[arg-type]
        [own] = [span for span in _finished() if span.name == "own"]
        attributes = own.attributes or {}
        assert attributes[LD_JUDGE_TARGET_RUN_ID] == "own"
        assert attributes[LD_JUDGE_TARGET_CONFIG_KEY] == "pr-reviewer"


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

GRAPH_VARIATION = {
    "root": "root-node",
    "edges": {},
    "_ldMeta": {"enabled": True, "variationKey": "graph-v2", "version": 2},
}
NODE_INSTRUCTIONS = "I am the root node."
NODE_JUDGE_INSTRUCTIONS = "You judge the node."
GRAPH_JUDGE_INSTRUCTIONS = "You judge the graph."


def _graph_variation_for(key: str) -> dict[str, Any]:
    if key == "graph-key":
        return GRAPH_VARIATION
    judge_meta = {
        "enabled": True,
        "variationKey": "j1",
        "version": 1,
        "mode": "messages",
    }
    if key == "node-judge":
        return {
            **JUDGE_CONFIG,
            "provider": {"name": "TestProvider"},
            "instructions": NODE_JUDGE_INSTRUCTIONS,
            "evaluationMetricKey": "$ld:ai:judge:node",
            "_ldMeta": judge_meta,
        }
    if key == "graph-judge":
        return {
            **JUDGE_CONFIG,
            "provider": {"name": "TestProvider"},
            "instructions": GRAPH_JUDGE_INSTRUCTIONS,
            "evaluationMetricKey": "$ld:ai:judge:graph",
            "_ldMeta": judge_meta,
        }
    return {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": NODE_INSTRUCTIONS,
        "judgeConfiguration": {"judges": [{"key": "node-judge", "samplingRate": 1.0}]},
        "_ldMeta": {
            "enabled": True,
            "variationKey": "node-v1",
            "version": 1,
            "mode": "messages",
        },
    }


async def _graph_handler_fn(
    config: Any,
    user_input: Any,
    tool_handlers: Any,
    variables: Any,
    history: Any = None,
) -> dict[str, Any]:
    instructions = config.get("instructions", "")
    if instructions == NODE_JUDGE_INSTRUCTIONS:
        _open_judge_spans("node-judge")
        return JUDGE_OUTPUT
    if instructions == GRAPH_JUDGE_INSTRUCTIONS:
        _open_judge_spans("graph-judge")
        return JUDGE_OUTPUT
    _open_judge_spans("node")
    return {"output": "final", "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.fixture
def graph_ld_client() -> Iterator[MagicMock]:
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()

    async def variation(key: str, ctx: Any, default: Any) -> Any:
        return _graph_variation_for(key)

    client.variation = AsyncMock(side_effect=variation)
    lifecycle_module._set_client_for_testing(client)
    # graph.py opens the graph span on trace.get_tracer, and the global provider
    # is claimed process-wide, so hand it this file's tracer instead.
    with patch.object(trace, "get_tracer", return_value=_tracer):
        yield client
    lifecycle_module._reset_for_testing()


def _metric_track_data(client: MagicMock, event_key: str) -> dict[str, Any]:
    [call] = [c for c in client.track.call_args_list if c.args[0] == event_key]
    return dict(call.args[2])


def _labelled(label: str) -> list[ReadableSpan]:
    return [s for s in _finished() if (s.attributes or {}).get("test.label") == label]


class TestGraphJudgeTargets:
    async def test_node_and_graph_judges_point_at_the_runs_they_graded(
        self, graph_ld_client: MagicMock
    ) -> None:
        handler = ProviderHandler(
            fn=_graph_handler_fn,  # type: ignore[arg-type]
            provides_for=("TestProvider", "messages"),
        )
        await graph("graph-key", handlers=[handler], graph_judge="graph-judge").invoke(
            "hi", CONTEXT
        )

        # A per-node judge points at the node's own run: the run its metric event names.
        node_metric = _metric_track_data(graph_ld_client, "$ld:ai:judge:node")
        assert node_metric["configKey"] == "root-node"
        node_judge_spans = _labelled("node-judge")
        assert sorted(s.name for s in node_judge_spans) == ["chat", "invoke_agent"]
        for span in node_judge_spans:
            attributes = span.attributes or {}
            assert attributes[LD_JUDGE_TARGET_RUN_ID] == node_metric["runId"]
            assert attributes[LD_JUDGE_TARGET_CONFIG_KEY] == "root-node"
            assert attributes[LD_JUDGE_TARGET_VARIATION_KEY] == "node-v1"

        # The graph-level judge points at the graph run.
        [graph_span] = [s for s in _finished() if s.name == "launchdarkly.graph"]
        graph_run_id = (graph_span.attributes or {})["launchdarkly.run.id"]
        graph_metric = _metric_track_data(graph_ld_client, "$ld:ai:judge:graph")
        assert graph_metric["runId"] == graph_run_id
        assert graph_run_id != node_metric["runId"]
        graph_judge_spans = _labelled("graph-judge")
        assert sorted(s.name for s in graph_judge_spans) == ["chat", "invoke_agent"]
        for span in graph_judge_spans:
            attributes = span.attributes or {}
            assert attributes[LD_JUDGE_TARGET_RUN_ID] == graph_run_id
            assert attributes[LD_JUDGE_TARGET_CONFIG_KEY] == "graph-key"
            assert attributes[LD_JUDGE_TARGET_VARIATION_KEY] == "graph-v2"

        # The graded node's own spans are not a judge run.
        for span in _labelled("node"):
            attributes = span.attributes or {}
            assert not [key for key in TARGET_KEYS if key in attributes]
