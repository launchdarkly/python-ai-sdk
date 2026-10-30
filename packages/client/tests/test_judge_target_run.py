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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, run_judges
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


async def _judge_fn(
    config: Any,
    user_input: Any,
    tool_handlers: Any,
    variables: Any,
    history: Any = None,
) -> dict[str, Any]:
    # Stands in for a judge handler: it opens the judge's invoke_agent and a
    # chat span under it, on an SDK tracer, after the graded run is over.
    with _tracer.start_as_current_span("invoke_agent"):
        _tracer.start_span("chat").end()
    return {
        "output": '{"score": 1, "reasoning": "ok"}',
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


HANDLER = ProviderHandler(fn=_judge_fn, provides_for=("Anthropic", "messages"))  # type: ignore[arg-type]


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


def _expect_every_span_points_at_the_graded_run() -> None:
    spans = _exporter.get_finished_spans()
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
        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config={
                    **JUDGE_CONFIG,
                    "judgeConfiguration": {
                        "judges": [{"key": "no-nitpicks", "samplingRate": 1.0}]
                    },
                },
                user_context=CONTEXT,
                handler=HANDLER,
                user_input=None,
                llm_response="Verdict: approve",
                base_track_data=dict(GRADED_RUN),
            )
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
        [span] = _exporter.get_finished_spans()
        attributes = span.attributes or {}
        assert LD_JUDGE_TARGET_RUN_ID not in attributes
        assert LD_JUDGE_TARGET_CONFIG_KEY not in attributes
        assert LD_JUDGE_TARGET_VARIATION_KEY not in attributes
