"""Functional evaluators and evaluator workflows (prototype)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
import launchdarkly_ai_server.tracking as tracking_module
from launchdarkly_ai_server.evaluations import (
    EvaluationsError,
    EvaluatorContext,
    init_evaluations,
)
from launchdarkly_ai_server.evaluations.api import HttpResponse
from launchdarkly_ai_server.evaluations.workflow import (
    WorkflowStep,
    aggregate_scores,
    validate_steps,
)
from launchdarkly_ai_server.types import ProviderHandler

PII = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|[\w.]+@[\w.]+\.\w+")


def tool_response(key: str, definition: dict[str, Any], **extra: Any) -> HttpResponse:
    body = {
        "key": key,
        "version": 3,
        "description": f"{key} description",
        "schema": {},
        "customParameters": {"ldEvaluator": definition},
        **extra,
    }
    return HttpResponse(status=200, body=json.dumps(body))


class Router:
    """Answers GET .../ai-tools/<key> from a dict."""

    def __init__(self, tools: dict[str, HttpResponse]) -> None:
        self.tools = tools

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: Any, timeout: float
    ) -> HttpResponse:
        key = url.split("?")[0].rsplit("/", 1)[-1]
        return self.tools.get(key, HttpResponse(status=404, body="{}"))


@pytest.fixture
def evals(monkeypatch: pytest.MonkeyPatch) -> Any:
    lifecycle_module._reset_for_testing()
    monkeypatch.setenv("LD_API_TOKEN", "api-x")
    monkeypatch.setenv("LD_PROJECT_KEY", "proj")
    monkeypatch.setenv("LD_SDK_KEY", "sdk-x")
    tools = {
        "pii-regex": tool_response(
            "pii-regex",
            {
                "kind": "functional",
                "successDirection": "lower_is_better",
                "threshold": 0.0,
            },
        ),
        "topic-guard": tool_response(
            "topic-guard",
            {"kind": "functional", "successDirection": "higher_is_better"},
        ),
        "support-quality": tool_response(
            "support-quality",
            {
                "kind": "workflow",
                "threshold": 0.7,
                "steps": [
                    {"id": "topics", "type": "judge", "ref": "extract-topics"},
                    {"id": "pii", "type": "functional", "ref": "pii-regex"},
                    {
                        "id": "topic_quality",
                        "type": "judge",
                        "ref": "judge-topics",
                        "uses": ["topics"],
                    },
                    {
                        "id": "final",
                        "type": "aggregate",
                        "method": "weighted_mean",
                        "inputs": [
                            {"step": "pii", "weight": 1},
                            {"step": "topic_quality", "weight": 3},
                        ],
                    },
                ],
            },
        ),
        "plain-tool": HttpResponse(
            status=200, body=json.dumps({"key": "plain-tool", "version": 1})
        ),
    }
    module = init_evaluations(transport=Router(tools))
    yield module
    lifecycle_module._reset_for_testing()


def pii_count(ctx: EvaluatorContext) -> float:
    # 0 hits -> 0.0, anything else -> 1.0. lower_is_better, so 0.0 is the pass.
    return 1.0 if PII.search(str(ctx.output)) else 0.0


@pytest.fixture
def fake_judges(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace variation lookup and the LLM call; record what each judge saw."""
    seen: list[dict[str, Any]] = []

    async def extract_variation(key: str, context: Any) -> dict[str, Any]:
        return {
            "config": {"provider": {"name": "openai"}, "messages": []},
            "meta": {"mode": "completion"},
            "value": None,
        }

    async def execute_and_track(**kwargs: Any) -> dict[str, Any]:
        key = kwargs["config_key"]
        seen.append({"key": key, "history": kwargs["variables"]["message_history"]})
        answers = {
            "extract-topics": {
                "score": 1.0,
                "reasoning": "ok",
                "topics": ["refunds", "shipping"],
            },
            "judge-topics": {"score": 0.8, "reasoning": "on topic"},
        }
        return {
            "response": json.dumps(answers[key]),
            "usage": {"input": 1, "output": 1, "total": 2},
            "track_data": {},
        }

    monkeypatch.setattr(lifecycle_module, "extract_variation", extract_variation)
    monkeypatch.setattr(tracking_module, "execute_and_track", execute_and_track)
    return seen


def handler() -> ProviderHandler:
    async def fn(*args: Any, **kwargs: Any) -> Any:  # never called: judges are faked
        raise AssertionError

    return ProviderHandler(fn, provides_for=("openai", "messages"))


@pytest.mark.asyncio
async def test_functional_evaluator_binds_to_ld_object(evals: Any) -> None:
    pii = await evals.evaluators.get("pii-regex", implementation=pii_count)
    assert pii.success_direction == "lower_is_better"
    assert pii.threshold == 0.0
    assert pii.version == 3
    score, _reasoning, _ = await pii.run(
        EvaluatorContext(input="q", output="mail me a@b.co", steps={})
    )
    assert score == 1.0
    assert pii.normalize(score) == 0.0


@pytest.mark.asyncio
async def test_a_plain_tool_is_not_an_evaluator(evals: Any) -> None:
    with pytest.raises(EvaluationsError, match="not a functional"):
        await evals.evaluators.get("plain-tool", implementation=pii_count)
    with pytest.raises(EvaluationsError, match="not found"):
        await evals.evaluators.get("nope", implementation=pii_count)


@pytest.mark.asyncio
async def test_implementation_may_return_mapping_bool_or_async(evals: Any) -> None:
    async def fn(ctx: EvaluatorContext) -> dict[str, Any]:
        return {"score": True, "reasoning": "fine", "extra": 1}

    guard = await evals.evaluators.get("topic-guard", implementation=fn)
    score, reasoning, raw = await guard.run(
        EvaluatorContext(input=None, output="x", steps={})
    )
    assert (score, reasoning, raw["extra"]) == (1.0, "fine", 1)

    bad = await evals.evaluators.get("topic-guard", implementation=lambda ctx: 7)
    with pytest.raises(ValueError, match="number from 0 to 1"):
        await bad.run(EvaluatorContext(input=None, output="x", steps={}))


@pytest.mark.asyncio
async def test_workflow_chains_judge_regex_judge_and_aggregate(
    evals: Any, fake_judges: list[dict[str, Any]]
) -> None:
    workflow = await evals.workflows.get(
        "support-quality", implementations={"pii-regex": pii_count}
    )
    result = await workflow.run(
        input="Where is my order?",
        output="Your refund ships Monday.",
        handlers=[handler()],
        context={"kind": "user", "key": "u1"},
    )

    assert list(result.steps) == ["topics", "pii", "topic_quality", "final"]
    assert result.steps["topics"].output["topics"] == ["refunds", "shipping"]
    # pii_count returns 0.0 (no PII); lower_is_better normalizes it to 1.0
    assert result.steps["pii"].score == 0.0
    assert result.steps["pii"].normalized == 1.0
    # the second judge was shown the first judge's extracted topics
    topic_judge = next(call for call in fake_judges if call["key"] == "judge-topics")
    assert "refunds" in topic_judge["history"]
    assert topic_judge["history"].rstrip().endswith("Do not include ```json tags.")
    # weighted mean: (1.0 * 1 + 0.8 * 3) / 4 = 0.85
    assert result.score == pytest.approx(0.85)
    assert result.passed is True
    assert result.to_dict()["steps"]["final"]["type"] == "aggregate"


@pytest.mark.asyncio
async def test_pii_hit_drags_the_final_score_down(
    evals: Any, fake_judges: list[dict[str, Any]]
) -> None:
    workflow = await evals.workflows.get(
        "support-quality", implementations={"pii-regex": pii_count}
    )
    result = await workflow.run(
        input="hi",
        output="Sure, it's jane@example.com",
        handlers=[handler()],
        context={"kind": "user", "key": "u1"},
    )
    # (0.0 * 1 + 0.8 * 3) / 4 = 0.6 < 0.7
    assert result.score == pytest.approx(0.6)
    assert result.passed is False


@pytest.mark.asyncio
async def test_a_failing_step_blocks_what_reads_it_and_never_passes(
    evals: Any, fake_judges: list[dict[str, Any]]
) -> None:
    def boom(ctx: EvaluatorContext) -> float:
        raise RuntimeError("regex service down")

    workflow = await evals.workflows.get(
        "support-quality", implementations={"pii-regex": boom}
    )
    result = await workflow.run(
        input="hi",
        output="ok",
        handlers=[handler()],
        context={"kind": "user", "key": "u1"},
    )
    assert "regex service down" in (result.steps["pii"].error or "")
    assert "skipped" in (result.steps["final"].error or "")
    assert result.score is None
    assert result.passed is None


@pytest.mark.asyncio
async def test_missing_implementation_fails_at_get_not_mid_run(evals: Any) -> None:
    with pytest.raises(EvaluationsError, match="pii-regex"):
        await evals.workflows.get("support-quality", implementations={})


@pytest.mark.asyncio
async def test_workflow_is_usable_as_an_offline_scorer(
    evals: Any, fake_judges: list[dict[str, Any]]
) -> None:
    from launchdarkly_ai_server.evaluations.types import DatasetRow

    workflow = await evals.workflows.get(
        "support-quality", implementations={"pii-regex": pii_count}
    )
    scorer = workflow.as_scorer(
        handlers=[handler()], context={"kind": "user", "key": "u"}
    )
    assert scorer.name == "support-quality"
    assert scorer.threshold == 0.7
    score = await scorer.fn(DatasetRow(row_index=0, input="q"), "answer")  # type: ignore[misc]
    assert score == pytest.approx(0.85)


def test_step_validation() -> None:
    def steps(*raw: Mapping[str, Any]) -> list[WorkflowStep]:
        return [WorkflowStep.from_wire(r) for r in raw]

    with pytest.raises(EvaluationsError, match="not an earlier step"):
        validate_steps(steps({"id": "a", "type": "judge", "ref": "j", "uses": ["b"]}))
    with pytest.raises(EvaluationsError, match="used twice"):
        validate_steps(
            steps(
                {"id": "a", "type": "judge", "ref": "j"},
                {"id": "a", "type": "judge", "ref": "j"},
            )
        )
    with pytest.raises(EvaluationsError, match="needs a ref"):
        validate_steps(steps({"id": "a", "type": "functional"}))
    with pytest.raises(EvaluationsError, match="aggregates nothing"):
        validate_steps(steps({"id": "a", "type": "aggregate"}))
    with pytest.raises(EvaluationsError, match="unknown method"):
        steps(
            {
                "id": "a",
                "type": "aggregate",
                "method": "median",
                "inputs": [{"step": "x"}],
            }
        )


def test_aggregate_methods() -> None:
    pairs = [(1.0, 1.0), (0.5, 1.0), (0.0, 2.0)]
    assert aggregate_scores("mean", pairs) == pytest.approx(0.5)
    assert aggregate_scores("min", pairs) == 0.0
    assert aggregate_scores("max", pairs) == 1.0
    assert aggregate_scores("weighted_mean", pairs) == pytest.approx(0.375)
