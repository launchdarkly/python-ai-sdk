"""Functional evaluators and evaluator workflows. PROTOTYPE.

A *functional evaluator* is a judge that runs your code instead of an LLM. Like a
tool, the object is created in LaunchDarkly (a key, a description, a success
direction and threshold) and the implementation lives in your application. It is
bound by key::

    pii = await evals.evaluators.get("pii-regex", implementation=check_pii)

A *workflow* is an ordered chain of steps, also defined in LaunchDarkly. A step is
one of:

``judge``
    Runs an existing LaunchDarkly judge (an LLM call) through the normal judge
    path. A judge may return extra JSON beside ``score`` and ``reasoning`` (for
    example the topics it extracted); that whole object is the step's output.
``functional``
    Runs a bound functional evaluator.
``aggregate``
    Combines earlier steps' scores: ``mean``, ``weighted_mean``, ``min`` or ``max``.

A step lists earlier steps in ``uses``. A judge step is shown their outputs in its
``message_history``; a functional step gets them as ``ctx.steps``. The score of the
last step is the workflow's score.

Both kinds are stored as ordinary LaunchDarkly AI tools, tagged by the
``ldEvaluator`` key of their ``customParameters``, so no new API is needed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

from ..canonical_json import canonical_json
from ..judge_scoring import build_message_history, judge_text, without_output_format
from ..types import LDContext, ProviderHandler, UsageDict
from ..utils import (
    collapse_messages_to_instructions,
    normalize_mode,
    parse_json_with_possible_fences,
    select_mode_handler,
    to_usage_dict,
)
from .api import EvaluationsError, LDApiClient, LDApiError, require_mapping, segment
from .criteria import Scorer
from .types import DatasetRow

logger = logging.getLogger(__name__)

#: The ``customParameters`` key that marks an AI tool as an evaluator or workflow.
CUSTOM_PARAMETERS_KEY = "ldEvaluator"

SuccessDirection = Literal["higher_is_better", "lower_is_better"]
StepType = Literal["judge", "functional", "aggregate"]

AGGREGATE_METHODS = ("mean", "weighted_mean", "min", "max")

#: What a functional evaluator may return: a score, or a mapping with a ``score``.
type EvaluatorFn = Callable[["EvaluatorContext"], Any | Awaitable[Any]]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class StepResult:
    """The outcome of one workflow step."""

    id: str
    type: StepType
    ref: str | None = None
    #: The score as the step produced it, 0 to 1, or ``None`` when it failed.
    score: float | None = None
    #: The score pointed so that higher is better. Aggregates use this one.
    normalized: float | None = None
    reasoning: str = ""
    #: A judge's whole JSON answer, or a functional evaluator's whole return value.
    output: Any = None
    error: str | None = None
    usage: UsageDict | None = None


@dataclass
class WorkflowResult:
    """The outcome of a workflow run."""

    key: str
    #: The last step's normalized score, or ``None`` when it failed or was skipped.
    score: float | None
    #: ``True`` when ``score`` meets the workflow's threshold. ``None`` without a score.
    passed: bool | None
    threshold: float
    steps: dict[str, StepResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "steps": {
                step_id: {
                    "type": step.type,
                    "ref": step.ref,
                    "score": step.score,
                    "reasoning": step.reasoning,
                    "output": step.output,
                    "error": step.error,
                }
                for step_id, step in self.steps.items()
            },
        }


@dataclass(frozen=True)
class EvaluatorContext:
    """What a functional evaluator receives."""

    input: Any
    output: Any
    #: Results of the steps this step lists in ``uses``, by step id.
    steps: Mapping[str, StepResult]
    #: The dataset row, when run as an offline-evaluation scorer.
    row: DatasetRow | None = None


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FunctionalEvaluator:
    """A functional evaluator: a LaunchDarkly object paired with your code."""

    key: str
    implementation: EvaluatorFn
    description: str = ""
    success_direction: SuccessDirection = "higher_is_better"
    threshold: float = 1.0
    version: int | None = None

    async def run(self, ctx: EvaluatorContext) -> tuple[float, str, Any]:
        """Call the implementation. Returns ``(score, reasoning, output)``.

        Raises ``ValueError`` when the return value is not a finite number from
        0 to 1, or a mapping whose ``score`` is one. A ``bool`` is a yes/no
        score. Exceptions from the implementation propagate.
        """
        result = self.implementation(ctx)
        if inspect.isawaitable(result):
            result = await result
        raw = result.get("score") if isinstance(result, Mapping) else result
        score = _as_score(raw)
        if score is None:
            raise ValueError(
                f"evaluator {self.key!r} must return a number from 0 to 1, or a "
                f"mapping with a 'score' of one; got {result!r}"
            )
        reasoning = ""
        if isinstance(result, Mapping):
            reasoning = judge_text(result.get("reasoning") or result.get("reason"))
        return score, reasoning, result

    def normalize(self, score: float) -> float:
        return 1.0 - score if self.success_direction == "lower_is_better" else score


def _as_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) and 0.0 <= number <= 1.0 else None


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    type: StepType
    ref: str | None = None
    uses: tuple[str, ...] = ()
    method: str = "mean"
    inputs: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> WorkflowStep:
        step_type = raw.get("type")
        if step_type not in ("judge", "functional", "aggregate"):
            raise EvaluationsError(f"unknown workflow step type {step_type!r}")
        step_id = raw.get("id")
        if not isinstance(step_id, str) or not step_id:
            raise EvaluationsError("every workflow step needs an id")
        inputs = tuple(
            (str(i["step"]), float(i.get("weight", 1.0)))
            for i in raw.get("inputs") or []
            if isinstance(i, Mapping) and "step" in i
        )
        method = str(raw.get("method") or "mean")
        if step_type == "aggregate" and method not in AGGREGATE_METHODS:
            raise EvaluationsError(f"step {step_id!r}: unknown method {method!r}")
        return cls(
            id=step_id,
            type=step_type,
            ref=raw.get("ref") or None,
            uses=tuple(str(u) for u in raw.get("uses") or []),
            method=method,
            inputs=inputs,
        )


def validate_steps(steps: Sequence[WorkflowStep]) -> None:
    """Raise ``EvaluationsError`` for a workflow that cannot run."""
    seen: set[str] = set()
    for step in steps:
        if step.id in seen:
            raise EvaluationsError(f"step id {step.id!r} is used twice")
        if step.type in ("judge", "functional") and not step.ref:
            raise EvaluationsError(f"step {step.id!r} needs a ref")
        if step.type == "aggregate" and not step.inputs:
            raise EvaluationsError(f"step {step.id!r} aggregates nothing")
        for dep in (*step.uses, *(name for name, _ in step.inputs)):
            if dep not in seen:
                raise EvaluationsError(
                    f"step {step.id!r} reads {dep!r}, which is not an earlier step"
                )
        seen.add(step.id)
    if not steps:
        raise EvaluationsError("a workflow needs at least one step")


def aggregate_scores(method: str, scored: Sequence[tuple[float, float]]) -> float:
    """Combine ``(normalized score, weight)`` pairs."""
    values = [score for score, _ in scored]
    if method == "min":
        return min(values)
    if method == "max":
        return max(values)
    if method == "weighted_mean":
        total = sum(weight for _, weight in scored)
        if total <= 0:
            raise ValueError("weights must sum to more than zero")
        return sum(score * weight for score, weight in scored) / total
    return sum(values) / len(values)


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #


@dataclass
class EvaluatorWorkflow:
    """A workflow read from LaunchDarkly, with its functional evaluators bound."""

    key: str
    steps: tuple[WorkflowStep, ...]
    evaluators: Mapping[str, FunctionalEvaluator]
    threshold: float = 0.5
    description: str = ""
    version: int | None = None

    async def run(
        self,
        *,
        input: Any,
        output: Any,
        handlers: Sequence[ProviderHandler],
        context: LDContext,
        row: DatasetRow | None = None,
    ) -> WorkflowResult:
        """Run every step in order. A failed step is recorded, never raised.

        A step that reads a failed step is skipped with an error of its own,
        so a broken PII check can not quietly pass a response.
        """
        results: dict[str, StepResult] = {}
        for step in self.steps:
            blocked = next(
                (
                    dep
                    for dep in (*step.uses, *(name for name, _ in step.inputs))
                    if results[dep].normalized is None
                ),
                None,
            )
            if blocked is not None:
                results[step.id] = StepResult(
                    id=step.id,
                    type=step.type,
                    ref=step.ref,
                    error=f"skipped: step {blocked!r} has no score",
                )
                continue
            try:
                results[step.id] = await self._run_step(
                    step, input, output, handlers, context, row, results
                )
            except Exception as exc:  # a step must never take down the run
                logger.error("workflow %r step %r failed: %s", self.key, step.id, exc)
                results[step.id] = StepResult(
                    id=step.id,
                    type=step.type,
                    ref=step.ref,
                    error=f"{type(exc).__name__}: {exc}",
                )
        final = results[self.steps[-1].id].normalized
        return WorkflowResult(
            key=self.key,
            score=final,
            passed=None if final is None else final >= self.threshold,
            threshold=self.threshold,
            steps=results,
        )

    async def _run_step(
        self,
        step: WorkflowStep,
        input: Any,
        output: Any,
        handlers: Sequence[ProviderHandler],
        context: LDContext,
        row: DatasetRow | None,
        results: Mapping[str, StepResult],
    ) -> StepResult:
        if step.type == "aggregate":
            score = aggregate_scores(
                step.method,
                [
                    (results[name].normalized or 0.0, weight)
                    for name, weight in step.inputs
                ],
            )
            return StepResult(
                id=step.id,
                type="aggregate",
                score=score,
                normalized=score,
                output={"method": step.method, "inputs": [n for n, _ in step.inputs]},
            )
        if step.type == "functional":
            assert step.ref is not None
            evaluator = self.evaluators[step.ref]
            ctx = EvaluatorContext(
                input=input,
                output=output,
                steps={name: results[name] for name in step.uses},
                row=row,
            )
            score, reasoning, raw = await evaluator.run(ctx)
            return StepResult(
                id=step.id,
                type="functional",
                ref=step.ref,
                score=score,
                normalized=evaluator.normalize(score),
                reasoning=reasoning,
                output=raw,
            )
        assert step.ref is not None
        return await _run_judge_step(
            step=step,
            judge_key=step.ref,
            input=input,
            output=output,
            handlers=handlers,
            context=context,
            prior={name: results[name] for name in step.uses},
        )

    def as_scorer(
        self, *, handlers: Sequence[ProviderHandler], context: LDContext
    ) -> Scorer:
        """This workflow as an offline-evaluation :class:`Scorer`.

        The scorer's name is the workflow key and it passes at the workflow's
        threshold. A run whose final score is missing raises, which the
        evaluation records as an invalid score for that row.
        """

        async def score_row(row: DatasetRow, output: Any) -> float:
            result = await self.run(
                input=row.input,
                output=output,
                handlers=handlers,
                context=context,
                row=row,
            )
            if result.score is None:
                failed = [
                    f"{s.id}: {s.error}" for s in result.steps.values() if s.error
                ]
                raise EvaluationsError(
                    f"workflow {self.key!r} produced no score ({'; '.join(failed)})"
                )
            return result.score

        return Scorer(name=self.key, fn=score_row, threshold=self.threshold)


async def _run_judge_step(
    *,
    step: WorkflowStep,
    judge_key: str,
    input: Any,
    output: Any,
    handlers: Sequence[ProviderHandler],
    context: LDContext,
    prior: Mapping[str, StepResult],
) -> StepResult:
    """Run a LaunchDarkly judge as one workflow step, through the normal judge path."""
    from ..lifecycle import extract_variation
    from ..tracking import execute_and_track

    variation = await extract_variation(judge_key, context)
    judge_config = variation["config"]
    judge_meta = variation["meta"]
    provider = (
        judge_config.get("provider", {}).get("name")
        if isinstance(judge_config, dict)
        else None
    )
    mode = normalize_mode(
        judge_meta.get("mode") if isinstance(judge_meta, dict) else None
    )
    handler = select_mode_handler(list(handlers), provider, mode)
    collapse = False
    if handler is None and mode == "messages":
        handler = select_mode_handler(list(handlers), provider, "agent")
        collapse = handler is not None
    if handler is None:
        raise EvaluationsError(
            f"no handler covers provider {provider!r} for judge {judge_key!r}"
        )
    effective = without_output_format(
        collapse_messages_to_instructions(judge_config) if collapse else judge_config,
        judge_key,
    )

    history = build_message_history(user_input=input, output=output)
    if prior:
        history = _with_prior_steps(history, prior)

    result = await execute_and_track(
        config_key=judge_key,
        config=effective,
        meta=judge_meta,
        user_context=context,
        handler=handler,
        user_input=output if isinstance(output, str) else judge_text(output),
        tool_handlers=None,
        variables={
            "message_history": history,
            "response_to_evaluate": judge_text(output),
            **({"input": judge_text(input)} if input else {}),
            # Earlier results as variables too, for templates that reference them.
            "steps": {name: _public(r) for name, r in prior.items()},
        },
        evaluated_value=variation.get("value"),
    )

    raw = result["response"]
    parsed = (
        raw
        if isinstance(raw, Mapping)
        else parse_json_with_possible_fences(raw)
        if isinstance(raw, str)
        else None
    )
    if not isinstance(parsed, Mapping) or not parsed:
        raise ValueError(f"judge {judge_key!r} did not return a JSON object")
    score = _as_score(parsed.get("score"))
    if score is None:
        raise ValueError(f"judge {judge_key!r} returned no score from 0 to 1")
    inverted = (
        bool(judge_config.get("isInverted"))
        if isinstance(judge_config, dict)
        else False
    )
    return StepResult(
        id=step.id,
        type="judge",
        ref=judge_key,
        score=score,
        normalized=1.0 - score if inverted else score,
        reasoning=judge_text(parsed.get("reasoning") or parsed.get("reason")),
        output=dict(parsed),
        usage=to_usage_dict(result["usage"]),
    )


def _public(result: StepResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "reasoning": result.reasoning,
        "output": result.output,
    }


def _with_prior_steps(history: str, prior: Mapping[str, StepResult]) -> str:
    """Append the results a judge step reads, after the conversation, before nothing else.

    The formatting block must stay last: the judge is told its answer shape there.
    """
    from ..judge_scoring import FORMATTING_INSTRUCTIONS

    lines = ["Results of earlier evaluation steps:"]
    for name, result in prior.items():
        lines.append(f"- {name}: {canonical_json(_public(result), sort_keys=False)}")
    block = "\n".join(lines)
    if history.endswith(FORMATTING_INSTRUCTIONS):
        head = history[: -len(FORMATTING_INSTRUCTIONS)].rstrip()
        return f"{head}\n\n{block}\n\n{FORMATTING_INSTRUCTIONS}"
    return f"{history}\n\n{block}"


# --------------------------------------------------------------------------- #
# Reading definitions from LaunchDarkly
# --------------------------------------------------------------------------- #


def _definition(raw: Mapping[str, Any], key: str, expected: str) -> Mapping[str, Any]:
    params = raw.get("customParameters")
    definition = (
        params.get(CUSTOM_PARAMETERS_KEY) if isinstance(params, Mapping) else None
    )
    kind = definition.get("kind") if isinstance(definition, Mapping) else None
    if kind != expected:
        raise EvaluationsError(
            f"LaunchDarkly AI tool {key!r} is not a {expected} "
            f"(kind: {kind or 'plain tool'}). Create it as one in the AI Library."
        )
    assert isinstance(definition, Mapping)
    return definition


class EvaluatorsClient:
    """Reads functional evaluators from LaunchDarkly."""

    def __init__(self, api_client: LDApiClient, project_key: str) -> None:
        self._api = api_client
        self._project_key = project_key

    async def _read(self, key: str) -> Mapping[str, Any]:
        path = f"projects/{segment(self._project_key)}/ai-tools/{segment(key)}"
        try:
            response = await asyncio.to_thread(self._api.get, path)
        except LDApiError as error:
            if error.status == 404:
                raise EvaluationsError(
                    f"LaunchDarkly AI tool {key!r} was not found in project "
                    f"{self._project_key!r}"
                ) from error
            raise
        return require_mapping(response, description=f"evaluator {key!r}")

    async def get(
        self, key: str, *, implementation: EvaluatorFn
    ) -> FunctionalEvaluator:
        """Return the functional evaluator ``key`` bound to ``implementation``.

        ``implementation`` receives an :class:`EvaluatorContext` and returns a
        number from 0 to 1, or a mapping with a ``score`` (and optionally
        ``reasoning``). It may be sync or async.
        """
        if not callable(implementation):
            raise EvaluationsError(f"evaluator {key!r} implementation must be callable")
        raw = await self._read(key)
        definition = _definition(raw, key, "functional")
        direction_raw = definition.get("successDirection") or "higher_is_better"
        direction: SuccessDirection
        if direction_raw == "higher_is_better":
            direction = "higher_is_better"
        elif direction_raw == "lower_is_better":
            direction = "lower_is_better"
        else:
            raise EvaluationsError(
                f"evaluator {key!r}: bad successDirection {direction_raw!r}"
            )
        threshold = definition.get("threshold")
        return FunctionalEvaluator(
            key=key,
            implementation=implementation,
            description=str(raw.get("description") or ""),
            success_direction=direction,
            threshold=float(threshold) if isinstance(threshold, (int, float)) else 1.0,
            version=raw.get("version") if isinstance(raw.get("version"), int) else None,
        )


class WorkflowsClient:
    """Reads evaluator workflows from LaunchDarkly."""

    def __init__(self, evaluators: EvaluatorsClient) -> None:
        self._evaluators = evaluators

    async def get(
        self,
        key: str,
        *,
        implementations: Mapping[str, EvaluatorFn] | None = None,
    ) -> EvaluatorWorkflow:
        """Return the workflow ``key``.

        ``implementations`` maps the key of every functional evaluator the
        workflow uses to your code. A missing one raises here, not mid-run.
        """
        raw = await self._evaluators._read(key)
        definition = _definition(raw, key, "workflow")
        steps = tuple(WorkflowStep.from_wire(s) for s in definition.get("steps") or [])
        validate_steps(steps)
        needed = sorted({s.ref for s in steps if s.type == "functional" and s.ref})
        provided = dict(implementations or {})
        missing = [ref for ref in needed if ref not in provided]
        if missing:
            raise EvaluationsError(
                f"workflow {key!r} uses functional evaluators with no implementation: "
                f"{', '.join(missing)}. Pass them in implementations={{...}}."
            )
        bound = await asyncio.gather(
            *(self._evaluators.get(ref, implementation=provided[ref]) for ref in needed)
        )
        threshold = definition.get("threshold")
        return EvaluatorWorkflow(
            key=key,
            steps=steps,
            evaluators={e.key: e for e in bound},
            threshold=float(threshold) if isinstance(threshold, (int, float)) else 0.5,
            description=str(raw.get("description") or ""),
            version=raw.get("version") if isinstance(raw.get("version"), int) else None,
        )
