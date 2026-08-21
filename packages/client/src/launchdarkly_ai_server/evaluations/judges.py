from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..judges import _FORMATTING_INSTRUCTIONS
from ..lifecycle import extract_variation, init_client
from ..types import AiConfigRep, LDContext, ProviderHandler, VariationMeta
from ..utils import (
    collapse_messages_to_instructions,
    normalize_mode,
    parse_json_with_possible_fences,
    parse_template,
    parse_usage,
)
from .api import EvaluationsError


class JudgeReference(BaseModel):
    """A reference to a LaunchDarkly judge config and its evaluation thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    pass_rate_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    ground_truth_context: str | None = None

    @field_validator("key")
    @classmethod
    def _key_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("judge key must not be blank")
        return value

    def to_criterion(self) -> dict[str, Any]:
        """Build the existing evaluation criteria wire representation."""
        options: dict[str, Any] = {
            "threshold": self.threshold,
            "passRateThreshold": self.pass_rate_threshold,
        }
        if self.ground_truth_context is not None:
            options["groundTruthContext"] = self.ground_truth_context
        return {"criterionType": self.key, "options": options}


class Judge(JudgeReference):
    """A reference to any customer or LaunchDarkly judge config."""


class Accuracy(JudgeReference):
    key: Literal["$ld:ai:judge:accuracy"] = "$ld:ai:judge:accuracy"


class AnswerRelevancy(JudgeReference):
    key: Literal["$ld:ai:judge:relevance"] = "$ld:ai:judge:relevance"


class Likeness(JudgeReference):
    key: Literal["$ld:ai:judge:likeness"] = "$ld:ai:judge:likeness"
    ground_truth_context: str | None = "{{expected_output}}"


class Bias(JudgeReference):
    key: Literal["$ld:ai:judge:bias"] = "$ld:ai:judge:bias"
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class Toxicity(JudgeReference):
    key: Literal["$ld:ai:judge:toxicity"] = "$ld:ai:judge:toxicity"


class Misinformation(JudgeReference):
    key: Literal["$ld:ai:judge:misinformation"] = "$ld:ai:judge:misinformation"
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    ground_truth_context: str | None = "{{expected_output}}"


class JudgeIdentity(BaseModel):
    """Pinned judge identity retained for later evaluation-results ingest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    variation_key: str
    version: int
    provider: str
    model: str
    mode: Literal["agent", "messages"]
    is_inverted: bool = False


class JudgeUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: int = 0
    output: int = 0
    total: int = 0


JudgeErrorCode = Literal[
    "rate_limit_exhausted", "judge_timeout", "judge_parse_error", "judge_error"
]


class JudgeEvaluationError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: JudgeErrorCode
    message: str


class JudgeEvaluationResult(BaseModel):
    """One offline score, including its row and pinned judge identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: int
    judge: JudgeIdentity
    status: Literal["complete", "error"]
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None
    usage: JudgeUsage = Field(default_factory=JudgeUsage)
    error: JudgeEvaluationError | None = None


class EvaluationMethod(Protocol):
    """Seam for evaluating one generation with its stable dataset-row context."""

    async def evaluate(
        self,
        generation_output: str,
        *,
        row_index: int,
        rendered_input: str | None,
        expected_output: str | None,
        variables: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> JudgeEvaluationResult: ...


JudgeVariationResolver = Callable[[str, LDContext], Awaitable[dict[str, Any]]]
JudgeClientInitializer = Callable[[dict[str, Any]], Awaitable[Any]]


def _required_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationsError(f"Resolved judge has invalid {description}")
    return value


def _required_non_blank_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationsError(f"Resolved judge has no {description}")
    return value


def _select_handler(
    handlers: Sequence[ProviderHandler], provider: str, mode: str
) -> tuple[ProviderHandler, bool] | None:
    exact = next(
        (handler for handler in handlers if handler.provides_for == (provider, mode)),
        None,
    )
    wildcard = next(
        (handler for handler in handlers if handler.provides_for == ("*", mode)),
        None,
    )
    selected = exact if exact is not None else wildcard
    if selected is not None:
        return selected, False

    if mode == "messages":
        exact_agent = next(
            (
                handler
                for handler in handlers
                if handler.provides_for == (provider, "agent")
            ),
            None,
        )
        wildcard_agent = next(
            (handler for handler in handlers if handler.provides_for == ("*", "agent")),
            None,
        )
        selected_agent = exact_agent if exact_agent is not None else wildcard_agent
        if selected_agent is not None:
            return selected_agent, True
    return None


def _usage(raw: Any) -> JudgeUsage:
    normalized = parse_usage(dict(raw) if isinstance(raw, Mapping) else {})
    return JudgeUsage(
        input=normalized["input"],
        output=normalized["output"],
        total=normalized["total"],
    )


def _error_code(error: Exception) -> JudgeErrorCode:
    if isinstance(error, TimeoutError):
        return "judge_timeout"
    if (
        getattr(error, "status", None) == 429
        or getattr(error, "status_code", None) == 429
    ):
        return "rate_limit_exhausted"
    return "judge_error"


class LaunchDarklyJudgeEvaluation:
    """Resolved, metric-free evaluation method backed by one LD judge config."""

    def __init__(
        self,
        *,
        reference: JudgeReference,
        config: AiConfigRep,
        identity: JudgeIdentity,
        handler: ProviderHandler,
        collapse_messages: bool,
    ) -> None:
        self.reference = reference
        self.identity = identity
        self._config = (
            collapse_messages_to_instructions(config) if collapse_messages else config
        )
        self._handler = handler

    async def evaluate(
        self,
        generation_output: str,
        *,
        row_index: int,
        rendered_input: str | None,
        expected_output: str | None,
        variables: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> JudgeEvaluationResult:
        """Evaluate a generation without emitting online evaluation metrics."""
        stable_variables: dict[str, Any] = {
            **variables,
            "row_index": row_index,
            "input": rendered_input,
            "expected_output": expected_output,
            "metadata": dict(metadata) if metadata is not None else None,
            "response_to_evaluate": generation_output,
        }
        history_parts = [rendered_input, generation_output, _FORMATTING_INSTRUCTIONS]
        stable_variables["message_history"] = "\n\n".join(
            part for part in history_parts if part
        )
        if self.reference.ground_truth_context is not None:
            stable_variables["ground_truth_context"] = parse_template(
                self.reference.ground_truth_context, stable_variables
            )

        try:
            response = await self._handler(
                self._config,
                generation_output,
                None,
                stable_variables,
                None,
            )
            if not isinstance(response, Mapping):
                raise TypeError("judge handler result must be a mapping")
            usage = _usage(response.get("usage"))
            output = response.get("output")
            parsed = parse_json_with_possible_fences(
                output if isinstance(output, str) else str(output or "")
            )
            if not isinstance(parsed, Mapping):
                return self._parse_error(
                    row_index, usage, "Judge returned invalid JSON"
                )
            score = parsed.get("score")
            reasoning = parsed.get("reasoning")
            if (
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
                or not isinstance(reasoning, str)
            ):
                return self._parse_error(
                    row_index,
                    usage,
                    "Judge response must contain a score from 0 to 1 and string reasoning",
                )
            return JudgeEvaluationResult(
                row_index=row_index,
                judge=self.identity,
                status="complete",
                score=float(score),
                reasoning=reasoning,
                usage=usage,
            )
        except Exception as error:
            return JudgeEvaluationResult(
                row_index=row_index,
                judge=self.identity,
                status="error",
                error=JudgeEvaluationError(
                    code=_error_code(error), message=f"Judge invocation failed: {error}"
                ),
            )

    def _parse_error(
        self, row_index: int, usage: JudgeUsage, message: str
    ) -> JudgeEvaluationResult:
        return JudgeEvaluationResult(
            row_index=row_index,
            judge=self.identity,
            status="error",
            usage=usage,
            error=JudgeEvaluationError(code="judge_parse_error", message=message),
        )


async def resolve_launchdarkly_judges(
    references: Sequence[JudgeReference],
    handlers: Sequence[ProviderHandler],
    *,
    sdk_key: str | None,
    context: LDContext | None = None,
    resolver: JudgeVariationResolver = extract_variation,
    initialize_client: JudgeClientInitializer = init_client,
) -> list[LaunchDarklyJudgeEvaluation]:
    """Resolve all judges before evaluation/run records are created.

    The integration layer should call this during preflight. Missing credentials,
    unknown/disabled judge keys, invalid variation metadata, and incompatible
    handlers are hard failures, so no partially configured offline run is started.
    """
    if any(not isinstance(reference, JudgeReference) for reference in references):
        raise EvaluationsError(
            "judges must contain typed JudgeReference objects, not strings or mappings"
        )
    if not references:
        return []
    if not sdk_key or not sdk_key.strip():
        raise EvaluationsError(
            "LaunchDarkly judging requires an SDK key. Set LD_SDK_KEY or pass "
            "sdk_key to init_evaluations()."
        )

    await initialize_client({"sdkKey": sdk_key})
    resolution_context = context or {
        "kind": "user",
        "key": "offline-evaluation-judge-resolution",
    }
    evaluations: list[LaunchDarklyJudgeEvaluation] = []
    for reference in references:
        try:
            variation = await resolver(reference.key, resolution_context)
        except Exception as error:
            raise EvaluationsError(
                f"LaunchDarkly judge {reference.key!r} was not found or is unavailable; "
                "create or enable it in the LaunchDarkly UI before starting the run"
            ) from error

        config = dict(_required_mapping(variation.get("config"), "config"))
        meta: VariationMeta = dict(
            _required_mapping(variation.get("meta"), "variation metadata")
        )
        provider = _required_non_blank_string(
            _required_mapping(config.get("provider"), "provider").get("name"),
            "provider name",
        )
        model = _required_non_blank_string(
            _required_mapping(config.get("model"), "model").get("name"),
            "model name",
        )
        variation_key = _required_non_blank_string(
            meta.get("variationKey"), "variation key"
        )
        version = meta.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise EvaluationsError(
                f"Resolved judge {reference.key!r} has no integer version"
            )
        mode = normalize_mode(meta.get("mode"))
        selected = _select_handler(handlers, provider, mode)
        if selected is None:
            raise EvaluationsError(
                f"No handler can execute LaunchDarkly judge {reference.key!r} "
                f"for provider {provider!r} in {mode!r} mode"
            )
        handler, collapse_messages = selected
        evaluations.append(
            LaunchDarklyJudgeEvaluation(
                reference=reference,
                config=config,
                identity=JudgeIdentity(
                    key=reference.key,
                    variation_key=variation_key,
                    version=version,
                    provider=provider,
                    model=model,
                    mode=mode,
                    is_inverted=bool(config.get("isInverted", False)),
                ),
                handler=handler,
                collapse_messages=collapse_messages,
            )
        )
    return evaluations
