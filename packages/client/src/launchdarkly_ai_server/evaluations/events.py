from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvaluationStatus(StrEnum):
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class EvaluationEventKind(StrEnum):
    JUDGE = "judge"
    SCORER = "scorer"


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by an LD Judge provider call."""

    input_tokens: int
    output_tokens: int

    def to_wire(self) -> dict[str, int]:
        return {
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
        }


@dataclass(frozen=True, kw_only=True)
class EvaluationEventPayload:
    """Common fields emitted for every SDK-run evaluation criterion result."""

    project_key: str
    evaluation_id: str
    evaluation_run_id: str
    run_id: str
    dataset_id: str
    row_index: int
    criterion_type: str
    kind: EvaluationEventKind
    event_id: str
    emitted_at: str
    evaluation_key: str
    dataset_key: str
    status: EvaluationStatus
    started_at: str
    evaluated_at: str
    latency_ms: int
    evaluation_version: int | None = None
    score: float | None = None
    reason: str | None = None
    error: dict[str, Any] | None = None
    error_message: str | None = None

    def to_track_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "projectKey": self.project_key,
            "evaluationId": self.evaluation_id,
            "evaluationRunId": self.evaluation_run_id,
            "runId": self.run_id,
            "datasetId": self.dataset_id,
            "rowIndex": self.row_index,
            "criterionType": self.criterion_type,
            "kind": self.kind.value,
            "eventId": self.event_id,
            "emittedAt": self.emitted_at,
            "evaluationKey": self.evaluation_key,
            "evaluationVersion": self.evaluation_version,
            "datasetKey": self.dataset_key,
            "status": self.status.value,
            "startedAt": self.started_at,
            "evaluatedAt": self.evaluated_at,
            "latencyMs": self.latency_ms,
            "score": self.score,
            "reason": self.reason,
            "error": self.error,
            "errorMessage": self.error_message,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, kw_only=True)
class LDJudgeEvaluationEventPayload(EvaluationEventPayload):
    """Payload for one LaunchDarkly AI Judge result on one dataset row."""

    kind: EvaluationEventKind = EvaluationEventKind.JUDGE
    judge_key: str
    variation_key: str
    version: int | None = None
    usage: TokenUsage | None = None

    def to_track_payload(self) -> dict[str, Any]:
        payload = super().to_track_payload()
        payload["judgeKey"] = self.judge_key
        payload["variationKey"] = self.variation_key
        if self.version is not None:
            payload["version"] = self.version
        if self.usage is not None:
            payload["usage"] = self.usage.to_wire()
        return payload


@dataclass(frozen=True, kw_only=True)
class DeterministicScorerEvaluationEventPayload(EvaluationEventPayload):
    """Payload for one local deterministic scorer result on one dataset row."""

    kind: EvaluationEventKind = EvaluationEventKind.SCORER
