from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationStatus(StrEnum):
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class EvaluationEventKind(StrEnum):
    JUDGE = "judge"
    SCORER = "scorer"


class TokenUsage(BaseModel):
    """Token usage reported by an LD Judge provider call."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")


class EvaluationEventPayload(BaseModel):
    """Common fields emitted for every SDK-run evaluation criterion result."""

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", use_enum_values=True
    )

    project_key: str = Field(alias="projectKey")
    evaluation_id: str = Field(alias="evaluationId")
    evaluation_run_id: str = Field(alias="evaluationRunId")
    run_id: str = Field(alias="runId")
    dataset_id: str = Field(alias="datasetId")
    row_index: int = Field(alias="rowIndex")
    criterion_type: str = Field(alias="criterionType")
    kind: EvaluationEventKind
    event_id: str = Field(alias="eventId")
    emitted_at: str = Field(alias="emittedAt")
    evaluation_key: str = Field(alias="evaluationKey")
    evaluation_version: int | None = Field(default=None, alias="evaluationVersion")
    dataset_key: str = Field(alias="datasetKey")
    status: EvaluationStatus
    started_at: str = Field(alias="startedAt")
    evaluated_at: str = Field(alias="evaluatedAt")
    latency_ms: int = Field(alias="latencyMs")
    score: float | int | None = None
    reason: str | None = None
    error: dict[str, Any] | None = None

    def to_track_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class LDJudgeEvaluationEventPayload(EvaluationEventPayload):
    """Payload for one LaunchDarkly AI Judge result on one dataset row."""

    kind: EvaluationEventKind = EvaluationEventKind.JUDGE
    judge_key: str = Field(alias="judgeKey")
    variation_key: str = Field(alias="variationKey")
    version: int | None = None
    usage: TokenUsage | None = None


class DeterministicScorerEvaluationEventPayload(EvaluationEventPayload):
    """Payload for one local deterministic scorer result on one dataset row."""

    kind: EvaluationEventKind = EvaluationEventKind.SCORER
