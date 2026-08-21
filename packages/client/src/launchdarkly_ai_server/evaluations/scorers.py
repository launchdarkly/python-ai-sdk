"""Deterministic function scorers for client-side evaluations."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal

ScoreValue = bool | int | float
ScorerFunction = Callable[["ScorerRow", str | None], ScoreValue | Awaitable[ScoreValue]]
ScorerStatus = Literal["COMPLETE", "ERROR"]
ScorerErrorCode = Literal["invalid_score", "scorer_error"]


@dataclass(frozen=True, slots=True)
class ScorerRow:
    """Complete rendered dataset-row context passed to a scorer function."""

    row_index: int
    input: str | None
    expected_output: str | None
    variables: Mapping[str, Any]
    metadata: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if type(self.row_index) is not int or self.row_index < 0:
            raise ValueError("row_index must be a non-negative integer")
        if self.input is not None and not isinstance(self.input, str):
            raise TypeError("input must be a string or None")
        if self.expected_output is not None and not isinstance(
            self.expected_output, str
        ):
            raise TypeError("expected_output must be a string or None")
        object.__setattr__(
            self, "variables", _validated_mapping(self.variables, name="variables")
        )
        if self.metadata is not None:
            object.__setattr__(
                self, "metadata", _validated_mapping(self.metadata, name="metadata")
            )


@dataclass(frozen=True, slots=True)
class ScorerError:
    """Structured scorer failure details for later evaluation-results ingest."""

    code: ScorerErrorCode
    message: str
    exception_type: str


@dataclass(frozen=True, slots=True)
class ScorerResult:
    """The normalized outcome and execution metadata for one row and scorer."""

    scorer_name: str
    row_index: int
    score: float | None
    started_at: datetime
    evaluated_at: datetime
    latency_ms: float
    status: ScorerStatus
    error: ScorerError | None = None

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETE", "ERROR"}:
            raise ValueError(f"unknown scorer result status: {self.status!r}")
        if self.status == "COMPLETE":
            if self.score is None or self.error is not None:
                raise ValueError(
                    "a COMPLETE scorer result requires a score and no error"
                )
        elif self.score is not None or self.error is None:
            raise ValueError("an ERROR scorer result requires an error and no score")


@dataclass(frozen=True, slots=True)
class Scorer:
    """A named deterministic scorer with an async execution method.

    The scorer function follows the Phase 3 protocol ``fn(row, output)`` and may
    be synchronous or asynchronous. ``execute`` converts function failures and
    invalid return values into typed error results so evaluation orchestration
    can continue processing the remaining rows.
    """

    name: str
    fn: ScorerFunction
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scorer name must not be blank")
        if not callable(self.fn):
            raise TypeError("scorer fn must be callable")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("scorer threshold must be numeric")
        normalized_threshold = float(self.threshold)
        if (
            not math.isfinite(normalized_threshold)
            or not 0 <= normalized_threshold <= 1
        ):
            raise ValueError("scorer threshold must be between 0 and 1")
        object.__setattr__(self, "threshold", normalized_threshold)

    async def execute(self, row: ScorerRow, output: str | None) -> ScorerResult:
        """Run this scorer for one generation and return a normalized result."""
        if not isinstance(row, ScorerRow):
            raise TypeError("row must be a ScorerRow")
        if output is not None and not isinstance(output, str):
            raise TypeError("output must be a string or None")

        started_at = datetime.now(UTC)
        started_clock = time.perf_counter()
        try:
            value = self.fn(row, output)
            if inspect.isawaitable(value):
                value = await value
            score = _normalize_score(value)
        except _InvalidScore as error:
            return self._error_result(
                row=row,
                started_at=started_at,
                started_clock=started_clock,
                code="invalid_score",
                error=error,
            )
        except Exception as error:
            return self._error_result(
                row=row,
                started_at=started_at,
                started_clock=started_clock,
                code="scorer_error",
                error=error,
            )

        evaluated_at = datetime.now(UTC)
        return ScorerResult(
            scorer_name=self.name,
            row_index=row.row_index,
            score=score,
            started_at=started_at,
            evaluated_at=evaluated_at,
            latency_ms=_elapsed_ms(started_clock),
            status="COMPLETE",
        )

    def _error_result(
        self,
        *,
        row: ScorerRow,
        started_at: datetime,
        started_clock: float,
        code: ScorerErrorCode,
        error: Exception,
    ) -> ScorerResult:
        evaluated_at = datetime.now(UTC)
        message = str(error) or type(error).__name__
        return ScorerResult(
            scorer_name=self.name,
            row_index=row.row_index,
            score=None,
            started_at=started_at,
            evaluated_at=evaluated_at,
            latency_ms=_elapsed_ms(started_clock),
            status="ERROR",
            error=ScorerError(
                code=code,
                message=message,
                exception_type=type(error).__name__,
            ),
        )


class _InvalidScore(ValueError):
    pass


def _validated_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return MappingProxyType(dict(value))


def _normalize_score(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise _InvalidScore(
            "scorer must return bool or a numeric score between 0 and 1; "
            f"got {type(value).__name__}"
        )
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise _InvalidScore(
            f"scorer must return a finite numeric score between 0 and 1; got {value!r}"
        )
    return score


def _elapsed_ms(started_clock: float) -> float:
    return round((time.perf_counter() - started_clock) * 1000, 3)
