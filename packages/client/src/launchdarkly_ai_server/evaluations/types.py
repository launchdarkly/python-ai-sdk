from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    """
    Token counts for a single generation, in the ingest wire shape. Handler
    results carry this dict verbatim, so nothing on the eval path adapts it.
    """

    input_tokens: int
    output_tokens: int

    def to_wire(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Usage:
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
        )


@dataclass
class RunSummary:
    """Row counts for a finished evaluation run."""

    total_rows: int = 0
    passed_rows: int = 0
    failed_rows: int = 0
    error_rows: int = 0

    @classmethod
    def from_wire(cls, data: dict[str, Any] | None) -> RunSummary:
        data = data or {}
        return cls(
            total_rows=int(data.get("total_rows") or 0),
            passed_rows=int(data.get("passed_rows") or 0),
            failed_rows=int(data.get("failed_rows") or 0),
            error_rows=int(data.get("error_rows") or 0),
        )


@dataclass
class EvalRunResult:
    """The verdict of an evaluation run, as computed and stored by LaunchDarkly."""

    passed: bool
    url: str
    run_id: str
    summary: RunSummary
