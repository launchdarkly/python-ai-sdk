from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


@dataclass
class Usage:
    """Token counts for one generation, using the ingest wire field names."""

    input_tokens: int
    output_tokens: int

    def to_wire(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> Usage:
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
        )


class GenerationConfig(TypedDict, total=False):
    """Generation settings stored on the evaluation and passed to its handler."""

    provider: str
    model: str
    parameters: dict[str, Any]
    instructions: str
    messages: list[dict[str, Any]]
    prompt_snippets: dict[str, str]
    output_format: dict[str, Any]


@dataclass
class DatasetRef:
    """Identifiers returned when resolving a dataset by key."""

    id: str
    key: str


@dataclass
class DatasetRow:
    """A rendered dataset row ready for handler invocation and ingest."""

    row_index: int
    input: str | None = None
    expected_output: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] | None = None


@dataclass
class InlineTool:
    """A tool defined in code instead of created in LaunchDarkly first.

    Use one as a value in ``run(tools=...)`` to supply a tool's ``schema`` and
    ``description`` with its executable. A callable in the same map names a
    tool to resolve by key, so one map may hold both kinds.

    ``implementation`` is the function the handler calls. Keys must be
    lowercase. A :class:`~launchdarkly_ai_server.NativeTool` is not accepted as
    the implementation.
    """

    implementation: Callable[..., Any]
    schema: dict[str, Any]
    description: str = ""


@dataclass
class ResolvedTool:
    """A tool a run uses, from the library or from an inline definition.

    ``version`` is set for a library tool and ``None`` for an inline one.
    """

    key: str
    version: int | None = None
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    source: Literal["library", "inline"] = "library"

    def to_create_wire(self) -> dict[str, Any]:
        """The entry this tool contributes to the evaluation-create body."""
        if self.source == "inline":
            return {
                "key": self.key,
                "schema": self.schema,
                "description": self.description,
                "source": "inline",
            }
        return {"key": self.key, "version": self.version, "source": "library"}


@dataclass
class ResolvedJudge:
    """A LaunchDarkly AI Judge config variation resolved for an evaluation run.

    ``provider`` and ``mode`` come from the judge's own variation, not the
    evaluation's generation config: a judge is an independent AI Config and may
    be served by a different provider in a different mode. They are kept here
    because they are what selects the handler that can actually run this config.
    """

    key: str
    config: dict[str, Any]
    variation_key: str = ""
    version: int | None = None
    provider: str | None = None
    mode: Literal["agent", "messages"] = "messages"


@dataclass
class EvaluationRef:
    """Identifiers returned after creating an evaluation."""

    id: str
    key: str
    version: int | None = None


@dataclass
class EvaluationRunRef:
    """Identifiers and state returned by the evaluation-run API."""

    id: str
    evaluation_id: str
    state: str
    status_reason: str | None = None


@dataclass
class RunSummary:
    """Row counts for an evaluation run.

    The summary endpoint does not return run state, so terminal completion
    is derived from row accounting instead.
    """

    total_rows: int = 0
    passed_rows: int = 0
    failed_rows: int = 0
    error_rows: int = 0
    pending_rows: int = 0

    @classmethod
    def from_wire(cls, data: Mapping[str, Any] | None) -> RunSummary:
        data = data or {}
        counts_value = data.get("statusCounts")
        counts = counts_value if isinstance(counts_value, Mapping) else data
        return cls(
            total_rows=int(counts.get("total", counts.get("total_rows", 0)) or 0),
            passed_rows=int(counts.get("passed", counts.get("passed_rows", 0)) or 0),
            failed_rows=int(counts.get("failed", counts.get("failed_rows", 0)) or 0),
            error_rows=int(counts.get("error", counts.get("error_rows", 0)) or 0),
            pending_rows=int(counts.get("pending", counts.get("pending_rows", 0)) or 0),
        )


@dataclass
class EvalRunResult:
    """The result of an evaluation run, derived from its row summary."""

    passed: bool
    url: str
    run_id: str
    summary: RunSummary
