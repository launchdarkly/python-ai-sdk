from __future__ import annotations

from collections.abc import Mapping
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
class ResolvedTool:
    """The schema and pinned version returned by the LaunchDarkly tool API."""

    key: str
    version: int
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIConfigVariation:
    """An AI Config variation read from the management API as run() defaults.

    ``generation`` holds only the fields the variation defines, so a caller's
    explicit arguments can be layered over it. ``tool_versions`` maps each
    attached tool key to the version the variation pins; ``judge_keys`` lists
    the judges attached to the variation.
    """

    generation: GenerationConfig
    tool_versions: dict[str, int] = field(default_factory=dict)
    judge_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_api(
        cls,
        data: Mapping[str, Any],
        model_config: Mapping[str, Any] | None = None,
    ) -> AIConfigVariation:
        """Build from one variation version and the model config it links.

        ``model_config`` is the linked model-config response, or ``None`` when
        the variation links none. Fetching it is the caller's job, so this stays
        a pure translation of API shapes. Provider and base parameters come from
        the model config; the variation's own parameters are layered over them,
        as the served flag payload layers them.
        """
        model = data.get("model")
        model = model if isinstance(model, Mapping) else {}
        model_name = model.get("modelName")
        variation_parameters = model.get("parameters")
        parameters: dict[str, Any] = dict(
            variation_parameters if isinstance(variation_parameters, Mapping) else {}
        )
        provider: Any = None
        if model_config is not None:
            provider = model_config.get("provider")
            base_parameters = model_config.get("params")
            if isinstance(base_parameters, Mapping):
                parameters = {**base_parameters, **parameters}
            if not model_name:
                model_name = model_config.get("id")

        generation = GenerationConfig()
        if isinstance(provider, str) and provider:
            generation["provider"] = provider
        if isinstance(model_name, str) and model_name:
            generation["model"] = model_name
        if parameters:
            generation["parameters"] = parameters
        instructions = data.get("instructions")
        messages = data.get("messages")
        if isinstance(instructions, str) and instructions:
            generation["instructions"] = instructions
        elif isinstance(messages, list) and messages:
            generation["messages"] = [
                dict(message) for message in messages if isinstance(message, Mapping)
            ]
        output_format = data.get("outputFormat")
        if isinstance(output_format, Mapping):
            generation["output_format"] = dict(output_format)

        tools = data.get("tools")
        tool_versions = {
            tool["key"]: tool["version"]
            for tool in (tools if isinstance(tools, list) else [])
            if isinstance(tool, Mapping)
            and isinstance(tool.get("key"), str)
            and isinstance(tool.get("version"), int)
        }
        judge_configuration = data.get("judgeConfiguration")
        judges = (
            judge_configuration.get("judges")
            if isinstance(judge_configuration, Mapping)
            else None
        )
        judge_keys = [
            judge["judgeConfigKey"]
            for judge in (judges if isinstance(judges, list) else [])
            if isinstance(judge, Mapping)
            and isinstance(judge.get("judgeConfigKey"), str)
        ]
        return cls(
            generation=generation,
            tool_versions=tool_versions,
            judge_keys=judge_keys,
        )


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
