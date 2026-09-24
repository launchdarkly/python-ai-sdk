from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from ..judge_scoring import (
    FORMATTING_INSTRUCTIONS,
    numeric_score,
    parse_judge_response,
)
from ..lifecycle import extract_variation
from ..types import NativeTool
from ..utils import (
    collapse_messages_to_instructions,
    normalize_mode,
    parse_template,
    parse_usage,
    to_ld_context,
)
from .api import EvaluationsError, LDApiClient, LDApiError
from .criteria import Criterion, Judge, Scorer
from .events import (
    CriterionEventPayload,
    CriterionStatus,
    DeterministicScorerCriterionEventPayload,
    LDJudgeCriterionEventPayload,
    TokenUsage,
)
from .types import (
    DatasetRef,
    DatasetRow,
    EvaluationRef,
    EvaluationRunRef,
    GenerationConfig,
    InlineTool,
    ResolvedJudge,
    ResolvedTool,
    RunSummary,
)

logger = logging.getLogger(__name__)

DATASET_PAGE_SIZE = 200
GENERATION_EVENT_NAME = "$ld:ai:offline-evals:generation"
CRITERION_EVENT_NAME = "$ld:ai:offline-evals:criterion"

EvalHandler = Callable[..., Awaitable[dict[str, Any]]]
ToolImplementation = Callable[..., Any] | NativeTool
# A value accepted by ``run(tools=...)``. A callable or ``NativeTool`` names a
# tool to resolve by key. An ``InlineTool`` carries its own definition.
ToolEntry = ToolImplementation | InlineTool


@dataclass(frozen=True)
class JudgeExecution:
    """A resolved judge paired with the handler selected to run its config."""

    resolved: ResolvedJudge
    handler: EvalHandler
    collapse_messages: bool = False


def _provides_for(
    handler: EvalHandler,
) -> tuple[str, Literal["agent", "messages"]] | None:
    provides_for = getattr(handler, "provides_for", None)
    if (
        isinstance(provides_for, tuple | list)
        and len(provides_for) == 2
        and isinstance(provides_for[0], str)
    ):
        return (provides_for[0], normalize_mode(provides_for[1]))
    return None


def _covers_provider(
    provides_for: tuple[str, Literal["agent", "messages"]],
    provider: str | None,
) -> bool:
    return provides_for[0] == provider or provides_for[0] == "*"


def _find_judge_handler(
    judge_handlers: list[EvalHandler],
    provider: str | None,
    mode: Literal["agent", "messages"],
) -> EvalHandler | None:
    """Find a handler for ``provider`` in ``mode``, exact match before wildcard.

    A wildcard handler is a fallback for multi-provider adapters, so it is only
    chosen when no handler names the provider outright -- the priority
    ``config()`` already applies to a generation config. Searching in one pass
    would instead let the order the caller happened to list its handlers in
    decide, sending an OpenAI judge through a LangChain adapter that was merely
    listed first.
    """
    for exact in (True, False):
        for candidate in judge_handlers:
            provides_for = _provides_for(candidate)
            if provides_for is None or provides_for[1] != mode:
                continue
            if exact:
                if provides_for[0] == provider:
                    return candidate
            elif provides_for[0] == "*":
                return candidate
    return None


def _select_judge_handler(
    resolved: ResolvedJudge,
    handler: EvalHandler,
    judge_handlers: list[EvalHandler],
) -> JudgeExecution | None:
    """Pick the handler that can run this judge's config, or ``None``.

    A judge is an independent AI Config: it may resolve to a different provider
    and mode than the evaluation's generation config, and a handler built for
    one provider cannot execute another's config. The priority mirrors the
    online path (``judges.run_judges``):

    1. a judge handler in the judge's mode, naming its provider outright
       before any wildcard adapter;
    2. an agent-mode judge handler for a messages-mode judge, whose messages
       are collapsed into a single instructions block;
    3. the generation handler, when it covers the judge's provider.

    A handler that declares no ``provides_for`` is a plain callable doing its
    own routing -- the same contract it already honours for the generation
    config -- so it is treated as covering every judge.
    """
    match = _find_judge_handler(judge_handlers, resolved.provider, resolved.mode)
    if match is not None:
        return JudgeExecution(resolved=resolved, handler=match)
    if resolved.mode == "messages":
        agent_fallback = _find_judge_handler(judge_handlers, resolved.provider, "agent")
        if agent_fallback is not None:
            return JudgeExecution(
                resolved=resolved, handler=agent_fallback, collapse_messages=True
            )
    generation_provides_for = _provides_for(handler)
    if generation_provides_for is None:
        return JudgeExecution(resolved=resolved, handler=handler)
    if _covers_provider(generation_provides_for, resolved.provider):
        return JudgeExecution(
            resolved=resolved,
            handler=handler,
            collapse_messages=(
                generation_provides_for[1] == "agent" and resolved.mode == "messages"
            ),
        )
    return None


def _segment(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationsError(
            f"LaunchDarkly returned an invalid {description} response"
        )
    return value


def _required_string(data: Mapping[str, Any], key: str, description: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationsError(
            f"LaunchDarkly {description} response is missing string field {key!r}"
        )
    return value


def _validate_inline_tool(key: str, tool: InlineTool) -> None:
    """Validate one inline tool definition. Raises ``EvaluationsError``.

    Checks the key, the implementation, the schema, and the description. Issues
    no requests.
    """
    # Keys are lowercase.
    if key != key.lower():
        raise EvaluationsError(
            f"Inline tool {key!r} key must not use uppercase letters. Use "
            f"{key.lower()!r} instead."
        )
    if isinstance(tool.implementation, NativeTool):
        raise EvaluationsError(
            f"Inline tool {key!r} cannot pair a NativeTool with an inline "
            "definition. A provider-native tool is implemented by the provider "
            "and has no schema of its own, so there is nothing for an inline "
            "schema to describe. Pass the NativeTool on its own to use the "
            "provider capability, or an InlineTool wrapping your own function."
        )
    if not callable(tool.implementation):
        raise EvaluationsError(
            f"Inline tool {key!r} implementation must be callable, got "
            f"{type(tool.implementation).__name__}"
        )
    if not isinstance(tool.schema, Mapping):
        raise EvaluationsError(
            f"Inline tool {key!r} schema must be a JSON object, got "
            f"{type(tool.schema).__name__}"
        )
    try:
        # allow_nan=False rejects NaN and Infinity, which are not valid JSON.
        json.dumps(dict(tool.schema), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise EvaluationsError(
            f"Inline tool {key!r} schema must be JSON-serializable: {error}"
        ) from error
    if not isinstance(tool.description, str):
        raise EvaluationsError(
            f"Inline tool {key!r} description must be a string, got "
            f"{type(tool.description).__name__}"
        )


def _validate_tools(tools: Mapping[str, ToolEntry]) -> None:
    """Validate the whole tools map. Raises ``EvaluationsError``.

    Issues no requests.
    """
    keys_by_identity: dict[str, str] = {}
    inline_identities: set[str] = set()
    for key, entry in tools.items():
        if not isinstance(key, str) or not key.strip():
            raise EvaluationsError("tool keys must not be blank")
        if isinstance(entry, InlineTool):
            is_inline = True
            _validate_inline_tool(key, entry)
        else:
            is_inline = False
            if not callable(entry) and not isinstance(entry, NativeTool):
                raise EvaluationsError(
                    f"Tool {key!r} must be callable, a NativeTool instance, or "
                    "an InlineTool"
                )
        # A key names either a library tool or an inline definition, not both.
        # Keys are compared case-insensitively when an inline definition is
        # involved. Two library keys that differ only by case are two tools.
        identity = key.strip().lower()
        collision = keys_by_identity.get(identity)
        if collision is not None and (is_inline or identity in inline_identities):
            if collision == key:
                raise EvaluationsError(
                    f"Tool {key!r} is defined more than once in tools. An "
                    "inline definition and a library tool cannot share a key."
                )
            raise EvaluationsError(
                f"Tool {key!r} collides with {collision!r}: both name the same "
                "tool in one run, and one of them is an inline definition. A "
                "tool key resolves either to a LaunchDarkly AI library tool or "
                "to an inline definition, never both."
            )
        keys_by_identity[identity] = key
        if is_inline:
            inline_identities.add(identity)


def _tool_handlers(tools: Mapping[str, ToolEntry]) -> dict[str, ToolImplementation]:
    """Return the tools map as ``{key: executable}``, unwrapping any ``InlineTool``."""
    return {
        key: entry.implementation if isinstance(entry, InlineTool) else entry
        for key, entry in tools.items()
    }


class ConcurrencyController:
    """Owns row-worker permits."""

    def __init__(self, limit: int = 10) -> None:
        if limit < 1:
            raise EvaluationsError("concurrency must be at least 1")
        self._semaphore = asyncio.Semaphore(limit)

    async def acquire(self, provider: str | None = None) -> None:
        del provider
        await self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()

    def record_success(
        self,
        provider: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        del provider, headers

    def record_rate_limit(
        self,
        provider: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        del provider, retry_after


class EvaluationsRunner:
    """Private API operations and orchestration used by EvaluationsModule.run()."""

    def __init__(self, api: LDApiClient) -> None:
        self._api = api

    def _resolve_tools(
        self,
        project_key: str,
        tools: Mapping[str, ToolEntry],
    ) -> dict[str, ResolvedTool]:
        """Resolve each tool in the map to a ``ResolvedTool``.

        Fetches a library entry from ``ai-tools`` and pins the version it
        returns. Takes an inline entry's definition as given, with no request.
        """
        resolved: dict[str, ResolvedTool] = {}
        for key, entry in tools.items():
            if isinstance(entry, InlineTool):
                resolved[key] = ResolvedTool(
                    key=key,
                    description=entry.description,
                    schema=dict(entry.schema),
                    source="inline",
                )
                continue
            path = f"projects/{_segment(project_key)}/ai-tools/{_segment(key)}"
            try:
                raw = _mapping(self._api.get(path), description=f"tool {key!r}")
            except LDApiError as error:
                if error.status == 404:
                    raise EvaluationsError(
                        f"LaunchDarkly AI tool {key!r} was not found in project {project_key!r}"
                    ) from error
                raise
            version = raw.get("version")
            if not isinstance(version, int):
                raise EvaluationsError(
                    f"LaunchDarkly AI tool {key!r} has no integer version"
                )
            schema = raw.get("schema")
            if not isinstance(schema, Mapping):
                schema = {}
            resolved[key] = ResolvedTool(
                key=key,
                version=version,
                description=str(raw.get("description") or ""),
                schema=dict(schema),
            )
        return resolved

    async def _resolve_judges(
        self,
        project_key: str,
        judges: list[Judge],
        handler: EvalHandler,
        judge_handlers: list[EvalHandler] | None = None,
    ) -> dict[str, JudgeExecution]:
        """Resolve LD Judge configs before any evaluation records are created.

        Each judge is paired with the handler that can execute its config here,
        rather than at scoring time, so a judge no handler covers fails the run
        before any records exist or any generation spend happens.
        """
        available_judge_handlers = list(judge_handlers or [])
        resolved: dict[str, JudgeExecution] = {}
        # variation() rejects a context without kind and key; use the same
        # context shape the emitted evaluation events are attributed to.
        context: dict[str, Any] = {"kind": "evaluation", "key": project_key}
        for judge in judges:
            try:
                variation = await extract_variation(judge.key, context)
            except Exception as error:
                raise EvaluationsError(
                    f"Failed to resolve LaunchDarkly judge {judge.key!r}: {error} "
                    f"If the judge does not exist in project {project_key!r}, "
                    "create it in the LaunchDarkly UI and try again."
                ) from error
            config = variation.get("config")
            meta_value = variation.get("meta")
            meta: Mapping[str, Any] = (
                meta_value if isinstance(meta_value, Mapping) else {}
            )
            if not isinstance(config, Mapping):
                raise EvaluationsError(
                    f"LaunchDarkly judge {judge.key!r} returned an invalid AI config variation"
                )
            provider_value = config.get("provider")
            provider = (
                provider_value.get("name")
                if isinstance(provider_value, Mapping)
                else None
            )
            resolved_judge = ResolvedJudge(
                key=judge.key,
                config=dict(config),
                variation_key=str(meta.get("variationKey") or ""),
                version=int(meta["version"])
                if isinstance(meta.get("version"), int)
                else None,
                provider=str(provider) if isinstance(provider, str) else None,
                mode=normalize_mode(
                    meta.get("mode") if isinstance(meta.get("mode"), str) else None
                ),
            )
            execution = _select_judge_handler(
                resolved_judge, handler, available_judge_handlers
            )
            if execution is None:
                raise EvaluationsError(
                    f"No handler can run LaunchDarkly judge {judge.key!r}: its "
                    f"config is served by provider {resolved_judge.provider!r} in "
                    f"{resolved_judge.mode!r} mode, which neither the generation "
                    "handler nor any judge_handlers entry provides for. Pass a "
                    "handler for that provider to run(judge_handlers=[...])."
                )
            resolved[judge.key] = execution
        return resolved

    def _fetch_dataset(self, project_key: str, dataset_key: str) -> DatasetRef:
        path = f"projects/{_segment(project_key)}/datasets/{_segment(dataset_key)}"
        try:
            raw = _mapping(self._api.get(path), description=f"dataset {dataset_key!r}")
        except LDApiError as error:
            if error.status == 404:
                raise EvaluationsError(
                    f"LaunchDarkly dataset {dataset_key!r} was not found in project {project_key!r}"
                ) from error
            raise
        dataset_id = _required_string(raw, "id", "dataset")
        response_key = raw.get("key", raw.get("name", dataset_key))
        return DatasetRef(id=dataset_id, key=str(response_key))

    def _fetch_dataset_rows_page(
        self,
        project_key: str,
        dataset_key: str,
        *,
        offset: int,
    ) -> Mapping[str, Any]:
        path = f"projects/{_segment(project_key)}/datasets/{_segment(dataset_key)}/rows"
        return _mapping(
            self._api.get(
                path,
                params={
                    "mode": "all",
                    "limit": DATASET_PAGE_SIZE,
                    "offset": offset,
                },
            ),
            description=f"rows for dataset {dataset_key!r}",
        )

    def _get_dataset_rows(self, project_key: str, dataset_key: str) -> list[DatasetRow]:
        rows: list[DatasetRow] = []
        offset = 0
        total: int | None = None
        while total is None or len(rows) < total:
            page = self._fetch_dataset_rows_page(
                project_key, dataset_key, offset=offset
            )
            items = page.get("items")
            page_total = page.get("totalCount")
            if not isinstance(items, list) or not isinstance(page_total, int):
                raise EvaluationsError(
                    f"LaunchDarkly returned invalid rows for dataset {dataset_key!r}"
                )
            total = page_total
            if not items:
                break
            for item_value in items:
                item = _mapping(item_value, description="dataset row")
                row_index = item.get("rowIndex")
                if not isinstance(row_index, int):
                    raise EvaluationsError(
                        "A dataset row is missing its integer rowIndex"
                    )
                variables_value = item.get("variables")
                variables = (
                    dict(variables_value)
                    if isinstance(variables_value, Mapping)
                    else {}
                )
                input_value = item.get("input")
                expected_value = item.get("expectedOutput")
                rendered_input = (
                    parse_template(input_value, variables)
                    if isinstance(input_value, str)
                    else None
                )
                rendered_expected = (
                    parse_template(expected_value, variables)
                    if isinstance(expected_value, str)
                    else None
                )
                variables["input"] = rendered_input
                variables["expected_output"] = rendered_expected
                metadata_value = item.get("metadata")
                rows.append(
                    DatasetRow(
                        row_index=row_index,
                        input=rendered_input,
                        expected_output=rendered_expected,
                        variables=variables,
                        metadata=(
                            dict(metadata_value)
                            if isinstance(metadata_value, Mapping)
                            else None
                        ),
                    )
                )
            offset += len(items)
        if not rows:
            raise EvaluationsError(f"Dataset {dataset_key!r} is empty")
        if total is not None and len(rows) != total:
            raise EvaluationsError(
                f"Dataset {dataset_key!r} returned {len(rows)} of {total} rows"
            )
        return rows

    def _create_evaluation(
        self,
        project_key: str,
        key: str,
        generation: GenerationConfig,
        tools: Mapping[str, ResolvedTool],
        criteria: list[Criterion] | None = None,
    ) -> EvaluationRef:
        body: dict[str, Any] = {
            "name": key,
            "generationProvider": generation["provider"],
            "generationModel": generation["model"],
        }
        if "parameters" in generation:
            body["parameters"] = generation["parameters"]
        if "instructions" in generation:
            body["messages"] = [
                {"role": "system", "content": generation["instructions"]}
            ]
        elif "messages" in generation:
            body["messages"] = generation["messages"]
        else:
            body["messages"] = []
        if "prompt_snippets" in generation:
            body["promptSnippets"] = generation["prompt_snippets"]
        if tools:
            body["tools"] = [tool.to_create_wire() for tool in tools.values()]
        if criteria:
            body["criteria"] = [criterion.to_criteria_wire() for criterion in criteria]

        path = f"projects/{_segment(project_key)}/evaluations"
        raw = _mapping(self._api.post(path, body=body), description="evaluation")
        evaluation_id = _required_string(raw, "id", "evaluation")
        response_key = raw.get("name", raw.get("label", key))
        version = raw.get("version")
        return EvaluationRef(
            id=evaluation_id,
            key=str(response_key),
            version=version if isinstance(version, int) else None,
        )

    def _create_evaluation_run(
        self,
        project_key: str,
        evaluation_id: str,
        dataset_id: str,
    ) -> EvaluationRunRef:
        path = (
            f"projects/{_segment(project_key)}/evaluations/"
            f"{_segment(evaluation_id)}/runs"
        )
        body: dict[str, Any] = {
            "source": "api",
            "datasetId": dataset_id,
        }
        raw = _mapping(
            self._api.post(path, body=body),
            description="evaluation run",
        )
        return self._run_ref(raw)

    def _run_ref(self, raw: Mapping[str, Any]) -> EvaluationRunRef:
        return EvaluationRunRef(
            id=_required_string(raw, "id", "evaluation run"),
            evaluation_id=_required_string(raw, "evaluationId", "evaluation run"),
            state=_required_string(raw, "state", "evaluation run"),
            status_reason=(
                str(raw["statusReason"])
                if raw.get("statusReason") is not None
                else None
            ),
        )

    def _build_handler_config(
        self,
        generation: GenerationConfig,
        tools: Mapping[str, ResolvedTool],
    ) -> dict[str, Any]:
        parameters = generation.get("parameters")
        config: dict[str, Any] = {
            "provider": {"name": generation["provider"]},
            "model": {"name": generation["model"], "parameters": parameters},
            "tools": {
                key: {
                    "description": tool.description,
                    "parameters": tool.schema,
                }
                for key, tool in tools.items()
            },
        }
        snippet_variables = {"snippet": generation.get("prompt_snippets", {})}
        if "instructions" in generation:
            config["instructions"] = parse_template(
                generation["instructions"], snippet_variables
            )
        elif "messages" in generation:
            config["messages"] = [
                {
                    **message,
                    "content": parse_template(message["content"], snippet_variables)
                    if isinstance(message.get("content"), str)
                    else message.get("content"),
                }
                for message in generation["messages"]
            ]
        if "output_format" in generation:
            config["outputFormat"] = generation["output_format"]
        return config

    async def _run_rows(
        self,
        rows: list[DatasetRow],
        handler: EvalHandler,
        config: dict[str, Any],
        tool_handlers: dict[str, ToolImplementation],
        concurrency: int,
    ) -> list[dict[str, Any]]:
        controller = ConcurrencyController(concurrency)

        async def invoke(row: DatasetRow) -> dict[str, Any]:
            await controller.acquire(config["provider"]["name"])
            started = datetime.now(UTC)
            started_clock = time.perf_counter()
            try:
                result = await handler(
                    config, row.input, tool_handlers, dict(row.variables)
                )
                if not isinstance(result, Mapping):
                    raise TypeError("handler result must be a mapping")
                completed = datetime.now(UTC)
                payload: dict[str, Any] = {
                    "row_index": row.row_index,
                    "input": row.input,
                    "expected_output": row.expected_output,
                    "variables": row.variables,
                    "metadata": row.metadata,
                    "output": result.get("output"),
                    "started_at": started.isoformat().replace("+00:00", "Z"),
                    "generated_at": completed.isoformat().replace("+00:00", "Z"),
                    "latency_ms": round((time.perf_counter() - started_clock) * 1000),
                    "status": "COMPLETE",
                }
                usage = result.get("usage")
                if isinstance(usage, Mapping):
                    payload["usage"] = dict(usage)
                controller.record_success(config["provider"]["name"])
                return payload
            except Exception as error:
                completed = datetime.now(UTC)
                return {
                    "row_index": row.row_index,
                    "input": row.input,
                    "expected_output": row.expected_output,
                    "variables": row.variables,
                    "metadata": row.metadata,
                    "started_at": started.isoformat().replace("+00:00", "Z"),
                    "generated_at": completed.isoformat().replace("+00:00", "Z"),
                    "latency_ms": round((time.perf_counter() - started_clock) * 1000),
                    "status": "ERROR",
                    "error": {"code": 5001, "message": f"handler raised: {error}"},
                }
            finally:
                controller.release()

        return list(await asyncio.gather(*(invoke(row) for row in rows)))

    def _emit_generation_events(
        self,
        client: Any,
        *,
        project_key: str,
        evaluation: EvaluationRef,
        evaluation_run: EvaluationRunRef,
        dataset: DatasetRef,
        results: list[dict[str, Any]],
    ) -> None:
        """Queue one LD custom event for each executed dataset row."""
        context = to_ld_context(
            client,
            {
                "kind": "evaluation",
                "key": evaluation_run.id,
                "projectKey": project_key,
                "evaluationId": evaluation.id,
            },
        )
        for result in results:
            identity = {
                "projectKey": project_key,
                "evaluationId": evaluation.id,
                "evaluationRunId": evaluation_run.id,
                "runId": evaluation_run.id,
                "datasetId": dataset.id,
                "rowIndex": result["row_index"],
            }
            event_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            error = result.get("error")
            generated = {
                "status": result["status"],
                "output": result.get("output"),
                "error": error,
            }
            if result["status"] == "ERROR":
                if isinstance(error, Mapping):
                    message = error.get("message")
                    generated["errorMessage"] = (
                        str(message) if message else "Unknown error"
                    )
                else:
                    generated["errorMessage"] = str(error) if error else "Unknown error"
            usage = result.get("usage")
            if isinstance(usage, Mapping):
                normalized_usage = parse_usage(dict(usage))
                generated["usage"] = {
                    "inputTokens": normalized_usage["input"],
                    "outputTokens": normalized_usage["output"],
                }
            content_hash = hashlib.sha256(
                json.dumps(
                    generated, sort_keys=True, separators=(",", ":"), default=str
                ).encode()
            ).hexdigest()
            emitted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            payload: dict[str, Any] = {
                **identity,
                "eventId": event_id,
                "contentHash": content_hash,
                "emittedAt": emitted_at,
                "evaluationKey": evaluation.key,
                "evaluationVersion": evaluation.version,
                "datasetKey": dataset.key,
                "status": result["status"],
                "startedAt": result["started_at"],
                "generatedAt": result["generated_at"],
                "latencyMs": result["latency_ms"],
            }
            if generated["output"] is not None:
                payload["output"] = generated["output"]
            if generated["error"] is not None:
                payload["error"] = generated["error"]
            if generated.get("errorMessage") is not None:
                payload["errorMessage"] = generated["errorMessage"]
            if "usage" in generated:
                payload["usage"] = generated["usage"]
            client.track(GENERATION_EVENT_NAME, context, payload, 1)
            logger.info(
                "%s emittedAt=%s eventId=%s",
                GENERATION_EVENT_NAME,
                emitted_at,
                event_id,
            )

    def _judge_variables(
        self,
        row_result: Mapping[str, Any],
        judge: Judge,
    ) -> dict[str, Any]:
        """Variables available to the judge config's ``{{...}}`` placeholders.

        Absent values become empty strings: ``parse_template`` leaves a
        placeholder with a ``None`` value as-is, and literal mustache text must
        not reach the judge model.
        """
        variables = dict(row_result.get("variables") or {})
        output = row_result.get("output")
        expected = row_result.get("expected_output")
        ground_truth = judge.ground_truth_context
        if ground_truth is not None:
            ground_truth = parse_template(ground_truth, variables)
        elif expected is not None:
            ground_truth = str(expected)
        # message_history carries FORMATTING_INSTRUCTIONS the same way the
        # online path builds it (judges.run_judges), because that -- not the
        # standalone formatting_instructions variable below -- is what every
        # judge built from the AI Library's default templates (accuracy,
        # relevance, toxicity, and any judge cloned from them) actually
        # references. A judge authored before this variable existed must keep
        # getting scored without edits.
        variables.update(
            {
                "input": row_result.get("input") or "",
                "response_to_evaluate": output if output is not None else "",
                "message_history": "\n\n".join(
                    str(value)
                    for value in (
                        row_result.get("input"),
                        output,
                        FORMATTING_INSTRUCTIONS,
                    )
                    if value
                ),
                "expected_output": expected if expected is not None else "",
                "ground_truth_context": (
                    ground_truth if ground_truth is not None else ""
                ),
            }
        )
        return variables

    def _criterion_error_result(
        self,
        base: Mapping[str, Any],
        started_clock: float,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        completed = datetime.now(UTC)
        return {
            **base,
            "status": "ERROR",
            "error": {"code": code, "message": message},
            "evaluated_at": completed.isoformat().replace("+00:00", "Z"),
            "latency_ms": round((time.perf_counter() - started_clock) * 1000),
        }

    async def _run_scorer_for_result(
        self,
        row: Mapping[str, Any],
        scorer: Scorer,
    ) -> dict[str, Any]:
        started = datetime.now(UTC)
        started_clock = time.perf_counter()
        base: dict[str, Any] = {
            "row_index": row["row_index"],
            "criterion_type": scorer.criterion_type,
            "kind": "scorer",
            "started_at": started.isoformat().replace("+00:00", "Z"),
        }
        if row.get("status") != "COMPLETE":
            return self._criterion_error_result(
                base,
                started_clock,
                "generation_incomplete",
                "generation did not complete",
            )
        dataset_row = DatasetRow(
            row_index=row["row_index"],
            input=row.get("input"),
            expected_output=row.get("expected_output"),
            variables=dict(row.get("variables") or {}),
            metadata=row.get("metadata"),
        )
        try:
            score_value = scorer.fn(dataset_row, row.get("output"))
            if inspect.isawaitable(score_value):
                score_value = await score_value
        except Exception as error:
            return self._criterion_error_result(
                base, started_clock, "scorer_raised", f"scorer fn raised: {error}"
            )
        if isinstance(score_value, bool):
            score: float = 1.0 if score_value else 0.0
        else:
            maybe_score = numeric_score(score_value)
            if maybe_score is None:
                return self._criterion_error_result(
                    base,
                    started_clock,
                    "invalid_score",
                    "scorer fn must return a bool or a finite number, "
                    f"got {score_value!r}",
                )
            score = maybe_score
        if score < 0 or score > 1:
            return self._criterion_error_result(
                base,
                started_clock,
                "invalid_score",
                f"scorer fn score must be between 0 and 1, got {score_value!r}",
            )
        completed = datetime.now(UTC)
        return {
            **base,
            "status": "COMPLETE",
            "score": score,
            "reason": None,
            "evaluated_at": completed.isoformat().replace("+00:00", "Z"),
            "latency_ms": round((time.perf_counter() - started_clock) * 1000),
        }

    async def _run_ld_judge_for_result(
        self,
        row: Mapping[str, Any],
        tool_handlers: dict[str, ToolImplementation],
        judge: Judge,
        execution: JudgeExecution,
    ) -> dict[str, Any]:
        resolved = execution.resolved
        started = datetime.now(UTC)
        started_clock = time.perf_counter()
        base: dict[str, Any] = {
            "row_index": row["row_index"],
            "criterion_type": judge.criterion_type,
            "kind": "judge",
            "judge_key": judge.key,
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "variation_key": resolved.variation_key,
            "version": resolved.version,
        }
        if row.get("status") != "COMPLETE":
            return self._criterion_error_result(
                base,
                started_clock,
                "generation_incomplete",
                "generation did not complete",
            )
        # The config is passed unrendered: the handler owns the single
        # parse_template pass, so ``{{...}}`` sequences inside generated output
        # or dataset values are never re-expanded into the judge prompt.
        variables = self._judge_variables(row, judge)
        # An agent-mode handler standing in for a messages-mode judge needs the
        # messages folded into one instructions block, exactly as the online
        # path does before handing a judge config to an agent handler.
        judge_config = (
            collapse_messages_to_instructions(resolved.config)
            if execution.collapse_messages
            else resolved.config
        )
        try:
            result = await execution.handler(
                dict(judge_config),
                row.get("output"),
                tool_handlers,
                {
                    **variables,
                    "formatting_instructions": FORMATTING_INSTRUCTIONS,
                },
            )
        except Exception as error:
            return self._criterion_error_result(
                base, started_clock, "handler_raised", f"judge handler raised: {error}"
            )
        if not isinstance(result, Mapping):
            return self._criterion_error_result(
                base,
                started_clock,
                "invalid_judge_output",
                "judge handler result must be a mapping",
            )
        try:
            raw_score, reason = parse_judge_response(
                result.get("output", result.get("response"))
            )
        except ValueError as error:
            return self._criterion_error_result(
                base, started_clock, "invalid_judge_output", str(error)
            )
        score = numeric_score(raw_score)
        if score is None or score < 0 or score > 1:
            return self._criterion_error_result(
                base,
                started_clock,
                "invalid_score",
                f"judge score must be a number between 0 and 1, got {raw_score!r}",
            )
        completed = datetime.now(UTC)
        # No verdict: the SDK reports the score and LaunchDarkly rules on it. The
        # criterion carries the threshold and the judge's success direction, and
        # ai-evaluator compares them at ingest -- so pass/fail policy is one
        # server-side implementation that applies to every SDK version and to runs
        # already recorded, rather than one frozen into each release of each
        # language's SDK.
        event = {
            **base,
            "status": "COMPLETE",
            "score": score,
            "reason": reason,
            "evaluated_at": completed.isoformat().replace("+00:00", "Z"),
            "latency_ms": round((time.perf_counter() - started_clock) * 1000),
        }
        usage = result.get("usage")
        if isinstance(usage, Mapping):
            event["usage"] = dict(usage)
        return event

    async def _run_criteria_for_results(
        self,
        rows: list[dict[str, Any]],
        tool_handlers: dict[str, ToolImplementation],
        criteria: list[Criterion],
        resolved_judges: Mapping[str, JudgeExecution],
        concurrency: int,
    ) -> list[dict[str, Any]]:
        """Run every (row, criterion) pair, bounded by the run's concurrency."""
        controller = ConcurrencyController(concurrency)

        async def run_one(
            row: Mapping[str, Any], criterion: Criterion
        ) -> dict[str, Any]:
            await controller.acquire()
            try:
                if isinstance(criterion, Scorer):
                    return await self._run_scorer_for_result(row, criterion)
                return await self._run_ld_judge_for_result(
                    row,
                    tool_handlers,
                    criterion,
                    resolved_judges[criterion.key],
                )
            finally:
                controller.release()

        return list(
            await asyncio.gather(
                *(run_one(row, criterion) for row in rows for criterion in criteria)
            )
        )

    def _emit_evaluation_events(
        self,
        client: Any,
        *,
        project_key: str,
        evaluation: EvaluationRef,
        evaluation_run: EvaluationRunRef,
        dataset: DatasetRef,
        results: list[dict[str, Any]],
    ) -> None:
        context = to_ld_context(
            client,
            {
                "kind": "evaluation",
                "key": evaluation_run.id,
                "projectKey": project_key,
                "evaluationId": evaluation.id,
            },
        )
        failures: list[str] = []
        for result in results:
            # One bad criterion result must not stop the remaining results from
            # being emitted, or drop the events already queued for the ones
            # before it -- so every result is attempted and the failures are
            # raised together once the loop is done.
            try:
                identity = {
                    "projectKey": project_key,
                    "evaluationId": evaluation.id,
                    "evaluationRunId": evaluation_run.id,
                    "runId": evaluation_run.id,
                    "datasetId": dataset.id,
                    "rowIndex": result["row_index"],
                    "criterionType": result["criterion_type"],
                }
                event_id = hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                emitted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                usage: TokenUsage | None = None
                if result["kind"] == "judge" and isinstance(
                    result.get("usage"), Mapping
                ):
                    normalized_usage = parse_usage(dict(result["usage"]))
                    usage = TokenUsage(
                        input_tokens=normalized_usage["input"],
                        output_tokens=normalized_usage["output"],
                    )
                error = result.get("error")
                error_message: str | None = None
                if result["status"] == "ERROR":
                    if isinstance(error, Mapping) and error.get("message"):
                        error_message = str(error["message"])
                    else:
                        error_message = str(error) if error else "Unknown error"
                common_payload: dict[str, Any] = {
                    "project_key": project_key,
                    "evaluation_id": evaluation.id,
                    "evaluation_run_id": evaluation_run.id,
                    "run_id": evaluation_run.id,
                    "dataset_id": dataset.id,
                    "row_index": result["row_index"],
                    "criterion_type": result["criterion_type"],
                    "event_id": event_id,
                    "emitted_at": emitted_at,
                    "evaluation_key": evaluation.key,
                    "evaluation_version": evaluation.version,
                    "dataset_key": dataset.key,
                    "status": CriterionStatus(result["status"]),
                    "started_at": result["started_at"],
                    "evaluated_at": result["evaluated_at"],
                    "latency_ms": result["latency_ms"],
                    "score": result.get("score"),
                    "reason": result.get("reason"),
                    "error": error,
                    "error_message": error_message,
                }
                payload_model: CriterionEventPayload
                if result["kind"] == "judge":
                    payload_model = LDJudgeCriterionEventPayload(
                        **common_payload,
                        judge_key=result["judge_key"],
                        variation_key=result["variation_key"],
                        version=result.get("version"),
                        usage=usage,
                    )
                else:
                    payload_model = DeterministicScorerCriterionEventPayload(
                        **common_payload
                    )
                client.track(
                    CRITERION_EVENT_NAME, context, payload_model.to_track_payload(), 1
                )
            except Exception as error:
                logger.exception(
                    "Failed to emit evaluation event for row %s criterion %s",
                    result.get("row_index"),
                    result.get("criterion_type"),
                )
                failures.append(
                    f"row {result.get('row_index')} criterion "
                    f"{result.get('criterion_type')!r}: {error}"
                )
                continue
            logger.info(
                "%s emittedAt=%s eventId=%s",
                CRITERION_EVENT_NAME,
                emitted_at,
                event_id,
            )
        if failures:
            # The evaluation was created with a fixed criterion list, so the
            # backend needs one result per (row, criterion) before it can finish
            # row accounting. A dropped event is never converted into an error
            # result by anything downstream -- swallowing it here would leave
            # run() polling to its timeout instead of reporting what failed.
            raise EvaluationsError(
                f"Failed to emit {len(failures)} of {len(results)} evaluation "
                "criterion events, so the run cannot be fully accounted for: "
                + "; ".join(failures)
            )

    def _get_summary(
        self, project_key: str, evaluation_id: str, run_id: str
    ) -> RunSummary:
        path = (
            f"projects/{_segment(project_key)}/evaluations/{_segment(evaluation_id)}"
            f"/runs/{_segment(run_id)}/summary"
        )
        return RunSummary.from_wire(
            _mapping(self._api.get(path), description="evaluation run summary")
        )
