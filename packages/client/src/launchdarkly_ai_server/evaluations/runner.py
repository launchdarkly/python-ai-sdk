from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
import urllib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from ..judge_scoring import (
    FORMATTING_INSTRUCTIONS,
    build_message_history,
    numeric_score,
    parse_judge_response,
)
from ..lifecycle import extract_variation
from ..trajectory import (
    TrajectoryRecorder,
    render_row_trajectory,
    row_fields,
)
from ..utils import (
    normalize_mode,
    parse_template,
    parse_usage,
    select_handler,
    to_ld_context,
)
from .api import (
    EvaluationsError,
    LDApiClient,
    LDApiError,
    require_mapping,
    require_string,
    segment,
)
from .criteria import Criterion, Judge, Scorer
from .events import (
    CriterionEventPayload,
    CriterionStatus,
    DeterministicScorerCriterionEventPayload,
    LDJudgeCriterionEventPayload,
    TokenUsage,
)
from .tools import (
    EvalTool,
    ToolImplementation,
    create_wire_tools,
    handler_config_tools,
)
from .types import (
    DatasetRef,
    DatasetRow,
    EvaluationRef,
    EvaluationRunRef,
    GenerationConfig,
    ResolvedJudge,
    RunSummary,
)

logger = logging.getLogger(__name__)

DATASET_PAGE_SIZE = 200
DATASET_UPLOAD_BATCH_SIZE = 500
GENERATION_EVENT_NAME = "$ld:ai:offline-evals:generation"
CRITERION_EVENT_NAME = "$ld:ai:offline-evals:criterion"

EvalHandler = Callable[..., Awaitable[dict[str, Any]]]


AI_CONFIG_MODES = ("agent", "completion", "judge")


@dataclass(frozen=True)
class JudgeExecution:
    """A resolved judge paired with the handler selected to run its config."""

    resolved: ResolvedJudge
    handler: EvalHandler


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


def describe_handlers(handlers: Sequence[EvalHandler]) -> str:
    """Return the ``provides_for`` of each handler, for an error message."""
    return ", ".join(repr(_provides_for(handler)) for handler in handlers)


def select_eval_handler(
    handlers: Sequence[EvalHandler],
    provider: str | None,
    mode: Literal["agent", "messages"],
) -> EvalHandler | None:
    """Select the handler for ``provider`` and ``mode`` with the ``config()`` rule.

    Returns ``None`` when no handler matches. There is no mode fallback.
    """
    if not provider:
        return None
    try:
        selected = select_handler(
            {"provider": {"name": provider}},
            {"mode": mode},
            list(handlers),  # type: ignore[arg-type]
        )
    except ValueError:
        return None
    return cast(EvalHandler, selected)


def prompt_mode_error(
    config: Mapping[str, Any],
    mode: Literal["agent", "messages"],
) -> str | None:
    """Return an error when the prompt field does not match the handler mode.

    A messages handler needs ``messages`` and an agent handler needs
    ``instructions``. A config with neither field is valid. Returns ``None``
    when the prompt matches.
    """
    if mode == "messages" and config.get("instructions"):
        return (
            "prompt needs 'messages' because the handler is a messages "
            "handler, but it has 'instructions'"
        )
    if mode == "agent" and config.get("messages"):
        return (
            "prompt needs 'instructions' because the handler is an agent "
            "handler, but it has 'messages'"
        )
    return None


def _segment(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def render_row(
    row_index: int,
    *,
    input_value: Any,
    expected_value: Any,
    variables_value: Any,
    metadata_value: Any,
) -> DatasetRow:
    """Render one stored dataset row for handler invocation.

    Shared by hosted and inline datasets so a row renders the same whichever
    path supplied it.
    """
    variables = dict(variables_value) if isinstance(variables_value, Mapping) else {}
    rendered_input = (
        parse_template(input_value, variables) if isinstance(input_value, str) else None
    )
    rendered_expected = (
        parse_template(expected_value, variables)
        if isinstance(expected_value, str)
        else None
    )
    variables["input"] = rendered_input
    variables["expected_output"] = rendered_expected
    return DatasetRow(
        row_index=row_index,
        input=rendered_input,
        expected_output=rendered_expected,
        variables=variables,
        metadata=dict(metadata_value) if isinstance(metadata_value, Mapping) else None,
    )


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

    def _fetch_ai_config_mode(self, project_key: str, config_key: str) -> str:
        """Read the mode of an AI Config from the management API.

        Returns ``"agent"``, ``"completion"`` or ``"judge"``. An absent mode is
        ``"completion"``. Raises :class:`EvaluationsError` for any other value,
        and when the AI Config does not exist.
        """
        description = f"AI Config {config_key!r}"
        path = f"projects/{segment(project_key)}/ai-configs/{segment(config_key)}"
        try:
            raw = require_mapping(self._api.get(path), description=description)
        except LDApiError as error:
            if error.status == 404:
                raise EvaluationsError(
                    f"LaunchDarkly {description} was not found in project {project_key!r}"
                ) from error
            raise
        mode = raw.get("mode", "completion")
        if mode is None:
            mode = "completion"
        if not isinstance(mode, str) or mode not in AI_CONFIG_MODES:
            raise EvaluationsError(
                f"LaunchDarkly {description} has an unknown mode {mode!r}. "
                "Expected one of: " + ", ".join(repr(m) for m in AI_CONFIG_MODES)
            )
        return mode

    def _fetch_config_variation(
        self,
        project_key: str,
        config_key: str,
        variation_key: str,
    ) -> Mapping[str, Any]:
        """Read the latest version of an AI Config variation by key.

        Raises :class:`EvaluationsError` when the variation does not exist,
        has no versions, or has a ``modelConfigKey`` that is not a string.
        """
        description = f"AI Config variation {config_key!r}/{variation_key!r}"
        path = (
            f"projects/{segment(project_key)}/ai-configs/{segment(config_key)}"
            f"/variations/{segment(variation_key)}"
        )
        try:
            raw = require_mapping(self._api.get(path), description=description)
        except LDApiError as error:
            if error.status == 404:
                raise EvaluationsError(
                    f"LaunchDarkly {description} was not found in project {project_key!r}"
                ) from error
            raise
        # The endpoint returns every version of the variation; evaluate the latest.
        items = raw.get("items")
        versions = [
            item
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, Mapping) and isinstance(item.get("version"), int)
        ]
        if not versions:
            raise EvaluationsError(f"LaunchDarkly {description} has no versions")
        latest = max(versions, key=lambda item: int(item["version"]))
        # Absent or empty means that the variation links no model config.
        model_config_key = latest.get("modelConfigKey")
        if model_config_key is not None and not isinstance(model_config_key, str):
            raise EvaluationsError(
                f"LaunchDarkly {description} has a non-string modelConfigKey: "
                f"{model_config_key!r}"
            )
        return latest

    def _fetch_linked_model_config(
        self, project_key: str, variation: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """Read the model config a variation links, or return ``None``."""
        model_config_key = variation.get("modelConfigKey")
        if not isinstance(model_config_key, str) or not model_config_key:
            return None
        return self._fetch_model_config(
            project_key, model_config_key, variation.get("modelConfigVersion")
        )

    def _fetch_model_config(
        self,
        project_key: str,
        model_config_key: str,
        version: Any,
    ) -> Mapping[str, Any]:
        path = (
            f"projects/{segment(project_key)}/ai-configs/model-configs/"
            f"{segment(model_config_key)}"
        )
        # A pinned variation names the model-config version it was built against.
        params = {"version": version} if isinstance(version, int) else None
        try:
            return require_mapping(
                self._api.get(path, params=params),
                description=f"model config {model_config_key!r}",
            )
        except LDApiError as error:
            if error.status == 404:
                raise EvaluationsError(
                    f"LaunchDarkly model config {model_config_key!r} was not found "
                    f"in project {project_key!r}"
                ) from error
            raise

    async def _resolve_judges(
        self,
        project_key: str,
        judges: list[Judge],
        handlers: Sequence[EvalHandler],
    ) -> dict[str, JudgeExecution]:
        """Resolve LD Judge configs and select a handler for each one.

        Runs before any evaluation record is created. Raises one
        :class:`EvaluationsError` that lists every judge with no handler and
        every judge whose prompt does not match its handler's mode.
        """
        resolved: dict[str, JudgeExecution] = {}
        problems: list[str] = []
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
            # Judges cannot use tools.
            judge_config = {
                name: value for name, value in config.items() if name != "tools"
            }
            resolved_judge = ResolvedJudge(
                key=judge.key,
                config=judge_config,
                variation_key=str(meta.get("variationKey") or ""),
                version=int(meta["version"])
                if isinstance(meta.get("version"), int)
                else None,
                provider=str(provider) if isinstance(provider, str) else None,
                mode=normalize_mode(
                    meta.get("mode") if isinstance(meta.get("mode"), str) else None
                ),
            )
            handler = select_eval_handler(
                handlers, resolved_judge.provider, resolved_judge.mode
            )
            if handler is None:
                problems.append(
                    f"judge {judge.key!r} needs a handler for provider "
                    f"{resolved_judge.provider!r} in {resolved_judge.mode!r} mode"
                )
                continue
            prompt_error = prompt_mode_error(judge_config, resolved_judge.mode)
            if prompt_error is not None:
                problems.append(f"judge {judge.key!r}: {prompt_error}")
                continue
            resolved[judge.key] = JudgeExecution(
                resolved=resolved_judge, handler=handler
            )
        if problems:
            raise EvaluationsError(
                "Cannot run the judges of this evaluation: "
                + "; ".join(problems)
                + f". The handlers provide for: {describe_handlers(handlers)}."
            )
        return resolved

    def _fetch_dataset(self, project_key: str, dataset_key: str) -> DatasetRef:
        path = f"projects/{segment(project_key)}/datasets/{segment(dataset_key)}"
        try:
            raw = require_mapping(
                self._api.get(path), description=f"dataset {dataset_key!r}"
            )
        except LDApiError as error:
            if error.status == 404:
                raise EvaluationsError(
                    f"LaunchDarkly dataset {dataset_key!r} was not found in project {project_key!r}"
                ) from error
            raise
        dataset_id = require_string(raw, "id", "dataset")
        response_key = raw.get("key", raw.get("name", dataset_key))
        return DatasetRef(id=dataset_id, key=str(response_key))

    def _fetch_dataset_rows_page(
        self,
        project_key: str,
        dataset_key: str,
        *,
        offset: int,
    ) -> Mapping[str, Any]:
        path = f"projects/{segment(project_key)}/datasets/{segment(dataset_key)}/rows"
        return require_mapping(
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
                item = require_mapping(item_value, description="dataset row")
                row_index = item.get("rowIndex")
                if not isinstance(row_index, int):
                    raise EvaluationsError(
                        "A dataset row is missing its integer rowIndex"
                    )
                rows.append(
                    render_row(
                        row_index,
                        input_value=item.get("input"),
                        expected_value=item.get("expectedOutput"),
                        variables_value=item.get("variables"),
                        metadata_value=item.get("metadata"),
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
        tools: Sequence[EvalTool],
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
            body["tools"] = create_wire_tools(tools)
        if criteria:
            body["criteria"] = [criterion.to_criteria_wire() for criterion in criteria]

        path = f"projects/{segment(project_key)}/evaluations"
        raw = require_mapping(self._api.post(path, body=body), description="evaluation")
        evaluation_id = require_string(raw, "id", "evaluation")
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
        dataset_id: str | None,
    ) -> EvaluationRunRef:
        path = (
            f"projects/{segment(project_key)}/evaluations/{segment(evaluation_id)}/runs"
        )
        body: dict[str, Any] = {"source": "api"}
        if dataset_id is not None:
            body["datasetId"] = dataset_id
        raw = require_mapping(
            self._api.post(path, body=body),
            description="evaluation run",
        )
        return self._run_ref(raw)

    def _upload_dataset_rows(
        self,
        project_key: str,
        evaluation_id: str,
        run_id: str,
        rows: list[DatasetRow],
    ) -> None:
        """Upload an inline dataset's raw rows to its run, in bounded batches.

        Uploads are idempotent per ``rowIdx``, so each batch is retried like a
        GET. Batches go one at a time so a failure names exactly which rows
        did not land.
        """
        path = (
            f"projects/{_segment(project_key)}/evaluations/"
            f"{_segment(evaluation_id)}/runs/{_segment(run_id)}/dataset-rows"
        )
        for start in range(0, len(rows), DATASET_UPLOAD_BATCH_SIZE):
            batch = rows[start : start + DATASET_UPLOAD_BATCH_SIZE]
            body = {
                "rows": [
                    {
                        "rowIdx": row.row_index,
                        "input": row.input,
                        "expectedOutput": row.expected_output,
                        "variables": row.variables,
                        "metadata": row.metadata,
                    }
                    for row in batch
                ]
            }
            try:
                self._api.post(path, body=body, idempotent=True)
            except EvaluationsError as error:
                raise EvaluationsError(
                    f"Failed to upload inline dataset rows {start}-"
                    f"{start + len(batch) - 1} of {len(rows)} to evaluation run "
                    f"{run_id}: {error}"
                ) from error

    def _cancel_evaluation_run(
        self,
        project_key: str,
        evaluation_id: str,
        run_id: str,
    ) -> None:
        """Cancel a run, retried like a GET: replaying a cancel cannot change
        its outcome."""
        path = (
            f"projects/{_segment(project_key)}/evaluations/"
            f"{_segment(evaluation_id)}/runs/{_segment(run_id)}/cancel"
        )
        self._api.post(path, idempotent=True)

    def _run_ref(self, raw: Mapping[str, Any]) -> EvaluationRunRef:
        return EvaluationRunRef(
            id=require_string(raw, "id", "evaluation run"),
            evaluation_id=require_string(raw, "evaluationId", "evaluation run"),
            state=require_string(raw, "state", "evaluation run"),
            status_reason=(
                str(raw["statusReason"])
                if raw.get("statusReason") is not None
                else None
            ),
        )

    def _build_handler_config(
        self,
        generation: GenerationConfig,
        tools: Sequence[EvalTool],
    ) -> dict[str, Any]:
        parameters = generation.get("parameters")
        config: dict[str, Any] = {
            "provider": {"name": generation["provider"]},
            "model": {"name": generation["model"], "parameters": parameters},
            "tools": handler_config_tools(tools),
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
            # One per row, not per run: rows generate concurrently against the
            # same tool map, so a shared recorder would splice their calls.
            recorder = TrajectoryRecorder()
            row_tool_handlers = recorder.wrap(tool_handlers)
            started = datetime.now(UTC)
            started_clock = time.perf_counter()
            try:
                result = await handler(
                    config, row.input, row_tool_handlers, dict(row.variables)
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
                    **row_fields(recorder),
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
                    # The calls that ran are what explain why it raised.
                    **row_fields(recorder),
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
            if dataset.id is None:
                del identity["datasetId"]
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
                "status": result["status"],
                "startedAt": result["started_at"],
                "generatedAt": result["generated_at"],
                "latencyMs": result["latency_ms"],
            }
            if dataset.key is not None:
                payload["datasetKey"] = dataset.key
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
        # The calls the row made on its way to `output`, recorded during
        # generation. Sits between input and output in message_history, which
        # is where it happened.
        trajectory = render_row_trajectory(row_result)
        # Shared builder, not an inline join: this path and both online paths
        # must show a judge the same conversation. The trajectory goes into
        # message_history and nowhere else -- it is already the transcript
        # variable judges read, and a second one would just let a rubric
        # interpolate both and pay for the trajectory twice.
        variables.update(
            {
                "input": row_result.get("input") or "",
                "response_to_evaluate": output if output is not None else "",
                "message_history": build_message_history(
                    user_input=row_result.get("input"),
                    trajectory=trajectory,
                    output=output,
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
        try:
            result = await execution.handler(
                dict(resolved.config),
                row.get("output"),
                {},
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
                if dataset.id is None:
                    del identity["datasetId"]
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
            f"projects/{segment(project_key)}/evaluations/{segment(evaluation_id)}"
            f"/runs/{segment(run_id)}/summary"
        )
        return RunSummary.from_wire(
            require_mapping(self._api.get(path), description="evaluation run summary")
        )
