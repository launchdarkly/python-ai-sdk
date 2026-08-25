from __future__ import annotations

import asyncio
import hashlib
import json
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from ..types import NativeTool
from ..utils import parse_template, to_ld_context
from .api import EvaluationsError, LDApiClient, LDApiError
from .types import (
    DatasetRef,
    DatasetRow,
    EvaluationRef,
    EvaluationRunRef,
    GenerationConfig,
    ResolvedTool,
    RunSummary,
)

DATASET_PAGE_SIZE = 200
GENERATION_EVENT_NAME = "$ld:ai:offline-evals:generation"

EvalHandler = Callable[..., Awaitable[dict[str, Any]]]
ToolImplementation = Callable[..., Any] | NativeTool


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
        tools: Mapping[str, ToolImplementation],
    ) -> dict[str, ResolvedTool]:
        resolved: dict[str, ResolvedTool] = {}
        for key, implementation in tools.items():
            if not callable(implementation) and not isinstance(
                implementation, NativeTool
            ):
                raise EvaluationsError(
                    f"Tool {key!r} must be callable or a NativeTool instance"
                )
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
            body["tools"] = [
                {"key": tool.key, "version": tool.version} for tool in tools.values()
            ]

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
        row_count: int,
        dataset_id: str,
    ) -> EvaluationRunRef:
        path = (
            f"projects/{_segment(project_key)}/evaluations/"
            f"{_segment(evaluation_id)}/runs"
        )
        raw = _mapping(
            self._api.post(
                path,
                body={
                    "source": "api",
                    "rowCount": row_count,
                    "datasetId": dataset_id,
                },
            ),
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
                    "output": {"generation": result.get("output")},
                    "started_at": started.isoformat().replace("+00:00", "Z"),
                    "generated_at": completed.isoformat().replace("+00:00", "Z"),
                    "latency_ms": round((time.perf_counter() - started_clock) * 1000),
                    "status": "COMPLETE",
                }
                usage = result.get("usage")
                if isinstance(usage, Mapping):
                    payload["output"]["usage"] = dict(usage)
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
            generated = {
                "status": result["status"],
                "generationOutput": result.get("output", {}).get("generation"),
                "error": result.get("error"),
                "usage": result.get("output", {}).get("usage"),
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
            if generated["generationOutput"] is not None:
                payload["generationOutput"] = generated["generationOutput"]
            if generated["error"] is not None:
                payload["error"] = generated["error"]
            if generated["usage"] is not None:
                payload["usage"] = generated["usage"]
            client.track(GENERATION_EVENT_NAME, context, payload, 1)
            print(
                f"{GENERATION_EVENT_NAME} emittedAt={emitted_at} eventId={event_id}",
                flush=True,
            )

    async def _poll_run(
        self,
        project_key: str,
        evaluation_id: str,
        run_id: str,
        timeout: float,
    ) -> EvaluationRunRef:
        path = (
            f"projects/{_segment(project_key)}/evaluations/{_segment(evaluation_id)}"
            f"/runs/{_segment(run_id)}"
        )
        deadline = time.monotonic() + timeout
        delay = 0.25
        while True:
            run = self._run_ref(
                _mapping(self._api.get(path), description="evaluation run")
            )
            if run.state == "COMPLETE":
                return run
            if run.state in {"CANCELLED", "TEMPORARY_ERROR", "PERMANENT_ERROR"}:
                reason = f": {run.status_reason}" if run.status_reason else ""
                raise EvaluationsError(
                    f"Evaluation run {run_id!r} failed in state {run.state}{reason}"
                )
            if time.monotonic() >= deadline:
                raise EvaluationsError(
                    f"Evaluation run {run_id!r} is still in progress after {timeout} seconds"
                )
            await asyncio.sleep(delay)
            delay = min(5.0, delay * 2)

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
