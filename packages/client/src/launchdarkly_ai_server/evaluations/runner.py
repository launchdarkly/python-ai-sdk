from __future__ import annotations

import asyncio
import time
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any

from ..types import NativeTool
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
                row_input = input_value if isinstance(input_value, str) else None
                expected_output = (
                    expected_value if isinstance(expected_value, str) else None
                )
                metadata_value = item.get("metadata")
                rows.append(
                    DatasetRow(
                        row_index=row_index,
                        input=row_input,
                        expected_output=expected_output,
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
