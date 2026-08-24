from __future__ import annotations

from typing import Any

import pytest

from launchdarkly_ai_server.evaluations.api import EvaluationsError
from launchdarkly_ai_server.evaluations.runner import EvaluationsRunner


class RecordingApi:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[tuple[str, dict[str, Any] | None]] = []
        self.responses: list[dict[str, Any]] = []

    def post(self, path: str, *, body: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, body))
        return self.responses.pop(0)

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.gets.append((path, params))
        return self.responses.pop(0)


def test_run_create_uses_landed_api_source_contract() -> None:
    api = RecordingApi()
    api.responses = [
        {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"}
    ]
    runner = EvaluationsRunner(api)  # type: ignore[arg-type]

    run = runner._create_evaluation_run("project", "evaluation-id", "dataset-id")

    assert run.id == "run-id"
    assert api.posts == [
        (
            "projects/project/evaluations/evaluation-id/runs",
            {"source": "api", "datasetId": "dataset-id"},
        )
    ]


def test_flat_public_summary_includes_pending_rows() -> None:
    api = RecordingApi()
    api.responses = [{"total": 4, "passed": 1, "failed": 1, "error": 1, "pending": 1}]
    runner = EvaluationsRunner(api)  # type: ignore[arg-type]

    summary = runner._get_summary("project", "evaluation-id", "run-id")

    assert summary.total_rows == 4
    assert summary.pending_rows == 1
    assert api.gets[0][0].endswith("/runs/run-id/summary")


@pytest.mark.asyncio
async def test_poll_uses_lifecycle_state_until_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = RecordingApi()
    api.responses = [
        {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
        {"id": "run-id", "evaluationId": "evaluation-id", "state": "COMPLETE"},
    ]
    runner = EvaluationsRunner(api)  # type: ignore[arg-type]

    async def no_sleep(delay: float) -> None:
        assert delay == 0.25

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    run = await runner._poll_run("project", "evaluation-id", "run-id", 1)

    assert run.state == "COMPLETE"
    assert len(api.gets) == 2


@pytest.mark.asyncio
async def test_poll_preserves_terminal_error_reason() -> None:
    api = RecordingApi()
    api.responses = [
        {
            "id": "run-id",
            "evaluationId": "evaluation-id",
            "state": "PERMANENT_ERROR",
            "statusReason": "invalid dataset",
        }
    ]
    runner = EvaluationsRunner(api)  # type: ignore[arg-type]

    with pytest.raises(EvaluationsError, match="invalid dataset"):
        await runner._poll_run("project", "evaluation-id", "run-id", 1)
