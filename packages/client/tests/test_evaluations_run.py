from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from launchdarkly_ai_server import create_handler
from launchdarkly_ai_server.evaluations import (
    DatasetRow,
    EvaluationsError,
    HttpResponse,
    Judge,
    Scorer,
    init_evaluations,
)
from launchdarkly_ai_server.evaluations.module import (
    MAX_INLINE_TEXT_BYTES,
    MAX_ROWS,
    EvaluationsModule,
)


@pytest.fixture(autouse=True)
def stub_sdk_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Give every run a resolvable SDK client, since one is now required."""
    monkeypatch.setenv("LD_SDK_KEY", "sdk-key")
    client = MagicMock()
    client.flush = AsyncMock()
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.get_client",
        MagicMock(side_effect=RuntimeError("client not initialized")),
    )
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.init_client",
        AsyncMock(return_value=client),
    )
    return client


class SequencedTransport:
    """Records requests and returns one response for each expected request."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse:
        index = len(self.requests)
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": json.loads(body) if body else None,
                "timeout": timeout,
            }
        )
        if index >= len(self.responses):
            raise AssertionError(f"unexpected request: {method} {url}")
        return self.responses[index]


def response(status: int, body: dict[str, Any] | None = None) -> HttpResponse:
    return HttpResponse(
        status=status, body=json.dumps(body) if body is not None else ""
    )


def failing_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    raise AssertionError("no network I/O expected")


def dataset_page(
    items: list[dict[str, Any]], total: int, next_href: str | None = None
) -> dict[str, Any]:
    links: dict[str, Any] = {"self": {"href": "https://api.test/current"}}
    if next_href:
        links["next"] = {"href": next_href}
    return {"items": items, "totalCount": total, "_links": links}


async def successful_handler(
    config: dict[str, Any],
    user_input: str | None,
    tool_handlers: dict[str, Callable[..., Any]],
    variables: dict[str, Any],
) -> dict[str, Any]:
    assert config["provider"] == {"name": "OpenAI"}
    assert config["model"] == {
        "name": "gpt-4o",
        "parameters": {"temperature": 0.2},
    }
    assert config["tools"]["lookup_order"] == {
        "description": "Look up an order",
        "parameters": {"type": "object"},
    }
    assert "lookup_order" in tool_handlers
    assert variables["input"] == user_input
    return {
        "output": f"generated: {user_input}",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


def lookup_order(order_id: str) -> str:
    return order_id


@pytest.mark.asyncio
async def test_complete_run_with_zero_failed_and_error_rows_passes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="launchdarkly_ai_server.evaluations.runner")
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
    init_client = AsyncMock()
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.init_client", init_client
    )
    transport = SequencedTransport(
        [
            response(
                200,
                {
                    "key": "lookup_order",
                    "version": 7,
                    "description": "Look up an order",
                    "schema": {"type": "object"},
                },
            ),
            response(
                200,
                {
                    "id": "33333333-3333-3333-3333-333333333333",
                    "name": "golden",
                },
            ),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 4,
                            "input": "Order {{order_id}}",
                            "expectedOutput": "Found {{order_id}}",
                            "variables": {"order_id": "A19"},
                            "metadata": {"suite": "orders"},
                        }
                    ],
                    total=2,
                    next_href="https://api.test/api/v2/projects/proj/datasets/key/golden/preview?limit=1&offset=1",
                ),
            ),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 9,
                            "input": "Order {{order_id}}",
                            "expectedOutput": None,
                            "variables": {"order_id": "B20"},
                            "metadata": None,
                        }
                    ],
                    total=2,
                ),
            ),
            response(
                201,
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "name": "support-qa-unique",
                    "version": 1,
                },
            ),
            response(
                201,
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "source": "api",
                    "state": "PENDING",
                    "createdAt": 1,
                },
            ),
            response(
                200,
                {
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "evaluationRunId": "22222222-2222-2222-2222-222222222222",
                    "state": "COMPLETE",
                    "statusCounts": {
                        "total": 2,
                        "passed": 2,
                        "failed": 0,
                        "error": 0,
                        "pending": 0,
                    },
                    "createdAt": 1,
                },
            ),
        ]
    )
    client = MagicMock()
    client.variation = AsyncMock(return_value=True)
    client.flush = AsyncMock()
    init_client.return_value = client
    evals = init_evaluations(
        api_token="token",
        sdk_key="sdk-key",
        base_uri="https://api.example.com",
        ui_base_uri="https://ui.example.com/",
        transport=transport,
    )
    assert evals.sdk_key == "sdk-key"

    result = await evals.run(
        project_key="proj",
        key="support-qa-unique",
        dataset="golden",
        handler=successful_handler,
        tools={"lookup_order": lookup_order},
        generation={
            "provider": "OpenAI",
            "model": "gpt-4o",
            "parameters": {"temperature": 0.2},
            "instructions": "Help the user.",
        },
        concurrency=2,
    )

    assert result.passed is True
    assert result.run_id == "22222222-2222-2222-2222-222222222222"
    assert result.url == (
        "https://ui.example.com/projects/proj/ai/evaluations/"
        "11111111-1111-1111-1111-111111111111/runs/"
        "22222222-2222-2222-2222-222222222222"
    )
    assert result.summary.total_rows == 2

    assert [request["method"] for request in transport.requests] == [
        "GET",
        "GET",
        "GET",
        "GET",
        "POST",
        "POST",
        "GET",
    ]
    assert transport.requests[0]["url"].endswith(
        "/api/v2/projects/proj/ai-tools/lookup_order"
    )
    assert transport.requests[1]["url"].endswith(
        "/api/v2/projects/proj/datasets/golden"
    )
    assert "/projects/proj/datasets/golden/rows" in transport.requests[2]["url"]
    assert "mode=all" in transport.requests[2]["url"]
    assert transport.requests[4]["body"] == {
        "name": "support-qa-unique",
        "generationProvider": "OpenAI",
        "generationModel": "gpt-4o",
        "parameters": {"temperature": 0.2},
        "messages": [{"role": "system", "content": "Help the user."}],
        "tools": [{"key": "lookup_order", "version": 7}],
    }
    assert transport.requests[5]["url"].endswith(
        "/api/v2/projects/proj/evaluations/11111111-1111-1111-1111-111111111111/runs"
    )
    assert transport.requests[5]["body"] == {
        "source": "api",
        "datasetId": "33333333-3333-3333-3333-333333333333",
    }

    assert not any(
        request["url"].endswith("/generation-results") for request in transport.requests
    )
    assert client.track.call_count == 2
    event_name, context, event, metric_value = client.track.call_args_list[0].args
    assert event_name == "$ld:ai:offline-evals:generation"
    assert context["key"] == "22222222-2222-2222-2222-222222222222"
    assert metric_value == 1
    assert event["projectKey"] == "proj"
    assert event["evaluationId"] == "11111111-1111-1111-1111-111111111111"
    assert event["evaluationRunId"] == "22222222-2222-2222-2222-222222222222"
    assert event["runId"] == event["evaluationRunId"]
    assert event["datasetId"] == "33333333-3333-3333-3333-333333333333"
    assert event["rowIndex"] == 4
    assert event["status"] == "COMPLETE"
    assert event["output"] == "generated: Order A19"
    assert event["usage"] == {"inputTokens": 10, "outputTokens": 4}
    assert "generationOutput" not in event
    assert "inputTokens" not in event
    assert "outputTokens" not in event
    assert len(event["eventId"]) == len(event["contentHash"]) == 64
    assert event["emittedAt"].endswith("Z")
    assert datetime.fromisoformat(event["emittedAt"]).tzinfo is not None
    assert {"input", "expected_output", "metadata", "variables"}.isdisjoint(event)
    emit_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "launchdarkly_ai_server.evaluations.runner"
    ]
    assert len(emit_logs) == 2
    assert emit_logs[0] == (
        "$ld:ai:offline-evals:generation "
        f"emittedAt={event['emittedAt']} eventId={event['eventId']}"
    )
    client.flush.assert_awaited_once_with()
    client.variation.assert_not_awaited()
    init_client.assert_awaited_once_with({"sdkKey": "sdk-key"})


@pytest.mark.asyncio
async def test_generation_events_always_emit_without_flag_or_run_status_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.SUMMARY_POLL_INTERVAL_SECONDS", 0
    )
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 3, "input": "hello", "variables": {}}],
                    total=1,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {
                    "id": "run-id",
                    "evaluationId": "evaluation-id",
                    "state": "PENDING",
                },
            ),
            response(
                200,
                {
                    "total": 1,
                    "passed": 0,
                    "failed": 0,
                    "error": 0,
                    "pending": 1,
                },
            ),
            response(
                200,
                {
                    "total": 1,
                    "passed": 0,
                    "failed": 0,
                    "error": 1,
                    "pending": 0,
                },
            ),
        ]
    )
    client = MagicMock()
    client.variation = AsyncMock(side_effect=AssertionError("flag must not be read"))

    def flush_before_summary() -> None:
        assert len(transport.requests) == 4
        assert transport.requests[-1]["url"].endswith("/evaluations/evaluation-id/runs")

    client.flush.side_effect = flush_before_summary

    async def fake_init_client(options: dict[str, Any]) -> MagicMock:
        assert options == {"sdkKey": "sdk-key"}
        return client

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.init_client", fake_init_client
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert result.passed is False
    assert result.summary.error_rows == 1
    assert result.summary.pending_rows == 0
    request_urls = [request["url"] for request in transport.requests]
    assert not any(url.endswith("/generation-results") for url in request_urls)
    client.variation.assert_not_awaited()
    client.track.assert_called_once()
    client.flush.assert_called_once_with()
    status_url = "/evaluations/evaluation-id/runs/run-id"
    assert not any(url.endswith(status_url) for url in request_urls)
    assert request_urls[-1].endswith(f"{status_url}/summary")


@pytest.mark.asyncio
async def test_summary_is_polled_until_rows_are_accounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.SUMMARY_POLL_INTERVAL_SECONDS", 0
    )
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 0, "input": "hello", "variables": {}}], total=1
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {
                    "state": "PENDING",
                    "statusCounts": {"total": 1, "passed": 0, "error": 0, "pending": 1},
                },
            ),
            response(
                200,
                {
                    "state": "COMPLETE",
                    "statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0},
                },
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    summary_requests = [
        request for request in transport.requests if request["url"].endswith("/summary")
    ]
    assert len(summary_requests) == 2
    assert result.passed is True


@pytest.mark.asyncio
async def test_summary_polling_completes_for_real_backend_summary_without_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.SUMMARY_POLL_INTERVAL_SECONDS", 0
    )
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 0, "input": "hello", "variables": {}}], total=1
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {
                    "id": "run-id",
                    "evaluationId": "evaluation-id",
                    "state": "COMPLETE",
                    "rowCount": 10,
                    "selectedRowCount": 10,
                },
            ),
            response(
                200,
                {
                    "statusCounts": {
                        "total": 10,
                        "passed": 10,
                        "failed": 0,
                        "error": 0,
                        "pending": 0,
                    },
                    "estimatedRemainingWindowMs": 0,
                },
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    summary_requests = [
        request for request in transport.requests if request["url"].endswith("/summary")
    ]
    assert len(summary_requests) == 1
    assert result.summary.total_rows == 10
    assert result.summary.pending_rows == 0
    assert result.summary.passed_rows == 10
    assert result.summary.failed_rows == 0
    assert result.summary.error_rows == 0


@pytest.mark.asyncio
async def test_summary_polling_ignores_missing_state_even_when_pending_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.SUMMARY_POLL_INTERVAL_SECONDS", 0
    )
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 0, "input": "hello", "variables": {}}], total=1
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(200, {}),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 0, "error": 0, "pending": 0}},
            ),
            response(
                200,
                {
                    "state": "COMPLETE",
                    "statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0},
                },
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    summary_requests = [
        request for request in transport.requests if request["url"].endswith("/summary")
    ]
    assert len(summary_requests) == 3
    assert result.passed is True


@pytest.mark.asyncio
async def test_summary_polling_times_out_waiting_for_rows_to_be_accounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.SUMMARY_POLL_TIMEOUT_SECONDS", 0
    )
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 0, "input": "hello", "variables": {}}], total=1
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {
                    "state": "PENDING",
                    "statusCounts": {"total": 1, "passed": 0, "error": 0, "pending": 1},
                },
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(
        EvaluationsError,
        match=r"Timed out after 0 seconds.*rows to be fully accounted.*pending_rows=1",
    ):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    summary_requests = [
        request for request in transport.requests if request["url"].endswith("/summary")
    ]
    assert len(summary_requests) == 1


@pytest.mark.asyncio
async def test_poll_timeout_and_interval_are_configurable_per_run() -> None:
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 0, "input": "hello", "variables": {}}], total=1
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 0, "error": 0, "pending": 1}},
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}},
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        poll_interval_seconds=0,
        poll_timeout_seconds=600,
    )

    assert result.passed is True
    summary_requests = [
        request for request in transport.requests if request["url"].endswith("/summary")
    ]
    assert len(summary_requests) == 2

    with pytest.raises(EvaluationsError, match="poll_timeout_seconds"):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            poll_timeout_seconds=-1,
        )


@pytest.mark.parametrize(
    ("poll_interval_seconds", "poll_timeout_seconds"),
    [(float("nan"), 1.0), (1.0, float("nan"))],
    ids=["interval", "timeout"],
)
@pytest.mark.asyncio
async def test_nan_poll_values_are_rejected(
    poll_interval_seconds: float, poll_timeout_seconds: float
) -> None:
    evals = init_evaluations(api_token="token", transport=failing_transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="must be a number"):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
        )


@pytest.mark.asyncio
async def test_run_uses_a_byoc_client_when_no_sdk_key_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
    byoc_client = MagicMock()
    byoc_client.flush = AsyncMock()
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.get_client",
        MagicMock(return_value=byoc_client),
    )
    init_client = AsyncMock(side_effect=AssertionError("must reuse the BYOC client"))
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.init_client", init_client
    )
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 0, "input": "hello", "variables": {}}], total=1
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}},
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)
    assert evals.sdk_key is None

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert result.passed is True
    init_client.assert_not_awaited()
    byoc_client.track.assert_called_once()
    byoc_client.flush.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_run_raises_when_no_sdk_key_and_no_initialized_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
    byoc_client = MagicMock()
    byoc_client.flush = AsyncMock()
    get_client = MagicMock(return_value=byoc_client)
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.get_client", get_client
    )
    evals = init_evaluations(api_token="token", transport=failing_transport)
    get_client.side_effect = RuntimeError("client not initialized")

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="no initialized LaunchDarkly client"):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )


@pytest.mark.asyncio
async def test_failed_rows_fail_the_result() -> None:
    """A row the server scored and marked failed must fail the gate.

    This reverses the previous assertion, which was written when runs were
    generation-only -- a row then either generated or errored, and nothing
    produced a "failed", so excluding failed_rows was unobservable. With
    criteria it is the normal way a run fails, and a gate that ignores it exits
    0 on a run where every row failed its judge.
    """
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [{"rowIndex": 0, "input": "hello", "variables": {}}], total=1
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "eval-key"}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {
                    "state": "COMPLETE",
                    "statusCounts": {
                        "total": 1,
                        "passed": 0,
                        "failed": 1,
                        "error": 0,
                        "pending": 0,
                    },
                },
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert result.summary.failed_rows == 1
    assert result.passed is False


@pytest.mark.asyncio
async def test_run_rejects_instructions_and_messages_before_network_io() -> None:
    transport = SequencedTransport([])
    evals = init_evaluations(api_token="token", transport=transport)

    with pytest.raises(EvaluationsError, match=r"instructions.*messages"):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            generation={
                "provider": "OpenAI",
                "model": "gpt-4o",
                "instructions": "System prompt",
                "messages": [{"role": "user", "content": "{{input}}"}],
            },
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_missing_tool_aborts_before_any_mutating_request() -> None:
    transport = SequencedTransport(
        [response(404, {"code": "not_found", "message": "not found"})]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    with pytest.raises(EvaluationsError, match="missing_tool"):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            tools={"missing_tool": lookup_order},
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert [request["method"] for request in transport.requests] == ["GET"]


@pytest.mark.asyncio
async def test_empty_dataset_fails_before_evaluation_or_run_creation() -> None:
    transport = SequencedTransport(
        [
            response(
                200,
                {
                    "id": "33333333-3333-3333-3333-333333333333",
                    "name": "golden",
                },
            ),
            response(200, dataset_page([], total=0)),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

    with pytest.raises(EvaluationsError, match="empty"):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert [request["method"] for request in transport.requests] == ["GET", "GET"]


@pytest.mark.asyncio
async def test_complete_run_with_error_rows_does_not_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_rows = 1
    calls: list[str | None] = []

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(user_input)
        if user_input == "bad":
            raise RuntimeError("provider failed")
        return {"output": "ok"}

    transport = SequencedTransport(
        [
            response(
                200,
                {
                    "id": "33333333-3333-3333-3333-333333333333",
                    "name": "golden",
                },
            ),
            response(
                200,
                dataset_page(
                    [
                        {"rowIndex": 0, "input": "bad", "variables": {}},
                        {"rowIndex": 1, "input": "good", "variables": {}},
                    ],
                    total=2,
                ),
            ),
            response(
                201,
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "name": "eval-key",
                    "version": 1,
                },
            ),
            response(
                201,
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "source": "api",
                    "state": "PENDING",
                    "createdAt": 1,
                },
            ),
            response(
                200,
                {
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "evaluationRunId": "22222222-2222-2222-2222-222222222222",
                    "state": "COMPLETE",
                    "statusCounts": {
                        "total": 2,
                        "passed": 2 - error_rows,
                        "error": error_rows,
                        "pending": 0,
                    },
                    "createdAt": 1,
                },
            ),
        ]
    )
    client = MagicMock()
    client.variation = AsyncMock(return_value=True)

    async def fake_init_client(options: dict[str, Any]) -> MagicMock:
        assert options == {"sdkKey": "sdk-key"}
        return client

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.init_client", fake_init_client
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    result = await evals.run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert set(calls) == {"bad", "good"}
    assert len(calls) == 2
    assert result.passed is False
    client.variation.assert_not_awaited()
    assert client.track.call_count == 2
    events = [call.args[2] for call in client.track.call_args_list]
    assert {event["status"] for event in events} == {"COMPLETE", "ERROR"}
    error_event = next(event for event in events if event["status"] == "ERROR")
    assert error_event["rowIndex"] == 0
    assert "provider failed" in error_event["error"]["message"]
    assert "provider failed" in error_event["errorMessage"]
    assert "generationOutput" not in error_event
    assert "output" not in error_event
    assert "usage" not in error_event
    assert "inputTokens" not in error_event
    assert "outputTokens" not in error_event
    assert {"input", "expected_output", "metadata", "variables"}.isdisjoint(error_event)


@pytest.mark.asyncio
async def test_run_with_ld_judge_emits_per_criterion_evaluation_event(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 42,
                            "input": "Question {{id}}",
                            "expectedOutput": "Answer {{id}}",
                            "variables": {"id": "A"},
                        }
                    ],
                    total=1,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}},
            ),
        ]
    )

    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        assert key == "$ld:ai:judge:accuracy"
        # An empty or kindless context is invalid to the real LD SDK and would
        # make every judge resolution fail.
        assert context == {"kind": "evaluation", "key": "proj"}
        return {
            "config": {
                "provider": {"name": "OpenAI"},
                "model": {"name": "gpt-4o"},
                "instructions": "Judge {{response_to_evaluate}} against {{expected_output}}",
            },
            "meta": {"variationKey": "default", "version": 12},
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            assert user_input == "generated"
            assert variables["response_to_evaluate"] == "generated"
            assert variables["expected_output"] == "Answer A"
            # The SDK hands the judge config over unrendered; the handler owns
            # the single template pass.
            assert config["instructions"] == (
                "Judge {{response_to_evaluate}} against {{expected_output}}"
            )
            assert variables["formatting_instructions"].startswith(
                "Your response MUST be in valid JSON"
            )
            # message_history must carry the formatting instructions the same
            # way judges.run_judges (the online path) builds it: every judge
            # built from the AI Library's default templates references
            # {{message_history}}, not the standalone formatting_instructions
            # variable above, to ask for the {score, reasoning} JSON shape.
            assert (
                "Your response MUST be in valid JSON format"
                in (variables["message_history"])
            )
            return {
                "output": '{"score": 0.86, "reasoning": "matches policy"}',
                "usage": {"input_tokens": 640, "output_tokens": 48},
            }
        return {
            "output": "generated",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True
    # kind and judgeKey are what let ai-evaluator store this as a judge rather
    # than default it to a deepeval metric and reject the key. successDirection
    # is deliberately absent: LaunchDarkly injects it from the judge's AI Config
    # on the way through, so the SDK must not assert a direction of its own.
    assert transport.requests[2]["body"]["criteria"] == [
        {
            "criterionType": "$ld:ai:judge:accuracy",
            "kind": "judge",
            "judgeKey": "$ld:ai:judge:accuracy",
            "options": {"threshold": 0.5},
        }
    ]
    assert transport.requests[3]["body"] == {"source": "api", "datasetId": "dataset-id"}
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    judge_event = next(event for event in events if event.get("kind") == "judge")
    assert judge_event["criterionType"] == "$ld:ai:judge:accuracy"
    assert judge_event["judgeKey"] == "$ld:ai:judge:accuracy"
    assert judge_event["status"] == "COMPLETE"
    assert judge_event["score"] == 0.86
    assert judge_event["reason"] == "matches policy"
    assert judge_event["usage"] == {"inputTokens": 640, "outputTokens": 48}
    assert judge_event["variationKey"] == "default"
    assert judge_event["version"] == 12
    assert len(judge_event["eventId"]) == 64
    # The SDK reports the score and never rules on it: ai-evaluator derives the
    # verdict at ingest from the criterion's stored threshold and direction.
    assert "verdict" not in judge_event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_inverted", "threshold"),
    [
        # The score is fixed at 0.86 below. Under every direction and on both
        # sides of the threshold, the SDK reports the same thing: a score.
        (False, 0.8),
        (False, 0.9),
        (True, 0.9),
        (True, 0.5),
        (None, 0.8),
    ],
)
async def test_run_with_ld_judge_never_sends_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    is_inverted: bool | None,
    threshold: float,
) -> None:
    """Pass/fail is ai-evaluator's ruling, not the SDK's.

    Parametrized over isInverted -- including the served-payload value -- to
    pin that the SDK does not compare even when it could: verdict policy has to
    be able to change server-side and apply to runs already recorded, which it
    cannot if each SDK release freezes its own comparison.
    """
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 42,
                            "input": "Question {{id}}",
                            "expectedOutput": "Answer {{id}}",
                            "variables": {"id": "A"},
                        }
                    ],
                    total=1,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}},
            ),
        ]
    )

    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "provider": {"name": "OpenAI"},
            "model": {"name": "gpt-4o"},
            "instructions": "Judge {{response_to_evaluate}} against {{expected_output}}",
        }
        if is_inverted is not None:
            config["isInverted"] = is_inverted
        return {
            "config": config,
            "meta": {"variationKey": "default", "version": 12},
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            return {
                "output": '{"score": 0.86, "reasoning": "matches policy"}',
                "usage": {"input_tokens": 640, "output_tokens": 48},
            }
        return {
            "output": "generated",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy", threshold=threshold)],
    )

    assert result.passed is True
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    judge_event = next(event for event in events if event.get("kind") == "judge")
    assert judge_event["score"] == 0.86
    assert "verdict" not in judge_event
    assert "successDirection" not in judge_event


@pytest.mark.asyncio
async def test_judges_resolve_once_per_run_not_once_per_row(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """One resolution for the whole run, however many rows it has.

    extract_variation reads flag delivery, an in-memory store that updates
    within seconds of a UI edit, so resolving per row would let an edit
    mid-run change the rubric text, judge model, and provider between one row
    and the next -- rows in a single run scored against different judges. The
    online path does resolve per invocation (judges.build_judge_tasks), so
    routing the offline runner through it for convenience is a live way to
    reintroduce this.
    """
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {"rowIndex": 0, "input": "one", "variables": {}},
                        {"rowIndex": 1, "input": "two", "variables": {}},
                        {"rowIndex": 2, "input": "three", "variables": {}},
                    ],
                    total=3,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa"}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 3, "passed": 3, "error": 0, "pending": 0}},
            ),
        ]
    )

    resolutions: list[str] = []

    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        resolutions.append(key)
        return {
            "config": {
                "provider": {"name": "OpenAI"},
                "model": {"name": "gpt-4o"},
                "instructions": "Judge {{response_to_evaluate}}",
            },
            "meta": {"variationKey": "default", "version": 12},
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            return {"output": '{"score": 0.9, "reasoning": "fine"}'}
        return {"output": "generated"}

    await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert resolutions == ["$ld:ai:judge:accuracy"]


def test_scorer_lower_is_better_reaches_the_criteria_wire() -> None:
    """A scorer counting something unwanted -- regex hits, edit distance --
    inverts, and only the SDK knows: there is no AI Config for the proxy to
    read a scorer's direction off, so what the caller declares is the sole
    source ai-evaluator derives its verdict from."""

    def count_violations(row: DatasetRow, output: Any) -> float:
        return 0.0

    scorer = Scorer(
        name="policy-violations",
        fn=count_violations,
        threshold=0.0,
        success_direction="lower_is_better",
    )

    assert scorer.to_criteria_wire() == {
        "criterionType": "policy-violations",
        "kind": "scorer",
        "successDirection": "lower_is_better",
        "options": {"threshold": 0.0},
    }


def test_judge_threshold_defaults_so_a_criterion_is_always_rulable() -> None:
    """A judge with no threshold gives LaunchDarkly nothing to compare against,
    so the criterion would be stored and never ruled on."""
    assert Judge(key="$ld:ai:judge:accuracy").to_criteria_wire() == {
        "criterionType": "$ld:ai:judge:accuracy",
        "kind": "judge",
        "judgeKey": "$ld:ai:judge:accuracy",
        "options": {"threshold": 0.5},
    }


@pytest.mark.asyncio
async def test_missing_ld_judge_aborts_before_mutating_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = SequencedTransport([])

    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        raise RuntimeError("not found")

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(
        EvaluationsError,
        match=r"Failed to resolve LaunchDarkly judge 'security-judge': not found",
    ):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            criteria=[Judge(key="security-judge")],
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_run_with_deterministic_scorer_emits_scorer_evaluation_event(
    stub_sdk_client: MagicMock,
) -> None:
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "support-golden-v3"}),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 42,
                            "input": "Ticket {{id}}",
                            "expectedOutput": "refund row",
                            "variables": {"id": "A"},
                        }
                    ],
                    total=1,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}},
            ),
        ]
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {
            "output": "refund exists",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }

    def check_refund(row: DatasetRow, output: Any) -> bool:
        assert row.row_index == 42
        assert row.input == "Ticket A"
        assert output == "refund exists"
        return "refund" in str(output)

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="support-golden-v3",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Scorer(name="refund-exists", fn=check_refund)],
    )

    assert result.passed is True
    # A scorer has no LaunchDarkly-side config, so unlike a judge it declares
    # its own direction and the proxy leaves it alone.
    assert transport.requests[2]["body"]["criteria"] == [
        {
            "criterionType": "refund-exists",
            "kind": "scorer",
            "successDirection": "higher_is_better",
            "options": {"threshold": 1.0},
        }
    ]
    assert transport.requests[3]["body"] == {"source": "api", "datasetId": "dataset-id"}
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    scorer_event = next(event for event in events if event.get("kind") == "scorer")
    assert scorer_event["projectKey"] == "proj"
    assert scorer_event["evaluationId"] == "evaluation-id"
    assert scorer_event["evaluationRunId"] == "run-id"
    assert scorer_event["runId"] == "run-id"
    assert scorer_event["datasetId"] == "dataset-id"
    assert scorer_event["rowIndex"] == 42
    assert scorer_event["criterionType"] == "refund-exists"
    assert scorer_event["evaluationKey"] == "support-qa"
    assert scorer_event["evaluationVersion"] == 3
    assert scorer_event["datasetKey"] == "support-golden-v3"
    assert scorer_event["status"] == "COMPLETE"
    assert scorer_event["score"] == 1
    assert "reason" not in scorer_event
    assert "usage" not in scorer_event
    assert scorer_event["latencyMs"] >= 0
    assert scorer_event["startedAt"].endswith("Z")
    assert scorer_event["evaluatedAt"].endswith("Z")
    assert "judgeKey" not in scorer_event
    assert "variationKey" not in scorer_event
    assert "version" not in scorer_event


def judge_run_transport(*, summary: dict[str, Any] | None = None) -> SequencedTransport:
    """Transport for a one-row run that resolves a dataset, evaluation, and run."""
    return SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 7,
                            "input": "Question {{id}}",
                            "expectedOutput": "Answer {{id}}",
                            "variables": {"id": "A"},
                        }
                    ],
                    total=1,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                summary
                or {
                    "statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}
                },
            ),
        ]
    )


def accuracy_judge_variation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "config": {
                "provider": {"name": "OpenAI"},
                "model": {"name": "gpt-4o"},
                "instructions": "Judge {{response_to_evaluate}} against {{expected_output}}",
            },
            "meta": {"variationKey": "default", "version": 12},
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )


@pytest.mark.parametrize(
    ("judge_output", "expected_code"),
    [
        ('{"score": "high (0.9)", "reasoning": "confident"}', "invalid_score"),
        ('{"score": 3, "reasoning": "confident"}', "invalid_score"),
        ('{"score": NaN, "reasoning": "confident"}', "invalid_score"),
        ("the answer looks right to me", "invalid_judge_output"),
    ],
)
@pytest.mark.asyncio
async def test_bad_judge_output_emits_error_event_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    judge_output: str,
    expected_code: str,
) -> None:
    transport = judge_run_transport()
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            return {"output": judge_output}
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    judge_event = next(event for event in events if event.get("kind") == "judge")
    assert judge_event["status"] == "ERROR"
    assert judge_event["error"]["code"] == expected_code
    assert judge_event["errorMessage"] == judge_event["error"]["message"]
    assert "score" not in judge_event
    stub_sdk_client.flush.assert_awaited()


@pytest.mark.asyncio
async def test_generated_placeholders_are_not_expanded_into_judge_prompt(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    from launchdarkly_ai_server import parse_template

    transport = judge_run_transport()
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            rendered = parse_template(config["instructions"], variables)
            # The placeholder smuggled in via the generated output must stay
            # literal text after the handler's single render pass.
            assert rendered == "Judge {{expected_output}} leaked? against Answer A"
            return {"output": '{"score": 1, "reasoning": "ok"}'}
        return {"output": "{{expected_output}} leaked?"}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    judge_event = next(event for event in events if event.get("kind") == "judge")
    assert judge_event["status"] == "COMPLETE"
    assert judge_event["score"] == 1.0


@pytest.mark.asyncio
async def test_missing_expected_output_renders_empty_judge_variables(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page([{"rowIndex": 7, "input": "Question"}], total=1),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}},
            ),
        ]
    )
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            assert variables["expected_output"] == ""
            assert variables["ground_truth_context"] == ""
            return {"output": '{"score": 1, "reasoning": "ok"}'}
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )
    assert result.passed is True


@pytest.mark.asyncio
async def test_duplicate_criteria_rejected_before_any_request() -> None:
    transport = SequencedTransport([])
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="Duplicate evaluation criteria"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            criteria=[
                Judge(key="accuracy"),
                Scorer(name="accuracy", fn=lambda row, output: True),
            ],
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_duplicate_criteria_rejected_case_insensitively() -> None:
    """Matches the API's own dedup, which lowercases criterionType before
    comparing: the worker's retry gate does the same, so criteria differing
    only by case would still collide there even though they'd look distinct
    to a case-sensitive check."""
    transport = SequencedTransport([])
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="Duplicate evaluation criteria"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            criteria=[
                Judge(key="Accuracy"),
                Scorer(name="accuracy", fn=lambda row, output: True),
            ],
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_errored_generation_row_emits_generation_incomplete_criterion_event(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    transport = judge_run_transport(
        summary={"statusCounts": {"total": 1, "passed": 0, "error": 1, "pending": 0}}
    )
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            raise AssertionError("judges must not run for errored generations")
        raise RuntimeError("provider unavailable")

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is False
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    judge_event = next(event for event in events if event.get("kind") == "judge")
    assert judge_event["status"] == "ERROR"
    assert judge_event["error"]["code"] == "generation_incomplete"


@pytest.mark.asyncio
async def test_failed_evaluation_event_tracking_raises_after_attempting_every_result(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """A dropped criterion event is a delivery failure, not a silent skip.

    The evaluation is created with a fixed criterion list, so the backend needs
    one result per (row, criterion) before row accounting can finish. Swallowing
    the failure leaves run() polling to its timeout and hides the cause, so the
    run attempts every result, flushes what it queued, and then raises.
    """
    transport = judge_run_transport()
    accuracy_judge_variation(monkeypatch)

    attempted: list[str] = []

    def track(event_name: str, *args: Any) -> None:
        if event_name == "$ld:ai:offline-evals:criterion":
            attempted.append(args[1]["criterionType"])
            raise RuntimeError("event pipeline unavailable")

    stub_sdk_client.track = MagicMock(side_effect=track)
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            return {"output": '{"score": 1, "reasoning": "ok"}'}
        return {"output": "generated"}

    with pytest.raises(EvaluationsError) as error:
        await evals.run(
            project_key="proj",
            key="support-qa",
            dataset="golden",
            handler=handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            criteria=[
                Judge(key="$ld:ai:judge:accuracy"),
                Scorer(name="nonempty", fn=lambda row, output: bool(output)),
            ],
        )

    # Every criterion is attempted before the failures are reported together,
    # so one bad result never drops the ones behind it.
    assert attempted == ["$ld:ai:judge:accuracy", "nonempty"]
    assert "Failed to emit 2 of 2 evaluation criterion events" in str(error.value)
    assert "event pipeline unavailable" in str(error.value)
    stub_sdk_client.flush.assert_awaited()


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1])
@pytest.mark.parametrize("field", ["threshold", "pass_rate_threshold"])
def test_criteria_reject_thresholds_outside_zero_to_one(
    field: str, value: float
) -> None:
    """NaN passes both range comparisons, so it needs its own rejection.

    Left in, it is serialized into the criteria wire payload as a bare ``NaN``
    literal and the management API rejects the whole evaluation.
    """
    with pytest.raises(ValueError, match=f"{field} must be a number between 0 and 1"):
        Judge(key="$ld:ai:judge:accuracy", **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be a number between 0 and 1"):
        Scorer(name="nonempty", fn=lambda row, output: True, **{field: value})


def judge_variation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: str,
    mode: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Serve one judge variation for the given provider and mode."""

    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {"variationKey": "default", "version": 12}
        if mode is not None:
            meta["mode"] = mode
        return {
            "config": {
                "provider": {"name": provider},
                "model": {"name": "judge-model"},
                **(config or {"instructions": "Judge {{response_to_evaluate}}"}),
            },
            "meta": meta,
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )


async def _generation_only(
    config: dict[str, Any],
    user_input: str | None = None,
    tool_handlers: dict[str, Callable[..., Any]] | None = None,
    variables: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"output": "generated"}


@pytest.mark.asyncio
async def test_judge_on_another_provider_fails_before_any_records_are_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider handler cannot execute another provider's judge config.

    Passing it anyway spent the generation budget and then recorded every row
    as handler_raised, so the mismatch is caught while it is still only a
    configuration error: before the dataset is read or any record is created.
    """
    transport = judge_run_transport()
    judge_variation(monkeypatch, provider="Anthropic")
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    with pytest.raises(EvaluationsError) as error:
        await evals.run(
            project_key="proj",
            key="support-qa",
            dataset="golden",
            handler=create_handler(("OpenAI", "messages"), _generation_only),
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            criteria=[Judge(key="$ld:ai:judge:accuracy")],
        )

    assert "No handler can run LaunchDarkly judge" in str(error.value)
    assert "'Anthropic'" in str(error.value)
    assert transport.requests == []


@pytest.mark.asyncio
async def test_judge_handlers_route_a_judge_to_its_own_provider(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    transport = judge_run_transport()
    judge_variation(monkeypatch, provider="Anthropic")
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    judged: list[dict[str, Any]] = []

    async def anthropic_judge(
        config: dict[str, Any],
        user_input: str | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        judged.append(config)
        return {"output": '{"score": 0.75, "reasoning": "ok"}'}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=create_handler(("OpenAI", "messages"), _generation_only),
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
        judge_handlers=[create_handler(("Anthropic", "messages"), anthropic_judge)],
    )

    assert result.passed is True
    assert [config["provider"]["name"] for config in judged] == ["Anthropic"]
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    judge_event = next(event for event in events if event.get("kind") == "judge")
    assert judge_event["status"] == "COMPLETE"
    assert judge_event["score"] == 0.75


@pytest.mark.asyncio
@pytest.mark.parametrize("wildcard_first", [True, False])
async def test_exact_provider_judge_handler_beats_a_wildcard_adapter(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    wildcard_first: bool,
) -> None:
    """A wildcard is a fallback, so the order handlers are listed in cannot decide.

    Taking the first provider-or-wildcard match would send an Anthropic judge
    through a multi-provider adapter that merely happened to be listed first.
    """
    transport = judge_run_transport()
    judge_variation(monkeypatch, provider="Anthropic")
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    chosen: list[str] = []

    def judge_handler(name: str) -> Any:
        async def run(
            config: dict[str, Any],
            user_input: str | None = None,
            tool_handlers: dict[str, Callable[..., Any]] | None = None,
            variables: dict[str, Any] | None = None,
            history: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            chosen.append(name)
            return {"output": '{"score": 1, "reasoning": "ok"}'}

        return run

    wildcard = create_handler(("*", "messages"), judge_handler("wildcard"))
    exact = create_handler(("Anthropic", "messages"), judge_handler("exact"))

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=create_handler(("OpenAI", "messages"), _generation_only),
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
        judge_handlers=[wildcard, exact] if wildcard_first else [exact, wildcard],
    )

    assert result.passed is True
    assert chosen == ["exact"]


@pytest.mark.asyncio
async def test_wildcard_judge_handler_runs_a_judge_no_handler_names(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    transport = judge_run_transport()
    judge_variation(monkeypatch, provider="Anthropic")
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    judged: list[dict[str, Any]] = []

    async def wildcard_judge(
        config: dict[str, Any],
        user_input: str | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        judged.append(config)
        return {"output": '{"score": 1, "reasoning": "ok"}'}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=create_handler(("OpenAI", "messages"), _generation_only),
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
        judge_handlers=[create_handler(("*", "messages"), wildcard_judge)],
    )

    assert result.passed is True
    assert [config["provider"]["name"] for config in judged] == ["Anthropic"]


@pytest.mark.asyncio
async def test_agent_handler_runs_a_messages_mode_judge_with_collapsed_messages(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """Mirrors the online path's agent-mode fallback for a messages-mode judge."""
    transport = judge_run_transport()
    judge_variation(
        monkeypatch,
        provider="Anthropic",
        mode="messages",
        config={
            "messages": [
                {"role": "system", "content": "Grade strictly."},
                {"role": "user", "content": "Judge {{response_to_evaluate}}"},
            ]
        },
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    judged: list[dict[str, Any]] = []

    async def anthropic_agent_judge(
        config: dict[str, Any],
        user_input: str | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        judged.append(config)
        return {"output": '{"score": 1, "reasoning": "ok"}'}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=create_handler(("OpenAI", "messages"), _generation_only),
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
        judge_handlers=[create_handler(("Anthropic", "agent"), anthropic_agent_judge)],
    )

    assert result.passed is True
    assert judged[0]["instructions"] == (
        "Grade strictly.\n\nJudge {{response_to_evaluate}}"
    )
    assert judged[0]["messages"] == []


@pytest.mark.asyncio
async def test_generation_handler_runs_a_judge_on_the_same_provider(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    transport = judge_run_transport()
    judge_variation(monkeypatch, provider="OpenAI")
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    calls: list[str | None] = []

    async def openai_handler(
        config: dict[str, Any],
        user_input: str | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        calls.append(config.get("instructions"))
        if "Judge" in (config.get("instructions") or ""):
            return {"output": '{"score": 0.9, "reasoning": "ok"}'}
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=create_handler(("OpenAI", "messages"), openai_handler),
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True
    assert any("Judge" in (instructions or "") for instructions in calls)


@pytest.mark.asyncio
async def test_judge_handlers_must_declare_the_provider_they_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrouted judge handler would silently never be selected."""
    transport = judge_run_transport()
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    with pytest.raises(EvaluationsError, match="does not declare provides_for"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            dataset="golden",
            handler=_generation_only,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            criteria=[Judge(key="$ld:ai:judge:accuracy")],
            judge_handlers=[_generation_only],
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_criteria_run_concurrently_within_the_concurrency_bound(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    import asyncio

    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {"rowIndex": index, "input": f"Question {index}"}
                        for index in range(3)
                    ],
                    total=3,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {"statusCounts": {"total": 3, "passed": 3, "error": 0, "pending": 0}},
            ),
        ]
    )
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    in_flight = 0
    max_in_flight = 0

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal in_flight, max_in_flight
        if "Judge" in config.get("instructions", ""):
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return {"output": '{"score": 1, "reasoning": "ok"}'}
        return {"output": "generated"}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
        concurrency=2,
    )

    assert result.passed is True
    assert max_in_flight == 2


def inline_run_transport(
    *,
    with_tool: bool = False,
    summary: dict[str, Any] | None = None,
    evaluation_version: int = 3,
) -> SequencedTransport:
    responses = []
    if with_tool:
        responses.append(
            response(
                200,
                {
                    "key": "lookup_order",
                    "version": 7,
                    "description": "Look up an order",
                    "schema": {"type": "object"},
                },
            )
        )
    responses.extend(
        [
            response(
                201,
                {
                    "id": "evaluation-id",
                    "name": "support-qa",
                    "version": evaluation_version,
                },
            ),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                summary
                or {
                    "statusCounts": {
                        "total": 1,
                        "passed": 1,
                        "failed": 0,
                        "error": 0,
                        "pending": 0,
                    }
                },
            ),
        ]
    )
    return SequencedTransport(responses)


async def echo_handler(
    config: dict[str, Any],
    user_input: str | None,
    tool_handlers: dict[str, Callable[..., Any]],
    variables: dict[str, Any],
) -> dict[str, Any]:
    return {
        "output": f"generated: {user_input}",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


def digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.asyncio
async def test_inline_rows_run_skips_dataset_requests_entirely(
    stub_sdk_client: MagicMock,
) -> None:
    transport = inline_run_transport(with_tool=True)
    evals = init_evaluations(
        api_token="token",
        sdk_key="sdk-key",
        ui_base_uri="https://ui.example.com",
        transport=transport,
    )

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[DatasetRow(row_index=0, input="Order A19")],
        handler=successful_handler,
        tools={"lookup_order": lookup_order},
        generation={
            "provider": "OpenAI",
            "model": "gpt-4o",
            "parameters": {"temperature": 0.2},
            "instructions": "Help the user.",
        },
    )

    assert [request["method"] for request in transport.requests] == [
        "GET",
        "POST",
        "POST",
        "GET",
    ]
    # The transport raises on a request past the end of its list, so a surviving
    # _fetch_dataset would already have failed -- but it tolerates surplus
    # responses, so assert the absence outright.
    assert not any("/datasets" in request["url"] for request in transport.requests)
    assert transport.requests[0]["url"].endswith(
        "/api/v2/projects/proj/ai-tools/lookup_order"
    )
    # The evaluation body is unchanged by the row source.
    assert transport.requests[1]["body"] == {
        "name": "support-qa",
        "generationProvider": "OpenAI",
        "generationModel": "gpt-4o",
        "parameters": {"temperature": 0.2},
        "messages": [{"role": "system", "content": "Help the user."}],
        "tools": [{"key": "lookup_order", "version": 7}],
    }
    # Equality, not a subset check: no datasetId, and no row count either.
    assert transport.requests[2]["body"] == {"source": "api"}
    assert result.passed is True
    assert result.run_id == "run-id"
    assert result.url == (
        "https://ui.example.com/projects/proj/ai/evaluations/evaluation-id/runs/run-id"
    )


@pytest.mark.asyncio
async def test_inline_rows_without_tools_issues_no_get_before_creation() -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[DatasetRow(row_index=0, input="Order A19")],
        handler=echo_handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert [request["method"] for request in transport.requests] == [
        "POST",
        "POST",
        "GET",
    ]
    assert result.passed is True


@pytest.mark.asyncio
async def test_inline_generation_event_carries_row_data_and_no_dataset_identifiers(
    stub_sdk_client: MagicMock,
) -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[
            DatasetRow(
                row_index=4,
                input="Order {{order_id}}",
                expected_output="Found {{order_id}}",
                variables={"order_id": "A19"},
                metadata={"suite": "orders"},
            )
        ],
        handler=echo_handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    event = stub_sdk_client.track.call_args_list[0].args[2]
    assert "datasetId" not in event
    assert "datasetKey" not in event
    # The row the dataset would otherwise own, rendered.
    assert event["input"] == "Order A19"
    assert event["expectedOutput"] == "Found A19"
    assert event["variables"] == {
        "order_id": "A19",
        "input": "Order A19",
        "expected_output": "Found A19",
    }
    assert event["metadata"] == {"suite": "orders"}
    # Everything else about the event is unchanged.
    assert event["projectKey"] == "proj"
    assert event["rowIndex"] == 4
    assert event["evaluationKey"] == "support-qa"
    assert event["evaluationVersion"] == 3
    assert event["status"] == "COMPLETE"
    assert event["output"] == "generated: Order A19"
    assert event["usage"] == {"inputTokens": 10, "outputTokens": 4}
    assert len(event["eventId"]) == len(event["contentHash"]) == 64


@pytest.mark.asyncio
async def test_inline_generation_event_id_covers_five_identity_fields(
    stub_sdk_client: MagicMock,
) -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[DatasetRow(row_index=4, input="Order A19")],
        handler=echo_handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    event = stub_sdk_client.track.call_args_list[0].args[2]
    assert event["eventId"] == digest(
        {
            "projectKey": "proj",
            "evaluationId": "evaluation-id",
            "evaluationRunId": "run-id",
            "runId": "run-id",
            "rowIndex": 4,
        }
    )
    # A null-valued datasetId is not the same thing as an absent one.
    assert event["eventId"] != digest(
        {
            "projectKey": "proj",
            "evaluationId": "evaluation-id",
            "evaluationRunId": "run-id",
            "runId": "run-id",
            "rowIndex": 4,
            "datasetId": None,
        }
    )


@pytest.mark.asyncio
async def test_inline_content_hash_ignores_row_data(
    stub_sdk_client: MagicMock,
) -> None:
    hashes = []
    for variables, metadata in (
        ({"a": "1"}, {"suite": "one"}),
        ({"b": "2"}, {"suite": "two"}),
    ):
        transport = inline_run_transport()
        evals = init_evaluations(
            api_token="token", sdk_key="sdk-key", transport=transport
        )
        client = MagicMock()
        client.flush = AsyncMock()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "launchdarkly_ai_server.evaluations.module.init_client",
                AsyncMock(return_value=client),
            )
            await evals.run(
                project_key="proj",
                key="support-qa",
                rows=[
                    DatasetRow(
                        row_index=4,
                        input="fixed",
                        variables=variables,
                        metadata=metadata,
                    )
                ],
                handler=echo_handler,
                generation={"provider": "OpenAI", "model": "gpt-4o"},
            )
        hashes.append(client.track.call_args_list[0].args[2]["contentHash"])

    assert hashes[0] == hashes[1]


@pytest.mark.asyncio
async def test_inline_criterion_event_omits_dataset_and_row_data(
    stub_sdk_client: MagicMock,
) -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    def check_refund(row: DatasetRow, output: Any) -> bool:
        assert row.row_index == 42
        assert row.input == "Ticket A"
        return "refund" in str(output)

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[
            DatasetRow(
                row_index=42,
                input="Ticket {{id}}",
                expected_output="refund row",
                variables={"id": "A"},
                metadata={"suite": "orders"},
            )
        ],
        handler=refund_handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Scorer(name="refund-exists", fn=check_refund)],
    )

    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    scorer_event = next(event for event in events if event.get("kind") == "scorer")
    assert "datasetId" not in scorer_event
    assert "datasetKey" not in scorer_event
    assert {"input", "expectedOutput", "variables", "metadata", "output"}.isdisjoint(
        scorer_event
    )
    assert scorer_event["criterionType"] == "refund-exists"
    assert scorer_event["rowIndex"] == 42
    assert scorer_event["score"] == 1
    assert scorer_event["eventId"] == digest(
        {
            "projectKey": "proj",
            "evaluationId": "evaluation-id",
            "evaluationRunId": "run-id",
            "runId": "run-id",
            "rowIndex": 42,
            "criterionType": "refund-exists",
        }
    )
    # The generation event for the same row still carries the row data.
    generation_event = next(
        event for event in events if event.get("status") and "kind" not in event
    )
    assert generation_event["input"] == "Ticket A"


async def refund_handler(*args: object) -> dict[str, Any]:
    return {"output": "refund exists"}


@pytest.mark.asyncio
async def test_inline_rows_render_templates_and_inject_variables() -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    seen: dict[str, Any] = {}

    async def capture(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        seen["user_input"] = user_input
        seen["variables"] = variables
        return {"output": "ok"}

    def scorer(row: DatasetRow, output: Any) -> float:
        seen["row"] = row
        return 1.0

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[
            DatasetRow(
                row_index=0,
                input="Order {{order_id}}",
                expected_output="Found {{order_id}}",
                variables={"order_id": "A19"},
            )
        ],
        handler=capture,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Scorer(name="always", fn=scorer)],
    )

    assert seen["user_input"] == "Order A19"
    assert seen["variables"] == {
        "order_id": "A19",
        "input": "Order A19",
        "expected_output": "Found A19",
    }
    assert seen["row"].input == "Order A19"
    assert seen["row"].expected_output == "Found A19"


@pytest.mark.asyncio
async def test_inline_unresolved_placeholder_is_left_literal() -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    seen: dict[str, Any] = {}

    async def capture(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        seen["user_input"] = user_input
        return {"output": "ok"}

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[DatasetRow(row_index=0, input="Order {{missing}}")],
        handler=capture,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert seen["user_input"] == "Order {{missing}}"


@pytest.mark.asyncio
async def test_inline_row_without_input_or_expected_output_injects_none(
    stub_sdk_client: MagicMock,
) -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    seen: dict[str, Any] = {}

    async def capture(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        seen["user_input"] = user_input
        seen["variables"] = variables
        return {"output": "ok"}

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[DatasetRow(row_index=0)],
        handler=capture,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert seen["user_input"] is None
    assert seen["variables"]["input"] is None
    assert seen["variables"]["expected_output"] is None
    # Null-valued fields are omitted from the payload rather than serialized.
    event = stub_sdk_client.track.call_args_list[0].args[2]
    assert "input" not in event
    assert "expectedOutput" not in event
    assert "metadata" not in event


@pytest.mark.asyncio
async def test_inline_variables_accept_a_non_dict_mapping() -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    seen: dict[str, Any] = {}

    async def capture(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        seen["user_input"] = user_input
        return {"output": "ok"}

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[
            DatasetRow(
                row_index=0,
                input="Order {{order_id}}",
                variables=MappingProxyType({"order_id": "A19"}),  # type: ignore[arg-type]
            )
        ],
        handler=capture,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert seen["user_input"] == "Order A19"


@pytest.mark.asyncio
async def test_inline_rows_are_not_mutated_by_the_run() -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    variables = {"order_id": "A19"}
    metadata = {"suite": "orders"}
    rows = [
        DatasetRow(
            row_index=0,
            input="Order {{order_id}}",
            expected_output="Found {{order_id}}",
            variables=variables,
            metadata=metadata,
        )
    ]

    def mutating_scorer(row: DatasetRow, output: Any) -> float:
        row.variables["injected-by-scorer"] = True
        if row.metadata is not None:
            row.metadata["injected-by-scorer"] = True
        return 1.0

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=rows,
        handler=echo_handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Scorer(name="mutates", fn=mutating_scorer)],
    )

    assert rows[0].input == "Order {{order_id}}"
    assert rows[0].expected_output == "Found {{order_id}}"
    assert rows[0].variables == {"order_id": "A19"}
    assert rows[0].metadata == {"suite": "orders"}
    assert variables == {"order_id": "A19"}
    assert metadata == {"suite": "orders"}


@pytest.mark.asyncio
async def test_inline_rows_can_be_reused_across_two_runs() -> None:
    rows = [
        DatasetRow(
            row_index=0, input="Order {{order_id}}", variables={"order_id": "A19"}
        )
    ]
    rendered = []
    for _ in range(2):
        transport = inline_run_transport()
        evals = init_evaluations(
            api_token="token", sdk_key="sdk-key", transport=transport
        )
        client = MagicMock()
        client.flush = AsyncMock()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "launchdarkly_ai_server.evaluations.module.init_client",
                AsyncMock(return_value=client),
            )
            await evals.run(
                project_key="proj",
                key="support-qa",
                rows=rows,
                handler=echo_handler,
                generation={"provider": "OpenAI", "model": "gpt-4o"},
            )
        rendered.append(client.track.call_args_list[0].args[2]["input"])

    assert rendered == ["Order A19", "Order A19"]


@pytest.mark.asyncio
async def test_inline_metadata_mapping_is_copied_before_the_scorer_sees_it() -> None:
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    metadata = MappingProxyType({"suite": "orders"})
    seen: dict[str, Any] = {}

    def scorer(row: DatasetRow, output: Any) -> float:
        seen["metadata"] = row.metadata
        return 1.0

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[
            DatasetRow(row_index=0, input="x", metadata=metadata),  # type: ignore[arg-type]
        ],
        handler=echo_handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Scorer(name="always", fn=scorer)],
    )

    assert seen["metadata"] == {"suite": "orders"}
    assert isinstance(seen["metadata"], dict)


@pytest.mark.asyncio
async def test_inline_error_row_emits_row_data_without_dataset_identifiers(
    stub_sdk_client: MagicMock,
) -> None:
    transport = inline_run_transport(
        summary={
            "statusCounts": {
                "total": 1,
                "passed": 0,
                "failed": 0,
                "error": 1,
                "pending": 0,
            }
        }
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)

    async def failing(*args: object) -> dict[str, Any]:
        raise RuntimeError("provider exploded")

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[
            DatasetRow(
                row_index=4,
                input="Order {{order_id}}",
                variables={"order_id": "A19"},
                metadata={"suite": "orders"},
            )
        ],
        handler=failing,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert result.passed is False
    event = stub_sdk_client.track.call_args_list[0].args[2]
    assert event["status"] == "ERROR"
    assert event["error"] == {
        "code": 5001,
        "message": "handler raised: provider exploded",
    }
    assert event["errorMessage"] == "handler raised: provider exploded"
    assert "datasetId" not in event
    assert "datasetKey" not in event
    assert event["input"] == "Order A19"
    assert event["metadata"] == {"suite": "orders"}


@pytest.mark.asyncio
async def test_inline_rows_respect_the_concurrency_bound() -> None:
    transport = inline_run_transport(
        summary={
            "statusCounts": {
                "total": 5,
                "passed": 5,
                "failed": 0,
                "error": 0,
                "pending": 0,
            }
        }
    )
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    in_flight = 0
    max_in_flight = 0

    async def handler(*args: object) -> dict[str, Any]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {"output": "ok"}

    result = await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[DatasetRow(row_index=index, input="x") for index in range(5)],
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        concurrency=2,
    )

    assert result.passed is True
    assert max_in_flight == 2


@pytest.mark.asyncio
async def test_inline_rows_run_with_judge_emits_criterion_event_without_dataset_id(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    from launchdarkly_ai_server import parse_template

    accuracy_judge_variation(monkeypatch)
    transport = inline_run_transport()
    evals = init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)
    judge_prompts: list[str] = []

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        instructions = config.get("instructions", "")
        if "Judge" in instructions:
            judge_prompts.append(parse_template(instructions, variables))
            return {"output": '{"score": 1, "reasoning": "accurate"}'}
        return {"output": "generated"}

    await evals.run(
        project_key="proj",
        key="support-qa",
        rows=[
            DatasetRow(
                row_index=7,
                input="Question {{id}}",
                expected_output="Answer {{id}}",
                variables={"id": "A"},
            )
        ],
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    # The judge prompt renders against the inline row's injected variables.
    assert judge_prompts == ["Judge generated against Answer A"]
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    judge_event = next(event for event in events if event.get("kind") == "judge")
    assert "datasetId" not in judge_event
    assert "datasetKey" not in judge_event
    assert judge_event["judgeKey"] == "$ld:ai:judge:accuracy"
    assert judge_event["variationKey"] == "default"
    assert judge_event["version"] == 12
    assert judge_event["score"] == 1
    assert judge_event["reason"] == "accurate"


def no_request_evals(transport: SequencedTransport) -> Any:
    return init_evaluations(api_token="token", sdk_key="sdk-key", transport=transport)


@pytest.mark.asyncio
async def test_run_requires_dataset_or_rows() -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)

    with pytest.raises(EvaluationsError) as error:
        await evals.run(
            project_key="proj",
            key="support-qa",
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert "dataset" in str(error.value)
    assert "rows" in str(error.value)
    assert transport.requests == []


@pytest.mark.asyncio
async def test_run_rejects_both_dataset_and_rows() -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)

    with pytest.raises(EvaluationsError, match="mutually exclusive"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            dataset="golden",
            rows=[DatasetRow(row_index=0, input="x")],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_run_rejects_blank_dataset_when_rows_omitted() -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)

    with pytest.raises(EvaluationsError, match="dataset must not be blank"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            dataset="   ",
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_run_rejects_empty_rows_list() -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)

    with pytest.raises(EvaluationsError, match="rows must not be empty"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_run_rejects_non_dataset_row_entries() -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)

    with pytest.raises(EvaluationsError, match=r"rows\[0\] must be a DatasetRow"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[{"row_index": 0, "input": "x"}],  # type: ignore[list-item]
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.parametrize("row_index", [-1, 1.5, "0", True])
@pytest.mark.asyncio
async def test_run_rejects_invalid_row_index(row_index: Any) -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)

    with pytest.raises(
        EvaluationsError, match=r"rows\[0\].row_index must be a non-negative integer"
    ):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[DatasetRow(row_index=row_index, input="x")],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_run_rejects_duplicate_row_index() -> None:
    """
    Load-bearing, not tidiness: ingest keys an inline row off (run, rowIndex)
    alone, so a duplicate collapses two rows into one stored row and the run can
    never account for its total -- surfacing only as a polling timeout.
    """
    transport = SequencedTransport([])
    evals = no_request_evals(transport)

    with pytest.raises(EvaluationsError, match=r"rows\[1\].row_index 3 duplicates"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[
                DatasetRow(row_index=3, input="first"),
                DatasetRow(row_index=3, input="second"),
            ],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.parametrize(
    "bad_value",
    [
        datetime(2026, 9, 17),
        {1, 2},
        float("nan"),
        float("inf"),
        object(),
    ],
)
@pytest.mark.parametrize("field", ["variables", "metadata"])
@pytest.mark.asyncio
async def test_run_rejects_unserializable_row_values(
    field: str, bad_value: Any
) -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)
    row = DatasetRow(row_index=0, input="x", **{field: {"bad": bad_value}})

    with pytest.raises(
        EvaluationsError, match=rf"rows\[0\].{field} must be JSON-serializable"
    ):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[row],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.parametrize("field", ["variables", "metadata"])
@pytest.mark.asyncio
async def test_run_rejects_non_mapping_row_values(field: str) -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)
    row = DatasetRow(row_index=0, input="x", **{field: ["not", "a", "mapping"]})

    with pytest.raises(EvaluationsError, match=rf"rows\[0\].{field} must be a mapping"):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[row],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.parametrize("field", ["input", "expected_output"])
@pytest.mark.asyncio
async def test_run_rejects_non_string_inline_input(field: str) -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)
    row = DatasetRow(row_index=0, **{field: 42})

    with pytest.raises(
        EvaluationsError, match=rf"rows\[0\].{field} must be a string or None"
    ):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[row],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_run_rejects_more_rows_than_the_limit() -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)
    rows = [DatasetRow(row_index=index, input="x") for index in range(MAX_ROWS + 1)]

    with pytest.raises(
        EvaluationsError, match=rf"rows has {MAX_ROWS + 1} entries, over the {MAX_ROWS}"
    ):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=rows,
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


def _sized_field_value(field: str, text: str) -> Any:
    return {"k": text} if field in {"variables", "metadata"} else text


def _sized_field_text(field: str, total_bytes: int) -> str:
    """ASCII text that makes ``field`` encode to exactly ``total_bytes``."""
    if field not in {"variables", "metadata"}:
        return "a" * total_bytes
    # Derived rather than hardcoded so the wrapper in sized_field_value stays
    # free to change: the mapping's braces, quotes and key are measured too.
    overhead = len(json.dumps(_sized_field_value(field, ""), ensure_ascii=False))
    return "a" * (total_bytes - overhead)


@pytest.mark.parametrize("field", ["input", "expected_output", "variables", "metadata"])
@pytest.mark.asyncio
async def test_run_rejects_oversized_inline_text(field: str) -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)
    oversize = MAX_INLINE_TEXT_BYTES + 1
    row = DatasetRow(
        row_index=0,
        **{field: _sized_field_value(field, _sized_field_text(field, oversize))},
    )

    with pytest.raises(
        EvaluationsError,
        match=rf"rows\[0\].{field} is {oversize} bytes, over the "
        rf"{MAX_INLINE_TEXT_BYTES} byte limit",
    ):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[row],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.parametrize("field", ["input", "expected_output", "variables", "metadata"])
@pytest.mark.asyncio
async def test_run_rejects_oversized_multibyte_inline_text(field: str) -> None:
    transport = SequencedTransport([])
    evals = no_request_evals(transport)
    # Two bytes per character, so half as many characters as the byte limit. The
    # mapping wrapper only adds ASCII, so a mapping field is over on the same text.
    text = "é" * (MAX_INLINE_TEXT_BYTES // 2 + 1)
    assert len(text) < MAX_INLINE_TEXT_BYTES
    row = DatasetRow(row_index=0, **{field: _sized_field_value(field, text)})

    with pytest.raises(
        EvaluationsError, match=rf"rows\[0\].{field} is \d+ bytes, over the"
    ):
        await evals.run(
            project_key="proj",
            key="support-qa",
            rows=[row],
            handler=echo_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.parametrize("field", ["input", "expected_output", "variables", "metadata"])
def test_validate_rows_accepts_the_limits_exactly(field: str) -> None:
    EvaluationsModule._validate_rows(
        [DatasetRow(row_index=index, input="x") for index in range(MAX_ROWS)]
    )
    at_limit = _sized_field_value(
        field, _sized_field_text(field, MAX_INLINE_TEXT_BYTES)
    )
    EvaluationsModule._validate_rows([DatasetRow(row_index=0, **{field: at_limit})])
