from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from launchdarkly_ai_server import NativeTool, create_handler
from launchdarkly_ai_server.evaluations import (
    AIConfig,
    DatasetRow,
    EvalTool,
    EvaluationsError,
    HttpResponse,
    Judge,
    Scorer,
    init_evaluations,
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


def refund_order(order_id: str) -> str:
    return f"refunded {order_id}"


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
        project_key="proj",
        api_key="token",
        sdk_key="sdk-key",
        base_uri="https://api.example.com",
        ui_base_uri="https://ui.example.com/",
        transport=transport,
    )
    assert evals.sdk_key == "sdk-key"

    result = await evals.run(
        key="support-qa-unique",
        dataset="golden",
        handler=successful_handler,
        tools=[await evals.tools.get("lookup_order", implementation=lookup_order)],
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
        "tools": [{"key": "lookup_order", "version": 7, "source": "library"}],
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
    # The captured tool trajectory reaches LaunchDarkly only inside the prompt a
    # judge was shown, never as a generation wire field the backend has not
    # specified.
    assert {
        "toolCalls",
        "tool_calls",
        "toolTrajectory",
        "tool_trajectory",
        "observableTools",
        "observable_tools",
    }.isdisjoint(event)
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(
        EvaluationsError,
        match=r"Timed out after 0 seconds.*rows to be fully accounted.*pending_rows=1",
    ):
        await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", transport=failing_transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="must be a number"):
        await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)
    assert evals.sdk_key is None

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", transport=failing_transport
    )
    get_client.side_effect = RuntimeError("client not initialized")

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="no initialized LaunchDarkly client"):
        await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    result = await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    with pytest.raises(EvaluationsError, match=r"instructions.*messages"):
        await evals.run(
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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    with pytest.raises(EvaluationsError, match="missing_tool"):
        await evals.tools.get("missing_tool", implementation=lookup_order)

    assert [request["method"] for request in transport.requests] == ["GET"]


ORDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}


def recorded_paths(transport: SequencedTransport) -> list[tuple[str, str]]:
    """The recorded requests as ``(method, path)`` pairs, query strings dropped."""
    return [
        (
            request["method"],
            request["url"].split("/api/v2/", 1)[-1].split("?", 1)[0],
        )
        for request in transport.requests
    ]


def hosted_dataset_responses() -> list[HttpResponse]:
    """Canned responses for a one-row hosted dataset run that passes."""
    return [
        response(200, {"id": "dataset-id", "name": "golden"}),
        response(
            200,
            dataset_page(
                [{"rowIndex": 0, "input": "Order A19", "variables": {}}], total=1
            ),
        ),
        response(201, {"id": "evaluation-id", "name": "eval-key", "version": 3}),
        response(
            201,
            {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
        ),
        response(
            200,
            {
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


@pytest.mark.asyncio
async def test_inline_tool_runs_without_reading_the_tool_api() -> None:
    transport = SequencedTransport(hosted_dataset_responses())
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
    seen: dict[str, Any] = {}

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        seen["config_tools"] = config["tools"]
        seen["tool_handlers"] = tool_handlers
        return {"output": "ok"}

    result = await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        tools=[
            EvalTool(
                key="lookup_order",
                implementation=lookup_order,
                schema=ORDER_SCHEMA,
                description="Look up an order",
            )
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert result.passed is True
    # The full sequence, so a stray request anywhere in the run is visible.
    assert recorded_paths(transport) == [
        ("GET", "projects/proj/datasets/golden"),
        ("GET", "projects/proj/datasets/golden/rows"),
        ("POST", "projects/proj/evaluations"),
        ("POST", "projects/proj/evaluations/evaluation-id/runs"),
        ("GET", "projects/proj/evaluations/evaluation-id/runs/run-id/summary"),
    ]
    # Asserted explicitly rather than left to the sequence above: a transport
    # that tolerated a surplus request would not fail on a stray tool GET, and
    # skipping that request is the whole point of an inline definition.
    assert not any("/ai-tools" in request["url"] for request in transport.requests), (
        "an inline tool must not be looked up in the AI library"
    )

    assert transport.requests[2]["body"]["tools"] == [
        {
            "key": "lookup_order",
            "schema": ORDER_SCHEMA,
            "description": "Look up an order",
            "source": "inline",
        }
    ]
    # Same config shape a library tool produces, fed from the inline body, so a
    # handler cannot tell the two sources apart.
    assert seen["config_tools"] == {
        "lookup_order": {
            "description": "Look up an order",
            "parameters": ORDER_SCHEMA,
        }
    }
    # The handler is passed the executable, not the definition wrapping it.
    assert list(seen["tool_handlers"]) == ["lookup_order"]
    assert seen["tool_handlers"]["lookup_order"]("A1") == "A1"


@pytest.mark.asyncio
async def test_inline_tool_description_defaults_to_empty_string() -> None:
    transport = SequencedTransport(hosted_dataset_responses())
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "ok"}

    await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        tools=[
            EvalTool(
                key="lookup_order", implementation=lookup_order, schema=ORDER_SCHEMA
            )
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert transport.requests[2]["body"]["tools"] == [
        {
            "key": "lookup_order",
            "schema": ORDER_SCHEMA,
            "description": "",
            "source": "inline",
        }
    ]


@pytest.mark.asyncio
async def test_mixed_library_and_inline_tools_each_keep_their_own_source() -> None:
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
            *hosted_dataset_responses(),
        ]
    )
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
    seen: dict[str, Any] = {}

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        seen["config_tools"] = config["tools"]
        seen["tool_handlers"] = tool_handlers
        return {"output": "ok"}

    await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        tools=[
            await evals.tools.get("lookup_order", implementation=lookup_order),
            EvalTool(
                key="refund_order",
                implementation=refund_order,
                schema=ORDER_SCHEMA,
                description="Refund an order",
            ),
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    # Exactly one tool GET, for the library key only.
    assert [
        path for method, path in recorded_paths(transport) if "ai-tools" in path
    ] == ["projects/proj/ai-tools/lookup_order"]
    assert transport.requests[3]["body"]["tools"] == [
        {"key": "lookup_order", "version": 7, "source": "library"},
        {
            "key": "refund_order",
            "schema": ORDER_SCHEMA,
            "description": "Refund an order",
            "source": "inline",
        },
    ]
    assert seen["config_tools"] == {
        "lookup_order": {
            "description": "Look up an order",
            "parameters": {"type": "object"},
        },
        "refund_order": {
            "description": "Refund an order",
            "parameters": ORDER_SCHEMA,
        },
    }
    assert sorted(seen["tool_handlers"]) == ["lookup_order", "refund_order"]
    assert seen["tool_handlers"]["lookup_order"]("A1") == "A1"
    assert seen["tool_handlers"]["refund_order"]("A1") == "refunded A1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "match"),
    [
        pytest.param(
            [EvalTool(key="  ", implementation=lookup_order, schema=ORDER_SCHEMA)],
            "must not be blank",
            id="blank_key",
        ),
        pytest.param(
            [
                EvalTool(
                    key="Lookup_Order", implementation=lookup_order, schema=ORDER_SCHEMA
                )
            ],
            "must not use uppercase letters",
            id="uppercase_key",
        ),
        pytest.param(
            [EvalTool(key="lookup_order", implementation=lookup_order, schema=None)],  # type: ignore[arg-type]
            "schema must be a JSON object",
            id="schema_is_none",
        ),
        pytest.param(
            [
                EvalTool(
                    key="lookup_order",
                    implementation=lookup_order,
                    schema=[{"type": "object"}],
                )
            ],  # type: ignore[arg-type]
            "schema must be a JSON object",
            id="schema_is_a_list",
        ),
        pytest.param(
            [
                EvalTool(
                    key="lookup_order",
                    implementation=lookup_order,
                    schema={"default": object()},
                )
            ],
            "schema must be JSON-serializable",
            id="schema_is_not_serializable",
        ),
        pytest.param(
            [
                EvalTool(
                    key="lookup_order",
                    implementation=lookup_order,
                    schema={"default": float("nan")},
                )
            ],
            "schema must be JSON-serializable",
            id="schema_holds_nan",
        ),
        pytest.param(
            [
                EvalTool(
                    key="lookup_order",
                    implementation=lookup_order,
                    schema={"default": float("inf")},
                )
            ],
            "schema must be JSON-serializable",
            id="schema_holds_infinity",
        ),
        pytest.param(
            [
                EvalTool(
                    key="lookup_order",
                    implementation="not a function",
                    schema=ORDER_SCHEMA,
                )
            ],  # type: ignore[arg-type]
            "implementation must be callable",
            id="implementation_is_not_callable",
        ),
        pytest.param(
            [
                EvalTool(
                    key="lookup_order",
                    implementation=lookup_order,
                    schema=ORDER_SCHEMA,
                    description=None,
                )
            ],  # type: ignore[arg-type]
            "description must be a string",
            id="description_is_not_a_string",
        ),
        pytest.param(
            ["not a tool"],  # type: ignore[dict-item]
            "each entry in tools must be an EvalTool",
            id="entry_is_not_a_tool",
        ),
    ],
)
async def test_bad_tool_entry_is_rejected_with_zero_requests(
    tools: list[Any], match: str
) -> None:
    transport = SequencedTransport([])
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    with pytest.raises(EvaluationsError, match=match):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            tools=tools,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_native_tool_paired_with_an_inline_definition_is_rejected() -> None:
    transport = SequencedTransport([])
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    with pytest.raises(EvaluationsError, match=r"lookup_order.*NativeTool"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            tools=[
                EvalTool(
                    key="lookup_order",
                    implementation=NativeTool("WebSearch"),
                    schema=ORDER_SCHEMA,
                )
            ],
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_native_tool_on_its_own_still_resolves_from_the_library() -> None:
    transport = SequencedTransport(
        [
            response(200, {"key": "web_search", "version": 2}),
            *hosted_dataset_responses(),
        ]
    )
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "ok"}

    await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        tools=[
            await evals.tools.get("web_search", implementation=NativeTool("WebSearch"))
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    assert recorded_paths(transport)[0] == ("GET", "projects/proj/ai-tools/web_search")
    assert transport.requests[3]["body"]["tools"] == [
        {"key": "web_search", "version": 2, "source": "library"}
    ]


@pytest.mark.asyncio
async def test_a_repeated_tool_key_is_rejected_with_zero_requests() -> None:
    """One key names one tool, whichever source each entry came from."""
    transport = SequencedTransport([])
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    with pytest.raises(
        EvaluationsError, match=r"'lookup_order' appears more than once"
    ):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            tools=[
                EvalTool(
                    key="lookup_order",
                    implementation=lookup_order,
                    schema=ORDER_SCHEMA,
                ),
                EvalTool(
                    key="lookup_order",
                    implementation=refund_order,
                    schema=ORDER_SCHEMA,
                ),
            ],
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_tools_get_refuses_an_uppercase_key_before_any_request() -> None:
    """A tool key is lowercase, whether it is constructed or read from the library."""
    transport = SequencedTransport([])
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    with pytest.raises(EvaluationsError, match="must not use uppercase letters"):
        await evals.tools.get("Lookup_Order", implementation=lookup_order)

    assert transport.requests == []


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
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    with pytest.raises(EvaluationsError, match="empty"):
        await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    result = await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(
        EvaluationsError,
        match=r"Failed to resolve LaunchDarkly judge 'security-judge': not found",
    ):
        await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="Duplicate evaluation criteria"):
        await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    with pytest.raises(EvaluationsError, match="Duplicate evaluation criteria"):
        await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    with pytest.raises(EvaluationsError) as error:
        await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    with pytest.raises(EvaluationsError, match="does not declare provides_for"):
        await evals.run(
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
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

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
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
        concurrency=2,
    )

    assert result.passed is True
    assert max_in_flight == 2


def tool_run_transport(*, rows: int = 1) -> SequencedTransport:
    """Transport for a run that resolves one tool before its dataset."""
    return SequencedTransport(
        [
            response(200, {"key": "lookup_order", "version": 4, "schema": {}}),
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": index,
                            "input": f"Question {index}",
                            "expectedOutput": "Answer",
                        }
                        for index in range(rows)
                    ],
                    total=rows,
                ),
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
            ),
            response(
                200,
                {
                    "statusCounts": {
                        "total": rows,
                        "passed": rows,
                        "error": 0,
                        "pending": 0,
                    }
                },
            ),
        ]
    )


@pytest.mark.asyncio
async def test_tool_trajectory_reaches_the_judge_via_message_history(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """The calls a row made are what let a judge grade its tool use.

    Handler packages return only {output, usage}, so without the runner
    recording the trajectory itself a judge sees the answer and nothing about
    how the agent arrived at it.
    """
    transport = tool_run_transport()
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
    seen: dict[str, str] = {}

    def lookup_order(args: dict[str, Any]) -> str:
        return f"order {args['id']} shipped"

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            seen["message_history"] = variables["message_history"]
            # The trajectory lives in message_history and nowhere else: this is
            # already the transcript variable every judge reads, so a second
            # overlapping variable only invited a rubric to pay for the
            # trajectory twice.
            assert "tool_trajectory" not in variables
            return {"output": '{"score": 1, "reasoning": "used the right tool"}'}
        # Called without await: a sync tool stays sync through the recorder.
        assert tool_handlers["lookup_order"]({"id": "A1"}) == "order A1 shipped"
        return {"output": "Your order shipped."}

    result = await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        tools=[await evals.tools.get("lookup_order", implementation=lookup_order)],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True
    history = seen["message_history"]
    assert "Tools available: lookup_order" in history
    assert '1. lookup_order\n   arguments: {"id":"A1"}' in history
    assert "result: order A1 shipped" in history
    # The trajectory sits between the request and the answer, because that is
    # where it happened: a judge reading the history sees the question, what the
    # agent did about it, then what it replied.
    assert history.index("Question 0") < history.index("Tools available")
    assert history.index("Tools available") < history.index("Your order shipped.")
    assert history.index("Your order shipped.") < history.index(
        "Your response MUST be in valid JSON"
    )


@pytest.mark.asyncio
async def test_each_row_gets_only_its_own_tool_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """Rows generate concurrently against one shared tool map.

    A recorder shared across rows would splice row 0's calls into row 1's
    trajectory and hand the judge a conversation that never happened.
    """
    import asyncio

    transport = tool_run_transport(rows=2)
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
    histories: dict[str, str] = {}
    both_started = asyncio.Barrier(2)

    def lookup_order(args: dict[str, Any]) -> str:
        return f"order {args['id']}"

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            histories[str(user_input)] = variables["message_history"]
            return {"output": '{"score": 1, "reasoning": "ok"}'}
        row = str(user_input).split()[-1]
        # Interleave the two rows' tool calls so a shared recorder would be
        # caught rather than merely be possible.
        await both_started.wait()
        tool_handlers["lookup_order"]({"id": row})
        return {"output": f"answered {row}"}

    result = await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        tools=[await evals.tools.get("lookup_order", implementation=lookup_order)],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
        concurrency=2,
    )

    assert result.passed is True
    assert '{"id":"0"}' in histories["answered 0"]
    assert '{"id":"1"}' not in histories["answered 0"]
    assert '{"id":"1"}' in histories["answered 1"]
    assert '{"id":"0"}' not in histories["answered 1"]


@pytest.mark.asyncio
async def test_a_row_that_called_no_tools_says_so_to_the_judge(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """A judge grading tool selection needs to see the tool that went unused."""
    transport = tool_run_transport()
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
    seen: dict[str, str] = {}

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            seen["message_history"] = variables["message_history"]
            return {"output": '{"score": 0, "reasoning": "should have looked it up"}'}
        return {"output": "I do not know."}

    await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        tools=[
            await evals.tools.get("lookup_order", implementation=lambda args: "unused")
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert (
        "Tools available: lookup_order\n"
        "No tool calls were made while producing the response."
    ) in seen["message_history"]


@pytest.mark.asyncio
async def test_a_run_without_tools_leaves_message_history_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """Judges authored before trajectories existed must read the same history.

    With no observable tools there is nothing to report, so no trajectory block
    is added rather than one saying no tools were called.
    """
    transport = judge_run_transport()
    accuracy_judge_variation(monkeypatch)
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
    seen: dict[str, str] = {}

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            seen["message_history"] = variables["message_history"]
            return {"output": '{"score": 1, "reasoning": "ok"}'}
        return {"output": "generated"}

    await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert seen["message_history"].startswith("Question A\n\ngenerated\n\n")
    assert "Tools available" not in seen["message_history"]


@pytest.mark.asyncio
async def test_tool_result_placeholders_are_not_expanded_into_the_judge_prompt(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
) -> None:
    """A tool result is now judge-prompt input, so it is an injection surface.

    It stays literal for the same reason the generated output does: the judge
    config is handed over unrendered and the handler makes exactly one template
    pass, so a substituted value is never rescanned for placeholders.
    """
    from launchdarkly_ai_server import parse_template

    transport = tool_run_transport()

    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "config": {
                "provider": {"name": "OpenAI"},
                "model": {"name": "gpt-4o"},
                "instructions": "Judge this history: {{message_history}}",
            },
            "meta": {"variationKey": "default", "version": 12},
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge this history" in config.get("instructions", ""):
            rendered = parse_template(config["instructions"], variables)
            assert "result: {{expected_output}} leaked?" in rendered
            assert "Answer leaked?" not in rendered
            return {"output": '{"score": 1, "reasoning": "ok"}'}
        tool_handlers["lookup_order"]({"id": "A1"})
        return {"output": "done"}

    result = await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        tools=[
            await evals.tools.get(
                "lookup_order",
                implementation=lambda args: "{{expected_output}} leaked?",
            )
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_failed_row_keeps_the_calls_made_before_the_handler_raised() -> None:
    """The trajectory of a row that errored is what explains why it errored."""
    from launchdarkly_ai_server.evaluations.api import LDApiClient
    from launchdarkly_ai_server.evaluations.runner import EvaluationsRunner

    runner = EvaluationsRunner(
        LDApiClient(api_key="token", transport=failing_transport)
    )

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        tool_handlers["lookup_order"]({"id": "A1"})
        raise RuntimeError("model refused")

    results = await runner._run_rows(
        [DatasetRow(row_index=0, input="Question")],
        handler,
        {"provider": {"name": "OpenAI"}, "model": {"name": "gpt-4o"}},
        {"lookup_order": lambda args: "shipped"},
        1,
    )

    assert results[0]["status"] == "ERROR"
    assert [invocation.name for invocation in results[0]["tool_calls"]] == [
        "lookup_order"
    ]
    assert results[0]["tool_calls"][0].result == "shipped"


def config_variation_page(**overrides: Any) -> dict[str, Any]:
    """A getAIConfigVariation response holding two versions of one variation."""
    latest: dict[str, Any] = {
        "_id": "variation-id",
        "key": "control",
        "name": "Control",
        "version": 2,
        "createdAt": 2,
        "model": {"modelName": "gpt-4o", "parameters": {"temperature": 0.7}},
        "modelConfigKey": "OpenAI.gpt-4o",
        "modelConfigVersion": 3,
        "instructions": "You are a support agent.",
        **overrides,
    }
    stale = {
        **latest,
        "version": 1,
        "createdAt": 1,
        "instructions": "stale prompt",
    }
    return {"items": [stale, latest], "totalCount": 2}


MODEL_CONFIG = {
    "key": "OpenAI.gpt-4o",
    "id": "gpt-4o",
    "name": "GPT-4o",
    "provider": "OpenAI",
    "params": {"max_tokens": 100, "temperature": 1.0},
    "version": 3,
}


def fetched_run_responses(variation_page: dict[str, Any]) -> list[HttpResponse]:
    """Every response a run seeded from an AI Config variation needs, in order."""
    return [
        response(200, variation_page),
        response(200, MODEL_CONFIG),
        response(200, {"id": "dataset-id", "name": "golden"}),
        response(
            200,
            dataset_page([{"rowIndex": 0, "input": "hello", "variables": {}}], total=1),
        ),
        response(201, {"id": "evaluation-id", "name": "eval-key"}),
        response(
            201,
            {"id": "run-id", "evaluationId": "evaluation-id", "state": "PENDING"},
        ),
        response(
            200,
            {
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


def evaluation_post(transport: SequencedTransport) -> dict[str, Any]:
    posts = [
        request
        for request in transport.requests
        if request["method"] == "POST" and request["url"].endswith("/evaluations")
    ]
    assert len(posts) == 1
    body: dict[str, Any] = posts[0]["body"]
    return body


@pytest.mark.asyncio
async def test_run_seeds_generation_from_the_latest_ai_config_variation() -> None:
    transport = SequencedTransport(fetched_run_responses(config_variation_page()))
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)
    seen_configs: list[dict[str, Any]] = []

    async def handler(config: dict[str, Any], *args: object) -> dict[str, Any]:
        seen_configs.append(config)
        return {"output": "generated"}

    result = await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        ai_config=AIConfig(key="support-agent", variation="control"),
    )

    assert result.passed is True
    assert transport.requests[0]["url"] == (
        "https://app.launchdarkly.com/api/v2/projects/proj/ai-configs/"
        "support-agent/variations/control"
    )
    # The pinned model-config version is the one read.
    assert transport.requests[1]["url"].endswith(
        "/projects/proj/ai-configs/model-configs/OpenAI.gpt-4o?version=3"
    )
    body = evaluation_post(transport)
    assert body["generationProvider"] == "OpenAI"
    assert body["generationModel"] == "gpt-4o"
    # Model-config parameters sit under the variation's own, as flag delivery layers them.
    assert body["parameters"] == {"max_tokens": 100, "temperature": 0.7}
    assert body["messages"] == [
        {"role": "system", "content": "You are a support agent."}
    ]
    assert seen_configs[0]["provider"] == {"name": "OpenAI"}
    assert seen_configs[0]["instructions"] == "You are a support agent."


@pytest.mark.asyncio
async def test_explicit_generation_overrides_the_fetched_variation() -> None:
    transport = SequencedTransport(fetched_run_responses(config_variation_page()))
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        ai_config=AIConfig(key="support-agent", variation="control"),
        generation={
            "model": "gpt-4o-mini",
            "parameters": {"temperature": 0.1},
            "messages": [{"role": "system", "content": "Candidate prompt"}],
        },
    )

    body = evaluation_post(transport)
    assert body["generationProvider"] == "OpenAI"
    assert body["generationModel"] == "gpt-4o-mini"
    # parameters merge key by key rather than replacing the fetched set.
    assert body["parameters"] == {"max_tokens": 100, "temperature": 0.1}
    # messages replace the fetched instructions instead of clashing with them.
    assert body["messages"] == [{"role": "system", "content": "Candidate prompt"}]


@pytest.mark.asyncio
async def test_variation_tools_without_implementations_fail_before_mutating_requests() -> (
    None
):
    transport = SequencedTransport(
        [
            response(
                200,
                config_variation_page(tools=[{"key": "lookup_order", "version": 4}]),
            ),
            response(200, MODEL_CONFIG),
        ]
    )
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    with pytest.raises(EvaluationsError, match="'lookup_order'"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            ai_config=AIConfig(key="support-agent", variation="control"),
        )

    assert [request["method"] for request in transport.requests] == ["GET", "GET"]


@pytest.mark.asyncio
async def test_variation_judges_become_the_default_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = SequencedTransport(
        [
            response(
                200,
                config_variation_page(
                    judgeConfiguration={
                        "judges": [
                            {"judgeConfigKey": "security-judge", "samplingRate": 1.0}
                        ]
                    }
                ),
            ),
            response(200, MODEL_CONFIG),
        ]
    )

    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        raise RuntimeError("not found")

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    # Resolving the attached judge is what fails, so it was picked up as a criterion.
    with pytest.raises(EvaluationsError, match="'security-judge'"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=handler,
            ai_config=AIConfig(key="support-agent", variation="control"),
        )

    assert [request["method"] for request in transport.requests] == ["GET", "GET"]


@pytest.mark.asyncio
async def test_unknown_variation_fails_before_any_records_are_created() -> None:
    transport = SequencedTransport(
        [response(404, {"code": "not_found", "message": "not found"})]
    )
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    with pytest.raises(EvaluationsError, match="was not found"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            ai_config=AIConfig(key="support-agent", variation="missing"),
        )

    assert [request["method"] for request in transport.requests] == ["GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "Pass generation"),
        (
            {"ai_config": AIConfig(key=" ", variation="control")},
            "ai_config.key must not be blank",
        ),
        (
            {"ai_config": AIConfig(key="support-agent", variation="")},
            "ai_config.variation must not be blank",
        ),
    ],
)
async def test_config_source_is_validated_before_network_io(
    source: dict[str, AIConfig], message: str
) -> None:
    transport = SequencedTransport([])
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    with pytest.raises(EvaluationsError, match=message):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            **source,  # type: ignore[arg-type]
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_tool_version_drift_from_the_variation_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="launchdarkly_ai_server.evaluations.module")
    responses = fetched_run_responses(
        config_variation_page(tools=[{"key": "lookup_order", "version": 4}])
    )
    responses.insert(
        0,
        response(
            200,
            {"key": "lookup_order", "version": 7, "schema": {"type": "object"}},
        ),
    )
    transport = SequencedTransport(responses)
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        ai_config=AIConfig(key="support-agent", variation="control"),
        tools=[await evals.tools.get("lookup_order", implementation=lookup_order)],
    )

    assert "pins tool 'lookup_order' at version 4" in caplog.text
    assert "the run uses version 7" in caplog.text
    assert evaluation_post(transport)["tools"] == [
        {"key": "lookup_order", "version": 7, "source": "library"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_config_key", [42, ["OpenAI.gpt-4o"], {"key": "x"}])
async def test_non_string_model_config_key_fails_loudly(
    model_config_key: object,
) -> None:
    transport = SequencedTransport(
        [response(200, config_variation_page(modelConfigKey=model_config_key))]
    )
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    with pytest.raises(EvaluationsError, match="non-string modelConfigKey"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            ai_config=AIConfig(key="support-agent", variation="control"),
        )

    assert [request["method"] for request in transport.requests] == ["GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_config_key", [None, ""])
async def test_variation_without_a_model_config_needs_an_explicit_provider(
    model_config_key: str | None,
) -> None:
    transport = SequencedTransport(
        [response(200, config_variation_page(modelConfigKey=model_config_key))]
    )
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)

    # No model config is linked, so none is fetched and no provider is known.
    with pytest.raises(EvaluationsError, match=r"generation\.provider is required"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            ai_config=AIConfig(key="support-agent", variation="control"),
        )

    assert [request["method"] for request in transport.requests] == ["GET"]


def test_ai_config_variation_from_api_layers_the_model_config() -> None:
    from launchdarkly_ai_server.evaluations.types import AIConfigVariation

    latest = config_variation_page(
        tools=[{"key": "lookup_order", "version": 4}],
        judgeConfiguration={
            "judges": [{"judgeConfigKey": "security-judge", "samplingRate": 1.0}]
        },
    )["items"][1]

    linked = AIConfigVariation.from_api(latest, MODEL_CONFIG)
    assert linked.generation == {
        "provider": "OpenAI",
        "model": "gpt-4o",
        "parameters": {"max_tokens": 100, "temperature": 0.7},
        "instructions": "You are a support agent.",
    }
    assert linked.tool_versions == {"lookup_order": 4}
    assert linked.judge_keys == ["security-judge"]

    unlinked = AIConfigVariation.from_api(latest)
    assert "provider" not in unlinked.generation
    assert unlinked.generation["parameters"] == {"temperature": 0.7}


@pytest.mark.asyncio
async def test_tools_get_pins_the_version_it_reads() -> None:
    transport = SequencedTransport(
        [
            response(
                200,
                {
                    "key": "lookup_order",
                    "version": 7,
                    "description": "Look up an order",
                    "schema": ORDER_SCHEMA,
                },
            )
        ]
    )
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    tool = await evals.tools.get("lookup_order", implementation=lookup_order)

    assert tool.source == "library"
    assert tool.version == 7
    assert tool.description == "Look up an order"
    assert tool.schema == ORDER_SCHEMA
    assert tool.implementation is lookup_order
    assert recorded_paths(transport) == [("GET", "projects/proj/ai-tools/lookup_order")]


def test_a_constructed_tool_is_always_inline() -> None:
    """``source`` and ``version`` are not constructor arguments."""
    tool = EvalTool(
        key="lookup_order", implementation=lookup_order, schema=ORDER_SCHEMA
    )

    assert tool.source == "inline"
    assert tool.version is None
    with pytest.raises(TypeError):
        EvalTool(  # type: ignore[call-arg]
            key="lookup_order",
            implementation=lookup_order,
            schema=ORDER_SCHEMA,
            source="library",
        )


def test_an_inline_tool_requires_a_schema() -> None:
    with pytest.raises(TypeError, match="schema"):
        EvalTool(key="lookup_order", implementation=lookup_order)  # type: ignore[call-arg]

    tool = EvalTool(key="lookup_order", implementation=lookup_order, schema={})
    assert tool.schema == {}


def test_a_tool_cannot_be_reassigned_to_a_library_tool() -> None:
    tool = EvalTool(
        key="lookup_order", implementation=lookup_order, schema=ORDER_SCHEMA
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.source = "library"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.version = 99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.schema = {}  # type: ignore[misc]


@pytest.mark.asyncio
async def test_a_library_tool_without_a_project_is_rejected() -> None:
    forged = EvalTool._library(
        "lookup_order",
        lookup_order,
        version=1,
        schema={},
        description="",
        project_key="proj",
    )
    object.__setattr__(forged, "project_key", None)
    transport = SequencedTransport([])
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    with pytest.raises(EvaluationsError, match="has no project"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            tools=[forged],
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )

    assert transport.requests == []


@pytest.mark.asyncio
async def test_tools_get_reads_in_a_worker_thread() -> None:
    main_thread = threading.get_ident()
    threads: list[int] = []

    class RecordingTransport(SequencedTransport):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            threads.append(threading.get_ident())
            return super().__call__(*args, **kwargs)

    transport = RecordingTransport(
        [response(200, {"key": "lookup_order", "version": 7, "schema": {}})]
    )
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    await evals.tools.get("lookup_order", implementation=lookup_order)

    assert threads
    assert main_thread not in threads


@pytest.mark.asyncio
async def test_run_reads_no_tool_from_the_api() -> None:
    """``run`` resolves nothing: a library tool was already read by ``tools.get``."""
    transport = SequencedTransport(
        [
            response(200, {"key": "lookup_order", "version": 7}),
            *hosted_dataset_responses(),
        ]
    )
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )
    library_tool = await evals.tools.get("lookup_order", implementation=lookup_order)
    requests_before_run = len(transport.requests)

    await evals.run(
        key="eval-key",
        dataset="golden",
        handler=successful_handler,
        tools=[
            library_tool,
            EvalTool(
                key="refund_order", implementation=refund_order, schema=ORDER_SCHEMA
            ),
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
    )

    run_paths = [path for _, path in recorded_paths(transport)[requests_before_run:]]
    assert not any("ai-tools" in path for path in run_paths), run_paths


@pytest.mark.asyncio
async def test_an_empty_tools_list_runs_a_variation_with_no_tools() -> None:
    """A caller who passes tools= replaces the variation's list."""
    transport = SequencedTransport(
        fetched_run_responses(
            config_variation_page(tools=[{"key": "lookup_order", "version": 4}])
        )
    )
    evals = init_evaluations(
        project_key="proj", api_key="token", sdk_key="sdk-key", transport=transport
    )

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "generated"}

    await evals.run(
        key="eval-key",
        dataset="golden",
        handler=handler,
        ai_config=AIConfig(key="support-agent", variation="control"),
        tools=[],
    )

    assert "tools" not in evaluation_post(transport)


@pytest.mark.asyncio
async def test_a_tool_from_another_project_is_rejected() -> None:
    transport = SequencedTransport(
        [response(200, {"key": "lookup_order", "version": 4, "schema": {}})]
    )
    other = init_evaluations(
        project_key="other-proj",
        api_key="token",
        sdk_key="sdk-key",
        transport=transport,
    )
    foreign_tool = await other.tools.get("lookup_order", implementation=lookup_order)
    evals = init_evaluations(
        project_key="proj",
        api_key="token",
        sdk_key="sdk-key",
        transport=SequencedTransport([]),
    )

    with pytest.raises(EvaluationsError, match="cannot run in project 'proj'"):
        await evals.run(
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            tools=[foreign_tool],
            generation={"provider": "OpenAI", "model": "gpt-4o"},
        )


INLINE_EVALUATION = response(
    201, {"id": "evaluation-id", "name": "inline-eval", "version": 1}
)


INLINE_RUN = response(
    201,
    {
        "id": "run-id",
        "evaluationId": "evaluation-id",
        "source": "api",
        "state": "PENDING",
    },
)


def inline_summary(total: int) -> HttpResponse:
    return response(
        200,
        {
            "statusCounts": {
                "total": total,
                "passed": total,
                "failed": 0,
                "error": 0,
                "pending": 0,
            }
        },
    )


async def echo_handler(
    config: dict[str, Any],
    user_input: str | None,
    tool_handlers: dict[str, Callable[..., Any]],
    variables: dict[str, Any],
) -> dict[str, Any]:
    return {"output": f"generated: {user_input}"}


INLINE_GENERATION: Any = {"provider": "OpenAI", "model": "gpt-4o"}


def is_upload(request: dict[str, Any]) -> bool:
    return bool(
        request["method"] == "POST" and request["url"].endswith("/dataset-rows")
    )


@pytest.mark.asyncio
async def test_inline_dataset_uploads_rows_before_any_event_and_omits_dataset_id(
    stub_sdk_client: MagicMock,
) -> None:
    calls: list[str] = []
    transport = SequencedTransport(
        [INLINE_EVALUATION, INLINE_RUN, response(200, {}), inline_summary(2)]
    )

    def logging_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse:
        calls.append(f"{method} {url.rsplit('/', 1)[-1]}")
        return transport(method, url, headers, body, timeout)

    stub_sdk_client.track.side_effect = lambda *args: calls.append("track")
    received_inputs: list[str | None] = []

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        received_inputs.append(user_input)
        return {"output": f"generated: {user_input}"}

    result = await init_evaluations(
        project_key="proj", api_key="token", transport=logging_transport
    ).run(
        key="inline-eval",
        rows=[
            {
                "input": "How do I reset my password?",
                "expectedOutput": "Use the link.",
            },
            DatasetRow(
                row_index=1,
                input="Where is order {{order_id}}?",
                variables={"order_id": "A-17"},
                metadata={"suite": "orders"},
            ),
        ],
        handler=handler,
        generation=INLINE_GENERATION,
    )

    assert result.passed is True
    assert not any("/datasets/" in request["url"] for request in transport.requests)
    assert transport.requests[1]["url"].endswith("/evaluations/evaluation-id/runs")
    assert transport.requests[1]["body"] == {"source": "api"}
    upload = transport.requests[2]
    assert upload["method"] == "POST"
    assert upload["url"].endswith(
        "/projects/proj/evaluations/evaluation-id/runs/run-id/dataset-rows"
    )
    # Rows are stored raw; the server renders them as it renders a hosted dataset.
    assert upload["body"] == {
        "rows": [
            {
                "rowIdx": 0,
                "input": "How do I reset my password?",
                "expectedOutput": "Use the link.",
                "variables": {},
                "metadata": None,
            },
            {
                "rowIdx": 1,
                "input": "Where is order {{order_id}}?",
                "expectedOutput": None,
                "variables": {"order_id": "A-17"},
                "metadata": {"suite": "orders"},
            },
        ]
    }
    assert sorted(received_inputs, key=str) == [
        "How do I reset my password?",
        "Where is order A-17?",
    ]
    # Rows land before anything is counted, or the placeholder row count of 1
    # would let the first event mark the run complete.
    assert calls.index("track") > calls.index("POST dataset-rows")
    events = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    assert sorted(event["rowIndex"] for event in events) == [0, 1]
    for event in events:
        assert "datasetId" not in event
        assert "datasetKey" not in event


@pytest.mark.asyncio
async def test_inline_dataset_uploads_in_batches_of_500() -> None:
    transport = SequencedTransport(
        [
            INLINE_EVALUATION,
            INLINE_RUN,
            response(200, {}),
            response(200, {}),
            response(200, {}),
            inline_summary(1001),
        ]
    )

    await init_evaluations(
        project_key="proj", api_key="token", transport=transport
    ).run(
        key="inline-eval",
        rows=[{"rowIdx": index, "input": f"row {index}"} for index in range(1001)],
        handler=echo_handler,
        generation=INLINE_GENERATION,
    )

    uploads = [
        request["body"]["rows"] for request in transport.requests if is_upload(request)
    ]
    assert [len(batch) for batch in uploads] == [500, 500, 1]
    assert [row["rowIdx"] for batch in uploads for row in batch] == list(range(1001))


@pytest.mark.asyncio
async def test_inline_dataset_criterion_events_omit_dataset_id(
    stub_sdk_client: MagicMock,
) -> None:
    transport = SequencedTransport(
        [INLINE_EVALUATION, INLINE_RUN, response(200, {}), inline_summary(1)]
    )

    await init_evaluations(
        project_key="proj", api_key="token", transport=transport
    ).run(
        key="inline-eval",
        rows=[{"input": "hello"}],
        handler=echo_handler,
        generation=INLINE_GENERATION,
        criteria=[Scorer(name="non-empty", fn=lambda row, output: bool(output))],
    )

    criterion_events = [
        call.args[2]
        for call in stub_sdk_client.track.call_args_list
        if call.args[0] == "$ld:ai:offline-evals:criterion"
    ]
    assert len(criterion_events) == 1
    assert criterion_events[0]["rowIndex"] == 0
    assert criterion_events[0]["criterionType"] == "non-empty"
    assert "datasetId" not in criterion_events[0]
    assert "datasetKey" not in criterion_events[0]


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        ([], "Inline dataset is empty"),
        ([DatasetRow(row_index=3, input="hi")], "has row_index 3"),
        (
            [{"input": "hi", "expected_output": "x"}],
            "unknown fields: 'expected_output'",
        ),
        ([{"rowIdx": 1, "input": "hi"}], "has rowIdx 1"),
        ([{"rowIdx": True, "input": "hi"}], "has rowIdx True"),
        (["just a string"], "must be a DatasetRow or a mapping"),
        ([{"input": 7}], "input must be a string"),
        ([{"variables": ["a"]}], "variables must be a mapping"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_inline_rows_fail_before_any_request(
    dataset: list[Any], message: str
) -> None:
    evals = init_evaluations(
        project_key="proj", api_key="token", transport=failing_transport
    )

    with pytest.raises(EvaluationsError, match=message):
        await evals.run(
            key="inline-eval",
            rows=dataset,
            handler=echo_handler,
            generation=INLINE_GENERATION,
        )


@pytest.mark.asyncio
async def test_inline_upload_is_retried_after_a_server_error() -> None:
    transport = SequencedTransport(
        [
            INLINE_EVALUATION,
            INLINE_RUN,
            response(503, {"message": "unavailable"}),
            response(200, {}),
            inline_summary(1),
        ]
    )
    evals = init_evaluations(project_key="proj", api_key="token", transport=transport)
    evals.api._sleep = lambda _: None

    result = await evals.run(
        key="inline-eval",
        rows=[{"input": "hello"}],
        handler=echo_handler,
        generation=INLINE_GENERATION,
    )

    assert result.passed is True
    assert sum(is_upload(request) for request in transport.requests) == 2


@pytest.mark.asyncio
async def test_failed_inline_upload_stops_the_run_before_generation(
    stub_sdk_client: MagicMock,
) -> None:
    transport = SequencedTransport(
        [INLINE_EVALUATION, INLINE_RUN, response(400, {"message": "bad rows"})]
    )
    handler = AsyncMock()

    with pytest.raises(
        EvaluationsError, match=r"rows 0-0 of 1 to evaluation run run-id"
    ):
        await init_evaluations(
            project_key="proj", api_key="token", transport=transport
        ).run(
            key="inline-eval",
            rows=[{"input": "hello"}],
            handler=handler,
            generation=INLINE_GENERATION,
        )

    handler.assert_not_awaited()
    stub_sdk_client.track.assert_not_called()


@pytest.mark.asyncio
async def test_hosted_dataset_event_identity_is_unchanged(
    stub_sdk_client: MagicMock,
) -> None:
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "key": "golden"}),
            response(200, dataset_page([{"rowIndex": 2, "input": "hello"}], total=1)),
            INLINE_EVALUATION,
            INLINE_RUN,
            inline_summary(1),
        ]
    )

    await init_evaluations(
        project_key="proj", api_key="token", transport=transport
    ).run(
        key="inline-eval",
        dataset="golden",
        handler=echo_handler,
        generation=INLINE_GENERATION,
    )

    assert transport.requests[3]["body"] == {"source": "api", "datasetId": "dataset-id"}
    assert not any(is_upload(request) for request in transport.requests)
    event = stub_sdk_client.track.call_args.args[2]
    identity = {
        "projectKey": "proj",
        "evaluationId": "evaluation-id",
        "evaluationRunId": "run-id",
        "runId": "run-id",
        "datasetId": "dataset-id",
        "rowIndex": 2,
    }
    assert (
        event["eventId"]
        == hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert event["datasetId"] == "dataset-id"
    assert event["datasetKey"] == "golden"


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ({}, "Pass dataset, a LaunchDarkly dataset key, or rows"),
        ({"dataset": "golden", "rows": [{"input": "hi"}]}, "mutually exclusive"),
        ({"dataset": "golden", "rows": []}, "mutually exclusive"),
        ({"dataset": [{"input": "hi"}]}, "pass inline rows with rows="),
        ({"rows": "golden"}, "pass a LaunchDarkly dataset key with dataset="),
        ({"rows": {"input": "hi"}}, "rows must be a sequence of rows"),
    ],
)
@pytest.mark.asyncio
async def test_exactly_one_dataset_source_is_required_before_any_request(
    sources: dict[str, Any], message: str
) -> None:
    evals = init_evaluations(
        project_key="proj", api_key="token", transport=failing_transport
    )

    with pytest.raises(EvaluationsError, match=message):
        await evals.run(
            key="inline-eval",
            handler=echo_handler,
            generation=INLINE_GENERATION,
            **sources,
        )
