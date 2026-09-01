from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from launchdarkly_ai_server.evaluations import (
    EvaluationsError,
    HttpResponse,
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


@pytest.mark.asyncio
async def test_complete_run_with_zero_failed_and_error_rows_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        "rowCount": 2,
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
    output_lines = capsys.readouterr().out.splitlines()
    assert len(output_lines) == 2
    assert output_lines[0] == (
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
async def test_generation_failed_rows_do_not_fail_the_result() -> None:
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
    assert result.passed is True


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
