from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from launchdarkly_ai_server.evaluations import (
    EvaluationsError,
    HttpResponse,
    init_evaluations,
)


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
                    "id": "22222222-2222-2222-2222-222222222222",
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "source": "api",
                    "state": "COMPLETE",
                    "createdAt": 1,
                },
            ),
            response(
                200,
                {
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "evaluationRunId": "22222222-2222-2222-2222-222222222222",
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
    assert event["generationOutput"] == "generated: Order A19"
    assert event["usage"] == {"input_tokens": 10, "output_tokens": 4}
    assert len(event["eventId"]) == len(event["contentHash"]) == 64
    assert {"input", "expected_output", "metadata", "variables"}.isdisjoint(event)
    client.flush.assert_awaited_once_with()
    init_client.assert_awaited_once_with({"sdkKey": "sdk-key"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flag_value", "expected_poll"),
    [
        pytest.param(True, True, id="enabled"),
        pytest.param(False, False, id="disabled-default"),
        pytest.param("true", False, id="malformed"),
        pytest.param(
            RuntimeError("delivery unavailable"), False, id="evaluation-error"
        ),
    ],
)
async def test_batch_ingest_flag_controls_generation_result_polling(
    monkeypatch: pytest.MonkeyPatch,
    flag_value: object,
    expected_poll: bool,
) -> None:
    responses = [
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
    ]
    if expected_poll:
        responses.extend(
            [
                response(
                    200,
                    {
                        "id": "run-id",
                        "evaluationId": "evaluation-id",
                        "state": "COMPLETE",
                    },
                ),
                response(
                    200,
                    {
                        "statusCounts": {
                            "total": 1,
                            "passed": 1,
                            "pending": 0,
                        }
                    },
                ),
            ]
        )
    else:
        responses.append(
            response(
                200,
                {
                    "total": 1,
                    "passed": 0,
                    "failed": 0,
                    "error": 0,
                    "pending": 1,
                },
            )
        )
    transport = SequencedTransport(responses)
    client = MagicMock()
    if isinstance(flag_value, Exception):
        client.variation = AsyncMock(side_effect=flag_value)
    else:
        client.variation = AsyncMock(return_value=flag_value)

    def flush_before_poll_or_summary() -> None:
        assert len(transport.requests) == 4
        assert transport.requests[-1]["url"].endswith("/evaluations/evaluation-id/runs")

    client.flush.side_effect = flush_before_poll_or_summary

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

    assert result.passed is expected_poll
    assert result.summary.pending_rows == (0 if expected_poll else 1)
    request_urls = [request["url"] for request in transport.requests]
    assert not any(url.endswith("/generation-results") for url in request_urls)
    client.track.assert_called_once()
    client.flush.assert_called_once_with()
    status_url = "/evaluations/evaluation-id/runs/run-id"
    assert any(url.endswith(status_url) for url in request_urls) is expected_poll
    assert request_urls[-1].endswith(f"{status_url}/summary")
    assert sum(url.endswith(f"{status_url}/summary") for url in request_urls) == 1


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
@pytest.mark.parametrize(
    ("failed_rows", "error_rows"),
    [
        pytest.param(1, 0, id="failed-row"),
        pytest.param(0, 1, id="error-row"),
    ],
)
async def test_complete_run_with_failed_or_error_rows_does_not_pass(
    monkeypatch: pytest.MonkeyPatch, failed_rows: int, error_rows: int
) -> None:
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
                    "id": "22222222-2222-2222-2222-222222222222",
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "source": "api",
                    "state": "COMPLETE",
                    "createdAt": 1,
                },
            ),
            response(
                200,
                {
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "evaluationRunId": "22222222-2222-2222-2222-222222222222",
                    "statusCounts": {
                        "total": 2,
                        "passed": 2 - failed_rows - error_rows,
                        "failed": failed_rows,
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
    assert client.track.call_count == 2
    events = [call.args[2] for call in client.track.call_args_list]
    assert {event["status"] for event in events} == {"COMPLETE", "ERROR"}
    error_event = next(event for event in events if event["status"] == "ERROR")
    assert error_event["rowIndex"] == 0
    assert "provider failed" in error_event["error"]["message"]
    assert "generationOutput" not in error_event
    assert {"input", "expected_output", "metadata", "variables"}.isdisjoint(error_event)
