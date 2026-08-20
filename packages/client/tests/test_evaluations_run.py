from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

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
async def test_run_calls_private_operations_in_order_and_returns_server_verdict() -> (
    None
):
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
                    "source": "client",
                    "state": "PENDING",
                    "createdAt": 1,
                },
            ),
            response(202, {}),
            response(
                200,
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "source": "client",
                    "state": "COMPLETE",
                    "verdict": "passed",
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
    evals = init_evaluations(api_token="token", transport=transport)

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
    assert result.summary.total_rows == 2

    assert [request["method"] for request in transport.requests] == [
        "GET",
        "GET",
        "GET",
        "GET",
        "POST",
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
        "source": "client",
        "rowCount": 2,
        "datasetId": "33333333-3333-3333-3333-333333333333",
    }

    ingested = transport.requests[6]["body"]["results"]
    assert [row["row_index"] for row in ingested] == [4, 9]
    assert ingested[0]["input"] == "Order A19"
    assert ingested[0]["expected_output"] == "Found A19"
    assert ingested[0]["variables"]["input"] == "Order A19"
    assert ingested[0]["variables"]["expected_output"] == "Found A19"


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
async def test_handler_error_is_ingested_and_other_rows_continue() -> None:
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
                    "source": "client",
                    "state": "PENDING",
                    "createdAt": 1,
                },
            ),
            response(202, {}),
            response(
                200,
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "evaluationId": "11111111-1111-1111-1111-111111111111",
                    "evaluationVersion": 1,
                    "source": "client",
                    "state": "COMPLETE",
                    "verdict": "failed",
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
                        "passed": 1,
                        "failed": 0,
                        "error": 1,
                        "pending": 0,
                    },
                    "createdAt": 1,
                },
            ),
        ]
    )
    evals = init_evaluations(api_token="token", transport=transport)

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
    rows = transport.requests[4]["body"]["results"]
    assert {row["status"] for row in rows} == {"COMPLETE", "ERROR"}
    error_row = next(row for row in rows if row["status"] == "ERROR")
    assert error_row["row_index"] == 0
    assert "provider failed" in error_row["error"]["message"]
