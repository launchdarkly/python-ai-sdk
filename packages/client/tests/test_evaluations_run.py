from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from launchdarkly_ai_server.evaluations import (
    EvaluationsError,
    HttpResponse,
    Judge,
    JudgeEvaluationResult,
    JudgeIdentity,
    JudgeReference,
    LaunchDarklyJudgeEvaluation,
    Scorer,
    ScorerResult,
    ScorerRow,
    init_evaluations,
)
from launchdarkly_ai_server.types import ProviderHandler
from launchdarkly_ai_server.utils import create_handler


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
async def test_run_calls_private_operations_in_order_and_returns_server_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
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
    assert evals.sdk_key is None

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
async def test_disabled_batch_ingest_flag_skips_generation_result_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                    "id": "run-id",
                    "evaluationId": "evaluation-id",
                    "state": "COMPLETE",
                    "verdict": "passed",
                },
            ),
            response(200, {"statusCounts": {"total": 1, "passed": 1}}),
        ]
    )
    client = MagicMock()
    client.variation = AsyncMock(return_value=False)

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

    assert result.passed is True
    assert not any(
        request["url"].endswith("/generation-results") for request in transport.requests
    )


@pytest.mark.asyncio
async def test_run_executes_scorer_with_generation_and_complete_row_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 8,
                            "input": "Order {{order_id}}",
                            "expectedOutput": "Found {{order_id}}",
                            "variables": {"order_id": "A19"},
                            "metadata": {"suite": "orders"},
                        }
                    ],
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
            response(202, {}),
            response(
                200,
                {
                    "id": "run-id",
                    "evaluationId": "evaluation-id",
                    "state": "COMPLETE",
                    "verdict": "passed",
                },
            ),
            response(200, {"statusCounts": {"total": 1, "passed": 1}}),
        ]
    )
    received: dict[str, Any] = {}

    async def handler(*args: object) -> dict[str, Any]:
        return {"output": "Found A19"}

    def score(row: ScorerRow, output: str | None) -> bool:
        received.update(
            row_index=row.row_index,
            input=row.input,
            expected_output=row.expected_output,
            variables=dict(row.variables),
            metadata=dict(row.metadata or {}),
            output=output,
        )
        return True

    result = await init_evaluations(api_token="token", transport=transport).run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        judges=[Scorer(name="exact-match", fn=score)],
    )

    assert received == {
        "row_index": 8,
        "input": "Order A19",
        "expected_output": "Found A19",
        "variables": {
            "order_id": "A19",
            "input": "Order A19",
            "expected_output": "Found A19",
        },
        "metadata": {"suite": "orders"},
        "output": "Found A19",
    }
    assert len(result.evaluation_results) == 1
    scorer_result = result.evaluation_results[0]
    assert isinstance(scorer_result, ScorerResult)
    assert scorer_result.score == 1.0


@pytest.mark.asyncio
async def test_run_resolves_and_executes_launchdarkly_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                dataset_page(
                    [
                        {
                            "rowIndex": 5,
                            "input": "Question",
                            "expectedOutput": "Expected",
                            "variables": {"account": "enterprise"},
                            "metadata": {"suite": "judge"},
                        }
                    ],
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
            response(202, {}),
            response(
                200,
                {
                    "id": "run-id",
                    "evaluationId": "evaluation-id",
                    "state": "COMPLETE",
                    "verdict": "passed",
                },
            ),
            response(200, {"statusCounts": {"total": 1, "passed": 1}}),
        ]
    )
    received: dict[str, Any] = {}

    async def generation_handler(*args: object) -> dict[str, Any]:
        return {"output": "Generated answer"}

    async def judge_handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        del config, tool_handlers, history
        received.update(user_input=user_input, variables=variables)
        return {"output": '{"score": 0.9, "reasoning": "good"}'}

    reference = Judge(key="security-judge")
    resolved = LaunchDarklyJudgeEvaluation(
        reference=reference,
        config={
            "provider": {"name": "OpenAI"},
            "model": {"name": "judge-model"},
            "instructions": "Judge the response",
        },
        identity=JudgeIdentity(
            key="security-judge",
            variation_key="variation-key",
            version=3,
            provider="OpenAI",
            model="judge-model",
            mode="messages",
        ),
        handler=create_handler(("OpenAI", "messages"), judge_handler),
        collapse_messages=False,
    )
    generation = create_handler(("OpenAI", "messages"), generation_handler)
    flag_client = MagicMock()
    flag_client.variation = AsyncMock(return_value=True)

    async def fake_init_client(options: dict[str, Any]) -> MagicMock:
        assert options == {"sdkKey": "sdk-key"}
        return flag_client

    async def fake_resolve(
        references: Sequence[JudgeReference],
        handlers: Sequence[ProviderHandler],
        *,
        sdk_key: str | None,
    ) -> list[LaunchDarklyJudgeEvaluation]:
        assert references == [reference]
        assert handlers == [generation]
        assert sdk_key == "sdk-key"
        return [resolved]

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.init_client", fake_init_client
    )
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.resolve_launchdarkly_judges",
        fake_resolve,
    )

    result = await init_evaluations(
        api_token="token", sdk_key="sdk-key", transport=transport
    ).run(
        project_key="proj",
        key="eval-key",
        dataset="golden",
        handler=generation,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        judges=[reference],
    )

    judge_result = result.evaluation_results[0]
    assert isinstance(judge_result, JudgeEvaluationResult)
    assert judge_result.score == 0.9
    assert received["user_input"] == "Generated answer"
    judge_variables = cast(Mapping[str, Any], received["variables"])
    assert judge_variables["row_index"] == 5
    assert judge_variables["input"] == "Question"
    assert judge_variables["expected_output"] == "Expected"
    assert judge_variables["account"] == "enterprise"
    assert judge_variables["metadata"] == {"suite": "judge"}
    assert judge_variables["response_to_evaluate"] == "Generated answer"


@pytest.mark.asyncio
async def test_run_rejects_untyped_judges_before_network_io() -> None:
    transport = SequencedTransport([])
    evals = init_evaluations(api_token="token", transport=transport)

    with pytest.raises(EvaluationsError, match="typed JudgeReference or Scorer"):
        await evals.run(
            project_key="proj",
            key="eval-key",
            dataset="golden",
            handler=successful_handler,
            generation={"provider": "OpenAI", "model": "gpt-4o"},
            judges=["accuracy"],  # type: ignore[list-item]
        )

    assert transport.requests == []


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
