from __future__ import annotations

import json
from typing import Any

import pytest

from launchdarkly_ai_server.evaluations.api import (
    DEFAULT_BASE_URI,
    HttpResponse,
    LDApiClient,
    LDApiError,
)
from launchdarkly_ai_server.evaluations.types import EvalRunResult, RunSummary, Usage


class RecordingTransport:
    """Mocked LD API — records requests and replays canned responses."""

    def __init__(self, responses: list[HttpResponse] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses = responses or [HttpResponse(status=200, body="{}")]

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": json.loads(body) if body else None,
                "timeout": timeout,
            }
        )
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[index]


def failing_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    raise AssertionError("no network I/O expected")


def test_requests_carry_token_auth_and_json_body() -> None:
    transport = RecordingTransport([HttpResponse(status=201, body='{"key": "run-1"}')])
    client = LDApiClient(api_token="api-token", transport=transport)

    result = client.post("projects/proj/evaluations", body={"key": "support-qa"})

    assert result == {"key": "run-1"}
    request = transport.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == f"{DEFAULT_BASE_URI}/api/v2/projects/proj/evaluations"
    assert request["headers"]["Authorization"] == "api-token"
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["body"] == {"key": "support-qa"}


def test_get_encodes_query_params_and_omits_none() -> None:
    transport = RecordingTransport([HttpResponse(status=200, body='{"items": []}')])
    client = LDApiClient(
        api_token="api-token", base_uri="https://ld.example.com", transport=transport
    )

    client.get("projects/proj/datasets/golden", params={"limit": 50, "offset": None})

    request = transport.requests[0]
    assert (
        request["url"]
        == "https://ld.example.com/api/v2/projects/proj/datasets/golden?limit=50"
    )
    assert "Content-Type" not in request["headers"]


def test_rate_limit_retries_and_honors_retry_after() -> None:
    transport = RecordingTransport(
        [
            HttpResponse(
                status=429,
                body='{"message": "slow down"}',
                headers={"retry-after": "2"},
            ),
            HttpResponse(status=200, body='{"items": []}'),
        ]
    )
    sleeps: list[float] = []
    client = LDApiClient(
        api_token="api-token",
        transport=transport,
        max_retries=1,
        sleep=sleeps.append,
        random_value=lambda: 0.0,
    )

    assert client.get("projects/proj/datasets") == {"items": []}
    assert len(transport.requests) == 2
    assert sleeps == [2.0]


def test_forbidden_response_is_not_retried() -> None:
    transport = RecordingTransport(
        [HttpResponse(status=403, body='{"message": "forbidden"}')]
    )
    client = LDApiClient(api_token="api-token", transport=transport, max_retries=3)

    with pytest.raises(LDApiError) as excinfo:
        client.get("projects/proj/evaluations")

    assert excinfo.value.status == 403
    assert len(transport.requests) == 1


def test_error_response_raises_ld_api_error() -> None:
    transport = RecordingTransport(
        [HttpResponse(status=404, body='{"message": "nope"}')]
    )
    client = LDApiClient(api_token="api-token", transport=transport)

    with pytest.raises(LDApiError) as excinfo:
        client.get("projects/proj/ai-tools/missing")

    assert excinfo.value.status == 404
    assert excinfo.value.path == "projects/proj/ai-tools/missing"


def test_empty_response_body_is_none() -> None:
    transport = RecordingTransport([HttpResponse(status=204, body="")])
    client = LDApiClient(api_token="api-token", transport=transport)

    assert client.post("projects/proj/evaluations/support-qa/runs") is None


def test_usage_matches_ingest_wire_shape() -> None:
    usage = Usage(input_tokens=812, output_tokens=96)

    assert usage.to_wire() == {"input_tokens": 812, "output_tokens": 96}
    assert Usage.from_wire({"input_tokens": 1, "output_tokens": 2}) == Usage(1, 2)
    assert Usage.from_wire({}) == Usage(0, 0)


def test_run_summary_and_result() -> None:
    summary = RunSummary.from_wire(
        {
            "total_rows": 500,
            "passed_rows": 497,
            "failed_rows": 1,
            "error_rows": 1,
            "pending_rows": 1,
        }
    )
    result = EvalRunResult(
        passed=False,
        url="https://app.launchdarkly.com/run",
        run_id="run-1",
        summary=summary,
    )

    assert summary.total_rows == 500
    assert summary.error_rows == 1
    assert summary.pending_rows == 1
    assert RunSummary.from_wire({"pending": 2}).pending_rows == 2
    assert RunSummary.from_wire(None) == RunSummary()
    assert result.passed is False
    assert result.run_id == "run-1"
