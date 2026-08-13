from __future__ import annotations

import json
from typing import Any

import pytest

from launchdarkly_ai_server.evaluations import (
    DEFAULT_BASE_URI,
    EvalRunResult,
    EvaluationsError,
    HttpResponse,
    LDApiClient,
    LDApiError,
    RunSummary,
    Usage,
    init_evaluations,
)


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


def test_init_resolves_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token-from-env")
    monkeypatch.setenv("LD_SDK_KEY", "sdk-key-from-env")

    evals = init_evaluations(transport=RecordingTransport())

    assert evals.api.api_token == "api-token-from-env"
    assert evals.sdk_key == "sdk-key-from-env"
    assert evals.api.base_uri == DEFAULT_BASE_URI


def test_init_prefers_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token-from-env")
    monkeypatch.setenv("LD_SDK_KEY", "sdk-key-from-env")

    evals = init_evaluations(
        api_token="explicit-token",
        sdk_key="explicit-sdk-key",
        transport=RecordingTransport(),
    )

    assert evals.api.api_token == "explicit-token"
    assert evals.sdk_key == "explicit-sdk-key"


def test_missing_api_token_raises_before_network_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LD_API_TOKEN", raising=False)
    monkeypatch.setenv("LD_SDK_KEY", "sdk-key")

    with pytest.raises(EvaluationsError, match="LD_API_TOKEN"):
        init_evaluations(transport=failing_transport)


def test_blank_api_token_env_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "   ")

    with pytest.raises(EvaluationsError):
        init_evaluations(transport=failing_transport)


def test_missing_sdk_key_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.delenv("LD_SDK_KEY", raising=False)

    evals = init_evaluations(transport=RecordingTransport())

    assert evals.sdk_key is None


def test_base_uri_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.setenv("LD_BASE_URI", "https://ld.internal.example.com/")

    from_env = init_evaluations(transport=RecordingTransport())
    explicit = init_evaluations(
        base_uri="https://other.example.com", transport=RecordingTransport()
    )

    assert from_env.api.base_uri == "https://ld.internal.example.com"
    assert explicit.api.base_uri == "https://other.example.com"


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
        {"total_rows": 500, "passed_rows": 498, "failed_rows": 1, "error_rows": 1}
    )
    result = EvalRunResult(
        passed=False,
        url="https://app.launchdarkly.com/run",
        run_id="run-1",
        summary=summary,
    )

    assert summary.total_rows == 500
    assert summary.error_rows == 1
    assert RunSummary.from_wire(None) == RunSummary()
    assert result.passed is False
    assert result.run_id == "run-1"
