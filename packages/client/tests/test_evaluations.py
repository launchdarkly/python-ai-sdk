from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
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


@pytest.fixture(autouse=True)
def reset_sdk_singleton() -> Iterator[None]:
    lifecycle_module._reset_for_testing()
    yield
    lifecycle_module._reset_for_testing()


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
    assert evals.ui_base_uri == "https://app.launchdarkly.com"


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


def test_missing_sdk_key_raises_before_network_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.delenv("LD_SDK_KEY", raising=False)

    with pytest.raises(EvaluationsError, match="LD_SDK_KEY"):
        init_evaluations(transport=failing_transport)


def test_blank_sdk_key_env_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.setenv("LD_SDK_KEY", "   ")

    with pytest.raises(EvaluationsError, match="LD_SDK_KEY"):
        init_evaluations(transport=failing_transport)


def test_missing_sdk_key_is_allowed_with_a_byoc_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
    byoc_client = MagicMock()
    byoc_client.track = MagicMock()
    byoc_client.flush = AsyncMock()
    lifecycle_module._set_client_for_testing(byoc_client)

    evals = init_evaluations(transport=failing_transport)

    assert evals.sdk_key is None


def test_missing_sdk_key_raises_when_the_byoc_client_cannot_emit_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
    lifecycle_module._set_client_for_testing(object())

    with pytest.raises(EvaluationsError, match="LD_SDK_KEY"):
        init_evaluations(transport=failing_transport)


def test_base_uri_override_isolated_from_sdk_delivery_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.setenv("LD_SDK_KEY", "sdk-key")
    monkeypatch.setenv("LD_API_BASE_URI", "https://api.staging.example.com/")
    monkeypatch.setenv("LD_BASE_URI", "https://relay.example.com/")

    from_env = init_evaluations(transport=RecordingTransport())
    explicit = init_evaluations(
        base_uri="https://other.example.com", transport=RecordingTransport()
    )

    assert from_env.api.base_uri == "https://api.staging.example.com"
    assert explicit.api.base_uri == "https://other.example.com"


def test_ui_base_uri_precedence_and_api_base_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_API_TOKEN", "api-token")
    monkeypatch.setenv("LD_SDK_KEY", "sdk-key")
    monkeypatch.setenv("LD_API_BASE_URI", "https://api.staging.example.com")
    monkeypatch.setenv("LD_UI_BASE_URI", "https://ld-stg.launchdarkly.com/")

    from_env = init_evaluations(transport=RecordingTransport())
    explicit = init_evaluations(
        ui_base_uri="https://ui.example.com/", transport=RecordingTransport()
    )

    assert from_env.api.base_uri == "https://api.staging.example.com"
    assert from_env.ui_base_uri == "https://ld-stg.launchdarkly.com"
    assert explicit.ui_base_uri == "https://ui.example.com"


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


def test_server_error_retries_get_but_not_post() -> None:
    server_error = HttpResponse(status=503, body='{"message": "unavailable"}')
    get_transport = RecordingTransport(
        [server_error, HttpResponse(200, '{"ok": true}')]
    )
    client = LDApiClient(
        api_token="api-token",
        transport=get_transport,
        max_retries=2,
        sleep=lambda _: None,
        random_value=lambda: 0.0,
    )

    assert client.get("projects/proj/datasets") == {"ok": True}
    assert len(get_transport.requests) == 2

    post_transport = RecordingTransport([server_error])
    client = LDApiClient(
        api_token="api-token",
        transport=post_transport,
        max_retries=2,
        sleep=lambda _: None,
        random_value=lambda: 0.0,
    )

    with pytest.raises(LDApiError) as excinfo:
        client.post("projects/proj/evaluations", body={"name": "eval"})

    assert excinfo.value.status == 503
    assert len(post_transport.requests) == 1


def test_transport_failure_is_not_replayed_for_post() -> None:
    attempts: list[str] = []

    def timing_out_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse:
        attempts.append(method)
        raise TimeoutError("timed out")

    client = LDApiClient(
        api_token="api-token",
        transport=timing_out_transport,
        max_retries=2,
        sleep=lambda _: None,
        random_value=lambda: 0.0,
    )

    with pytest.raises(EvaluationsError):
        client.post("projects/proj/evaluations", body={"name": "eval"})

    assert attempts == ["POST"]


def test_rate_limited_post_is_retried() -> None:
    transport = RecordingTransport(
        [
            HttpResponse(status=429, body='{"message": "slow down"}'),
            HttpResponse(status=201, body='{"id": "eval-id"}'),
        ]
    )
    client = LDApiClient(
        api_token="api-token",
        transport=transport,
        max_retries=1,
        sleep=lambda _: None,
        random_value=lambda: 0.0,
    )

    assert client.post("projects/proj/evaluations", body={"name": "eval"}) == {
        "id": "eval-id"
    }
    assert len(transport.requests) == 2


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
