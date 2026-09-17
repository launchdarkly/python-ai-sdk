from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

DEFAULT_BASE_URI = "https://app.launchdarkly.com"

# Only these methods are replayed after a 5xx or a transport failure: a POST that
# timed out may still have created a record server-side.
RETRY_SAFE_METHODS = frozenset({"GET", "HEAD"})


class EvaluationsError(Exception):
    """Base error for the evaluations harness."""


class LDApiError(EvaluationsError):
    """A non-2xx response from the LaunchDarkly API."""

    def __init__(self, status: int, method: str, path: str, body: str) -> None:
        super().__init__(
            f"LaunchDarkly API {method} {path} failed with {status}: {body}"
        )
        self.status = status
        self.method = method
        self.path = path
        self.body = body


@dataclass
class HttpResponse:
    status: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)


class Transport(Protocol):
    """Seam the API client sends requests through; replaced in tests."""

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse: ...


def urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                body=response.read().decode("utf-8"),
                headers={k.lower(): v for k, v in response.headers.items()},
            )
    except urllib.error.HTTPError as error:
        return HttpResponse(
            status=error.code,
            body=error.read().decode("utf-8"),
            headers={k.lower(): v for k, v in error.headers.items()},
        )


class LDApiClient:
    """Minimal retrying client for the LaunchDarkly public management API."""

    def __init__(
        self,
        api_token: str,
        base_uri: str = DEFAULT_BASE_URI,
        transport: Transport = urllib_transport,
        timeout: float = 30.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.api_token = api_token
        self.base_uri = base_uri.rstrip("/")
        self._transport = transport
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._sleep = sleep
        self._random_value = random_value

    def url_for(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_uri}/api/v2/{path.lstrip('/')}"
        if params:
            query = {k: str(v) for k, v in params.items() if v is not None}
            if query:
                url = f"{url}?{urllib.parse.urlencode(query)}"
        return url

    def _retry_delay(self, attempt: int, response: HttpResponse | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after") or response.headers.get(
                "Retry-After"
            )
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    try:
                        when: datetime = parsedate_to_datetime(retry_after)
                        now = datetime.now(UTC)
                        return max(0.0, (when - now).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
        exponential = float(min(30.0, 0.5 * (2**attempt)))
        jitter = float(self._random_value()) * min(1.0, exponential)
        return exponential + jitter

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        headers = {
            "Authorization": self.api_token,
            "Accept": "application/json",
            "User-Agent": "launchdarkly-ai-evaluations-python",
        }
        payload: bytes | None = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")

        response: HttpResponse | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._transport(
                    method, self.url_for(path, params), headers, payload, self._timeout
                )
            except (TimeoutError, urllib.error.URLError) as error:
                if (
                    method.upper() not in RETRY_SAFE_METHODS
                    or attempt >= self._max_retries
                ):
                    raise EvaluationsError(
                        f"LaunchDarkly API {method} {path} failed after retries: {error}"
                    ) from error
                self._sleep(self._retry_delay(attempt))
                continue

            # A 429 is rejected before the server acts on it, so it is safe to
            # replay for any method.
            retryable = response.status == 429 or (
                response.status >= 500 and method.upper() in RETRY_SAFE_METHODS
            )
            if retryable and attempt < self._max_retries:
                self._sleep(self._retry_delay(attempt, response))
                continue
            break

        if response is None:
            raise EvaluationsError(
                f"LaunchDarkly API {method} {path} returned no response"
            )
        if response.status < 200 or response.status >= 300:
            raise LDApiError(response.status, method, path, response.body)
        if not response.body:
            return None
        try:
            return json.loads(response.body)
        except json.JSONDecodeError as error:
            raise EvaluationsError(
                f"LaunchDarkly API {method} {path} returned invalid JSON"
            ) from error

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, body: Any = None) -> Any:
        return self.request("POST", path, body=body)
