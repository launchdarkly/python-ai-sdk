from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_BASE_URI = "https://app.launchdarkly.com"


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
    """
    Minimal client for the LaunchDarkly public ``/api/v2`` surface used by the
    evaluations harness. Every request carries the API access token; the base
    URI is overridable for non-default instances.
    """

    def __init__(
        self,
        api_token: str,
        base_uri: str = DEFAULT_BASE_URI,
        transport: Transport = urllib_transport,
        timeout: float = 30.0,
    ) -> None:
        self.api_token = api_token
        self.base_uri = base_uri.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    def url_for(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_uri}/api/v2/{path.lstrip('/')}"
        if params:
            query = {k: str(v) for k, v in params.items() if v is not None}
            if query:
                url = f"{url}?{urllib.parse.urlencode(query)}"
        return url

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

        response = self._transport(
            method, self.url_for(path, params), headers, payload, self._timeout
        )
        if response.status < 200 or response.status >= 300:
            raise LDApiError(response.status, method, path, response.body)
        if not response.body:
            return None
        return json.loads(response.body)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, body: Any = None) -> Any:
        return self.request("POST", path, body=body)
