from __future__ import annotations

import logging
import os

from .api import (
    DEFAULT_BASE_URI,
    EvaluationsError,
    LDApiClient,
    Transport,
    urllib_transport,
)

logger = logging.getLogger(__name__)


def _env(name: str) -> str | None:
    """Read an env var, treating blank/whitespace-only values as unset."""
    value = os.environ.get(name, "").strip()
    return value if value else None


class EvaluationsModule:
    """
    Entry point for running LaunchDarkly evaluations from code. Holds the
    resolved credentials and the LaunchDarkly API client; ``run()`` arrives with
    the harness.
    """

    def __init__(self, api_client: LDApiClient, sdk_key: str | None = None) -> None:
        self._api = api_client
        self._sdk_key = sdk_key

    @property
    def api(self) -> LDApiClient:
        return self._api

    @property
    def sdk_key(self) -> str | None:
        """SDK key used for observability traces; ``None`` disables tracing."""
        return self._sdk_key


def init_evaluations(
    api_token: str | None = None,
    sdk_key: str | None = None,
    base_uri: str | None = None,
    transport: Transport = urllib_transport,
) -> EvaluationsModule:
    """
    Resolves credentials and builds the evaluations module.

    ``api_token`` (``LD_API_TOKEN``) authenticates every ``/api/v2`` call and is
    required — a missing token raises before any network I/O rather than
    surfacing as an opaque 401 mid-run. ``sdk_key`` (``LD_SDK_KEY``) is optional
    and only makes handler calls emit observability traces. Both credentials
    must point at the same project.
    """
    token = api_token or _env("LD_API_TOKEN")
    if not token:
        raise EvaluationsError(
            "No LaunchDarkly API access token provided. Set the LD_API_TOKEN "
            "environment variable or pass api_token to init_evaluations()."
        )

    resolved_sdk_key = sdk_key or _env("LD_SDK_KEY")
    if not resolved_sdk_key:
        logger.info(
            "No LaunchDarkly SDK key provided; evaluation runs will not emit traces."
        )

    api_client = LDApiClient(
        api_token=token,
        base_uri=base_uri or _env("LD_BASE_URI") or DEFAULT_BASE_URI,
        transport=transport,
    )
    return EvaluationsModule(api_client=api_client, sdk_key=resolved_sdk_key)
