from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time
from collections.abc import Mapping
from typing import Any

from ..lifecycle import get_client, init_client
from .api import (
    DEFAULT_BASE_URI,
    EvaluationsError,
    LDApiClient,
    Transport,
    urllib_transport,
)
from .judges import Judge, JudgeReference
from .runner import EvalHandler, EvaluationsRunner, ToolImplementation, _segment
from .types import EvalRunResult, GenerationConfig, RunSummary

logger = logging.getLogger(__name__)

DEFAULT_UI_BASE_URI = "https://app.launchdarkly.com"
SUMMARY_POLL_INTERVAL_SECONDS = 2.0
SUMMARY_POLL_TIMEOUT_SECONDS = 180.0


def _env(name: str) -> str | None:
    """Read an env var, treating blank/whitespace-only values as unset."""
    value = os.environ.get(name, "").strip()
    return value if value else None


def _initialized_client() -> Any | None:
    """Return the SDK singleton when one is initialized, otherwise ``None``."""
    try:
        return get_client()
    except RuntimeError:
        return None


def _can_emit_events(client: Any) -> bool:
    return callable(getattr(client, "track", None)) and callable(
        getattr(client, "flush", None)
    )


def _is_terminal_summary(summary: RunSummary) -> bool:
    accounted_rows = summary.passed_rows + summary.failed_rows + summary.error_rows
    return (
        summary.total_rows > 0
        and summary.pending_rows == 0
        and accounted_rows == summary.total_rows
    )


class EvaluationsModule:
    """Entry point for running LaunchDarkly evaluations from customer code."""

    def __init__(
        self,
        api_client: LDApiClient,
        sdk_key: str | None,
        ui_base_uri: str = DEFAULT_UI_BASE_URI,
    ) -> None:
        self._api = api_client
        self._sdk_key = sdk_key
        self._ui_base_uri = ui_base_uri.rstrip("/")
        self._runner = EvaluationsRunner(api_client)

    @property
    def api(self) -> LDApiClient:
        return self._api

    @property
    def sdk_key(self) -> str | None:
        """SDK key whose event transport carries generation results to LaunchDarkly."""
        return self._sdk_key

    @property
    def ui_base_uri(self) -> str:
        """LaunchDarkly application host used for evaluation-run links."""
        return self._ui_base_uri

    async def run(
        self,
        *,
        project_key: str,
        key: str,
        dataset: str,
        handler: EvalHandler,
        generation: GenerationConfig,
        tools: Mapping[str, ToolImplementation] | None = None,
        judges: list[JudgeReference] | None = None,
        concurrency: int = 10,
        poll_interval_seconds: float | None = None,
        poll_timeout_seconds: float | None = None,
    ) -> EvalRunResult:
        """
        Create and run a generation-only evaluation in the caller's process.

        The returned pass/fail result is derived from LaunchDarkly's run summary.
        A CI script can exit with ``0 if result.passed else 1`` after awaiting
        this method. Large datasets may need a longer ``poll_timeout_seconds``
        and a wider ``poll_interval_seconds``; both default to
        ``SUMMARY_POLL_TIMEOUT_SECONDS`` / ``SUMMARY_POLL_INTERVAL_SECONDS``.
        """
        if poll_interval_seconds is None:
            poll_interval_seconds = SUMMARY_POLL_INTERVAL_SECONDS
        if poll_timeout_seconds is None:
            poll_timeout_seconds = SUMMARY_POLL_TIMEOUT_SECONDS
        self._validate_run_args(
            project_key=project_key,
            key=key,
            dataset=dataset,
            handler=handler,
            generation=generation,
            concurrency=concurrency,
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
        )
        run_tools = dict(tools or {})
        run_judges = list(judges or [])
        ld_judges = [judge for judge in run_judges if isinstance(judge, Judge)]
        client = await self._resolve_client()

        # The management API client is synchronous; running it in a worker thread
        # keeps the caller's event loop free.
        # Tool/judge verification is deliberately first: a typo must not create records.
        resolved_tools = await asyncio.to_thread(
            self._runner._resolve_tools, project_key, run_tools
        )
        resolved_judges = await self._runner._resolve_judges(ld_judges)
        dataset_ref = await asyncio.to_thread(
            self._runner._fetch_dataset, project_key, dataset
        )
        rows = await asyncio.to_thread(
            self._runner._get_dataset_rows, project_key, dataset
        )
        evaluation = await asyncio.to_thread(
            self._runner._create_evaluation,
            project_key,
            key,
            generation,
            resolved_tools,
            run_judges,
        )
        evaluation_run = await asyncio.to_thread(
            self._runner._create_evaluation_run,
            project_key,
            evaluation.id,
            dataset_ref.id,
        )
        config = self._runner._build_handler_config(generation, resolved_tools)
        results = await self._runner._run_rows(
            rows,
            handler,
            config,
            run_tools,
            concurrency,
        )
        self._runner._emit_generation_events(
            client,
            project_key=project_key,
            evaluation=evaluation,
            evaluation_run=evaluation_run,
            dataset=dataset_ref,
            results=results,
        )
        if run_judges:
            judge_results = await self._runner._run_judges_for_results(
                results,
                handler,
                run_tools,
                run_judges,
                resolved_judges,
            )
            self._runner._emit_evaluation_events(
                client,
                project_key=project_key,
                evaluation=evaluation,
                evaluation_run=evaluation_run,
                dataset=dataset_ref,
                results=judge_results,
            )
        flush_result = client.flush()
        if inspect.isawaitable(flush_result):
            await flush_result
        summary = await self._poll_summary_until_terminal(
            project_key,
            evaluation.id,
            evaluation_run.id,
            poll_interval_seconds,
            poll_timeout_seconds,
        )
        url = (
            f"{self._ui_base_uri}/projects/{_segment(project_key)}/ai/evaluations/"
            f"{_segment(evaluation.id)}/runs/{_segment(evaluation_run.id)}"
        )
        return EvalRunResult(
            passed=(summary.error_rows == 0 and summary.pending_rows == 0),
            url=url,
            run_id=evaluation_run.id,
            summary=summary,
        )

    async def _poll_summary_until_terminal(
        self,
        project_key: str,
        evaluation_id: str,
        run_id: str,
        poll_interval_seconds: float,
        poll_timeout_seconds: float,
    ) -> RunSummary:
        deadline = time.monotonic() + poll_timeout_seconds
        last_summary = None
        while True:
            last_summary = await asyncio.to_thread(
                self._runner._get_summary, project_key, evaluation_id, run_id
            )
            if _is_terminal_summary(last_summary):
                return last_summary
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                accounted_rows = (
                    last_summary.passed_rows
                    + last_summary.failed_rows
                    + last_summary.error_rows
                )
                raise EvaluationsError(
                    "Timed out after "
                    f"{poll_timeout_seconds:g} seconds waiting for evaluation "
                    f"run {run_id} summary rows to be fully accounted "
                    f"(total_rows={last_summary.total_rows}, "
                    f"accounted_rows={accounted_rows}, "
                    f"pending_rows={last_summary.pending_rows})"
                )
            await asyncio.sleep(min(poll_interval_seconds, remaining))

    async def _resolve_client(self) -> Any:
        """
        Return the SDK client used for generation events.

        ``init_client`` is idempotent, so an application that already holds a
        client keeps it and the evaluations SDK key is not applied.
        """
        existing = _initialized_client()
        if existing is not None:
            if self._sdk_key:
                logger.warning(
                    "A LaunchDarkly client is already initialized; evaluation "
                    "events are sent with it and the evaluations SDK key is "
                    "ignored. Both must point at the project under evaluation."
                )
            return existing
        if not self._sdk_key:
            raise EvaluationsError(
                "No LaunchDarkly SDK key provided and no initialized "
                "LaunchDarkly client is available to deliver generation events."
            )
        return await init_client({"sdkKey": self._sdk_key})

    @staticmethod
    def _validate_run_args(
        *,
        project_key: str,
        key: str,
        dataset: str,
        handler: EvalHandler,
        generation: GenerationConfig,
        concurrency: int,
        poll_interval_seconds: float,
        poll_timeout_seconds: float,
    ) -> None:
        for name, value in (
            ("project_key", project_key),
            ("key", key),
            ("dataset", dataset),
        ):
            if not value.strip():
                raise EvaluationsError(f"{name} must not be blank")
        if not callable(handler):
            raise EvaluationsError("handler must be callable")
        provider = generation.get("provider")
        model = generation.get("model")
        if not isinstance(provider, str) or not provider.strip():
            raise EvaluationsError("generation.provider is required")
        if not isinstance(model, str) or not model.strip():
            raise EvaluationsError("generation.model is required")
        if "instructions" in generation and "messages" in generation:
            raise EvaluationsError(
                "generation.instructions and generation.messages are mutually exclusive"
            )
        if concurrency < 1:
            raise EvaluationsError("concurrency must be at least 1")
        for name, seconds in (
            ("poll_interval_seconds", poll_interval_seconds),
            ("poll_timeout_seconds", poll_timeout_seconds),
        ):
            # NaN comparisons are always false, so a NaN would poll forever.
            if math.isnan(seconds):
                raise EvaluationsError(f"{name} must be a number")
            if seconds < 0:
                raise EvaluationsError(f"{name} must not be negative")


def init_evaluations(
    api_token: str | None = None,
    sdk_key: str | None = None,
    base_uri: str | None = None,
    ui_base_uri: str | None = None,
    transport: Transport = urllib_transport,
) -> EvaluationsModule:
    """Resolve credentials and construct the evaluations module."""
    token = api_token or _env("LD_API_TOKEN")
    if not token:
        raise EvaluationsError(
            "No LaunchDarkly API access token provided. Set the LD_API_TOKEN "
            "environment variable or pass api_token to init_evaluations()."
        )

    resolved_sdk_key = sdk_key or _env("LD_SDK_KEY")
    if not resolved_sdk_key:
        byoc_client = _initialized_client()
        if byoc_client is None or not _can_emit_events(byoc_client):
            raise EvaluationsError(
                "No LaunchDarkly SDK key provided and no initialized "
                "LaunchDarkly client to emit events with. Generation results "
                "reach LaunchDarkly through the SDK event transport, so a run "
                "cannot complete without one: set the LD_SDK_KEY environment "
                "variable, pass sdk_key to init_evaluations(), or initialize a "
                "client first with init_client(client=...)."
            )

    api_client = LDApiClient(
        api_token=token,
        base_uri=base_uri or _env("LD_API_BASE_URI") or DEFAULT_BASE_URI,
        transport=transport,
    )
    return EvaluationsModule(
        api_client=api_client,
        sdk_key=resolved_sdk_key,
        ui_base_uri=ui_base_uri or _env("LD_UI_BASE_URI") or DEFAULT_UI_BASE_URI,
    )
