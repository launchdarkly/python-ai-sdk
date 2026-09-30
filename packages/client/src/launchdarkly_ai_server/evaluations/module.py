from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time
from collections.abc import Mapping
from typing import Any, cast

from ..lifecycle import get_client, init_client
from .api import (
    DEFAULT_BASE_URI,
    EvaluationsError,
    LDApiClient,
    Transport,
    urllib_transport,
)
from .criteria import Criterion, Judge
from .runner import (
    EvalHandler,
    EvaluationsRunner,
    ToolImplementation,
    _provides_for,
    _segment,
    _select_handler,
)
from .types import AIConfig, EvalRunResult, GenerationConfig, RunSummary

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


def _merge_generation(
    base: GenerationConfig, override: GenerationConfig | None
) -> GenerationConfig:
    """Layer a caller's generation settings over a fetched variation's.

    Keys the caller sets replace the fetched ones, except ``parameters``, which
    merge key by key so overriding ``temperature`` keeps a fetched
    ``max_tokens``. ``instructions`` and ``messages`` are one prompt slot:
    supplying either discards both fetched values, so a caller swapping an
    agent prompt for a message list does not trip the mutual-exclusion check.
    """
    merged: dict[str, Any] = dict(base)
    if not override:
        return cast(GenerationConfig, merged)
    if "instructions" in override or "messages" in override:
        merged.pop("instructions", None)
        merged.pop("messages", None)
    for field_name, value in override.items():
        if field_name == "parameters" and isinstance(value, Mapping):
            base_parameters = merged.get("parameters")
            merged["parameters"] = {
                **(base_parameters if isinstance(base_parameters, Mapping) else {}),
                **value,
            }
        else:
            merged[field_name] = value
    return cast(GenerationConfig, merged)


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
        handlers: list[EvalHandler],
        generation: GenerationConfig | None = None,
        ai_config: AIConfig | None = None,
        tools: Mapping[str, ToolImplementation] | None = None,
        criteria: list[Criterion] | None = None,
        concurrency: int = 10,
        poll_interval_seconds: float | None = None,
        poll_timeout_seconds: float | None = None,
    ) -> EvalRunResult:
        """
        Create and run an evaluation in the caller's process.

        run() generates each dataset row with a handler from ``handlers``.
        Every entry in ``criteria`` (a LaunchDarkly :class:`Judge` reference
        or a local :class:`Scorer` function) then runs against each
        generated row. run() emits one evaluation event per
        ``(row, criterion)`` result.

        Pass one handler in ``handlers`` when generation and every judge use
        the same provider. Pass more than one handler otherwise. Build each
        handler with ``create_handler()`` (or a provider package's
        ``create_*_handler()``) and give it a ``provides_for`` tag for its
        provider and mode. With more than one handler, run() picks the
        handler tagged for the provider a call needs. For a judge, or for
        generation when ``ai_config`` supplies a known mode, run() also
        matches on mode. run() raises an error before any record exists if no
        handler matches a call, or if more than one handler could match and
        the call has no mode to break the tie.

        The returned pass/fail result is derived from LaunchDarkly's run summary.
        A CI script can exit with ``0 if result.passed else 1`` after awaiting
        this method. Large datasets may need a longer ``poll_timeout_seconds``
        and a wider ``poll_interval_seconds``; both default to
        ``SUMMARY_POLL_TIMEOUT_SECONDS`` / ``SUMMARY_POLL_INTERVAL_SECONDS``.

        Pass ``ai_config`` to start from an existing AI Config
        variation instead of a hand-built ``generation``. Its model, provider,
        parameters, prompt and output format become the defaults, and anything
        set in ``generation`` overrides them field by field (``parameters``
        merge key by key). When ``tools`` is omitted the variation's tools are
        used, so each needs an implementation; pass ``tools`` to replace the
        set. When ``criteria`` is omitted the variation's attached judges run;
        pass ``criteria`` (even ``[]``) to replace them.
        """
        if poll_interval_seconds is None:
            poll_interval_seconds = SUMMARY_POLL_INTERVAL_SECONDS
        if poll_timeout_seconds is None:
            poll_timeout_seconds = SUMMARY_POLL_TIMEOUT_SECONDS
        self._validate_run_args(
            project_key=project_key,
            key=key,
            dataset=dataset,
            handlers=handlers,
            concurrency=concurrency,
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
        )
        self._validate_config_source(generation=generation, ai_config=ai_config)
        pinned_tool_versions: dict[str, int] = {}
        config_label = ""
        if ai_config is not None:
            config_label = f"{ai_config.key!r}/{ai_config.variation!r}"
            ai_config_variation = await asyncio.to_thread(
                self._runner._fetch_config_variation,
                project_key,
                ai_config.key,
                ai_config.variation,
            )
            generation = _merge_generation(ai_config_variation.generation, generation)
            if tools is None and ai_config_variation.tool_versions:
                raise EvaluationsError(
                    f"AI Config variation {config_label} uses tools "
                    "with no implementation: "
                    + ", ".join(
                        repr(name) for name in ai_config_variation.tool_versions
                    )
                    + ". Pass tools= with an implementation for each."
                )
            pinned_tool_versions = ai_config_variation.tool_versions
            if criteria is None:
                criteria = [
                    Judge(key=judge_key) for judge_key in ai_config_variation.judge_keys
                ]
        generation = self._validate_generation(generation)
        # Checkable as soon as generation is known, and always before any
        # network request the run still needs to make. A fetched AI Config
        # variation may carry a mode; a hand-built generation has none.
        generation_handler = _select_handler(
            generation["provider"], handlers, mode=generation.get("mode")
        )
        run_tools = dict(tools or {})
        run_criteria = list(criteria or [])
        self._validate_criteria(run_criteria)
        ld_judges = [
            criterion for criterion in run_criteria if isinstance(criterion, Judge)
        ]
        client = await self._resolve_client()

        # The management API client is synchronous; running it in a worker thread
        # keeps the caller's event loop free.
        # Tool/judge verification is deliberately first: a typo must not create records.
        resolved_tools = await asyncio.to_thread(
            self._runner._resolve_tools, project_key, run_tools
        )
        # The tool API serves only the latest version, so a variation pinned to
        # an older one is evaluated against the current schema.
        for tool_key, pinned_version in pinned_tool_versions.items():
            resolved_tool = resolved_tools.get(tool_key)
            if resolved_tool is not None and resolved_tool.version != pinned_version:
                logger.warning(
                    "AI Config variation %s pins tool %r at version %d; "
                    "evaluating against the latest version %d.",
                    config_label,
                    tool_key,
                    pinned_version,
                    resolved_tool.version,
                )
        resolved_judges = await self._runner._resolve_judges(
            project_key, ld_judges, handlers
        )
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
            run_criteria,
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
            generation_handler,
            config,
            run_tools,
            concurrency,
        )
        try:
            self._runner._emit_generation_events(
                client,
                project_key=project_key,
                evaluation=evaluation,
                evaluation_run=evaluation_run,
                dataset=dataset_ref,
                results=results,
            )
            if run_criteria:
                criterion_results = await self._runner._run_criteria_for_results(
                    results,
                    run_tools,
                    run_criteria,
                    resolved_judges,
                    concurrency,
                )
                self._runner._emit_evaluation_events(
                    client,
                    project_key=project_key,
                    evaluation=evaluation,
                    evaluation_run=evaluation_run,
                    dataset=dataset_ref,
                    results=criterion_results,
                )
        finally:
            # Generation results already queued on the SDK event buffer must
            # reach LaunchDarkly even when the criteria phase fails.
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
            # failed_rows counts rows whose criteria were scored and did not
            # meet their threshold, so a gate that ignores it exits 0 on a run
            # where every row failed its judge. It was omissible while runs were
            # generation-only -- a row either generated or errored, and nothing
            # produced a fail -- and stops being so the moment criteria exist.
            passed=(
                summary.error_rows == 0
                and summary.failed_rows == 0
                and summary.pending_rows == 0
            ),
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
    def _validate_criteria(criteria: list[Criterion]) -> None:
        """Reject duplicate criterion identities before any records are created.

        A judge key and a scorer name that collide would share a criterionType,
        and with it the deterministic event identity of their results. Case-
        insensitive, matching the API's own dedup: the worker's retry gate
        lowercases criterion types, so two criteria differing only by case
        would still collide there even though they look distinct here.
        """
        seen: set[str] = set()
        duplicates: list[str] = []
        for criterion in criteria:
            criterion_type = criterion.criterion_type
            normalized = criterion_type.lower()
            if normalized in seen and criterion_type not in duplicates:
                duplicates.append(criterion_type)
            seen.add(normalized)
        if duplicates:
            raise EvaluationsError(
                "Duplicate evaluation criteria: "
                + ", ".join(repr(name) for name in duplicates)
                + ". Judge keys and scorer names must be unique within a run "
                "(case-insensitive)."
            )

    @staticmethod
    def _validate_handlers(handlers: list[EvalHandler]) -> None:
        """Reject an empty or invalid ``handlers`` list before any request.

        Two entries tagged for the identical provider and mode could never be
        told apart later, so that check also runs here rather than at
        selection time.
        """
        if not handlers:
            raise EvaluationsError("handlers must not be empty")
        seen: dict[tuple[str, str], int] = {}
        for index, candidate in enumerate(handlers):
            if not callable(candidate):
                raise EvaluationsError(f"handlers[{index}] must be callable")
            tag = _provides_for(candidate)
            if tag is None:
                continue
            if tag in seen:
                raise EvaluationsError(
                    f"handlers[{seen[tag]}] and handlers[{index}] both declare "
                    f"provides_for {tag[0]!r} in {tag[1]!r} mode. Only one "
                    "handler may serve a given provider and mode."
                )
            seen[tag] = index

    @staticmethod
    def _validate_run_args(
        *,
        project_key: str,
        key: str,
        dataset: str,
        handlers: list[EvalHandler],
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
        EvaluationsModule._validate_handlers(handlers)
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

    @staticmethod
    def _validate_config_source(
        *,
        generation: GenerationConfig | None,
        ai_config: AIConfig | None,
    ) -> None:
        """Require a generation source before any request is made."""
        if ai_config is None:
            if generation is None:
                raise EvaluationsError(
                    "Pass generation, or ai_config to evaluate an existing AI "
                    "Config variation"
                )
            return
        for name, value in (
            ("ai_config.key", ai_config.key),
            ("ai_config.variation", ai_config.variation),
        ):
            if not value.strip():
                raise EvaluationsError(f"{name} must not be blank")

    @staticmethod
    def _validate_generation(generation: GenerationConfig | None) -> GenerationConfig:
        """Check the final generation settings, after any fetched variation is merged."""
        if generation is None:
            raise EvaluationsError(
                "Pass generation, or ai_config to evaluate an existing AI "
                "Config variation"
            )
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
        return generation


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
