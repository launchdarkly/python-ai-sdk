from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..lifecycle import get_client, init_client
from .api import (
    DEFAULT_BASE_URI,
    EvaluationsError,
    LDApiClient,
    Transport,
    segment,
    urllib_transport,
)
from .criteria import Criterion, Judge
from .runner import EvalHandler, EvaluationsRunner, _provides_for
from .tools import Tool, ToolsClient, tool_handlers, validate_tools
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
        project_key: str,
        sdk_key: str | None,
        ui_base_uri: str = DEFAULT_UI_BASE_URI,
    ) -> None:
        self._api = api_client
        self._project_key = project_key
        self._sdk_key = sdk_key
        self._ui_base_uri = ui_base_uri.rstrip("/")
        self._runner = EvaluationsRunner(api_client)
        self._tools = ToolsClient(api_client, project_key)

    @property
    def api(self) -> LDApiClient:
        return self._api

    @property
    def project_key(self) -> str:
        """Project that holds this module's evaluations, tools, and datasets."""
        return self._project_key

    @property
    def tools(self) -> ToolsClient:
        """Reader for tools in the LaunchDarkly tool library."""
        return self._tools

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
        key: str,
        dataset: str,
        handler: EvalHandler,
        generation: GenerationConfig | None = None,
        ai_config: AIConfig | None = None,
        tools: Sequence[Tool] | None = None,
        criteria: list[Criterion] | None = None,
        judge_handlers: list[EvalHandler] | None = None,
        concurrency: int = 10,
        poll_interval_seconds: float | None = None,
        poll_timeout_seconds: float | None = None,
    ) -> EvalRunResult:
        """
        Create and run an evaluation in the caller's process.

        Each dataset row is generated with ``handler``; every entry in
        ``criteria`` — LaunchDarkly :class:`Judge` references and local
        deterministic :class:`Scorer` functions — is then run against each
        generated row, and one evaluation event is emitted per
        ``(row, criterion)`` result.

        ``tools`` is a list of :class:`Tool`. Construct one to define a tool
        in code. Call ``evals.tools.get(key, implementation=...)`` to use a
        tool from the LaunchDarkly tool library, which reads the tool and pins
        its version at that point. One list may hold both kinds. ``run`` reads
        no tool from the API, and it checks the list before any network I/O.
        Handlers receive a ``{key: executable}`` map either way.

        A :class:`Judge` is an independent AI Config and may be served by a
        different provider or mode than ``generation``. ``handler`` runs a judge
        only when it provides for that judge's provider; pass handlers for any
        other providers your judges use in ``judge_handlers``. A judge no
        handler covers fails the run before any records are created.

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
            key=key,
            dataset=dataset,
            handler=handler,
            concurrency=concurrency,
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
        )
        self._validate_config_source(generation=generation, ai_config=ai_config)
        run_tools = list(tools or [])
        validate_tools(run_tools)
        run_tool_handlers = tool_handlers(run_tools)
        pinned_tool_versions: dict[str, int] = {}
        config_label = ""
        if ai_config is not None:
            config_label = f"{ai_config.key!r}/{ai_config.variation!r}"
            ai_config_variation = await asyncio.to_thread(
                self._runner._fetch_config_variation,
                self._project_key,
                ai_config.key,
                ai_config.variation,
            )
            generation = _merge_generation(ai_config_variation.generation, generation)
            supplied = {tool.key for tool in run_tools}
            missing = [
                name
                for name in ai_config_variation.tool_versions
                if name not in supplied
            ]
            if missing:
                raise EvaluationsError(
                    f"AI Config variation {config_label} uses tools "
                    "with no implementation: "
                    + ", ".join(repr(name) for name in missing)
                    + ". Pass tools= with a Tool for each."
                )
            pinned_tool_versions = ai_config_variation.tool_versions
            if criteria is None:
                criteria = [
                    Judge(key=judge_key) for judge_key in ai_config_variation.judge_keys
                ]
        generation = self._validate_generation(generation)
        run_criteria = list(criteria or [])
        run_judge_handlers = list(judge_handlers or [])
        self._validate_criteria(run_criteria)
        self._validate_judge_handlers(run_judge_handlers)
        ld_judges = [
            criterion for criterion in run_criteria if isinstance(criterion, Judge)
        ]
        client = await self._resolve_client()

        # The management API client is synchronous; running it in a worker thread
        # keeps the caller's event loop free.
        # Judge verification is first: a typo must not create records.
        # A variation pins a tool version. Compare it with the version the run
        # uses, which tools.get() already read.
        tools_by_key = {tool.key: tool for tool in run_tools}
        for tool_key, pinned_version in pinned_tool_versions.items():
            tool = tools_by_key.get(tool_key)
            if tool is None or tool.version is None or tool.version == pinned_version:
                continue
            logger.warning(
                "AI Config variation %s pins tool %r at version %d; "
                "the run uses version %d.",
                config_label,
                tool_key,
                pinned_version,
                tool.version,
            )
        resolved_judges = await self._runner._resolve_judges(
            self._project_key, ld_judges, handler, run_judge_handlers
        )
        dataset_ref = await asyncio.to_thread(
            self._runner._fetch_dataset, self._project_key, dataset
        )
        rows = await asyncio.to_thread(
            self._runner._get_dataset_rows, self._project_key, dataset
        )
        evaluation = await asyncio.to_thread(
            self._runner._create_evaluation,
            self._project_key,
            key,
            generation,
            run_tools,
            run_criteria,
        )
        evaluation_run = await asyncio.to_thread(
            self._runner._create_evaluation_run,
            self._project_key,
            evaluation.id,
            dataset_ref.id,
        )
        config = self._runner._build_handler_config(generation, run_tools)
        results = await self._runner._run_rows(
            rows,
            handler,
            config,
            run_tool_handlers,
            concurrency,
        )
        try:
            self._runner._emit_generation_events(
                client,
                project_key=self._project_key,
                evaluation=evaluation,
                evaluation_run=evaluation_run,
                dataset=dataset_ref,
                results=results,
            )
            if run_criteria:
                criterion_results = await self._runner._run_criteria_for_results(
                    results,
                    run_tool_handlers,
                    run_criteria,
                    resolved_judges,
                    concurrency,
                )
                self._runner._emit_evaluation_events(
                    client,
                    project_key=self._project_key,
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
            evaluation.id,
            evaluation_run.id,
            poll_interval_seconds,
            poll_timeout_seconds,
        )
        url = (
            f"{self._ui_base_uri}/projects/{segment(self._project_key)}/ai/evaluations/"
            f"{segment(evaluation.id)}/runs/{segment(evaluation_run.id)}"
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
        evaluation_id: str,
        run_id: str,
        poll_interval_seconds: float,
        poll_timeout_seconds: float,
    ) -> RunSummary:
        deadline = time.monotonic() + poll_timeout_seconds
        last_summary = None
        while True:
            last_summary = await asyncio.to_thread(
                self._runner._get_summary, self._project_key, evaluation_id, run_id
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
    def _validate_judge_handlers(judge_handlers: list[EvalHandler]) -> None:
        """Reject judge handlers that cannot be routed by provider and mode.

        A judge handler is only ever chosen by matching its ``provides_for``
        against the judge's resolved provider and mode. One without that
        metadata could never be selected, so it would silently fall through to
        the generation handler instead of running the judge it was passed for.
        """
        for index, candidate in enumerate(judge_handlers):
            if not callable(candidate):
                raise EvaluationsError(f"judge_handlers[{index}] must be callable")
            if _provides_for(candidate) is None:
                raise EvaluationsError(
                    f"judge_handlers[{index}] does not declare provides_for. "
                    "Build judge handlers with create_handler() (or a provider "
                    "package's create_*_handler()) so they can be matched to a "
                    "judge's provider and mode."
                )

    @staticmethod
    def _validate_run_args(
        *,
        key: str,
        dataset: str,
        handler: EvalHandler,
        concurrency: int,
        poll_interval_seconds: float,
        poll_timeout_seconds: float,
    ) -> None:
        for name, value in (
            ("key", key),
            ("dataset", dataset),
        ):
            if not value.strip():
                raise EvaluationsError(f"{name} must not be blank")
        if not callable(handler):
            raise EvaluationsError("handler must be callable")
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
    project_key: str | None = None,
    api_key: str | None = None,
    sdk_key: str | None = None,
    base_uri: str | None = None,
    ui_base_uri: str | None = None,
    transport: Transport = urllib_transport,
) -> EvaluationsModule:
    """Resolve credentials and construct the evaluations module.

    ``project_key`` names the project that holds the evaluations, the tools,
    and the datasets this module uses.
    """
    resolved_project_key = (project_key or "").strip() or _env("LD_PROJECT_KEY")
    if not resolved_project_key:
        raise EvaluationsError(
            "No LaunchDarkly project key provided. Set the LD_PROJECT_KEY "
            "environment variable or pass project_key to init_evaluations()."
        )

    token = api_key or _env("LD_API_TOKEN")
    if not token:
        raise EvaluationsError(
            "No LaunchDarkly API access token provided. Set the LD_API_TOKEN "
            "environment variable or pass api_key to init_evaluations()."
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
        api_key=token,
        base_uri=base_uri or _env("LD_API_BASE_URI") or DEFAULT_BASE_URI,
        transport=transport,
    )
    return EvaluationsModule(
        api_client=api_client,
        project_key=resolved_project_key,
        sdk_key=resolved_sdk_key,
        ui_base_uri=ui_base_uri or _env("LD_UI_BASE_URI") or DEFAULT_UI_BASE_URI,
    )
