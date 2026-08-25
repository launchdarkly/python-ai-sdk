from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Mapping

from ..lifecycle import init_client
from .api import (
    DEFAULT_BASE_URI,
    EvaluationsError,
    LDApiClient,
    Transport,
    urllib_transport,
)
from .flags import is_generation_result_batch_ingest_enabled
from .runner import EvalHandler, EvaluationsRunner, ToolImplementation, _segment
from .types import EvalRunResult, GenerationConfig

logger = logging.getLogger(__name__)

DEFAULT_UI_BASE_URI = "https://app.launchdarkly.com"


def _env(name: str) -> str | None:
    """Read an env var, treating blank/whitespace-only values as unset."""
    value = os.environ.get(name, "").strip()
    return value if value else None


class EvaluationsModule:
    """Entry point for running LaunchDarkly evaluations from customer code."""

    def __init__(
        self,
        api_client: LDApiClient,
        sdk_key: str | None = None,
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
        """SDK key used for observability traces; ``None`` disables tracing."""
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
        concurrency: int = 10,
        timeout: float = 300.0,
    ) -> EvalRunResult:
        """
        Create and run a generation-only evaluation in the caller's process.

        The returned pass/fail result is derived from LaunchDarkly's run summary.
        A CI script can exit with ``0 if result.passed else 1`` after awaiting
        this method.
        """
        self._validate_run_args(
            project_key=project_key,
            key=key,
            dataset=dataset,
            handler=handler,
            generation=generation,
            concurrency=concurrency,
            timeout=timeout,
        )
        run_tools = dict(tools or {})
        client = None
        batch_ingest_enabled = False
        if self._sdk_key:
            client = await init_client({"sdkKey": self._sdk_key})
            batch_ingest_enabled = await is_generation_result_batch_ingest_enabled(
                client, project_key
            )

        # Tool verification is deliberately first: a typo must not create records.
        resolved_tools = self._runner._resolve_tools(project_key, run_tools)
        dataset_ref = self._runner._fetch_dataset(project_key, dataset)
        rows = self._runner._get_dataset_rows(project_key, dataset)
        evaluation = self._runner._create_evaluation(
            project_key, key, generation, resolved_tools
        )
        evaluation_run = self._runner._create_evaluation_run(
            project_key, evaluation.id, len(rows), dataset_ref.id
        )
        config = self._runner._build_handler_config(generation, resolved_tools)
        results = await self._runner._run_rows(
            rows,
            handler,
            config,
            run_tools,
            concurrency,
        )
        if client is not None:
            self._runner._emit_generation_events(
                client,
                project_key=project_key,
                evaluation=evaluation,
                evaluation_run=evaluation_run,
                dataset=dataset_ref,
                results=results,
            )
            flush_result = client.flush()
            if inspect.isawaitable(flush_result):
                await flush_result
        if batch_ingest_enabled:
            await self._runner._poll_run(
                project_key, evaluation.id, evaluation_run.id, timeout
            )
        summary = self._runner._get_summary(
            project_key, evaluation.id, evaluation_run.id
        )
        url = (
            f"{self._ui_base_uri}/projects/{_segment(project_key)}/ai/evaluations/"
            f"{_segment(evaluation.id)}/runs/{_segment(evaluation_run.id)}"
        )
        return EvalRunResult(
            passed=(
                summary.failed_rows == 0
                and summary.error_rows == 0
                and summary.pending_rows == 0
            ),
            url=url,
            run_id=evaluation_run.id,
            summary=summary,
        )

    @staticmethod
    def _validate_run_args(
        *,
        project_key: str,
        key: str,
        dataset: str,
        handler: EvalHandler,
        generation: GenerationConfig,
        concurrency: int,
        timeout: float,
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
        if timeout <= 0:
            raise EvaluationsError("timeout must be greater than zero")


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
        logger.info(
            "No LaunchDarkly SDK key provided; evaluation runs will not emit traces."
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
