from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import cast

from ..lifecycle import init_client
from ..types import ProviderHandler
from .api import (
    DEFAULT_BASE_URI,
    EvaluationsError,
    LDApiClient,
    Transport,
    urllib_transport,
)
from .flags import is_generation_result_batch_ingest_enabled
from .judges import (
    JudgeReference,
    LaunchDarklyJudgeEvaluation,
    resolve_launchdarkly_judges,
)
from .runner import (
    EvalHandler,
    EvaluationsRunner,
    OfflineEvaluation,
    ToolImplementation,
    _segment,
)
from .scorers import Scorer
from .types import EvalRunResult, GenerationConfig

logger = logging.getLogger(__name__)


def _env(name: str) -> str | None:
    """Read an env var, treating blank/whitespace-only values as unset."""
    value = os.environ.get(name, "").strip()
    return value if value else None


class EvaluationsModule:
    """Entry point for running LaunchDarkly evaluations from customer code."""

    def __init__(self, api_client: LDApiClient, sdk_key: str | None = None) -> None:
        self._api = api_client
        self._sdk_key = sdk_key
        self._runner = EvaluationsRunner(api_client)

    @property
    def api(self) -> LDApiClient:
        return self._api

    @property
    def sdk_key(self) -> str | None:
        """SDK key used for observability traces; ``None`` disables tracing."""
        return self._sdk_key

    async def run(
        self,
        *,
        project_key: str,
        key: str,
        dataset: str,
        handler: EvalHandler,
        generation: GenerationConfig,
        tools: Mapping[str, ToolImplementation] | None = None,
        judges: Sequence[JudgeReference | Scorer] | None = None,
        concurrency: int = 10,
        timeout: float = 300.0,
    ) -> EvalRunResult:
        """
        Create and run an evaluation in the caller's process.

        Typed judges and scorers run after each successful generation. The
        returned verdict is computed by LaunchDarkly. A CI script can exit
        with ``0 if result.passed else 1`` after awaiting this method.
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
        requested_evaluations = list(judges or [])
        self._validate_evaluations(requested_evaluations)
        batch_ingest_enabled = True
        if self._sdk_key:
            client = await init_client({"sdkKey": self._sdk_key})
            batch_ingest_enabled = await is_generation_result_batch_ingest_enabled(
                client, project_key
            )
        evaluation_methods = await self._resolve_evaluations(
            requested_evaluations, handler
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
        results, evaluation_results = await self._runner._run_rows(
            rows,
            handler,
            config,
            run_tools,
            concurrency,
            evaluation_methods,
        )
        self._runner._ingest_results(
            project_key,
            evaluation.id,
            evaluation_run.id,
            results,
            batch_ingest_enabled=batch_ingest_enabled,
        )
        completed = await self._runner._poll_run(
            project_key, evaluation.id, evaluation_run.id, timeout
        )
        summary = self._runner._get_summary(
            project_key, evaluation.id, evaluation_run.id
        )
        url = (
            f"{self._api.base_uri}/projects/{_segment(project_key)}/ai/evaluations/"
            f"{_segment(evaluation.id)}/runs/{_segment(evaluation_run.id)}"
        )
        return EvalRunResult(
            passed=completed.verdict == "passed",
            url=url,
            run_id=evaluation_run.id,
            summary=summary,
            evaluation_results=evaluation_results,
        )

    def _validate_evaluations(
        self, evaluations: Sequence[JudgeReference | Scorer]
    ) -> None:
        if any(
            not isinstance(evaluation, (JudgeReference, Scorer))
            for evaluation in evaluations
        ):
            raise EvaluationsError(
                "judges must contain typed JudgeReference or Scorer objects"
            )

    async def _resolve_evaluations(
        self,
        evaluations: Sequence[JudgeReference | Scorer],
        generation_handler: EvalHandler,
    ) -> list[OfflineEvaluation]:
        judge_references = [
            evaluation
            for evaluation in evaluations
            if isinstance(evaluation, JudgeReference)
        ]
        resolved_judges: list[LaunchDarklyJudgeEvaluation] = []
        if judge_references:
            if not hasattr(generation_handler, "provides_for"):
                raise EvaluationsError(
                    "LaunchDarkly judges require a ProviderHandler created with "
                    "create_handler()"
                )
            resolved_judges = await resolve_launchdarkly_judges(
                judge_references,
                [cast(ProviderHandler, generation_handler)],
                sdk_key=self._sdk_key,
            )

        judge_iterator = iter(resolved_judges)
        return [
            next(judge_iterator)
            if isinstance(evaluation, JudgeReference)
            else evaluation
            for evaluation in evaluations
        ]

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
    return EvaluationsModule(api_client=api_client, sdk_key=resolved_sdk_key)
