"""
Shadow requests (experimental).

Use ``shadow_config()`` in place of ``config()`` to also run the shadow variations
LaunchDarkly attaches to the matched targeting rule or the fallthrough. The served
variation runs exactly as ``config()`` runs it. Once it succeeds, each sampled shadow
runs on the same input in the background, or is handed to ``on_task`` to run later with
``run_shadow()``. A shadow's response is discarded; it is tracked on the normal
``$ld:ai:*`` events with ``isShadow``. These names may change in a minor release, so
review the changelog when you upgrade.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import math
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context

from ..client import ConfigInstance, _InternalConfigInstance
from ..conversation import _conversation_id_from, conversation_id
from ..lifecycle import _fetch_variation, _parse_variation
from ..registry import resolve_handlers
from ..sdk_usage import report_usage
from ..shutdown_hooks import register_shutdown_hook
from ..tracking import execute_and_track, make_track_data
from ..types import (
    AiConfigRep,
    LDContext,
    NativeTool,
    ProviderHandler,
    ProviderResponse,
    VariationMeta,
)
from ..utils import (
    _context_identity_from_ld_context,
    parse_usage,
    select_handler,
)

__all__ = ["ShadowResult", "ShadowTask", "run_shadow", "shadow_config"]

logger = logging.getLogger("launchdarkly_ai_server.experimental.shadow_requests")

_FALLTHROUGH = "fallthrough"
_PENDING_SHADOWS = "__ld_pending_shadows"
_SHADOW_ONLY_KEYS = ("tools", "judgeConfiguration", "shadowConfiguration")
_BUCKET_SCALE = float(0xFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class ShadowTask:
    """
    Everything needed to run and track one sampled shadow, with no flag evaluation.
    Plain data, so it survives ``pickle``. Run it with ``run_shadow(task, handlers)``.
    """

    config_key: str
    shadow_request_id: str
    shadow_config: AiConfigRep
    shadow_meta: VariationMeta
    user_input: str | None
    history: list[dict[str, Any]] | None
    variables: dict[str, Any] | None
    user_context: LDContext
    timeout_s: float
    track_data: dict[str, Any]
    span_attributes: dict[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None


@dataclass(frozen=True)
class ShadowResult:
    """The outcome of one shadow run. There is no response text: it is discarded."""

    usage: dict[str, int]
    track_data: dict[str, Any]
    error: str | None = None


_running: set[asyncio.Task[ShadowResult]] = set()


def shadow_config(
    *,
    key: str,
    handler: ProviderHandler | list[ProviderHandler] | None = None,
    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
    registry: Any = None,
    skip_judges: bool = False,
    shadow_timeout_s: float = 30.0,
    on_task: Callable[[ShadowTask], None] | None = None,
) -> ConfigInstance:
    """
    Same as ``config()``, and also runs the shadows of the matched rule or fallthrough.

    Shadows start only after the served variation succeeds, and never delay or change
    its response. Each one gets *shadow_timeout_s*. Pass *on_task* to run them
    yourself: it receives a ``ShadowTask`` per sampled shadow and should only enqueue
    it.
    """
    report_usage("experimental.shadowConfig")
    if not (math.isfinite(shadow_timeout_s) and shadow_timeout_s > 0):
        raise ValueError("shadow_timeout_s must be a finite number greater than 0")
    return _ShadowConfigInstance(
        key=key,
        handler=handler,
        tool_handlers=tool_handlers,
        registry=registry,
        skip_judges=skip_judges,
        timeout_s=shadow_timeout_s,
        on_task=on_task,
    )


async def run_shadow(task: ShadowTask, handlers: list[ProviderHandler]) -> ShadowResult:
    """Runs one ``ShadowTask``. Never raises; a failure is returned in ``error``."""
    report_usage("experimental.runShadow")
    return await _run(task, handlers)


def _cancel_running() -> None:
    for task in list(_running):
        task.cancel()
    _running.clear()


register_shutdown_hook("Shadow requests", _cancel_running)


async def _wait_for_shadows_for_testing() -> None:
    while _running:
        await asyncio.gather(*list(_running), return_exceptions=True)


@dataclass(frozen=True)
class _Evaluation:
    variation: dict[str, Any]
    reason: Any
    shadow_configuration: Any


class _ShadowConfigInstance(_InternalConfigInstance):
    def __init__(
        self,
        *,
        timeout_s: float,
        on_task: Callable[[ShadowTask], None] | None,
        **config_args: Any,
    ) -> None:
        super().__init__(**config_args)
        self._timeout_s = timeout_s
        self._on_task = on_task

    async def _invoke(
        self,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse[Any]:
        evaluation = await _evaluate(self._key, context)
        response = await _Evaluated(self, evaluation.variation)._invoke(
            user_input, context, variables, history
        )
        self._start_shadows(
            evaluation, response.track_data, user_input, context, variables, history
        )
        return response

    async def _execute_stream(
        self,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
        sink: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        evaluation = await _evaluate(self._key, context)
        primary = _Evaluated(self, evaluation.variation)
        async with aclosing(
            primary._execute_stream(user_input, context, variables, history, sink)
        ) as source:
            async for event in source:
                if event.get("type") == "done":
                    sink[_PENDING_SHADOWS] = (
                        evaluation,
                        event.get("track_data"),
                        user_input,
                        variables,
                        history,
                    )
                yield event

    async def _judge_stream(
        self,
        sink: dict[str, Any],
        done: dict[str, Any],
        user_input: str | None,
        context: LDContext,
    ) -> dict[str, Any]:
        judge_results = await super()._judge_stream(sink, done, user_input, context)
        pending = sink.pop(_PENDING_SHADOWS, None)
        if pending is not None:
            evaluation, track_data, user_input, variables, history = pending
            self._start_shadows(
                evaluation, track_data, user_input, context, variables, history
            )
        return judge_results

    def _start_shadows(
        self,
        evaluation: _Evaluation,
        primary_track_data: dict[str, Any] | None,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> None:
        if not primary_track_data:
            return
        handlers = resolve_handlers(self._registry, self._normalize_handlers()) or []
        try:
            tasks = _select_tasks(
                config_key=self._key,
                evaluation=evaluation,
                primary_track_data=primary_track_data,
                handlers=handlers,
                user_input=user_input,
                context=context,
                variables=variables,
                history=history,
                timeout_s=self._timeout_s,
            )
        except Exception:
            logger.warning(
                "Skipping shadow requests: they could not be prepared", exc_info=True
            )
            return
        for task in tasks:
            if self._on_task is not None:
                try:
                    self._on_task(task)
                except Exception:
                    logger.warning(
                        "The shadow requests on_task callback failed", exc_info=True
                    )
                continue
            running = asyncio.get_running_loop().create_task(_run(task, handlers))
            _running.add(running)
            running.add_done_callback(_running.discard)


class _Evaluated(_InternalConfigInstance):
    """Runs ``config()``'s invoke and stream on a variation that is already evaluated."""

    _FORWARDED = ("key", "handler", "tool_handlers", "registry", "skip_judges")

    def __init__(self, source: ConfigInstance, variation: dict[str, Any]) -> None:
        super().__init__(
            **{name: getattr(source, f"_{name}") for name in self._FORWARDED}
        )
        self._variation = variation

    async def _evaluate(self, context: LDContext) -> dict[str, Any]:
        return self._variation


async def _evaluate(config_key: str, context: LDContext) -> _Evaluation:
    raw, reason = await _fetch_variation(config_key, context, with_reason=True)
    shadow_configuration = None
    if isinstance(raw, dict):
        shadow_configuration = raw.get("shadowConfiguration")
        raw = {k: v for k, v in raw.items() if k != "shadowConfiguration"}
    return _Evaluation(
        variation=_parse_variation(config_key, raw),
        reason=reason,
        shadow_configuration=shadow_configuration,
    )


def _slot(reason: Any) -> str | None:
    if not isinstance(reason, dict):
        return None
    kind = reason.get("kind")
    if kind == "FALLTHROUGH":
        return _FALLTHROUGH
    if kind == "RULE_MATCH" and isinstance(reason.get("ruleId"), str):
        return str(reason["ruleId"])
    return None


def _valid_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    numbers_ok = all(
        isinstance(entry.get(k), (int, float)) and not isinstance(entry.get(k), bool)
        for k in ("samplingRate", "endsAt")
    )
    strings_ok = all(
        isinstance(entry.get(k), str) and entry.get(k) for k in ("id", "variationKey")
    )
    return numbers_ok and strings_ok


def _in_sample(shadow_id: str, context: LDContext, sampling_rate: float) -> bool:
    identity = _context_identity_from_ld_context(context)
    if identity is None:
        return False
    canonical, _ = identity
    digest = hashlib.sha1(f"{shadow_id}.{canonical}".encode(), usedforsecurity=False)
    bucket = int(digest.hexdigest()[:15], 16) / _BUCKET_SCALE
    return bucket < sampling_rate


def _shadow_variation(
    config_key: str, raw: Any
) -> tuple[AiConfigRep, VariationMeta] | None:
    if not isinstance(raw, dict):
        return None
    try:
        parsed = _parse_variation(
            config_key, {k: v for k, v in raw.items() if k not in _SHADOW_ONLY_KEYS}
        )
    except RuntimeError:
        return None
    return parsed["config"], parsed["meta"]


def _select_tasks(
    *,
    config_key: str,
    evaluation: _Evaluation,
    primary_track_data: dict[str, Any],
    handlers: list[ProviderHandler],
    user_input: str | None,
    context: LDContext,
    variables: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    timeout_s: float,
) -> list[ShadowTask]:
    slot = _slot(evaluation.reason)
    configuration = evaluation.shadow_configuration
    if slot is None or not isinstance(configuration, dict):
        return []
    shadows = configuration.get("shadows")
    variations = configuration.get("variations")
    if not isinstance(shadows, dict) or not isinstance(variations, dict):
        return []
    entries = shadows.get(slot)
    if not isinstance(entries, list):
        return []

    primary_variation_key = primary_track_data.get("variationKey", "")
    conversation = _conversation_id_from(otel_context.get_current())
    now_ms = time.time() * 1000
    tasks: list[ShadowTask] = []
    for entry in entries:
        if not _valid_entry(entry) or entry["endsAt"] <= now_ms:
            continue
        if not _in_sample(entry["id"], context, float(entry["samplingRate"])):
            continue
        parsed = _shadow_variation(config_key, variations.get(entry["variationKey"]))
        if parsed is None:
            continue
        shadow_config, shadow_meta = parsed
        try:
            select_handler(shadow_config, shadow_meta, handlers)
        except Exception:
            logger.debug("No handler for shadow request %s", entry["id"])
            continue
        tasks.append(
            ShadowTask(
                config_key=config_key,
                shadow_request_id=entry["id"],
                shadow_config=shadow_config,
                shadow_meta=shadow_meta,
                user_input=user_input,
                history=list(history) if history else None,
                variables=dict(variables) if variables else None,
                user_context=copy.deepcopy(context),
                timeout_s=timeout_s,
                track_data={
                    "runId": primary_track_data.get("runId"),
                    "isShadow": True,
                    "primaryVariationKey": primary_variation_key,
                    "shadowRequestId": entry["id"],
                    "ownerKind": entry.get("ownerKind", ""),
                    "ownerId": entry.get("ownerId", ""),
                },
                span_attributes={
                    "launchdarkly.is_shadow": True,
                    "launchdarkly.shadow.request.id": entry["id"],
                    "launchdarkly.shadow.primary.variation.key": primary_variation_key,
                },
                conversation_id=conversation,
            )
        )
    return tasks


async def _run(task: ShadowTask, handlers: list[ProviderHandler]) -> ShadowResult:
    run_id = task.track_data.get("runId")
    extra = {k: v for k, v in task.track_data.items() if k != "runId"}
    try:
        handler = select_handler(task.shadow_config, task.shadow_meta, handlers)
        with conversation_id(task.conversation_id):
            result = await execute_and_track(
                config_key=task.config_key,
                config=task.shadow_config,
                meta=task.shadow_meta,
                user_context=task.user_context,
                handler=_with_deadline(handler, task.timeout_s),
                user_input=task.user_input,
                tool_handlers=None,
                variables=task.variables,
                history=list(task.history) if task.history else None,
                run_id=run_id,
                extra_track_data=extra,
                span_attributes=task.span_attributes,
                feature_flag_event=False,
            )
    except _ShadowTimeout:
        return _failed(task, "timeout")
    except Exception as err:
        return _failed(task, str(err))
    return ShadowResult(usage=result["usage"], track_data=result["track_data"])


def _failed(task: ShadowTask, error: str) -> ShadowResult:
    track_data = make_track_data(
        config_key=task.config_key,
        config=task.shadow_config,
        meta=task.shadow_meta,
        run_id=task.track_data.get("runId"),
        extra_track_data=task.track_data,
    )
    return ShadowResult(usage=parse_usage({}), track_data=dict(track_data), error=error)


class _ShadowTimeout(Exception):
    pass


def _with_deadline(handler: ProviderHandler, timeout_s: float) -> ProviderHandler:
    async def timed(*args: Any) -> Any:
        deadline = asyncio.timeout(timeout_s)
        try:
            async with deadline:
                return await handler(*args)
        except TimeoutError:
            if deadline.expired():
                raise _ShadowTimeout("timeout") from None
            raise

    return ProviderHandler(
        fn=timed,
        provides_for=handler.provides_for,
        capture_content=handler.capture_content,
    )
