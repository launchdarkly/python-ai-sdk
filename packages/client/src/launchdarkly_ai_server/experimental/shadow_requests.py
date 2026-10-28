"""
Shadow Requests (experimental).

Call ``enable_shadow_requests()`` to run the shadow variations LaunchDarkly attaches
to a targeting rule or the fallthrough. Each sampled shadow runs in the background on
the same input, and is tracked on the normal ``$ld:ai:*`` events with ``isShadow``.
Its response is discarded, and the caller's response, latency and events are
unchanged. These names may change in a minor release, so review the changelog when
you upgrade.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..invocation_hooks import InvocationEvent, register_invocation_hook
from ..sdk_usage import report_usage
from ..shutdown_hooks import register_shutdown_hook
from ..tracking import build_track_data, execute_and_track
from ..types import AiConfigRep, LDContext, ProviderHandler, VariationMeta
from ..types_validation import parse_ai_config
from ..utils import (
    _context_identity_from_ld_context,
    model_stamps_from_meta,
    parse_usage,
    select_handler,
    to_ld_context,
)

__all__ = [
    "ShadowResult",
    "ShadowTask",
    "disable_shadow_requests",
    "enable_shadow_requests",
    "run_shadow",
]

logger = logging.getLogger("launchdarkly_ai_server.experimental.shadow_requests")

_FEATURE = "Shadow Requests"
_FALLTHROUGH = "fallthrough"
_STRIPPED_KEYS = ("_ldMeta", "tools", "judgeConfiguration", "shadowConfiguration")
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


@dataclass(frozen=True)
class ShadowResult:
    """The outcome of one shadow run. There is no response text: it is discarded."""

    usage: dict[str, int]
    track_data: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class _Settings:
    timeout_s: float
    on_task: Callable[[ShadowTask], None] | None


_settings: _Settings | None = None
_pending: set[asyncio.Task[ShadowResult]] = set()


def enable_shadow_requests(
    *,
    timeout_s: float = 30.0,
    on_task: Callable[[ShadowTask], None] | None = None,
) -> None:
    """
    Turns Shadow Requests on for this process. Calling it again replaces the settings.

    By default each sampled shadow runs in the background with *timeout_s*. Pass
    *on_task* to run them yourself instead: it receives a ``ShadowTask`` for each
    sampled shadow, before the real handler runs, and must only enqueue it.
    """
    report_usage("experimental.enableShadowRequests")
    if not timeout_s > 0:
        raise ValueError("timeout_s must be greater than 0")
    global _settings
    _settings = _Settings(timeout_s=timeout_s, on_task=on_task)
    register_invocation_hook(_FEATURE, _on_invocation)


def disable_shadow_requests() -> None:
    """Turns Shadow Requests off. Shadows already running finish."""
    global _settings
    _settings = None
    register_invocation_hook(_FEATURE, None)


async def run_shadow(task: ShadowTask, handlers: list[ProviderHandler]) -> ShadowResult:
    """Runs one ``ShadowTask``. Never raises; a failure is returned in ``error``."""
    report_usage("experimental.runShadow")
    return await _run(task, handlers)


def _reset() -> None:
    for pending in list(_pending):
        pending.cancel()
    _pending.clear()
    disable_shadow_requests()


register_shutdown_hook(_FEATURE, _reset)


async def _wait_for_shadows_for_testing() -> None:
    while _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


def _in_sample(shadow_id: str, context: LDContext, sampling_rate: float) -> bool:
    identity = _context_identity_from_ld_context(context)
    if identity is None:
        return False
    canonical, _ = identity
    digest = hashlib.sha1(f"{shadow_id}.{canonical}".encode(), usedforsecurity=False)
    bucket = int(digest.hexdigest()[:15], 16) / _BUCKET_SCALE
    return bucket < sampling_rate


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


def _parse_variation(raw: Any) -> tuple[AiConfigRep, VariationMeta] | None:
    if not isinstance(raw, dict):
        return None
    ld_meta = raw.get("_ldMeta")
    if not isinstance(ld_meta, dict) or not ld_meta.get("enabled"):
        return None
    parsed = parse_ai_config({k: v for k, v in raw.items() if k not in _STRIPPED_KEYS})
    if not parsed.success:
        return None
    meta: VariationMeta = {
        "enabled": True,
        "variationKey": ld_meta.get("variationKey", ""),
        "version": ld_meta.get("version", 1),
        "mode": ld_meta.get("mode"),
        **model_stamps_from_meta(ld_meta),
    }
    return parsed.data, meta


def _on_invocation(event: InvocationEvent) -> None:
    settings = _settings
    if settings is None:
        return
    slot = _slot(event.reason)
    configuration = event.value.get("shadowConfiguration")
    if slot is None or not isinstance(configuration, dict):
        return
    shadows = configuration.get("shadows")
    variations = configuration.get("variations")
    if not isinstance(shadows, dict) or not isinstance(variations, dict):
        return
    entries = shadows.get(slot)
    if not isinstance(entries, list):
        return

    now_ms = time.time() * 1000
    for entry in entries:
        if not _valid_entry(entry) or entry["endsAt"] <= now_ms:
            continue
        if not _in_sample(entry["id"], event.context, float(entry["samplingRate"])):
            continue
        parsed = _parse_variation(variations.get(entry["variationKey"]))
        if parsed is None:
            continue
        shadow_config, shadow_meta = parsed
        try:
            select_handler(shadow_config, shadow_meta, event.handlers)
        except Exception:
            logger.debug("No handler for shadow request %s", entry["id"])
            continue
        task = ShadowTask(
            config_key=event.config_key,
            shadow_request_id=entry["id"],
            shadow_config=shadow_config,
            shadow_meta=shadow_meta,
            user_input=event.user_input,
            history=list(event.history) if event.history else None,
            variables=dict(event.variables) if event.variables else None,
            user_context=dict(event.context),
            timeout_s=settings.timeout_s,
            track_data={
                "runId": event.run_id,
                "isShadow": True,
                "primaryVariationKey": event.variation_key,
                "shadowRequestId": entry["id"],
                "ownerKind": entry.get("ownerKind", ""),
                "ownerId": entry.get("ownerId", ""),
            },
            span_attributes={
                "launchdarkly.is_shadow": True,
                "launchdarkly.shadow.request.id": entry["id"],
                "launchdarkly.shadow.primary.variation.key": event.variation_key,
            },
        )
        if settings.on_task is not None:
            try:
                settings.on_task(task)
            except Exception:
                logger.warning(
                    "The Shadow Requests on_task callback failed", exc_info=True
                )
            continue
        running = asyncio.get_running_loop().create_task(_run(task, event.handlers))
        _pending.add(running)
        running.add_done_callback(_pending.discard)


async def _run(task: ShadowTask, handlers: list[ProviderHandler]) -> ShadowResult:
    extra = {k: v for k, v in task.track_data.items() if k != "runId"}
    run_id = task.track_data.get("runId")
    track_data = build_track_data(
        config_key=task.config_key,
        config=task.shadow_config,
        meta=task.shadow_meta,
        run_id=run_id,
        extra_track_data=extra,
    )
    try:
        handler = select_handler(task.shadow_config, task.shadow_meta, handlers)
    except Exception as err:
        return ShadowResult(
            usage=parse_usage({}), track_data=track_data, error=str(err)
        )

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            execute_and_track(
                config_key=task.config_key,
                config=task.shadow_config,
                meta=task.shadow_meta,
                user_context=task.user_context,
                handler=handler,
                user_input=task.user_input,
                tool_handlers=None,
                variables=task.variables,
                history=list(task.history) if task.history else None,
                run_id=run_id,
                extra_track_data=extra,
                span_attributes=task.span_attributes,
            ),
            timeout=task.timeout_s,
        )
    except TimeoutError:
        _track_timeout(task, track_data, int((time.monotonic() - start) * 1000))
        return ShadowResult(
            usage=parse_usage({}), track_data=track_data, error="timeout"
        )
    except Exception as err:
        return ShadowResult(
            usage=parse_usage({}), track_data=track_data, error=str(err)
        )
    return ShadowResult(usage=result["usage"], track_data=result["track_data"])


def _track_timeout(
    task: ShadowTask, track_data: dict[str, Any], elapsed_ms: int
) -> None:
    from ..lifecycle import get_client

    try:
        client = get_client()
        ld_ctx = to_ld_context(client, task.user_context)
        client.track("$ld:ai:duration:total", ld_ctx, track_data, elapsed_ms)
        client.track("$ld:ai:generation:error", ld_ctx, track_data, 1)
    except Exception:
        logger.debug("Could not track a shadow timeout", exc_info=True)
