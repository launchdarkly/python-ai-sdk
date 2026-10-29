"""
Shadow requests: run another variation of the AI Config in the background on
the same input, and track it on the normal ``$ld:ai:*`` events with
``isShadow``. The shadow's response is discarded, and the caller's response,
latency and events are unchanged.

LaunchDarkly attaches shadow requests to a targeting rule or the fallthrough
and sends them in the served variation as ``shadowConfiguration``.
``config().invoke()`` and ``.stream()`` call :func:`build_shadow_tasks`, then
either :func:`start_shadows` or hand the tasks back when ``skip_shadows=True``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from .sdk_usage import report_usage
from .shutdown_hooks import register_shutdown_hook
from .tracking import build_track_data, execute_and_track
from .types import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    ShadowResult,
    ShadowTask,
    VariationMeta,
)
from .types_validation import parse_ai_config
from .utils import (
    _context_identity_from_ld_context,
    model_stamps_from_meta,
    parse_usage,
    select_handler,
)

logger = logging.getLogger("launchdarkly_ai_server.shadows")

DEFAULT_SHADOW_TIMEOUT_S = 30.0
_FALLTHROUGH = "fallthrough"
_STRIPPED_KEYS = ("_ldMeta", "tools", "judgeConfiguration", "shadowConfiguration")
_BUCKET_SCALE = float(0xFFFFFFFFFFFFFFF)

_running: set[asyncio.Task[ShadowResult]] = set()


def build_shadow_tasks(
    *,
    config_key: str,
    raw_variation: Any,
    reason: Any,
    primary_variation_key: str,
    context: LDContext,
    user_input: str | None,
    history: list[dict[str, Any]] | None,
    variables: dict[str, Any] | None,
    handlers: list[ProviderHandler],
    run_id: str,
    timeout_s: float,
) -> list[ShadowTask]:
    """
    Returns a task for each shadow request on the rule or fallthrough that
    served this call, skipping entries that have ended, are not sampled for this
    context, or have no matching handler. Never raises.
    """
    try:
        entries, variations = _entries_for(raw_variation, reason)
    except Exception:
        logger.debug("Ignoring a malformed shadowConfiguration", exc_info=True)
        return []

    tasks: list[ShadowTask] = []
    now_ms = time.time() * 1000
    for entry in entries:
        if not _valid_entry(entry) or entry["endsAt"] <= now_ms:
            continue
        if not _in_sample(entry["id"], context, float(entry["samplingRate"])):
            continue
        parsed = _parse_variation(variations.get(entry["variationKey"]))
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
                user_context=dict(context),
                timeout_s=timeout_s,
                track_data={
                    "runId": run_id,
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
            )
        )
    return tasks


def start_shadows(tasks: list[ShadowTask], handlers: list[ProviderHandler]) -> None:
    """Runs each task in the background. The caller does not wait for them."""
    if not tasks:
        return
    loop = asyncio.get_running_loop()
    for task in tasks:
        running = loop.create_task(_run(task, handlers))
        _running.add(running)
        running.add_done_callback(_running.discard)


async def run_shadow(task: ShadowTask, handlers: list[ProviderHandler]) -> ShadowResult:
    """Runs one :class:`ShadowTask`. Never raises; a failure is in ``error``."""
    report_usage("client.runShadow")
    return await _run(task, handlers)


def _cancel_running() -> None:
    for running in list(_running):
        running.cancel()
    _running.clear()


register_shutdown_hook("Shadow requests", _cancel_running)


async def _wait_for_shadows_for_testing() -> None:
    while _running:
        await asyncio.gather(*list(_running), return_exceptions=True)


def _entries_for(raw_variation: Any, reason: Any) -> tuple[list[Any], dict[str, Any]]:
    configuration = (
        raw_variation.get("shadowConfiguration")
        if isinstance(raw_variation, dict)
        else None
    )
    slot = _slot(reason)
    if slot is None or not isinstance(configuration, dict):
        return [], {}
    shadows = configuration.get("shadows")
    variations = configuration.get("variations")
    if not isinstance(shadows, dict) or not isinstance(variations, dict):
        return [], {}
    entries = shadows.get(slot)
    return (entries if isinstance(entries, list) else []), variations


def _slot(reason: Any) -> str | None:
    if not isinstance(reason, dict):
        return None
    kind = reason.get("kind")
    if kind == "FALLTHROUGH":
        return _FALLTHROUGH
    if kind == "RULE_MATCH" and isinstance(reason.get("ruleId"), str):
        return str(reason["ruleId"])
    return None


def _in_sample(shadow_id: str, context: LDContext, sampling_rate: float) -> bool:
    identity = _context_identity_from_ld_context(context)
    if identity is None:
        return False
    canonical, _ = identity
    digest = hashlib.sha1(f"{shadow_id}.{canonical}".encode(), usedforsecurity=False)
    bucket = int(digest.hexdigest()[:15], 16) / _BUCKET_SCALE
    return bucket < sampling_rate


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

    try:
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
        )
    except _ShadowTimeout:
        return ShadowResult(
            usage=parse_usage({}), track_data=track_data, error="timeout"
        )
    except Exception as err:
        return ShadowResult(
            usage=parse_usage({}), track_data=track_data, error=str(err)
        )
    return ShadowResult(usage=result["usage"], track_data=result["track_data"])


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
