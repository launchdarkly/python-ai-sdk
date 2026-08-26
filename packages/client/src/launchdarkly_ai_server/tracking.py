from __future__ import annotations

import inspect
import os
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

from .types import (
    NATIVE_TOOL_KEY,
    AiConfigRep,
    ExecuteStreamEvent,
    LDContext,
    NativeTool,
    ProviderHandler,
    TrackData,
    VariationMeta,
)
from .utils import parse_usage, to_ld_context


def _try_get_environment_id() -> str | None:
    """
    Returns the LaunchDarkly environment MongoDB ObjectId used to set
    ``feature_flag.set.id`` on OTel spans for AI Config Monitoring trace
    correlation.

    Checks the ``LD_ENVIRONMENT_ID`` environment variable.  The Python ldclient
    SDK does not expose the environment ID through any public or stable private
    API after initialisation, so manual configuration is required.
    """
    return os.environ.get("LD_ENVIRONMENT_ID") or None


def wrap_tool_handlers(
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None,
    user_context: LDContext,
    track_data: TrackData,
) -> dict[str, Callable[..., Any]]:
    """
    Wraps each tool handler so that invoking it emits a ``$ld:ai:tool_call``
    tracking event.

    - Regular callables are wrapped with a function that tracks before delegating.
    - ``NativeTool`` values become zero-arg tracking stubs; the original
      ``NativeTool`` instance is stashed on the stub under ``NATIVE_TOOL_KEY``
      so handler packages can recover the provider tool name.
    - Tools whose name starts with ``__handoff_`` skip tracking (synthetic
      graph-routing tools must not pollute ``$ld:ai:tool_call`` metrics).
    """
    if not tool_handlers:
        return {}

    from .lifecycle import get_client  # late import to avoid circular deps

    wrapped: dict[str, Callable[..., Any]] = {}

    for name, fn in tool_handlers.items():
        if isinstance(fn, NativeTool):
            native = fn

            def _make_native_stub(
                tool_name: str, native_tool: NativeTool
            ) -> Callable[..., Any]:
                def stub(*args: Any, **kwargs: Any) -> None:
                    get_client().track(
                        "$ld:ai:tool_call",
                        user_context,
                        {**track_data, "toolKey": tool_name},
                        1,
                    )

                setattr(stub, NATIVE_TOOL_KEY, native_tool)
                return stub

            wrapped[name] = _make_native_stub(name, native)
        else:

            def _make_regular_wrapper(
                tool_name: str, original: Callable[..., Any]
            ) -> Callable[..., Any]:
                async def wrapper(*args: Any, **kwargs: Any) -> Any:
                    if not tool_name.startswith("__handoff_"):
                        get_client().track(
                            "$ld:ai:tool_call",
                            user_context,
                            {**track_data, "toolKey": tool_name},
                            1,
                        )
                    result = original(*args, **kwargs)
                    return await result if inspect.isawaitable(result) else result

                return wrapper

            wrapped[name] = _make_regular_wrapper(name, fn)

    return wrapped


async def execute_and_track(
    *,
    config_key: str,
    config: AiConfigRep,
    meta: VariationMeta,
    user_context: LDContext,
    handler: ProviderHandler,
    user_input: str | None = None,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
    variables: dict[str, Any] | None = None,
    graph_key: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Calls *handler*, emits LD telemetry events, and returns
    ``{usage, response, track_data}``.
    """
    from .lifecycle import get_client

    environment_id = _try_get_environment_id()
    track_data: TrackData = {
        "runId": str(uuid.uuid4()),
        "configKey": config_key,
        "variationKey": meta.get("variationKey", "") if isinstance(meta, dict) else "",
        "version": meta.get("version", 1) if isinstance(meta, dict) else 1,
        "modelName": config.get("model", {}).get("name", "")
        if isinstance(config, dict)
        else "",
        "providerName": config.get("provider", {}).get("name", "")
        if isinstance(config, dict)
        else "",
    }
    if graph_key:
        track_data["graphKey"] = graph_key
    if environment_id:
        track_data["environmentId"] = environment_id

    client = get_client()
    ld_ctx = to_ld_context(client, user_context)

    tracked_tool_handlers = wrap_tool_handlers(tool_handlers, ld_ctx, track_data)
    merged_variables: dict[str, Any] = {
        **(variables or {}),
        "ldContext": {**user_context},
        "__ld": track_data,
    }

    start_time = time.monotonic()

    try:
        result = await handler(
            config, user_input, tracked_tool_handlers, merged_variables, history
        )
    except Exception:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        client.track("$ld:ai:duration:total", ld_ctx, track_data, elapsed_ms)
        client.track("$ld:ai:generation:error", ld_ctx, track_data, 1)
        raise

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    client.track("$ld:ai:duration:total", ld_ctx, track_data, elapsed_ms)
    client.track("$ld:ai:generation:success", ld_ctx, track_data, 1)

    usage = parse_usage(result.get("usage") or {})

    if usage["total"] > 0:
        client.track("$ld:ai:tokens:total", ld_ctx, track_data, usage["total"])
    if usage["input"] > 0:
        client.track("$ld:ai:tokens:input", ld_ctx, track_data, usage["input"])
    if usage["output"] > 0:
        client.track("$ld:ai:tokens:output", ld_ctx, track_data, usage["output"])

    raw_output = result.get("output")
    response = raw_output if raw_output is not None else ""
    return {"usage": usage, "response": response, "track_data": track_data}


async def execute_and_stream(
    *,
    config_key: str,
    config: AiConfigRep,
    meta: VariationMeta,
    user_context: LDContext,
    handler: ProviderHandler,
    user_input: str | None = None,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
    variables: dict[str, Any] | None = None,
    graph_key: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[ExecuteStreamEvent, None]:
    """
    Streaming counterpart to ``execute_and_track``. Yields ``chunk`` events
    from the handler, then a single ``done`` event once the stream completes
    and LD telemetry has been flushed.

    Falls back to calling the blocking handler when the handler has no
    ``stream`` implementation.
    """
    from .lifecycle import get_client

    environment_id = _try_get_environment_id()
    track_data: TrackData = {
        "runId": str(uuid.uuid4()),
        "configKey": config_key,
        "variationKey": meta.get("variationKey", "") if isinstance(meta, dict) else "",
        "version": meta.get("version", 1) if isinstance(meta, dict) else 1,
        "modelName": config.get("model", {}).get("name", "")
        if isinstance(config, dict)
        else "",
        "providerName": config.get("provider", {}).get("name", "")
        if isinstance(config, dict)
        else "",
    }
    if graph_key:
        track_data["graphKey"] = graph_key
    if environment_id:
        track_data["environmentId"] = environment_id

    client = get_client()
    ld_ctx = to_ld_context(client, user_context)

    tracked_tool_handlers = wrap_tool_handlers(tool_handlers, ld_ctx, track_data)
    merged_variables: dict[str, Any] = {
        **(variables or {}),
        "ldContext": {**user_context},
        "__ld": track_data,
    }

    start_time = time.monotonic()
    full_text = ""
    raw_usage: dict[str, Any] = {}

    try:
        if handler.has_stream:
            stream_gen = await handler.stream(
                config, user_input, tracked_tool_handlers, merged_variables, history
            )
            async for event in stream_gen:
                if event.get("type") == "chunk":
                    full_text += event.get("text", "")
                    yield {"type": "chunk", "text": event["text"]}
                elif event.get("type") == "done":
                    if event.get("output") is not None:
                        full_text = event["output"]
                    raw_usage = event.get("usage", {})
        else:
            result = await handler(
                config, user_input, tracked_tool_handlers, merged_variables, history
            )
            out = result.get("output")
            full_text = str(out) if out is not None else ""
            raw_usage = result.get("usage") or {}
            if full_text:
                yield {"type": "chunk", "text": full_text}
    except Exception:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        client.track("$ld:ai:duration:total", ld_ctx, track_data, elapsed_ms)
        client.track("$ld:ai:generation:error", ld_ctx, track_data, 1)
        raise

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    client.track("$ld:ai:duration:total", ld_ctx, track_data, elapsed_ms)
    client.track("$ld:ai:generation:success", ld_ctx, track_data, 1)

    usage = parse_usage(raw_usage)
    if usage["total"] > 0:
        client.track("$ld:ai:tokens:total", ld_ctx, track_data, usage["total"])
    if usage["input"] > 0:
        client.track("$ld:ai:tokens:input", ld_ctx, track_data, usage["input"])
    if usage["output"] > 0:
        client.track("$ld:ai:tokens:output", ld_ctx, track_data, usage["output"])

    yield {
        "type": "done",
        "response": full_text,
        "usage": usage,
        "track_data": track_data,
    }
