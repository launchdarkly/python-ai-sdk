from __future__ import annotations

import inspect
import os
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from .agui import HANDLER_RICH_TYPES, TEXT_MESSAGE_CONTENT, TEXT_TYPES
from .trajectory import HANDOFF_TOOL_PREFIX, TrajectoryRecorder, render_trajectory
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
from .utils import (
    SPAN_ATTRIBUTES_KEY,
    model_stamps_from_meta,
    parse_usage,
    to_ld_context,
)


@asynccontextmanager
async def _closing_stream(stream: Any) -> AsyncIterator[Any]:
    """Close a handler stream when it implements ``aclose``.

    ``contextlib.aclosing`` always awaits ``aclose``. A handler may return a
    plain async iterator, which has no such method, and that call then raises
    ``AttributeError`` after the loop has already succeeded.
    """
    try:
        yield stream
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


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


def build_track_data(
    *,
    config_key: str,
    config: AiConfigRep,
    meta: VariationMeta,
    graph_key: str | None = None,
    run_id: str | None = None,
    extra_track_data: dict[str, Any] | None = None,
) -> TrackData:
    """The tracking payload for one invocation."""
    environment_id = _try_get_environment_id()
    track_data: TrackData = {
        "runId": run_id or str(uuid.uuid4()),
        "configKey": config_key,
        "variationKey": meta.get("variationKey", "") if isinstance(meta, dict) else "",
        "version": meta.get("version", 1) if isinstance(meta, dict) else 1,
        "modelName": config.get("model", {}).get("name", "")
        if isinstance(config, dict)
        else "",
        "providerName": config.get("provider", {}).get("name", "")
        if isinstance(config, dict)
        else "",
        **model_stamps_from_meta(meta),
    }
    if graph_key:
        track_data["graphKey"] = graph_key
    if environment_id:
        track_data["environmentId"] = environment_id
    if extra_track_data:
        track_data.update(extra_track_data)
    return track_data


def _merge_variables(
    variables: dict[str, Any] | None,
    user_context: LDContext,
    track_data: TrackData,
    span_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        **{k: v for k, v in (variables or {}).items() if k != SPAN_ATTRIBUTES_KEY},
        "ldContext": {**user_context},
        "__ld": track_data,
    }
    if span_attributes:
        merged[SPAN_ATTRIBUTES_KEY] = dict(span_attributes)
    return merged


def wrap_tool_handlers(
    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None,
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
                # Synthetic graph handoff tools are sync and skipped for tool_call metrics.
                # Keep them sync so a bare handoff() records the chosen edge (stream and
                # invoke share the same surface; an async wrapper would leave `chosen` empty).
                if tool_name.startswith(HANDOFF_TOOL_PREFIX):

                    def handoff_wrapper(*args: Any, **kwargs: Any) -> Any:
                        return original(*args, **kwargs)

                    return handoff_wrapper

                async def wrapper(*args: Any, **kwargs: Any) -> Any:
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


def _exposed_tool_keys(
    config: AiConfigRep,
    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None,
) -> list[str]:
    """The tool keys this config offered the model, in declaration order.

    The map handed to a handler can be wider -- ``config()`` merges a
    ``Registry``'s tools in -- and only the offered set belongs in a trajectory.
    Doubles as the "Tools available" list itself: a tool the config declares
    but no implementation was registered for is still one the model could
    call, so it belongs there even though the recorder can never observe it.
    Excludes a routed graph node's synthetic ``__handoff_`` tools, for the
    same reason ``TrajectoryRecorder.wrap`` excludes them from what it
    records, and a ``NativeTool``: the provider executes it directly, so no
    local recording is possible and listing it as available would read as
    unused even when the model used it.
    """
    tools = config.get("tools") if isinstance(config, dict) else None
    if not isinstance(tools, dict):
        return []
    handlers = tool_handlers or {}
    return [
        key
        for key in tools
        if not key.startswith(HANDOFF_TOOL_PREFIX)
        and not isinstance(handlers.get(key), NativeTool)
    ]


async def execute_and_track(
    *,
    config_key: str,
    config: AiConfigRep,
    meta: VariationMeta,
    user_context: LDContext,
    handler: ProviderHandler,
    user_input: str | None = None,
    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
    variables: dict[str, Any] | None = None,
    graph_key: str | None = None,
    history: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    extra_track_data: dict[str, Any] | None = None,
    span_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calls *handler*, emits LD telemetry events, and returns
    ``{usage, response, track_data}``.

    *extra_track_data* is added to every event, and *span_attributes* to the
    handler's root span.
    """
    from .lifecycle import get_client

    track_data = build_track_data(
        config_key=config_key,
        config=config,
        meta=meta,
        graph_key=graph_key,
        run_id=run_id,
        extra_track_data=extra_track_data,
    )

    client = get_client()
    ld_ctx = to_ld_context(client, user_context)

    # Recording composes *inside* wrap_tool_handlers, on the original map, so
    # the recorder still sees a NativeTool and skips it -- wrapping the tracked
    # map would record the callable stub instead and show a judge a tool call
    # with an empty result. One recorder per invocation; they run concurrently.
    exposed_tool_keys = _exposed_tool_keys(config, tool_handlers)
    recorder = TrajectoryRecorder()
    tracked_tool_handlers = wrap_tool_handlers(
        recorder.wrap(tool_handlers or {}, exposed=exposed_tool_keys),
        ld_ctx,
        track_data,
    )
    merged_variables = _merge_variables(
        variables, user_context, track_data, span_attributes
    )

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
    return {
        "usage": usage,
        "response": response,
        "track_data": track_data,
        # Rendered, not structural: message_history is the only consumer, and
        # a JudgeTask must stay picklable.
        "trajectory": render_trajectory(
            recorder.invocations,
            observable_tools=exposed_tool_keys,
            omitted=recorder.omitted,
        ),
    }


async def execute_and_stream(
    *,
    config_key: str,
    config: AiConfigRep,
    meta: VariationMeta,
    user_context: LDContext,
    handler: ProviderHandler,
    user_input: str | None = None,
    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
    variables: dict[str, Any] | None = None,
    graph_key: str | None = None,
    history: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> AsyncGenerator[ExecuteStreamEvent, None]:
    """
    Streaming counterpart to ``execute_and_track``. Yields ``chunk`` events,
    the AG-UI text and tool subset when the handler emits it, then a single
    ``done`` event once the stream completes and LD telemetry has been flushed.
    ``config().stream()`` projects that subset back to ``chunk`` then ``done``.

    Falls back to calling the blocking handler when the handler has no
    ``stream`` implementation.
    """
    from .lifecycle import get_client

    track_data = build_track_data(
        config_key=config_key,
        config=config,
        meta=meta,
        graph_key=graph_key,
        run_id=run_id,
    )

    client = get_client()
    ld_ctx = to_ld_context(client, user_context)

    exposed_tool_keys = _exposed_tool_keys(config, tool_handlers)
    recorder = TrajectoryRecorder()
    tracked_tool_handlers = wrap_tool_handlers(
        recorder.wrap(tool_handlers or {}, exposed=exposed_tool_keys),
        ld_ctx,
        track_data,
    )
    merged_variables = _merge_variables(variables, user_context, track_data)

    start_time = time.monotonic()
    full_text = ""
    raw_usage: dict[str, Any] = {}
    had_usage = False
    saw_text = False

    try:
        if handler.has_stream:
            stream_gen = await handler.stream(
                config, user_input, tracked_tool_handlers, merged_variables, history
            )
            async with _closing_stream(stream_gen) as agen:
                async for event in agen:
                    typ = event.get("type")
                    if typ in TEXT_TYPES:
                        saw_text = True
                    if typ == "chunk":
                        if saw_text:
                            continue
                        text = event.get("text", "")
                        if not isinstance(text, str):
                            continue
                        full_text += text
                        yield {"type": "chunk", "text": text}
                    elif typ == TEXT_MESSAGE_CONTENT:
                        delta = event.get("delta") or ""
                        if isinstance(delta, str):
                            full_text += delta
                        yield event
                    elif typ in HANDLER_RICH_TYPES:
                        yield event
                    elif typ == "done":
                        if event.get("output") is not None:
                            full_text = event["output"]
                        usage_value = event.get("usage")
                        if isinstance(usage_value, dict) and usage_value:
                            raw_usage = usage_value
                            had_usage = True
        else:
            result = await handler(
                config, user_input, tracked_tool_handlers, merged_variables, history
            )
            out = result.get("output")
            full_text = str(out) if out is not None else ""
            usage_value = result.get("usage")
            if isinstance(usage_value, dict) and usage_value:
                raw_usage = usage_value
                had_usage = True
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
        "trajectory": render_trajectory(
            recorder.invocations,
            observable_tools=exposed_tool_keys,
            omitted=recorder.omitted,
        ),
        "_had_usage": had_usage,
    }
