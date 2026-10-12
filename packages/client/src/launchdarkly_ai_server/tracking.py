from __future__ import annotations

import inspect
import os
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

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
from .utils import model_stamps_from_meta, parse_usage, to_ld_context


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
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None,
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
        **model_stamps_from_meta(meta),
    }
    if graph_key:
        track_data["graphKey"] = graph_key
    if environment_id:
        track_data["environmentId"] = environment_id

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


# Private terminal of :func:`_drive_handler_stream`. Not a handler event and not
# forwarded by either reader. A handler that yields this type is ignored.
_SETTLED_EVENT = "__ld_settled"


async def _drive_handler_stream(
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
) -> AsyncGenerator[dict[str, Any], None]:
    """Run one handler stream and yield its events, then one settled record.

    The settled record carries the response, parsed usage, tracking payload and
    trajectory. Handler ``done`` is consumed here. Telemetry matches
    :func:`execute_and_stream`. A raise records the error metrics and propagates,
    with no settled record.
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
        **model_stamps_from_meta(meta),
    }
    if graph_key:
        track_data["graphKey"] = graph_key
    if environment_id:
        track_data["environmentId"] = environment_id

    client = get_client()
    ld_ctx = to_ld_context(client, user_context)

    exposed_tool_keys = _exposed_tool_keys(config, tool_handlers)
    recorder = TrajectoryRecorder()
    tracked_tool_handlers = wrap_tool_handlers(
        recorder.wrap(tool_handlers or {}, exposed=exposed_tool_keys),
        ld_ctx,
        track_data,
    )
    merged_variables: dict[str, Any] = {
        **(variables or {}),
        "ldContext": {**user_context},
        "__ld": track_data,
    }

    start_time = time.monotonic()
    full_text = ""
    raw_usage: dict[str, Any] = {}
    had_usage = False
    saw_text_message = False

    try:
        if handler.has_stream:
            stream_gen = await handler.stream(
                config, user_input, tracked_tool_handlers, merged_variables, history
            )
            async for event in stream_gen:
                kind = event.get("type")
                if kind == "chunk":
                    if not saw_text_message:
                        full_text += event.get("text", "")
                    yield event
                elif kind == "TEXT_MESSAGE_CONTENT":
                    saw_text_message = True
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        full_text += delta
                    yield event
                elif kind == "done":
                    if event.get("output") is not None:
                        full_text = event["output"]
                    raw_usage = event.get("usage", {})
                    had_usage = bool(event.get("usage"))
                elif kind != _SETTLED_EVENT:
                    yield event
        else:
            result = await handler(
                config, user_input, tracked_tool_handlers, merged_variables, history
            )
            out = result.get("output")
            full_text = str(out) if out is not None else ""
            raw_usage = result.get("usage") or {}
            had_usage = bool(result.get("usage"))
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
        "type": _SETTLED_EVENT,
        "response": full_text,
        "usage": usage,
        "had_usage": had_usage,
        "track_data": track_data,
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
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
    variables: dict[str, Any] | None = None,
    graph_key: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[ExecuteStreamEvent, None]:
    """
    Streaming counterpart to ``execute_and_track``. Yields ``chunk`` events
    from the handler, then a single ``done`` event once the stream completes
    and LD telemetry has been flushed.

    Assistant text arrives as ``chunk``. ``TEXT_MESSAGE_CONTENT`` contributes
    its ``delta`` as a ``chunk`` as well. A tool event, or any other handler
    event, is not forwarded and does not end the turn. Only the handler's
    ``done`` supplies output and usage. ``config().stream()`` therefore stays
    ``chunk`` then ``done``.

    Falls back to calling the blocking handler when the handler has no
    ``stream`` implementation.
    """
    settled: dict[str, Any] | None = None
    saw_text_message = False
    async for event in _drive_handler_stream(
        config_key=config_key,
        config=config,
        meta=meta,
        user_context=user_context,
        handler=handler,
        user_input=user_input,
        tool_handlers=tool_handlers,
        variables=variables,
        graph_key=graph_key,
        history=history,
    ):
        if event.get("type") == _SETTLED_EVENT:
            settled = event
            continue
        kind = event.get("type")
        if kind == "TEXT_MESSAGE_CONTENT":
            saw_text_message = True
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                yield {"type": "chunk", "text": delta}
        elif kind == "chunk" and not saw_text_message:
            yield {"type": "chunk", "text": event["text"]}
    if settled is None:
        return
    yield {
        "type": "done",
        "response": settled.get("response", ""),
        "usage": settled.get("usage"),
        "track_data": settled.get("track_data"),
        "trajectory": settled.get("trajectory", ""),
    }
