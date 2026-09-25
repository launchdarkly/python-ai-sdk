from __future__ import annotations

import base64
import inspect
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from typing import Any

import ai
from ai.types.tools import ToolSpec
from pydantic import BaseModel, ConfigDict, create_model

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    NativeTool,
    ProviderHandler,
    SpanUsage,
    compose_history,
    config,
    create_handler,
    parse_template,
)

from .model_id import gateway_model_id
from .spans import (
    fail_span,
    finish_model_span,
    finish_root_span,
    mark_ok,
    model_name,
    parent_context_of,
    start_model_span,
    start_root_span,
    start_tool_span,
)

OWNED_PARAMETERS = {
    "model",
    "messages",
    "prompt",
    "system",
    "tools",
    "stream",
    "output",
    "outputformat",
    "stopwhen",
    "maxsteps",
    "apikey",
    "baseurl",
}

ModelSource = Any | Callable[[AiConfigRep], Any]

MAX_STEPS = 10


def _owned(name: str) -> bool:
    return name.replace("_", "").lower() in OWNED_PARAMETERS


def _request_params(config: AiConfigRep) -> Any:
    raw = {
        key: value
        for key, value in (config.get("model", {}).get("parameters") or {}).items()
        if not _owned(key)
    }
    sampling: dict[type[Any], Any] = {}
    mappings = {
        "temperature": ("TemperatureSamplerParams", "temperature"),
        "top_p": ("TopPSamplerParams", "top_p"),
        "topP": ("TopPSamplerParams", "top_p"),
        "top_k": ("TopKSamplerParams", "top_k"),
        "topK": ("TopKSamplerParams", "top_k"),
        "min_p": ("MinPSamplerParams", "min_p"),
        "minP": ("MinPSamplerParams", "min_p"),
        "repetition_penalty": (
            "RepetitionPenaltyParams",
            "repetition_penalty",
        ),
        "repetitionPenalty": (
            "RepetitionPenaltyParams",
            "repetition_penalty",
        ),
        "seed": ("SeedSamplerParams", "seed"),
    }
    for key, (class_name, argument) in mappings.items():
        if key in raw:
            cls = getattr(ai, class_name)
            sampling[cls] = cls(**{argument: raw.pop(key)})
    kwargs: dict[str, Any] = {}
    if sampling:
        kwargs["sampling"] = sampling
    max_tokens = raw.pop("max_tokens", raw.pop("maxTokens", None))
    if max_tokens is not None:
        kwargs["output"] = ai.OutputParams(max_tokens=max_tokens)
    reasoning_effort = raw.pop("reasoning_effort", raw.pop("reasoningEffort", None))
    if reasoning_effort is not None:
        kwargs["reasoning"] = ai.ReasoningParams(effort=reasoning_effort)
    for direct in ("metadata", "safety_identifier", "extra_headers", "extra_query"):
        if direct in raw:
            kwargs[direct] = raw.pop(direct)
    if raw:
        kwargs["extra_body"] = raw
    return ai.InferenceRequestParams(**kwargs)


async def _resolve_model(source: ModelSource | None, cfg: AiConfigRep) -> Any:
    if source is None:
        return ai.get_model(gateway_model_id(cfg))
    value = source(cfg) if callable(source) else source
    return await value if inspect.isawaitable(value) else value


def _image_part(block: dict[str, Any]) -> Any:
    source = block.get("source") or {}
    media_type = source.get("media_type")
    data: str | bytes
    if source.get("type") == "base64":
        data = base64.b64decode(source.get("data", ""))
    else:
        data = source.get("url") or block.get("url") or ""
    return ai.file_part(data, media_type=media_type)


def _content(content: Any) -> Any:
    if not isinstance(content, list):
        return content if isinstance(content, str) else ""
    parts: list[Any] = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") in {"image", "file"}:
            parts.append(_image_part(block))
    return parts


def _messages(
    cfg: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> list[Any]:
    result: list[Any] = []
    config_messages: list[dict[str, Any]] = []
    if cfg.get("instructions"):
        result.append(ai.system_message(parse_template(cfg["instructions"], variables)))
    else:
        for message in cfg.get("messages") or []:
            content = message.get("content", "")
            mapped = (
                parse_template(content, variables)
                if isinstance(content, str)
                else content
            )
            if message.get("role") == "system":
                result.append(ai.system_message(mapped))
            else:
                config_messages.append({**message, "content": mapped})

    turns = compose_history(
        history=history or [],
        user_input=user_input,
        config_messages=config_messages,
    )
    for turn in turns:
        content = _content(turn.get("content", ""))
        args = content if isinstance(content, list) else [content]
        if turn.get("role") == "assistant":
            result.append(ai.assistant_message(*args))
        elif turn.get("role") == "user":
            result.append(ai.user_message(*args))
    return result


async def _call_tool(handler: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    value = handler(kwargs)
    return await value if inspect.isawaitable(value) else value


def _tool_executors(
    cfg: AiConfigRep, handlers: dict[str, Any] | None
) -> dict[str, Callable[..., Any]]:
    executors: dict[str, Callable[..., Any]] = {}
    for name in cfg.get("tools") or {}:
        handler = (handlers or {}).get(name)
        if callable(handler) and not isinstance(handler, NativeTool):
            executors[name] = handler
    return executors


def _tool_args(call: Any) -> dict[str, Any]:
    args = getattr(call, "tool_args", None)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


async def _tool_result(
    call: Any,
    executors: dict[str, Callable[..., Any]],
    parent: Any,
) -> Any:
    handler = executors.get(call.tool_name)
    if handler is None:
        return ai.tool_result_part(
            call.tool_call_id,
            tool_name=call.tool_name,
            result=f"No handler registered for tool {call.tool_name!r}",
            is_error=True,
        )
    tool_span = start_tool_span(call.tool_name, call.tool_call_id, parent)
    try:
        result = await _call_tool(handler, _tool_args(call))
    except BaseException as exc:
        if isinstance(exc, Exception):
            # Surface ordinary tool failures to the model so it can recover.
            fail_span(tool_span, exc)
            return ai.tool_result_part(
                call.tool_call_id,
                tool_name=call.tool_name,
                result=str(exc),
                is_error=True,
            )
        tool_span.set_attribute("launchdarkly.run.cancelled", True)
        tool_span.end()
        raise
    mark_ok(tool_span)
    tool_span.end()
    return ai.tool_result_part(
        call.tool_call_id, tool_name=call.tool_name, result=result
    )


def _final_output(stream: Any) -> str:
    output = stream.output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _tools(cfg: AiConfigRep, handlers: dict[str, Any] | None) -> list[Any]:
    result: list[Any] = []
    for name, definition in (cfg.get("tools") or {}).items():
        handler = (handlers or {}).get(name)
        if not callable(handler) or isinstance(handler, NativeTool):
            continue

        async def execute(_handler: Callable[..., Any] = handler, **kwargs: Any) -> Any:
            return await _call_tool(_handler, kwargs)

        spec = ToolSpec(
            description=definition.get("description"),
            params=definition.get("parameters") or {},
        )
        tool = ai.Tool(kind="function", name=name, spec=spec)
        # The model-facing object remains a native Tool. The attribute is useful to
        # custom executors and harmless to the frozen SDK type only when supported.
        try:
            object.__setattr__(tool, "name", name)
            object.__setattr__(tool, "execute", execute)
            object.__setattr__(tool, "input_schema", spec.params)
            object.__setattr__(tool, "description", spec.description)
        except (AttributeError, TypeError):
            pass
        result.append(tool)
    return result


def _python_type(schema: dict[str, Any], name: str) -> Any:
    kind = schema.get("type")
    if kind == "integer":
        return int
    if kind == "number":
        return float
    if kind == "boolean":
        return bool
    if kind == "array":
        items = schema.get("items")
        item_type = (
            _python_type(items, f"{name}Item") if isinstance(items, dict) else str
        )
        return list[item_type]  # type: ignore[valid-type]
    if kind == "object" or schema.get("properties"):
        return _output_type(schema, name=f"{name}Object")
    # Strict structured output rejects untyped members.
    return str


def _output_type(
    schema: dict[str, Any], *, name: str = "VercelStructuredOutput"
) -> type[BaseModel]:
    """Build the strict Pydantic model required by structured-output providers."""
    fields: dict[str, Any] = {
        field: (_python_type(field_schema, field.title().replace("_", "")), ...)
        for field, field_schema in (schema.get("properties") or {}).items()
    }
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def _usage(stream: Any) -> dict[str, int]:
    usage = getattr(stream, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", getattr(usage, "input", 0)) or 0)
    output_tokens = int(
        getattr(usage, "output_tokens", getattr(usage, "output", 0)) or 0
    )
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _text_delta(event: Any) -> str | None:
    if getattr(event, "kind", None) == "text_delta":
        return str(getattr(event, "chunk", ""))
    events = getattr(ai, "events", None)
    delta_type = getattr(events, "TextDelta", None)
    if delta_type is not None and isinstance(event, delta_type):
        return str(event.chunk)
    return None


async def _run_conversation(
    cfg: AiConfigRep,
    model: ModelSource | None,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    tool_handlers: dict[str, Any] | None,
    parent: Any,
    *,
    structured: bool,
) -> AsyncGenerator[dict[str, Any], None]:
    """Drive a request to its final answer, resolving tool calls between turns.

    ``ai.stream`` reports the tool calls a model asks for but never runs them, so a
    config with tools stalls on an assistant message that still has calls pending
    unless each round is executed and fed back.
    """
    resolved_model = await _resolve_model(model, cfg)
    messages = _messages(cfg, user_input, variables, history)
    tools = _tools(cfg, tool_handlers)
    executors = _tool_executors(cfg, tool_handlers)
    params = _request_params(cfg)
    output_type = (
        _output_type(cfg["outputFormat"])
        if structured and cfg.get("outputFormat")
        else None
    )

    input_tokens = 0
    output_tokens = 0
    for _ in range(MAX_STEPS):
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "tools": tools,
            "params": params,
        }
        if output_type is not None:
            kwargs["output_type"] = output_type
        async with ai.stream(**kwargs) as provider_stream:
            async for event in provider_stream:
                text = _text_delta(event)
                if text is not None:
                    yield {"type": "chunk", "text": text}
        usage = _usage(provider_stream)
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        message = provider_stream.message
        calls = list(message.tool_calls or [])
        if not calls:
            yield {
                "type": "done",
                "output": _final_output(provider_stream),
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            }
            return
        results = [await _tool_result(call, executors, parent) for call in calls]
        messages = [*messages, message, ai.tool_message(*results)]
    raise RuntimeError(
        f"Vercel messages run did not reach a final response within {MAX_STEPS} steps"
    )


def create_vercel_messages_handler(
    model: ModelSource | None = None,
    *,
    capture_content: bool = False,
) -> ProviderHandler:
    async def invoke(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        vs = variables or {}
        span = start_root_span(cfg, vs)
        parent = parent_context_of(span)
        model_span = start_model_span(cfg, parent)
        failed = False
        try:
            text = ""
            usage = {"input_tokens": 0, "output_tokens": 0}
            async with aclosing(
                _run_conversation(
                    cfg,
                    model,
                    user_input,
                    vs,
                    history,
                    tool_handlers,
                    parent,
                    structured=True,
                )
            ) as events:
                async for event in events:
                    if event["type"] == "done":
                        text = event["output"]
                        usage = event["usage"]
            span_usage = SpanUsage(
                input=usage["input_tokens"], output=usage["output_tokens"]
            )
            response_model = model_name(cfg)
            finish_model_span(model_span, response_model, span_usage)
            finish_root_span(span, response_model, span_usage)
            mark_ok(model_span)
            mark_ok(span)
            return {"output": text, "usage": usage}
        except BaseException as exc:
            if isinstance(exc, Exception):
                failed = True
                fail_span(model_span, exc)
                fail_span(span, exc)
            raise
        finally:
            if not failed:
                model_span.end()
                span.end()

    async def stream(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        vs = variables or {}
        span = start_root_span(cfg, vs)
        parent = parent_context_of(span)
        model_span = start_model_span(cfg, parent)
        completed = False
        failed = False
        try:
            output = ""
            usage = {"input_tokens": 0, "output_tokens": 0}
            # aclosing so that abandoning this generator also unwinds the provider
            # stream's context rather than leaving it open until finalization.
            async with aclosing(
                _run_conversation(
                    cfg,
                    model,
                    user_input,
                    vs,
                    history,
                    tool_handlers,
                    parent,
                    structured=False,
                )
            ) as events:
                async for event in events:
                    if event["type"] == "chunk":
                        yield event
                        continue
                    completed = True
                    output = event["output"]
                    usage = event["usage"]
            span_usage = SpanUsage(
                input=usage["input_tokens"], output=usage["output_tokens"]
            )
            response_model = model_name(cfg)
            finish_model_span(model_span, response_model, span_usage)
            finish_root_span(span, response_model, span_usage)
            mark_ok(model_span)
            mark_ok(span)
            yield {"type": "done", "output": output, "usage": usage}
        except BaseException as exc:
            if isinstance(exc, Exception):
                failed = True
                fail_span(model_span, exc)
                fail_span(span, exc)
            raise
        finally:
            if not completed and not failed:
                model_span.set_attribute("launchdarkly.stream.abandoned", True)
                span.set_attribute("launchdarkly.stream.abandoned", True)
            if not failed:
                model_span.end()
                span.end()

    return create_handler(
        ("*", "messages"),
        invoke,
        stream,
        capture_content=capture_content,
    )


def vercel_messages(
    config_key: str,
    user_input: str | None,
    context: LDContext,
    *,
    model: ModelSource | None = None,
    capture_content: bool = False,
    variables: dict[str, Any] | None = None,
    **options: Any,
) -> Any:
    handler = create_vercel_messages_handler(
        model=model, capture_content=capture_content
    )
    return config(key=config_key, handler=handler, **options).invoke(
        user_input, context, variables=variables
    )
