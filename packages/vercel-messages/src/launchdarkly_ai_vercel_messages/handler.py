from __future__ import annotations

import asyncio
import base64
import inspect
import json
from collections.abc import AsyncGenerator, Callable, Mapping
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
    _config,
    _create_handler,
    compose_history,
    parse_template,
    report_usage,
)
from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from launchdarkly_ai_server.utils import model_parameters

from ._version import PACKAGE_NAME, __version__
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

ModelSource = Any | Callable[[AiConfigRep], Any]

MAX_STEPS = 10


#: The ``model.parameters`` keys forwarded into ``ai.InferenceRequestParams``, after the renames in
#: :func:`_renamed_parameters`. Every other key is dropped. The handler sets ``model``,
#: ``messages``, ``tools`` and ``output_type`` itself, and runs the tool loop.
#:
#: This is the Vercel AI SDK list from TESTING.md section 1.12 as the Python ``ai`` library can
#: send it. ``stop_sequences`` is on the JS list but the Python library has no field for it.
#: ``parallel_tool_calls``, ``max_tool_calls``, ``service_tier`` and ``prompt_cache_key`` are
#: Python-only: the library takes them as typed request params, while the JS AI SDK can reach them
#: only through ``providerOptions``. ``TestInferenceRequestParamsAreClassified`` in this package's
#: tests asserts every field of ``InferenceRequestParams`` and its parts is classified.
_INFERENCE_PARAMS_FORWARDED_KEYS = frozenset(
    {
        "frequency_penalty",
        "max_output_tokens",
        "max_tool_calls",
        "parallel_tool_calls",
        "presence_penalty",
        "prompt_cache_key",
        "reasoning",
        "seed",
        "service_tier",
        "temperature",
        "tool_choice",
        "top_k",
        "top_p",
    }
)

#: Fields of ``ai.InferenceRequestParams`` and its parts that no config key reaches, and why:
#: * ``metadata``, ``safety_identifier``, ``tags``: identity and attribution.
#: * ``sampling`` ``min_p``, ``repetition_penalty``, ``consideration_window``: every built-in
#:   provider raises on them, so a config that set one would fail every call.
#: * ``output.include``: API-shape switch.
#: * ``output.text_verbosity``: the AI Gateway raises on it.
#: * ``output.reasoning_summary``: ``reasoning`` is read for its effort only.
#: * ``cache.mode``, ``cache.retention``: the mode is provider-specific with no shared config
#:   spelling, and retention is data retention.
#: * ``routing``: endpoint and routing selection.
#: * ``context_management``: server-side state.
#: * ``provider_params``: carries credentials, data-retention and timeout settings.
#: * ``extra_headers``, ``extra_query``, ``extra_body``: raw request injection.
_INFERENCE_PARAMS_EXCLUDED_FIELDS = frozenset(
    {
        "metadata",
        "safety_identifier",
        "tags",
        "min_p",
        "repetition_penalty",
        "consideration_window",
        "include",
        "text_verbosity",
        "reasoning_summary",
        "mode",
        "retention",
        "routing",
        "context_management",
        "provider_params",
        "extra_headers",
        "extra_query",
        "extra_body",
    }
)

#: Forwarded keys that become one sampler each: config key to the sampler class and its field.
_SAMPLERS = {
    "temperature": ("TemperatureSamplerParams", "temperature"),
    "top_p": ("TopPSamplerParams", "top_p"),
    "top_k": ("TopKSamplerParams", "top_k"),
    "seed": ("SeedSamplerParams", "seed"),
}

#: The string ``tool_choice`` values the library's ``ToolChoiceMode`` takes.
_TOOL_CHOICE_MODES = frozenset({"auto", "none", "required"})


def _renamed_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Applies the section 1.12 renames: ``max_output_tokens`` wins over ``max_completion_tokens``,
    which wins over ``max_tokens``, and an explicit ``reasoning`` wins over ``reasoning_effort``."""
    for alias in ("max_completion_tokens", "max_tokens"):
        value = params.pop(alias, None)
        if value is not None:
            params.setdefault("max_output_tokens", value)
    effort = params.pop("reasoning_effort", None)
    if effort is not None:
        params.setdefault("reasoning", effort)
    return params


def _reasoning_effort(value: Any) -> tuple[bool, str | None]:
    """The effort to send for a ``reasoning`` value, and whether to send one at all.

    ``reasoning`` is a string, or an object whose ``effort`` is read. ``"provider-default"``
    leaves it unset, and ``"none"`` becomes ``None``, the library's spelling of reasoning
    disabled. Any other shape is malformed and dropped.
    """
    effort = value.get("effort") if isinstance(value, Mapping) else value
    if not isinstance(effort, str) or effort == "provider-default":
        return False, None
    return True, None if effort == "none" else effort


def _request_params(config: AiConfigRep) -> Any:
    params = select_forwarded_parameters(
        _renamed_parameters(model_parameters(config)), _INFERENCE_PARAMS_FORWARDED_KEYS
    )
    kwargs: dict[str, Any] = {}
    sampling: dict[type[Any], Any] = {}
    for key, (class_name, argument) in _SAMPLERS.items():
        if key in params:
            cls = getattr(ai, class_name)
            sampling[cls] = cls(**{argument: params[key]})
    penalties = {
        key: params[key]
        for key in ("frequency_penalty", "presence_penalty")
        if key in params
    }
    if penalties:
        sampling[ai.RepetitionPenaltyParams] = ai.RepetitionPenaltyParams(**penalties)
    if sampling:
        kwargs["sampling"] = sampling
    if params.get("max_output_tokens") is not None:
        kwargs["output"] = ai.OutputParams(max_tokens=params["max_output_tokens"])
    send_effort, effort = _reasoning_effort(params.get("reasoning"))
    if send_effort:
        kwargs["reasoning"] = ai.ReasoningParams(effort=effort)
    tool_calling: dict[str, Any] = {}
    tool_choice = params.get("tool_choice")
    if isinstance(tool_choice, str) and tool_choice in _TOOL_CHOICE_MODES:
        tool_calling["tool_choice"] = ai.ToolChoiceMode(tool_choice)
    if isinstance(params.get("parallel_tool_calls"), bool):
        tool_calling["parallel_tool_calls"] = params["parallel_tool_calls"]
    max_tool_calls = params.get("max_tool_calls")
    if isinstance(max_tool_calls, int) and not isinstance(max_tool_calls, bool):
        tool_calling["max_tool_calls"] = max_tool_calls
    if tool_calling:
        # ``ToolCallingParams`` requires a policy. ``auto`` is the library's own default when only
        # the parallel or max-calls setting is given.
        tool_calling.setdefault("tool_choice", ai.ToolChoiceMode.AUTO)
        kwargs["tool_calling"] = ai.ToolCallingParams(**tool_calling)
    if isinstance(params.get("service_tier"), str):
        kwargs["provider_service"] = ai.ProviderServiceParams(
            service_tier=params["service_tier"]
        )
    if isinstance(params.get("prompt_cache_key"), str):
        kwargs["cache"] = ai.CacheParams(key=params["prompt_cache_key"])
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
    # A schema on a turn that can still call tools stops the model from emitting
    # those calls, or fails validation before a final answer exists. Tool turns
    # stay unconstrained; the schema is applied only once the model has stopped
    # calling tools (or immediately, when the config has no tools).
    constrain_output = output_type is not None and not tools
    open_model_span: Any = None
    try:
        # Tool rounds stop at MAX_STEPS. The structured follow-up is one extra
        # turn, and it only runs after the model has stopped calling tools.
        for step in range(MAX_STEPS + 1):
            if step == MAX_STEPS and not constrain_output:
                break
            model_span = start_model_span(cfg, parent)
            open_model_span = model_span
            try:
                kwargs: dict[str, Any] = {
                    "model": resolved_model,
                    "messages": messages,
                    "tools": [] if constrain_output else tools,
                    "params": params,
                }
                if constrain_output and output_type is not None:
                    kwargs["output_type"] = output_type
                async with ai.stream(**kwargs) as provider_stream:
                    async for event in provider_stream:
                        text = _text_delta(event)
                        if text is not None:
                            yield {"type": "chunk", "text": text}
                usage = _usage(provider_stream)
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                # This turn's counts, not the run total. The root span carries the sum.
                finish_model_span(
                    model_span,
                    model_name(cfg),
                    SpanUsage(
                        input=usage["input_tokens"], output=usage["output_tokens"]
                    ),
                )
                mark_ok(model_span)
                model_span.end()
                open_model_span = None
            except asyncio.CancelledError:
                model_span.set_attribute("launchdarkly.run.cancelled", True)
                model_span.end()
                open_model_span = None
                raise
            except Exception as exc:
                fail_span(model_span, exc)
                open_model_span = None
                raise
            message = provider_stream.message
            calls = list(message.tool_calls or [])
            if not calls or constrain_output:
                if output_type is not None and tools and not constrain_output:
                    constrain_output = True
                    continue
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
    finally:
        # A consumer that stops reading injects GeneratorExit, which neither
        # except above sees. The chat span for the in-flight turn is still open.
        if open_model_span is not None:
            open_model_span.set_attribute("launchdarkly.stream.abandoned", True)
            open_model_span.end()


def create_vercel_messages_handler(
    model: ModelSource | None = None,
    *,
    capture_content: bool | None = None,
) -> ProviderHandler:
    report_usage(
        "vercel-messages.createVercelMessagesHandler", PACKAGE_NAME, __version__
    )
    return _create_vercel_messages_handler(model, capture_content=capture_content)


def _create_vercel_messages_handler(
    model: ModelSource | None = None,
    *,
    capture_content: bool | None = None,
) -> ProviderHandler:
    """Non-reporting :func:`create_vercel_messages_handler`, used by this package's wrappers."""

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
        failed = False
        cancelled = False
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
            finish_root_span(span, model_name(cfg), span_usage)
            mark_ok(span)
            return {"output": text, "usage": usage}
        except asyncio.CancelledError:
            # Not an Exception: a timeout or task.cancel() must still mark the run
            # cancelled, matching the other handlers. The in-flight chat span is
            # marked inside _run_conversation, where that turn was opened.
            cancelled = True
            raise
        except Exception as exc:
            failed = True
            fail_span(span, exc)
            raise
        finally:
            if cancelled:
                span.set_attribute("launchdarkly.run.cancelled", True)
            if not failed:
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
        completed = False
        failed = False
        cancelled = False
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
            finish_root_span(span, model_name(cfg), span_usage)
            mark_ok(span)
            yield {"type": "done", "output": output, "usage": usage}
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            failed = True
            fail_span(span, exc)
            raise
        finally:
            if cancelled:
                span.set_attribute("launchdarkly.run.cancelled", True)
            elif not completed and not failed:
                span.set_attribute("launchdarkly.stream.abandoned", True)
            if not failed:
                span.end()

    return _create_handler(
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
    capture_content: bool | None = None,
    variables: dict[str, Any] | None = None,
    **options: Any,
) -> Any:
    report_usage("vercel-messages.vercelMessages", PACKAGE_NAME, __version__)
    handler = _create_vercel_messages_handler(
        model=model, capture_content=capture_content
    )
    return _config(key=config_key, handler=handler, **options).invoke(
        user_input, context, variables=variables
    )
