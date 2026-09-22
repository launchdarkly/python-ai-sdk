from __future__ import annotations

import base64
import inspect
import json
from collections.abc import AsyncGenerator, Callable
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


def _owned(name: str) -> bool:
    return name.replace("_", "").lower() in OWNED_PARAMETERS


def build_request_params(config: AiConfigRep, runtime: Any = None) -> Any:
    runtime = runtime or ai
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
            cls = getattr(runtime, class_name)
            sampling[cls] = cls(**{argument: raw.pop(key)})
    kwargs: dict[str, Any] = {}
    if sampling:
        kwargs["sampling"] = sampling
    max_tokens = raw.pop("max_tokens", raw.pop("maxTokens", None))
    if max_tokens is not None:
        kwargs["output"] = runtime.OutputParams(max_tokens=max_tokens)
    reasoning_effort = raw.pop("reasoning_effort", raw.pop("reasoningEffort", None))
    if reasoning_effort is not None:
        kwargs["reasoning"] = runtime.ReasoningParams(effort=reasoning_effort)
    for direct in ("metadata", "safety_identifier", "extra_headers", "extra_query"):
        if direct in raw:
            kwargs[direct] = raw.pop(direct)
    if raw:
        kwargs["extra_body"] = raw
    return runtime.InferenceRequestParams(**kwargs)


async def resolve_model(
    source: ModelSource | None, cfg: AiConfigRep, runtime: Any = None
) -> Any:
    runtime = runtime or ai
    if source is None:
        return runtime.get_model(gateway_model_id(cfg))
    value = source(cfg) if callable(source) else source
    return await value if inspect.isawaitable(value) else value


def _image_part(block: dict[str, Any], runtime: Any) -> Any:
    source = block.get("source") or {}
    value: str | bytes
    if source.get("type") == "base64":
        value = base64.b64decode(source.get("data", ""))
    else:
        value = source.get("url") or block.get("url") or ""
    return runtime.file_part(value, media_type=source.get("media_type"))


def _content(content: Any, runtime: Any) -> Any:
    if not isinstance(content, list):
        return content if isinstance(content, str) else ""
    parts: list[Any] = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") in {"image", "file"}:
            parts.append(_image_part(block, runtime))
    return parts


def build_messages(
    cfg: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    *,
    include_instructions: bool = True,
    runtime: Any = None,
) -> list[Any]:
    runtime = runtime or ai
    result: list[Any] = []
    config_messages: list[dict[str, Any]] = []
    if cfg.get("instructions"):
        if include_instructions:
            result.append(
                runtime.system_message(parse_template(cfg["instructions"], variables))
            )
    else:
        for message in cfg.get("messages") or []:
            content = message.get("content", "")
            mapped = (
                parse_template(content, variables)
                if isinstance(content, str)
                else content
            )
            if message.get("role") == "system":
                if include_instructions:
                    result.append(runtime.system_message(mapped))
            else:
                config_messages.append({**message, "content": mapped})
    for turn in compose_history(
        history=history or [],
        user_input=user_input,
        config_messages=config_messages,
    ):
        mapped_content = _content(turn.get("content", ""), runtime)
        if turn.get("role") == "assistant":
            args = (
                mapped_content if isinstance(mapped_content, list) else [mapped_content]
            )
            result.append(runtime.assistant_message(*args))
        elif turn.get("role") == "user":
            args = (
                mapped_content if isinstance(mapped_content, list) else [mapped_content]
            )
            result.append(runtime.user_message(*args))
    return result


async def _call_tool(handler: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    value = handler(kwargs)
    return await value if inspect.isawaitable(value) else value


def build_agent_tools(
    definitions: dict[str, Any] | None,
    handlers: dict[str, Any] | None,
    runtime: Any = None,
    *,
    parent: Any = None,
) -> list[Any]:
    runtime = runtime or ai
    result: list[Any] = []
    for name, definition in (definitions or {}).items():
        handler = (handlers or {}).get(name)
        if not callable(handler) or isinstance(handler, NativeTool):
            continue

        async def execute(
            _handler: Callable[..., Any] = handler,
            _name: str = name,
            **kwargs: Any,
        ) -> Any:
            tool_span = (
                start_tool_span(_name, "", parent) if parent is not None else None
            )
            try:
                value = await _call_tool(_handler, kwargs)
            except BaseException as exc:
                if tool_span is not None:
                    if isinstance(exc, Exception):
                        fail_span(tool_span, exc)
                    else:
                        tool_span.set_attribute("launchdarkly.run.cancelled", True)
                        tool_span.end()
                raise
            if tool_span is not None:
                mark_ok(tool_span)
                tool_span.end()
            return value

        spec = ToolSpec(
            description=definition.get("description"),
            params=definition.get("parameters") or {},
        )
        model_tool = runtime.Tool(kind="function", name=name, spec=spec)
        agent_tool = runtime.AgentTool(model_tool, execute)
        try:
            object.__setattr__(agent_tool, "name", name)
            object.__setattr__(agent_tool, "execute", execute)
            object.__setattr__(agent_tool, "input_schema", spec.params)
            object.__setattr__(agent_tool, "description", spec.description)
        except (AttributeError, TypeError):
            pass
        result.append(agent_tool)
    return result


def _usage_values(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input", 0)
    if output_tokens is None:
        output_tokens = getattr(usage, "output", 0)
    return int(input_tokens or 0), int(output_tokens or 0)


def usage_of(stream: Any) -> dict[str, int]:
    """Total token usage for an agent run.

    ``AgentStream`` carries no aggregate usage — each message in the run holds its
    own — so a tool loop only reports true totals when the per-message values are
    summed rather than read off the stream.
    """
    messages = getattr(stream, "messages", None)
    if messages:
        totals = [
            _usage_values(getattr(message, "usage", None)) for message in messages
        ]
        return {
            "input_tokens": sum(total[0] for total in totals),
            "output_tokens": sum(total[1] for total in totals),
        }
    input_tokens, output_tokens = _usage_values(getattr(stream, "usage", None))
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def output_of(stream: Any) -> str:
    """Final text of a completed run.

    ``AgentStream`` exposes ``output`` and no ``text``, so reading ``text``
    defensively would turn a real response into an empty one.
    """
    output = stream.output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, str):
        return output
    return json.dumps(output)


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
        return build_output_type(schema, name=f"{name}Object")
    # Strict structured output rejects an untyped member, so fall back to a string
    # rather than emitting a schema the provider refuses.
    return str


def build_output_type(
    schema: dict[str, Any], *, name: str = "VercelAgentOutput"
) -> type[BaseModel]:
    """Build the Pydantic model the run validates its final answer against.

    Every field is required and extras are forbidden because providers running
    strict structured output reject a schema that allows either.
    """
    fields: dict[str, Any] = {
        field: (_python_type(field_schema, field.title().replace("_", "")), ...)
        for field, field_schema in (schema.get("properties") or {}).items()
    }
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def text_delta(event: Any) -> str | None:
    if getattr(event, "kind", None) == "text_delta":
        return str(getattr(event, "chunk", ""))
    events = getattr(ai, "events", None)
    delta_type = getattr(events, "TextDelta", None)
    if delta_type is not None and isinstance(event, delta_type):
        return str(event.chunk)
    return None


def create_vercel_agents_handler(
    model: ModelSource | None = None,
    *,
    capture_content: bool = False,
) -> ProviderHandler:
    async def run(
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
            agent = ai.Agent(
                tools=build_agent_tools(cfg.get("tools"), tool_handlers, parent=parent)
            )
            run_kwargs: dict[str, Any] = {
                "model": await resolve_model(model, cfg),
                "messages": build_messages(cfg, user_input, vs, history),
                "params": build_request_params(cfg),
            }
            if cfg.get("outputFormat"):
                run_kwargs["output_type"] = build_output_type(cfg["outputFormat"])
            async with agent.run(**run_kwargs) as provider_stream:
                async for _ in provider_stream:
                    pass
            usage = usage_of(provider_stream)
            output = output_of(provider_stream)
            span_usage = SpanUsage(
                input=usage["input_tokens"], output=usage["output_tokens"]
            )
            response_model = model_name(cfg)
            finish_model_span(model_span, response_model, span_usage)
            finish_root_span(span, response_model, span_usage)
            mark_ok(model_span)
            mark_ok(span)
            return {"output": output, "usage": usage}
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
        provider_stream: Any = None
        try:
            agent = ai.Agent(
                tools=build_agent_tools(cfg.get("tools"), tool_handlers, parent=parent)
            )
            async with agent.run(
                model=await resolve_model(model, cfg),
                messages=build_messages(cfg, user_input, vs, history),
                params=build_request_params(cfg),
            ) as provider_stream:
                async for event in provider_stream:
                    text = text_delta(event)
                    if text is not None:
                        yield {"type": "chunk", "text": text}
            completed = True
            usage = usage_of(provider_stream)
            span_usage = SpanUsage(
                input=usage["input_tokens"], output=usage["output_tokens"]
            )
            response_model = model_name(cfg)
            finish_model_span(model_span, response_model, span_usage)
            finish_root_span(span, response_model, span_usage)
            mark_ok(model_span)
            mark_ok(span)
            yield {
                "type": "done",
                "output": output_of(provider_stream),
                "usage": usage,
            }
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

    return create_handler(("*", "agent"), run, stream, capture_content=capture_content)


def vercel_agents(
    config_key: str,
    user_input: str | None,
    context: LDContext,
    *,
    model: ModelSource | None = None,
    capture_content: bool = False,
    variables: dict[str, Any] | None = None,
    **options: Any,
) -> Any:
    handler = create_vercel_agents_handler(model=model, capture_content=capture_content)
    return config(key=config_key, handler=handler, **options).invoke(
        user_input, context, variables=variables
    )
