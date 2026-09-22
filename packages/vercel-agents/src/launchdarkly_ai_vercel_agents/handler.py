from __future__ import annotations

import base64
import inspect
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

import ai
from ai.types.tools import ToolSpec
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ConfigDict, create_model

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    NativeTool,
    ProviderHandler,
    compose_history,
    config,
    create_handler,
    parse_template,
    set_ld_span_attributes,
)

from .model_id import gateway_model_id

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
) -> list[Any]:
    runtime = runtime or ai
    result: list[Any] = []
    for name, definition in (definitions or {}).items():
        handler = (handlers or {}).get(name)
        if not callable(handler) or isinstance(handler, NativeTool):
            continue

        async def execute(_handler: Callable[..., Any] = handler, **kwargs: Any) -> Any:
            return await _call_tool(_handler, kwargs)

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


def _start_span(cfg: AiConfigRep, variables: dict[str, Any]) -> Any:
    span = trace.get_tracer("@launchdarkly/ai-vercel-agents").start_span("invoke_agent")
    provider = str(cfg.get("provider", {}).get("name") or "vercel").lower()
    model = str(cfg.get("model", {}).get("name") or "")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    span.set_attribute("gen_ai.system", provider)
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model)
    set_ld_span_attributes(span, variables)
    return span


def _start_model_span(cfg: AiConfigRep, root: Any) -> Any:
    model = str(cfg.get("model", {}).get("name") or "")
    provider = str(cfg.get("provider", {}).get("name") or "vercel").lower()
    span = trace.get_tracer("@launchdarkly/ai-vercel-agents").start_span(
        f"chat {model}", context=trace.set_span_in_context(root)
    )
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.system", provider)
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model)
    return span


def _set_usage(span: Any, usage: dict[str, int]) -> None:
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)
    span.set_attribute("gen_ai.usage.prompt_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", output_tokens)
    span.set_attribute("gen_ai.usage.cache_read.input_tokens", 0)
    span.set_attribute("gen_ai.usage.cache_creation.input_tokens", 0)


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
        span = _start_span(cfg, vs)
        model_span = _start_model_span(cfg, span)
        try:
            agent = ai.Agent(tools=build_agent_tools(cfg.get("tools"), tool_handlers))
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
            _set_usage(model_span, usage)
            _set_usage(span, usage)
            model_span.set_status(StatusCode.OK)
            span.set_status(StatusCode.OK)
            return {"output": output, "usage": usage}
        except BaseException as exc:
            if isinstance(exc, Exception):
                model_span.record_exception(exc)
                model_span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
            raise
        finally:
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
        span = _start_span(cfg, vs)
        model_span = _start_model_span(cfg, span)
        completed = False
        provider_stream: Any = None
        try:
            agent = ai.Agent(tools=build_agent_tools(cfg.get("tools"), tool_handlers))
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
            _set_usage(model_span, usage)
            _set_usage(span, usage)
            model_span.set_status(StatusCode.OK)
            span.set_status(StatusCode.OK)
            yield {
                "type": "done",
                "output": output_of(provider_stream),
                "usage": usage,
            }
        except BaseException as exc:
            if isinstance(exc, Exception):
                model_span.record_exception(exc)
                model_span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
            raise
        finally:
            if not completed:
                model_span.set_attribute("launchdarkly.stream.abandoned", True)
                span.set_attribute("launchdarkly.stream.abandoned", True)
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
