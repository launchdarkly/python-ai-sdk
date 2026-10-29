from __future__ import annotations

import asyncio
import base64
import inspect
import json
from collections.abc import AsyncGenerator, Callable, Mapping
from types import SimpleNamespace
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
from launchdarkly_ai_server.lifecycle import effective_capture_content
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


#: The ``model.parameters`` keys forwarded into ``ai.InferenceRequestParams``, after the renames in
#: :func:`_renamed_parameters`. Every other key is dropped. The handler sets ``model``,
#: ``messages``, ``tools`` and ``output_type`` itself, and runs the tool loop.
#:
#: This is the cross-SDK Vercel AI SDK list as the Python ``ai`` library can
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
    """Applies the cross-SDK renames: ``max_output_tokens`` wins over ``max_completion_tokens``,
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


def build_request_params(config: AiConfigRep, runtime: Any = None) -> Any:
    runtime = runtime or ai
    params = select_forwarded_parameters(
        _renamed_parameters(model_parameters(config)), _INFERENCE_PARAMS_FORWARDED_KEYS
    )
    kwargs: dict[str, Any] = {}
    sampling: dict[type[Any], Any] = {}
    for key, (class_name, argument) in _SAMPLERS.items():
        if key in params:
            cls = getattr(runtime, class_name)
            sampling[cls] = cls(**{argument: params[key]})
    penalties = {
        key: params[key]
        for key in ("frequency_penalty", "presence_penalty")
        if key in params
    }
    if penalties:
        sampling[runtime.RepetitionPenaltyParams] = runtime.RepetitionPenaltyParams(
            **penalties
        )
    if sampling:
        kwargs["sampling"] = sampling
    if params.get("max_output_tokens") is not None:
        kwargs["output"] = runtime.OutputParams(max_tokens=params["max_output_tokens"])
    send_effort, effort = _reasoning_effort(params.get("reasoning"))
    if send_effort:
        kwargs["reasoning"] = runtime.ReasoningParams(effort=effort)
    tool_calling: dict[str, Any] = {}
    tool_choice = params.get("tool_choice")
    if isinstance(tool_choice, str) and tool_choice in _TOOL_CHOICE_MODES:
        tool_calling["tool_choice"] = runtime.ToolChoiceMode(tool_choice)
    if isinstance(params.get("parallel_tool_calls"), bool):
        tool_calling["parallel_tool_calls"] = params["parallel_tool_calls"]
    max_tool_calls = params.get("max_tool_calls")
    if isinstance(max_tool_calls, int) and not isinstance(max_tool_calls, bool):
        tool_calling["max_tool_calls"] = max_tool_calls
    if tool_calling:
        # ``ToolCallingParams`` requires a policy. ``auto`` is the library's own default when only
        # the parallel or max-calls setting is given.
        tool_calling.setdefault("tool_choice", runtime.ToolChoiceMode.AUTO)
        kwargs["tool_calling"] = runtime.ToolCallingParams(**tool_calling)
    if isinstance(params.get("service_tier"), str):
        kwargs["provider_service"] = runtime.ProviderServiceParams(
            service_tier=params["service_tier"]
        )
    if isinstance(params.get("prompt_cache_key"), str):
        kwargs["cache"] = runtime.CacheParams(key=params["prompt_cache_key"])
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
    handlers: Mapping[str, Any] | None,
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


class _ChatTurns:
    """One ``chat`` span per model turn.

    ``Agent.run`` loops inferences inside a single call and brackets each one
    with ``stream_start`` / ``stream_end``. A run that never emits those
    boundaries — the in-memory test double — keeps the span opened up front
    and records the run's usage on it.
    """

    def __init__(self, cfg: AiConfigRep, parent: Any) -> None:
        self._cfg = cfg
        self._parent = parent
        self._open: Any = start_model_span(cfg, parent)
        self._closed_a_turn = False

    def observe(self, event: Any) -> str | None:
        kind = getattr(event, "kind", None)
        if kind == "stream_end":
            if self._open is None:
                self._open = start_model_span(self._cfg, self._parent)
            self._close(getattr(event, "usage", None))
            self._closed_a_turn = True
        elif kind == "stream_start" and self._open is None:
            self._open = start_model_span(self._cfg, self._parent)
        return text_delta(event)

    def finish(self, usage: dict[str, int]) -> None:
        """Close a turn that never emitted ``stream_end``."""
        if self._open is None:
            return
        if self._closed_a_turn:
            self._close(None)
            return
        self._close(
            SimpleNamespace(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )
        )

    def cancel(self) -> None:
        if self._open is None:
            return
        self._open.set_attribute("launchdarkly.run.cancelled", True)
        self._open.end()
        self._open = None

    def fail(self, exc: BaseException) -> None:
        if self._open is None:
            return
        fail_span(self._open, exc)
        self._open = None

    def abandon(self) -> None:
        if self._open is None:
            return
        self._open.set_attribute("launchdarkly.stream.abandoned", True)
        self._open.end()
        self._open = None

    def _close(self, usage: Any) -> None:
        incoming, outgoing = _usage_values(usage)
        finish_model_span(
            self._open,
            model_name(self._cfg),
            SpanUsage(input=incoming, output=outgoing),
        )
        mark_ok(self._open)
        self._open.end()
        self._open = None


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
    capture_content: bool | None = None,
) -> ProviderHandler:
    report_usage("vercel-agents.createVercelAgentsHandler", PACKAGE_NAME, __version__)
    return _create_vercel_agents_handler(model, capture_content=capture_content)


def _create_vercel_agents_handler(
    model: ModelSource | None = None,
    *,
    capture_content: bool | None = None,
) -> ProviderHandler:
    """Non-reporting :func:`create_vercel_agents_handler`, used by this package's wrappers."""

    async def run(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        vs = variables or {}
        span = start_root_span(
            cfg, vs, capture_content=effective_capture_content(capture_content)
        )
        parent = parent_context_of(span)
        turns = _ChatTurns(cfg, parent)
        failed = False
        cancelled = False
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
                async for event in provider_stream:
                    turns.observe(event)
            usage = usage_of(provider_stream)
            output = output_of(provider_stream)
            turns.finish(usage)
            span_usage = SpanUsage(
                input=usage["input_tokens"], output=usage["output_tokens"]
            )
            finish_root_span(span, model_name(cfg), span_usage)
            mark_ok(span)
            return {"output": output, "usage": usage}
        except asyncio.CancelledError:
            # Not an Exception: a timeout or task.cancel() must still mark the run
            # cancelled, matching the other handlers and the native graph path.
            cancelled = True
            turns.cancel()
            raise
        except Exception as exc:
            failed = True
            turns.fail(exc)
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
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        vs = variables or {}
        span = start_root_span(
            cfg, vs, capture_content=effective_capture_content(capture_content)
        )
        parent = parent_context_of(span)
        turns = _ChatTurns(cfg, parent)
        completed = False
        failed = False
        cancelled = False
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
                    text = turns.observe(event)
                    if text is not None:
                        yield {"type": "chunk", "text": text}
            completed = True
            usage = usage_of(provider_stream)
            turns.finish(usage)
            span_usage = SpanUsage(
                input=usage["input_tokens"], output=usage["output_tokens"]
            )
            finish_root_span(span, model_name(cfg), span_usage)
            mark_ok(span)
            yield {
                "type": "done",
                "output": output_of(provider_stream),
                "usage": usage,
            }
        except asyncio.CancelledError:
            cancelled = True
            turns.cancel()
            raise
        except Exception as exc:
            failed = True
            turns.fail(exc)
            fail_span(span, exc)
            raise
        finally:
            if cancelled:
                span.set_attribute("launchdarkly.run.cancelled", True)
            elif not completed and not failed:
                turns.abandon()
                span.set_attribute("launchdarkly.stream.abandoned", True)
            if not failed:
                span.end()

    return _create_handler(("*", "agent"), run, stream, capture_content=capture_content)


def vercel_agents(
    config_key: str,
    user_input: str | None,
    context: LDContext,
    *,
    model: ModelSource | None = None,
    capture_content: bool | None = None,
    variables: dict[str, Any] | None = None,
    **options: Any,
) -> Any:
    report_usage("vercel-agents.vercelAgents", PACKAGE_NAME, __version__)
    handler = _create_vercel_agents_handler(
        model=model, capture_content=capture_content
    )
    return _config(key=config_key, handler=handler, **options).invoke(
        user_input, context, variables=variables
    )
