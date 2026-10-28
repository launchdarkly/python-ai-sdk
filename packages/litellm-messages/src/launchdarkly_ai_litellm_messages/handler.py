from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    NativeTool,
    ProviderHandler,
    SpanUsage,
    _config,
    _create_handler,
    content_to_text,
    end_span_once,
    image_block_to_url,
    parse_template,
    report_usage,
    set_input_content_attributes,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    set_usage_span_attributes,
    text_message,
    to_semconv_finish_reason,
)
from launchdarkly_ai_server.lifecycle import effective_capture_content
from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from launchdarkly_ai_server.utils import model_parameters

from ._version import PACKAGE_NAME, __version__

Completion = Callable[..., Awaitable[Any]]


def _json_schema(output_format: dict[str, Any]) -> dict[str, Any]:
    schema = dict(output_format)
    if schema.get("type") in (None, "json_schema", "json"):
        schema["type"] = "object"
    return schema


#: The cross-SDK list for LiteLLM Messages (TESTING.md section 1.12): the ``model.parameters``
#: keys forwarded to ``litellm.acompletion``. Every other key is dropped. The handler sets
#: ``model``, ``messages``, ``stream``, ``stream_options``, ``tools`` and ``response_format`` itself,
#: and ``api_key`` (with ``api_base``) comes from the handler's options, never from the config.
#: ``TestAcompletionAcceptsExactlyTheseKeys`` in this package's tests asserts every named parameter
#: of ``acompletion`` is classified as forwarded, handler-owned or excluded.
_COMPLETION_FORWARDED_KEYS = frozenset(
    {
        "frequency_penalty",
        "max_completion_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_effort",
        "seed",
        "stop",
        "temperature",
        "tool_choice",
        "top_p",
    }
)

#: Accepted by ``litellm.acompletion`` but never forwarded, and why:
#: * ``store``: data retention.
#: * ``user``, ``safety_identifier``: identity and attribution.
#: * ``timeout``, ``base_url``, ``api_version``, ``deployment_id``, ``model_list``,
#:   ``shared_session``: credentials, endpoints and connection settings. ``shared_session`` is a
#:   live HTTP session object.
#: * ``extra_headers``: raw request injection.
#: * ``functions``, ``function_call``: the deprecated spelling of ``tools``, which the handler owns.
#: * ``enable_json_schema_validation``: runtime wiring.
#: * ``n``: switches the response shape. The handler reads only ``choices[0]``, and a streamed
#:   ``n > 1`` interleaves the choices.
#: * ``logprobs``, ``top_logprobs``, ``modalities``, ``audio``, ``prediction``,
#:   ``include_server_side_tool_invocations``: API-shape switches. They add or change output the
#:   handler does not read.
#: * ``web_search_options``: a remote, server-side tool.
#: * ``thinking``, ``logit_bias``, ``service_tier``, ``prompt_cache_key``, ``verbosity``: model
#:   settings LiteLLM translates per backing provider and may reject for one that lacks them. Left
#:   off until each is confirmed safe across the providers LiteLLM fronts.
#:
#: Everything ``acompletion`` reads only through ``**kwargs`` (``api_base``,
#: ``custom_llm_provider``, ``fallbacks``, ``num_retries``, ``mock_response``, ``aws_*``,
#: ``vertex_*``, ...) is dropped the same way. Named for the drift test and for review, not read at
#: runtime.
_COMPLETION_EXCLUDED_KEYS = frozenset(
    {
        "store",
        "user",
        "safety_identifier",
        "timeout",
        "base_url",
        "api_version",
        "deployment_id",
        "model_list",
        "shared_session",
        "extra_headers",
        "functions",
        "function_call",
        "enable_json_schema_validation",
        "n",
        "logprobs",
        "top_logprobs",
        "modalities",
        "audio",
        "prediction",
        "include_server_side_tool_invocations",
        "web_search_options",
        "thinking",
        "logit_bias",
        "service_tier",
        "prompt_cache_key",
        "verbosity",
    }
)
_MAX_TOOL_TURNS = 10


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _model_name(config_value: AiConfigRep) -> str:
    return str((config_value.get("model") or {}).get("name") or "")


def _provider_name(config_value: AiConfigRep) -> str:
    return str((config_value.get("provider") or {}).get("name") or "litellm").lower()


def _usage(value: Any) -> SpanUsage:
    raw = _value(value, "usage")
    return SpanUsage(
        input=int(_value(raw, "prompt_tokens", 0) or 0),
        output=int(_value(raw, "completion_tokens", 0) or 0),
    )


def _usage_reported(value: Any) -> bool:
    return _value(value, "usage") is not None


def _map_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            parts.append(
                {"type": "image_url", "image_url": {"url": image_block_to_url(block)}}
            )
    return parts


def _build_messages(
    config_value: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    configured = config_value.get("messages") or []
    if configured:
        for message in configured:
            content = message.get("content", "")
            if isinstance(content, str):
                content = parse_template(content, variables)
            messages.append(
                {"role": message.get("role", "user"), "content": _map_content(content)}
            )
    elif config_value.get("instructions"):
        messages.append(
            {
                "role": "system",
                "content": parse_template(config_value["instructions"], variables),
            }
        )

    if history:
        messages.extend(
            {
                "role": message.get("role", "user"),
                "content": _map_content(message.get("content", "")),
            }
            for message in history
            if message.get("role") != "system"
        )
        if user_input:
            messages.append({"role": "user", "content": user_input})
    elif not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": user_input or ""})
    return messages


def _build_tools(
    config_value: AiConfigRep,
    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool],
) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {},
            },
        }
        for name, tool in (config_value.get("tools") or {}).items()
        if callable(tool_handlers.get(name))
    ]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content_to_text(content)
    return "" if content is None else str(content)


def _span_messages(messages: list[dict[str, Any]]) -> list[Any]:
    return [
        text_message(str(message["role"]), _message_text(message.get("content")))
        for message in messages
    ]


def _request(
    config_value: AiConfigRep,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    stream: bool,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    parameters = select_forwarded_parameters(
        model_parameters(config_value), _COMPLETION_FORWARDED_KEYS
    )
    parameters.update(
        {
            "model": _model_name(config_value),
            "messages": messages,
            "stream": stream,
        }
    )
    # Connection settings come from the handler, never from model.parameters.
    if base_url is not None:
        parameters["api_base"] = base_url
    if api_key is not None:
        parameters["api_key"] = api_key
    if tools:
        parameters["tools"] = tools
    if stream:
        parameters["stream_options"] = {"include_usage": True}
    output_format = config_value.get("outputFormat")
    if output_format and not stream:
        parameters["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": False,
                "schema": _json_schema(output_format),
            },
        }
    return parameters


def _root_span(config_value: AiConfigRep, variables: dict[str, Any]) -> Any:
    span = trace.get_tracer("@launchdarkly/ai-litellm-messages").start_span(
        "invoke_agent"
    )
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    set_model_identity_attributes(
        span, _provider_name(config_value), _model_name(config_value), "litellm"
    )
    set_ld_span_attributes(span, variables)
    return span


def _parent_context(span: Any) -> Any:
    return trace.set_span_in_context(span)


def _model_span(config_value: AiConfigRep, parent: Any) -> Any:
    model = _model_name(config_value)
    span = trace.get_tracer("@launchdarkly/ai-litellm-messages").start_span(
        f"chat {model}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(span, _provider_name(config_value), model, "litellm")
    return span


def _tool_span(name: str, call_id: str, parent: Any) -> Any:
    span = trace.get_tracer("@launchdarkly/ai-litellm-messages").start_span(
        f"execute_tool {name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", name)
    span.set_attribute("gen_ai.tool.call.id", call_id)
    return span


def _finish_span(
    span: Any,
    usage: SpanUsage,
    ended: set[int],
    *,
    model: str | None = None,
    finish_reason: str | None = None,
) -> None:
    if model is not None:
        span.set_attribute("gen_ai.response.model", model)
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    set_usage_span_attributes(span, usage)
    span.set_status(StatusCode.OK)
    end_span_once(span, ended)


def _succeed_span(span: Any, ended: set[int]) -> None:
    span.set_status(StatusCode.OK)
    end_span_once(span, ended)


def _fail_span(span: Any, error: BaseException, ended: set[int]) -> None:
    span.record_exception(error)
    span.set_status(StatusCode.ERROR, str(error))
    end_span_once(span, ended)


def _finish_reason(response: Any) -> str | None:
    choices = _value(response, "choices", []) or []
    reason = _value(choices[0], "finish_reason") if choices else None
    return to_semconv_finish_reason(str(reason)) if reason else None


def _tool_call_dict(call: Any) -> dict[str, Any]:
    function = _value(call, "function")
    return {
        "id": str(_value(call, "id", "")),
        "type": "function",
        "function": {
            "name": str(_value(function, "name", "")),
            "arguments": str(_value(function, "arguments", "") or ""),
        },
    }


async def _invoke_tool(handler: Any, arguments: dict[str, Any]) -> Any:
    result = handler(arguments)
    return await result if inspect.isawaitable(result) else result


async def _execute_tools(
    calls: list[Any],
    handlers: Mapping[str, Any],
    parent: Any,
    ended: set[int],
    *,
    capture_content: bool,
) -> list[dict[str, Any]]:
    async def execute(call: Any) -> dict[str, Any]:
        call_dict = _tool_call_dict(call)
        function = call_dict["function"]
        name = function["name"]
        span = _tool_span(name, call_dict["id"], parent)
        try:
            try:
                arguments = json.loads(function["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid arguments for tool "{name}"') from exc
            if capture_content:
                set_tool_call_content_attributes(span, True, arguments=arguments)
            handler = handlers.get(name)
            if not callable(handler):
                raise ValueError(f'No handler registered for tool "{name}"')
            result = await _invoke_tool(handler, arguments)
            if capture_content:
                set_tool_call_content_attributes(span, True, result=result)
            content = result if isinstance(result, str) else json.dumps(result)
            _succeed_span(span, ended)
            return {
                "role": "tool",
                "tool_call_id": call_dict["id"],
                "content": content,
            }
        except Exception as exc:
            _fail_span(span, exc, ended)
            raise
        finally:
            end_span_once(span, ended, abandoned=True)

    return await asyncio.gather(*(execute(call) for call in calls))


def _response_message(response: Any) -> Any:
    choices = _value(response, "choices", []) or []
    return _value(choices[0], "message") if choices else None


def _output_value(content: Any, output_format: Any) -> Any:
    if output_format and isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    return content


def create_litellm_messages_handler(
    *,
    completion: Completion | None = None,
    capture_content: bool | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderHandler:
    report_usage(
        "litellm-messages.createLiteLLMMessagesHandler", PACKAGE_NAME, __version__
    )
    return _create_litellm_messages_handler(
        completion=completion,
        capture_content=capture_content,
        base_url=base_url,
        api_key=api_key,
    )


def _create_litellm_messages_handler(
    *,
    completion: Completion | None = None,
    capture_content: bool | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderHandler:
    """Non-reporting :func:`create_litellm_messages_handler`, used by this package's wrappers."""
    content_override = capture_content
    if completion is None:
        from litellm import acompletion

        completion = acompletion

    async def call(
        config_value: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        capture_content = effective_capture_content(content_override)
        handlers = tool_handlers or {}
        values = variables or {}
        messages = _build_messages(config_value, user_input, values, history)
        tools = _build_tools(config_value, handlers)
        root = _root_span(config_value, values)
        parent = _parent_context(root)
        total = SpanUsage()
        usage_reported = False
        ended: set[int] = set()
        open_model: Any = None
        try:
            if capture_content:
                set_input_content_attributes(
                    root,
                    True,
                    messages=_span_messages(messages),
                )
            for _ in range(_MAX_TOOL_TURNS + 1):
                model_span = _model_span(config_value, parent)
                open_model = model_span
                try:
                    if capture_content:
                        set_input_content_attributes(
                            model_span,
                            True,
                            messages=_span_messages(messages),
                        )
                    response = await completion(
                        **_request(
                            config_value,
                            messages,
                            tools,
                            stream=False,
                            base_url=base_url,
                            api_key=api_key,
                        )
                    )
                    turn_usage = _usage(response)
                    usage_reported = usage_reported or _usage_reported(response)
                    total.input += turn_usage.input
                    total.output += turn_usage.output
                    message = _response_message(response)
                    content = _value(message, "content")
                    calls = list(_value(message, "tool_calls", []) or [])
                    if capture_content:
                        set_output_content_attributes(
                            model_span,
                            True,
                            [text_message("assistant", str(content or ""))],
                        )
                    _finish_span(
                        model_span,
                        turn_usage,
                        ended,
                        model=_model_name(config_value),
                        finish_reason=_finish_reason(response),
                    )
                    open_model = None
                except Exception as exc:
                    _fail_span(model_span, exc, ended)
                    open_model = None
                    raise
                if not calls:
                    output = _output_value(content, config_value.get("outputFormat"))
                    if capture_content:
                        set_output_content_attributes(
                            root, True, [text_message("assistant", str(content or ""))]
                        )
                    _finish_span(root, total, ended, model=_model_name(config_value))
                    return {
                        "output": output,
                        "usage": {
                            "input_tokens": total.input,
                            "output_tokens": total.output,
                        },
                    }
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [_tool_call_dict(call) for call in calls],
                    }
                )
                messages.extend(
                    await _execute_tools(
                        calls,
                        handlers,
                        parent,
                        ended,
                        capture_content=capture_content,
                    )
                )
            raise RuntimeError("Tool loop exceeded the maximum number of turns")
        except Exception as exc:
            if usage_reported:
                root.set_attribute("gen_ai.response.model", _model_name(config_value))
                set_usage_span_attributes(root, total)
            _fail_span(root, exc, ended)
            raise
        finally:
            if open_model is not None:
                end_span_once(open_model, ended, cancelled=True)
            end_span_once(root, ended, cancelled=True)

    def stream(
        config_value: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream(
            completion,
            config_value,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=effective_capture_content(content_override),
            base_url=base_url,
            api_key=api_key,
        )

    return _create_handler(
        ("*", "messages"),
        call,
        stream,
        capture_content=content_override,
    )


async def _stream(
    completion: Completion,
    config_value: AiConfigRep,
    user_input: str | None,
    handlers: Mapping[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    *,
    capture_content: bool,
    base_url: str | None = None,
    api_key: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    messages = _build_messages(config_value, user_input, variables, history)
    tools = _build_tools(config_value, handlers)
    root = _root_span(config_value, variables)
    parent = _parent_context(root)
    total = SpanUsage()
    usage_reported = False
    provider_stream: Any = None
    completed = False
    full_output = ""
    ended: set[int] = set()
    open_model: Any = None
    cancelled = False
    try:
        if capture_content:
            set_input_content_attributes(
                root,
                True,
                messages=_span_messages(messages),
            )
        for _ in range(_MAX_TOOL_TURNS + 1):
            model_span = _model_span(config_value, parent)
            open_model = model_span
            turn_usage = SpanUsage()
            try:
                if capture_content:
                    set_input_content_attributes(
                        model_span,
                        True,
                        messages=_span_messages(messages),
                    )
                provider_stream = await completion(
                    **_request(
                        config_value,
                        messages,
                        tools,
                        stream=True,
                        base_url=base_url,
                        api_key=api_key,
                    )
                )
                fragments: dict[int, dict[str, Any]] = {}
                turn_text = ""
                async for chunk in provider_stream:
                    chunk_usage = _usage(chunk)
                    usage_reported = usage_reported or _usage_reported(chunk)
                    turn_usage.input += chunk_usage.input
                    turn_usage.output += chunk_usage.output
                    total.input += chunk_usage.input
                    total.output += chunk_usage.output
                    choices = _value(chunk, "choices", []) or []
                    delta = _value(choices[0], "delta") if choices else None
                    text = _value(delta, "content")
                    if isinstance(text, str) and text:
                        turn_text += text
                        yield {"type": "chunk", "text": text}
                    for call in _value(delta, "tool_calls", []) or []:
                        index = int(_value(call, "index", 0) or 0)
                        current = fragments.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        current["id"] += str(_value(call, "id", "") or "")
                        function = _value(call, "function")
                        current["name"] += str(_value(function, "name", "") or "")
                        current["arguments"] += str(
                            _value(function, "arguments", "") or ""
                        )
                await provider_stream.aclose()
                provider_stream = None
                if capture_content:
                    set_output_content_attributes(
                        model_span,
                        True,
                        [text_message("assistant", turn_text)],
                    )
                _finish_span(
                    model_span,
                    turn_usage,
                    ended,
                    model=_model_name(config_value),
                )
                open_model = None
            except Exception as exc:
                _fail_span(model_span, exc, ended)
                open_model = None
                raise
            if not fragments:
                full_output += turn_text
                completed = True
                if capture_content:
                    set_output_content_attributes(
                        root, True, [text_message("assistant", full_output)]
                    )
                _finish_span(root, total, ended, model=_model_name(config_value))
                yield {
                    "type": "done",
                    "output": full_output,
                    "usage": {
                        "input_tokens": total.input,
                        "output_tokens": total.output,
                    },
                }
                return
            calls = [
                {
                    "id": value["id"],
                    "type": "function",
                    "function": {
                        "name": value["name"],
                        "arguments": value["arguments"],
                    },
                }
                for _, value in sorted(fragments.items())
            ]
            messages.append(
                {"role": "assistant", "content": turn_text or None, "tool_calls": calls}
            )
            messages.extend(
                await _execute_tools(
                    calls,
                    handlers,
                    parent,
                    ended,
                    capture_content=capture_content,
                )
            )
        raise RuntimeError("Tool loop exceeded the maximum number of turns")
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        if usage_reported:
            root.set_attribute("gen_ai.response.model", _model_name(config_value))
            set_usage_span_attributes(root, total)
        _fail_span(root, exc, ended)
        completed = True
        raise
    finally:
        if provider_stream is not None:
            try:
                await provider_stream.aclose()
            except Exception:
                pass
        if not completed:
            if open_model is not None:
                end_span_once(open_model, ended, abandoned=True, cancelled=cancelled)
            if usage_reported:
                root.set_attribute("gen_ai.response.model", _model_name(config_value))
                set_usage_span_attributes(root, total)
            end_span_once(root, ended, abandoned=True, cancelled=cancelled)


def litellm_messages(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    variables = kwargs.pop("variables", None)
    capture_content = kwargs.pop("capture_content", None)
    base_url = kwargs.pop("base_url", None)
    api_key = kwargs.pop("api_key", None)
    report_usage("litellm-messages.litellmMessages", PACKAGE_NAME, __version__)
    return _config(
        key=config_key,
        handler=_create_litellm_messages_handler(
            capture_content=capture_content, base_url=base_url, api_key=api_key
        ),
        **kwargs,
    ).invoke(user_input, context, variables)
