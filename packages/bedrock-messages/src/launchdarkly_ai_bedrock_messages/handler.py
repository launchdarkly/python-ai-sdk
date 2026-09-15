"""Amazon Bedrock Runtime Converse handler."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    SpanMessage,
    SpanMessagePart,
    config,
    create_handler,
    end_span_once,
    end_unfinished_spans,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
)

from .spans import (
    fail_span,
    finish_model_span,
    finish_root_span,
    mark_ok,
    model_id,
    parent_context_of,
    span_messages,
    start_model_span,
    start_root_span,
    start_tool_span,
    succeed_span,
    tool_definitions,
    usage_of,
)

ConverseOptions = Callable[[AiConfigRep], dict[str, Any]]
_MAX_STEPS = 10
_BEARER_ENV_LOCK = asyncio.Lock()


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content or "")}]
    result: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            result.append({"text": str(block.get("text", ""))})
        elif block.get("type") == "image":
            source = block.get("source") or {}
            media_type = str(source.get("media_type", "image/png"))
            image_format = media_type.rsplit("/", 1)[-1].replace("jpg", "jpeg")
            if source.get("type") == "base64":
                result.append(
                    {
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": base64.b64decode(str(source.get("data", "")))
                            },
                        }
                    }
                )
            elif source.get("type") == "s3":
                result.append(
                    {
                        "image": {
                            "format": image_format,
                            "source": {"s3Location": {"uri": source.get("url", "")}},
                        }
                    }
                )
            elif source.get("type") == "url":
                raise ValueError(
                    "Bedrock Converse does not accept remote image URLs; use base64 or s3"
                )
    return result


def _build_prompt(
    cfg: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    *,
    output_format: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    messages: list[dict[str, Any]] = []
    system: str | None = None
    if cfg.get("messages"):
        systems = [m for m in cfg["messages"] if m.get("role") == "system"]
        if systems:
            system = parse_template(
                "\n".join(str(m.get("content", "")) for m in systems), variables
            )
        for message in cfg["messages"]:
            if message.get("role") not in ("user", "assistant"):
                continue
            messages.append(
                {
                    "role": message["role"],
                    "content": [
                        {
                            "text": parse_template(
                                str(message.get("content", "")), variables
                            )
                        }
                    ],
                }
            )
    elif cfg.get("instructions"):
        system = parse_template(str(cfg["instructions"]), variables)

    for message in history or []:
        if message.get("role") in ("user", "assistant"):
            messages.append(
                {
                    "role": message["role"],
                    "content": _content_blocks(message.get("content", "")),
                }
            )

    if user_input or not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": [{"text": user_input or ""}]})

    if output_format and cfg.get("outputFormat"):
        instruction = "Respond with valid JSON matching this schema:\n" + json.dumps(
            cfg["outputFormat"], separators=(",", ":")
        )
        system = f"{system}\n\n{instruction}" if system else instruction
    return messages, system


def _tools(cfg: AiConfigRep, handlers: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for name, definition in (cfg.get("tools") or {}).items():
        if not callable(handlers.get(name)):
            continue
        result.append(
            {
                "toolSpec": {
                    "name": name,
                    "description": definition.get("description", ""),
                    "inputSchema": {"json": definition.get("parameters") or {}},
                }
            }
        )
    return result


def _inference_config(cfg: AiConfigRep) -> dict[str, Any]:
    parameters = cfg.get("model", {}).get("parameters") or {}
    aliases = {
        "maxTokens": "maxTokens",
        "max_tokens": "maxTokens",
        "temperature": "temperature",
        "topP": "topP",
        "top_p": "topP",
        "stopSequences": "stopSequences",
        "stop_sequences": "stopSequences",
    }
    return {
        target: parameters[source]
        for source, target in aliases.items()
        if source in parameters
    }


def _request(
    cfg: AiConfigRep,
    messages: list[dict[str, Any]],
    system: str | None,
    tools: list[dict[str, Any]],
    options: ConverseOptions | None,
) -> dict[str, Any]:
    request = dict(options(cfg) or {}) if options else {}
    inference = _inference_config(cfg)
    if inference:
        request["inferenceConfig"] = inference
    if tools:
        request["toolConfig"] = {"tools": tools}
    request["modelId"] = model_id(cfg)
    request["messages"] = messages
    if system:
        request["system"] = [{"text": system}]
    else:
        request.pop("system", None)
    return request


async def _invoke_method(client: Any, name: str, kwargs: dict[str, Any]) -> Any:
    method = getattr(client, name)
    if inspect.iscoroutinefunction(method):
        return await method(**kwargs)
    value = await asyncio.to_thread(method, **kwargs)
    return await value if inspect.isawaitable(value) else value


async def _execute_tool(
    block: dict[str, Any],
    handlers: dict[str, Any],
    parent: Any,
    capture_content: bool,
    open_tool_spans: dict[str, Any] | None = None,
) -> dict[str, Any]:
    use = block["toolUse"]
    name = str(use.get("name", ""))
    call_id = str(use.get("toolUseId", ""))
    arguments = use.get("input") or {}
    span = start_tool_span(name, call_id, parent)
    if open_tool_spans is not None:
        open_tool_spans[call_id] = span
    try:
        set_tool_call_content_attributes(span, capture_content, arguments=arguments)
        handler = handlers.get(name)
        if not callable(handler):
            raise ValueError(f'No handler registered for tool "{name}"')
        result = handler(arguments)
        if inspect.isawaitable(result):
            result = await result
        set_tool_call_content_attributes(span, capture_content, result=result)
        succeed_span(span)
        if open_tool_spans is not None:
            open_tool_spans.pop(call_id, None)
        return {
            "toolResult": {
                "toolUseId": call_id,
                "content": [{"json": result}]
                if isinstance(result, (dict, list, int, float, bool)) or result is None
                else [{"text": str(result)}],
            }
        }
    except asyncio.CancelledError:
        if open_tool_spans is not None:
            open_tool_spans.pop(call_id, None)
        end_span_once(span, set(), abandoned=True, cancelled=True)
        raise
    except Exception as exc:
        fail_span(span, exc)
        if open_tool_spans is not None:
            open_tool_spans.pop(call_id, None)
        raise


class _Usage:
    def __init__(self) -> None:
        self.raw: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        from launchdarkly_ai_server import SpanUsage

        self.span = SpanUsage()
        self.reported = False

    def add(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        raw, span = usage_of(usage)
        self.reported = True
        for key, value in raw.items():
            self.raw[key] = self.raw.get(key, 0) + value
        self.span.input += span.input
        self.span.output += span.output
        self.span.cache_read += span.cache_read
        self.span.cache_creation += span.cache_creation


@asynccontextmanager
async def _client_scope(
    injected: Any, api_key: str | None, region: str | None
) -> AsyncIterator[Any]:
    if injected is not None:
        yield injected
        return

    import aioboto3  # type: ignore[import-untyped]

    kwargs: dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region
    if not api_key:
        async with aioboto3.Session().client("bedrock-runtime", **kwargs) as client:
            yield client
        return

    # Botocore exposes Bedrock bearer authentication through this environment provider rather than
    # as a create_client argument. Scope the override to client ownership and serialize explicit
    # token scopes so two handlers cannot overwrite one another.
    async with _BEARER_ENV_LOCK:
        previous = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        try:
            async with aioboto3.Session().client("bedrock-runtime", **kwargs) as client:
                yield client
        finally:
            if previous is None:
                os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            else:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = previous


async def _call(
    client: Any,
    cfg: AiConfigRep,
    user_input: str | None,
    handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    options: ConverseOptions | None,
    capture_content: bool,
) -> dict[str, Any]:
    messages, system = _build_prompt(
        cfg, user_input, variables, history, output_format=True
    )
    tools = _tools(cfg, handlers)
    root = start_root_span(cfg, variables)
    open_root: Any = root
    parent = parent_context_of(root)
    usage = _Usage()
    open_model: Any = None
    open_tool_spans: dict[str, Any] = {}
    try:
        set_input_content_attributes(
            root,
            capture_content,
            system_instructions=system,
            messages=span_messages(messages),
        )
        for step in range(_MAX_STEPS + 1):
            model_span = start_model_span(cfg, parent)
            open_model = model_span
            try:
                if capture_content:
                    set_input_content_attributes(
                        model_span,
                        capture_content,
                        system_instructions=system,
                        messages=span_messages(messages),
                        tool_definitions=tool_definitions(tools),
                    )
                response = await _invoke_method(
                    client, "converse", _request(cfg, messages, system, tools, options)
                )
                _, turn_usage = usage_of(response.get("usage"))
                usage.add(response.get("usage"))
                reason = response.get("stopReason")
                output_message = response.get("output", {}).get("message", {})
                content = output_message.get("content") or []
                if capture_content:
                    set_output_content_attributes(
                        model_span,
                        capture_content,
                        [
                            SpanMessage(
                                role="assistant",
                                parts=span_messages(
                                    [{"role": "assistant", "content": content}]
                                )[0].parts,
                                finish_reason=reason,
                            )
                        ],
                    )
                finish_model_span(model_span, cfg, turn_usage, reason)
                open_model = None
            except Exception as exc:
                fail_span(model_span, exc)
                open_model = None
                raise

            if reason != "tool_use":
                output = "".join(str(block.get("text", "")) for block in content)
                set_output_content_attributes(
                    root,
                    capture_content,
                    [
                        SpanMessage(
                            role="assistant",
                            parts=[SpanMessagePart(type="text", content=output)],
                        )
                    ],
                )
                finish_root_span(root, cfg, usage.span)
                succeed_span(root)
                open_root = None
                return {"output": output, "usage": usage.raw}
            if step == _MAX_STEPS:
                raise RuntimeError(f"Tool loop exceeded {_MAX_STEPS} steps")
            messages.append({"role": "assistant", "content": content})
            results = [
                await _execute_tool(
                    block,
                    handlers,
                    parent,
                    capture_content,
                    open_tool_spans,
                )
                for block in content
                if "toolUse" in block
            ]
            messages.append({"role": "user", "content": results})
        raise AssertionError("unreachable")
    except Exception as exc:
        if usage.reported:
            finish_root_span(root, cfg, usage.span)
        fail_span(root, exc)
        open_root = None
        raise
    finally:
        end_unfinished_spans(
            *open_tool_spans.values(),
            open_model,
            open_root,
        )


async def _stream_events(stream: Any) -> AsyncIterator[dict[str, Any]]:
    if hasattr(stream, "__aiter__"):
        async for event in stream:
            yield event
    else:
        iterator = iter(stream)

        def next_event() -> tuple[bool, Any]:
            try:
                return True, next(iterator)
            except StopIteration:
                return False, None

        while True:
            present, event = await asyncio.to_thread(next_event)
            if not present:
                break
            yield event


async def _stream(
    client: Any,
    cfg: AiConfigRep,
    user_input: str | None,
    handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    options: ConverseOptions | None,
    capture_content: bool,
) -> AsyncGenerator[dict[str, Any], None]:
    messages, system = _build_prompt(
        cfg, user_input, variables, history, output_format=False
    )
    tools = _tools(cfg, handlers)
    root = start_root_span(cfg, variables)
    parent = parent_context_of(root)
    usage = _Usage()
    full_output = ""
    ended: set[int] = set()
    open_model: Any = None
    open_tool_spans: dict[str, Any] = {}
    cancelled = False
    try:
        set_input_content_attributes(
            root,
            capture_content,
            system_instructions=system,
            messages=span_messages(messages),
        )
        for step in range(_MAX_STEPS + 1):
            model_span = start_model_span(cfg, parent)
            open_model = model_span
            content: list[dict[str, Any]] = []
            text = ""
            reason: str | None = None
            turn_usage_data: dict[str, Any] | None = None
            tool_by_index: dict[int, dict[str, Any]] = {}
            try:
                if capture_content:
                    set_input_content_attributes(
                        model_span,
                        capture_content,
                        system_instructions=system,
                        messages=span_messages(messages),
                        tool_definitions=tool_definitions(tools),
                    )
                response = await _invoke_method(
                    client,
                    "converse_stream",
                    _request(cfg, messages, system, tools, options),
                )
                async for event in _stream_events(response["stream"]):
                    if "contentBlockStart" in event:
                        start = event["contentBlockStart"]
                        if "toolUse" in start.get("start", {}):
                            tool_by_index[start.get("contentBlockIndex", 0)] = {
                                "toolUse": {
                                    **start["start"]["toolUse"],
                                    "input": "",
                                }
                            }
                    elif "contentBlockDelta" in event:
                        delta_event = event["contentBlockDelta"]
                        delta = delta_event.get("delta", {})
                        if "text" in delta:
                            chunk = str(delta["text"])
                            text += chunk
                            full_output += chunk
                            yield {"type": "chunk", "text": chunk}
                        elif "toolUse" in delta:
                            block = tool_by_index[
                                delta_event.get("contentBlockIndex", 0)
                            ]
                            block["toolUse"]["input"] += str(
                                delta["toolUse"].get("input", "")
                            )
                    elif "messageStop" in event:
                        reason = event["messageStop"].get("stopReason")
                    elif "metadata" in event:
                        turn_usage_data = event["metadata"].get("usage") or {}
                content.append({"text": text}) if text else None
                for block in tool_by_index.values():
                    raw_input = block["toolUse"].get("input", "")
                    try:
                        block["toolUse"]["input"] = json.loads(raw_input or "{}")
                    except json.JSONDecodeError:
                        block["toolUse"]["input"] = raw_input
                    content.append(block)
                _, turn_usage = usage_of(turn_usage_data)
                usage.add(turn_usage_data)
                if capture_content:
                    set_output_content_attributes(
                        model_span,
                        capture_content,
                        [
                            SpanMessage(
                                role="assistant",
                                parts=span_messages(
                                    [{"role": "assistant", "content": content}]
                                )[0].parts,
                                finish_reason=reason,
                            )
                        ],
                    )
                finish_model_span(model_span, cfg, turn_usage, reason)
                open_model = None
            except Exception as exc:
                fail_span(model_span, exc, ended)
                open_model = None
                raise

            if reason != "tool_use":
                set_output_content_attributes(
                    root,
                    capture_content,
                    [
                        SpanMessage(
                            role="assistant",
                            parts=[SpanMessagePart(type="text", content=full_output)],
                        )
                    ],
                )
                finish_root_span(root, cfg, usage.span)
                mark_ok(root)
                end_span_once(root, ended)
                yield {"type": "done", "output": full_output, "usage": usage.raw}
                return
            if step == _MAX_STEPS:
                raise RuntimeError(f"Tool loop exceeded {_MAX_STEPS} steps")
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        await _execute_tool(
                            block,
                            handlers,
                            parent,
                            capture_content,
                            open_tool_spans,
                        )
                        for block in content
                        if "toolUse" in block
                    ],
                }
            )
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        if usage.reported:
            finish_root_span(root, cfg, usage.span)
        fail_span(root, exc, ended)
        raise
    finally:
        for open_tool_span in open_tool_spans.values():
            end_span_once(
                open_tool_span,
                ended,
                abandoned=True,
                cancelled=cancelled,
            )
        open_tool_spans.clear()
        if open_model is not None:
            end_span_once(
                open_model,
                ended,
                abandoned=True,
                cancelled=cancelled,
            )
        if root is not None and id(root) not in ended and usage.reported:
            finish_root_span(root, cfg, usage.span)
        end_span_once(root, ended, abandoned=True, cancelled=cancelled)


def create_bedrock_messages_handler(
    *,
    client: Any = None,
    api_key: str | None = None,
    region: str | None = None,
    converse_options: ConverseOptions | None = None,
    capture_content: bool = False,
) -> ProviderHandler:
    """Create a Bedrock Converse handler.

    ``model.region`` is an inference-profile prefix. ``region`` configures only the AWS endpoint.
    Injected clients are used verbatim and are never closed.
    """

    async def invoke(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        async with _client_scope(client, api_key, region) as runtime:
            return await _call(
                runtime,
                cfg,
                user_input,
                tool_handlers or {},
                variables or {},
                history,
                converse_options,
                capture_content,
            )

    async def stream(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async with _client_scope(client, api_key, region) as runtime:
            provider_stream = _stream(
                runtime,
                cfg,
                user_input,
                tool_handlers or {},
                variables or {},
                history,
                converse_options,
                capture_content,
            )
            try:
                async for event in provider_stream:
                    yield event
            finally:
                await provider_stream.aclose()

    return create_handler(
        ("Bedrock", "messages"),
        invoke,
        stream,
        capture_content=capture_content,
    )


def bedrock_messages(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Invoke a LaunchDarkly AI config using Bedrock Converse."""
    variables = kwargs.pop("variables", None)
    factory_keys = (
        "client",
        "api_key",
        "region",
        "converse_options",
        "capture_content",
    )
    factory = {key: kwargs.pop(key) for key in factory_keys if key in kwargs}
    return config(
        key=config_key,
        handler=create_bedrock_messages_handler(**factory),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
