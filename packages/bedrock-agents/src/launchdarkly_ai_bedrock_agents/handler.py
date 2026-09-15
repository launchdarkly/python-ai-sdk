"""Amazon Bedrock agent handler built on Strands."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    SpanMessage,
    SpanMessagePart,
    SpanUsage,
    config,
    create_handler,
    end_span_once,
    end_unfinished_spans,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    text_message,
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
    usage_of,
)

ModelOptions = Callable[[AiConfigRep], dict[str, Any]]
_BEARER_ENV_LOCK = asyncio.Lock()


@asynccontextmanager
async def _bearer_scope(token: str | None) -> AsyncGenerator[None, None]:
    if not token:
        yield
        return
    async with _BEARER_ENV_LOCK:
        previous = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            else:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = previous


def _system_prompt(cfg: AiConfigRep, variables: dict[str, Any]) -> str | None:
    system: str | None
    if cfg.get("instructions"):
        system = parse_template(str(cfg["instructions"]), variables)
    else:
        systems = [
            str(message.get("content", ""))
            for message in cfg.get("messages") or []
            if message.get("role") == "system"
        ]
        system = parse_template("\n".join(systems), variables) if systems else None
    if cfg.get("outputFormat"):
        instruction = "Respond with valid JSON matching this schema:\n" + json.dumps(
            cfg["outputFormat"], separators=(",", ":")
        )
        system = f"{system}\n\n{instruction}" if system else instruction
    return system


def _native_content(content: Any) -> list[dict[str, Any]]:
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
            image_format = (
                str(source.get("media_type", "image/png"))
                .rsplit("/", 1)[-1]
                .replace("jpg", "jpeg")
            )
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
                    "Bedrock does not accept remote image URLs; use base64 or s3"
                )
    return result


def _prompt(
    cfg: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if not cfg.get("instructions"):
        for message in cfg.get("messages") or []:
            if message.get("role") in ("user", "assistant"):
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
    for message in history or []:
        if message.get("role") in ("user", "assistant"):
            messages.append(
                {
                    "role": message["role"],
                    "content": _native_content(message.get("content", "")),
                }
            )
    if not messages:
        return user_input or ""
    if user_input or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": [{"text": user_input or ""}]})
    return messages


def _tools(cfg: AiConfigRep, handlers: dict[str, Any]) -> list[Any]:
    from strands import tool

    result = []
    for name, definition in (cfg.get("tools") or {}).items():
        handler = handlers.get(name)
        if not callable(handler):
            continue

        async def execute(_handler: Any = handler, **kwargs: Any) -> Any:
            value = _handler(kwargs)
            return await value if inspect.isawaitable(value) else value

        result.append(
            tool(
                name=name,
                description=definition.get("description", ""),
                inputSchema=definition.get("parameters") or {},
            )(execute)
        )
    return result


def _model_config(cfg: AiConfigRep) -> dict[str, Any]:
    parameters = cfg.get("model", {}).get("parameters") or {}
    aliases = {
        "max_tokens": "max_tokens",
        "maxTokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "topP": "top_p",
        "stop_sequences": "stop_sequences",
        "stopSequences": "stop_sequences",
    }
    return {
        target: parameters[source]
        for source, target in aliases.items()
        if source in parameters
    }


def _result_usage(result: Any) -> dict[str, Any]:
    metrics = getattr(result, "metrics", None)
    usage = getattr(metrics, "accumulated_usage", None)
    return dict(usage) if isinstance(usage, dict) else {}


def _result_text(result: Any) -> str:
    message = getattr(result, "message", None)
    if not isinstance(message, dict):
        return str(result or "")
    content = message.get("content") or []
    return "".join(
        str(block.get("text", "")) for block in content if isinstance(block, dict)
    )


class _RunSpans:
    def __init__(self, parent: Any, capture_content: bool) -> None:
        self.parent = parent
        self.capture_content = capture_content
        self.open_model_span: Any = None
        self.open_tool_spans: dict[str, Any] = {}
        self.span_usage = SpanUsage()
        self.reported = False

    def add_usage(self, usage: Any) -> SpanUsage:
        _, span = usage_of(usage)
        if isinstance(usage, dict):
            self.reported = True
            self.span_usage.input += span.input
            self.span_usage.output += span.output
            self.span_usage.cache_read += span.cache_read
            self.span_usage.cache_creation += span.cache_creation
        return span

    def register_hooks(self, registry: Any, **_kwargs: Any) -> None:
        from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent

        registry.add_callback(BeforeToolCallEvent, self.before)
        registry.add_callback(AfterToolCallEvent, self.after)

    def before(self, event: Any) -> None:
        use = event.tool_use
        call_id = str(use.get("toolUseId", ""))
        span = start_tool_span(str(use.get("name", "")), call_id, self.parent)
        self.open_tool_spans[call_id] = span
        set_tool_call_content_attributes(
            span, self.capture_content, arguments=use.get("input")
        )

    def after(self, event: Any) -> None:
        call_id = str(event.tool_use.get("toolUseId", ""))
        span = self.open_tool_spans.get(call_id)
        if event.exception is not None:
            fail_span(span, event.exception)
            self.open_tool_spans.pop(call_id, None)
            return
        try:
            set_tool_call_content_attributes(
                span, self.capture_content, result=event.result
            )
            succeed_span(span)
            self.open_tool_spans.pop(call_id, None)
        except Exception as exc:
            fail_span(span, exc)
            self.open_tool_spans.pop(call_id, None)
            raise

    def close_open_spans(self, error: BaseException) -> None:
        """Fail child spans still open when Strands raises."""
        for span in self.open_tool_spans.values():
            fail_span(span, error)
        self.open_tool_spans.clear()
        if self.open_model_span is not None:
            fail_span(self.open_model_span, error)
            self.open_model_span = None

    def abandon_open_spans(self, ended: set[int], cancelled: bool = False) -> None:
        """End interrupted stream children without marking them as errors."""
        for span in self.open_tool_spans.values():
            end_span_once(span, ended, abandoned=True, cancelled=cancelled)
        self.open_tool_spans.clear()
        if self.open_model_span is not None:
            end_span_once(
                self.open_model_span,
                ended,
                abandoned=True,
                cancelled=cancelled,
            )
            self.open_model_span = None

    def cancel_open_spans(self) -> None:
        """End blocking-run children skipped by CancelledError."""
        end_unfinished_spans(
            *self.open_tool_spans.values(),
            self.open_model_span,
        )
        self.open_tool_spans.clear()
        self.open_model_span = None


def _instrument_model(
    model: Any,
    cfg: AiConfigRep,
    spans: _RunSpans,
) -> None:
    original = model.stream

    async def stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        messages = args[0] if args else kwargs.get("messages") or []
        system_prompt = args[2] if len(args) > 2 else kwargs.get("system_prompt")
        span = start_model_span(cfg, spans.parent)
        spans.open_model_span = span
        usage_data: dict[str, Any] | None = None
        reason: str | None = None
        text = ""
        content: list[dict[str, Any]] = []
        try:
            if spans.capture_content:
                set_input_content_attributes(
                    span,
                    True,
                    system_instructions=system_prompt,
                    messages=(
                        span_messages(messages) if isinstance(messages, list) else []
                    ),
                )
            async for event in original(*args, **kwargs):
                if isinstance(event, dict):
                    if "metadata" in event:
                        usage_data = event["metadata"].get("usage") or {}
                    if "messageStop" in event:
                        reason = event["messageStop"].get("stopReason")
                    delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
                    if "text" in delta:
                        text += str(delta["text"])
                    start = (event.get("contentBlockStart") or {}).get("start") or {}
                    if "toolUse" in start:
                        content.append({"toolUse": start["toolUse"]})
                yield event
            if text:
                content.insert(0, {"text": text})
            turn_usage = spans.add_usage(usage_data)
            if spans.capture_content:
                set_output_content_attributes(
                    span,
                    True,
                    [
                        SpanMessage(
                            role="assistant",
                            parts=span_messages(
                                [{"role": "assistant", "content": content}]
                            )[0].parts
                            if content
                            else [SpanMessagePart(type="text", content=text)],
                            finish_reason=reason,
                        )
                    ],
                )
            finish_model_span(span, cfg, turn_usage, reason)
            spans.open_model_span = None
        except Exception as exc:
            fail_span(span, exc)
            spans.open_model_span = None
            raise

    model.stream = stream


def _build_agent(
    cfg: AiConfigRep,
    handlers: dict[str, Any],
    variables: dict[str, Any],
    *,
    client: Any,
    boto_session: Any,
    region: str | None,
    model_options: ModelOptions | None,
    parent: Any,
    capture_content: bool,
) -> tuple[Any, str | None, _RunSpans]:
    from strands import Agent
    from strands.models import BedrockModel

    options = dict(model_options(cfg) or {}) if model_options else {}
    options.update(_model_config(cfg))
    options["model_id"] = model_id(cfg)
    if client is None:
        if boto_session is not None:
            options["boto_session"] = boto_session
        if region:
            options["region_name"] = region

    model = BedrockModel(**options)
    if client is not None:
        model.client = client
    run_spans = _RunSpans(parent, capture_content)
    _instrument_model(model, cfg, run_spans)
    system = _system_prompt(cfg, variables)
    agent = Agent(
        model=model,
        system_prompt=system,
        tools=_tools(cfg, handlers),
        hooks=[run_spans],
        callback_handler=None,
    )
    return agent, system, run_spans


def create_bedrock_agents_handler(
    *,
    client: Any = None,
    boto_session: Any = None,
    api_key: str | None = None,
    region: str | None = None,
    model_options: ModelOptions | None = None,
    capture_content: bool = False,
) -> ProviderHandler:
    """Create a Strands Agent backed by BedrockModel."""

    async def invoke(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        values = variables or {}
        prompt = _prompt(cfg, user_input, values, history)
        bearer = _bearer_scope(api_key if client is None else None)
        await bearer.__aenter__()
        root = start_root_span(cfg, values)
        open_root: Any = root
        parent = parent_context_of(root)
        run_spans: _RunSpans | None = None
        try:
            agent, system, run_spans = _build_agent(
                cfg,
                tool_handlers or {},
                values,
                client=client,
                boto_session=boto_session,
                region=region,
                model_options=model_options,
                parent=parent,
                capture_content=capture_content,
            )
            set_input_content_attributes(
                root,
                capture_content,
                system_instructions=system,
                messages=[text_message("user", prompt)]
                if isinstance(prompt, str)
                else span_messages(prompt),
            )
            result = await agent.invoke_async(prompt)
            output = _result_text(result)
            result_usage = _result_usage(result)
            raw, span_usage = usage_of(result_usage)
            if not result_usage and run_spans.reported:
                span_usage = run_spans.span_usage
            set_output_content_attributes(
                root, capture_content, [text_message("assistant", output)]
            )
            finish_root_span(root, cfg, span_usage)
            succeed_span(root)
            open_root = None
            return {"output": output, "usage": raw}
        except Exception as exc:
            if run_spans is not None:
                run_spans.close_open_spans(exc)
                if run_spans.reported:
                    finish_root_span(root, cfg, run_spans.span_usage)
            fail_span(root, exc)
            open_root = None
            raise
        finally:
            if run_spans is not None:
                run_spans.cancel_open_spans()
            end_unfinished_spans(open_root)
            await bearer.__aexit__(None, None, None)

    async def stream(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        values = variables or {}
        prompt = _prompt(cfg, user_input, values, history)
        bearer = _bearer_scope(api_key if client is None else None)
        await bearer.__aenter__()
        root = start_root_span(cfg, values)
        parent = parent_context_of(root)
        ended: set[int] = set()
        run_spans: _RunSpans | None = None
        cancelled = False
        try:
            agent, system, run_spans = _build_agent(
                cfg,
                tool_handlers or {},
                values,
                client=client,
                boto_session=boto_session,
                region=region,
                model_options=model_options,
                parent=parent,
                capture_content=capture_content,
            )
            set_input_content_attributes(
                root,
                capture_content,
                system_instructions=system,
                messages=[text_message("user", prompt)]
                if isinstance(prompt, str)
                else span_messages(prompt),
            )
            output = ""
            final_result: Any = None
            async for event in agent.stream_async(prompt):
                if isinstance(event, dict) and isinstance(event.get("data"), str):
                    text = event["data"]
                    output += text
                    yield {"type": "chunk", "text": text}
                if isinstance(event, dict) and event.get("result") is not None:
                    final_result = event["result"]
            if final_result is not None:
                output = _result_text(final_result) or output
            raw, span_usage = usage_of(
                _result_usage(final_result) if final_result is not None else {}
            )
            if (
                (final_result is None or not _result_usage(final_result))
                and run_spans is not None
                and run_spans.reported
            ):
                span_usage = run_spans.span_usage
            set_output_content_attributes(
                root, capture_content, [text_message("assistant", output)]
            )
            finish_root_span(root, cfg, span_usage)
            mark_ok(root)
            end_span_once(root, ended)
            yield {"type": "done", "output": output, "usage": raw}
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            if run_spans is not None:
                run_spans.close_open_spans(exc)
                if run_spans.reported:
                    finish_root_span(root, cfg, run_spans.span_usage)
            fail_span(root, exc, ended)
            raise
        finally:
            if run_spans is not None and root is not None and id(root) not in ended:
                if run_spans.reported:
                    finish_root_span(root, cfg, run_spans.span_usage)
                run_spans.abandon_open_spans(ended, cancelled=cancelled)
            end_span_once(
                root,
                ended,
                abandoned=True,
                cancelled=cancelled,
            )
            await bearer.__aexit__(None, None, None)

    return create_handler(
        ("Bedrock", "agent"),
        invoke,
        stream,
        capture_content=capture_content,
    )


def bedrock_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Invoke a LaunchDarkly AI config using a Strands Bedrock agent."""
    variables = kwargs.pop("variables", None)
    factory_keys = (
        "client",
        "boto_session",
        "api_key",
        "region",
        "model_options",
        "capture_content",
    )
    factory = {key: kwargs.pop(key) for key in factory_keys if key in kwargs}
    return config(
        key=config_key,
        handler=create_bedrock_agents_handler(**factory),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
