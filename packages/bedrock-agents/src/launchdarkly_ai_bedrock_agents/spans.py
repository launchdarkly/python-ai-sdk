"""OpenTelemetry span helpers for Bedrock Strands agents."""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanMessage,
    SpanMessagePart,
    SpanUsage,
    number_or_zero,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_usage_span_attributes,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    _HAS_OTEL = False

TRACER_NAME = "@launchdarkly/ai-bedrock-agents"
PROVIDER = "aws.bedrock"


def model_id(config: AiConfigRep) -> str:
    model = config.get("model", {})
    name = str(model.get("name", ""))
    prefix = str(model.get("region") or "")
    return name if not prefix or name.startswith(f"{prefix}.") else f"{prefix}.{name}"


def model_name(config: AiConfigRep) -> str:
    """Shared span surface name for Bedrock's resolved model ID."""
    return model_id(config)


def start_root_span(config: AiConfigRep, variables: dict[str, Any]) -> Any:
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span("invoke_agent")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    set_model_identity_attributes(span, PROVIDER, model_id(config))
    set_ld_span_attributes(span, variables)
    return span


def parent_context_of(span: Any) -> Any:
    return trace.set_span_in_context(span) if _HAS_OTEL and span is not None else None


def start_model_span(config: AiConfigRep, parent: Any) -> Any:
    if not _HAS_OTEL:
        return None
    name = model_id(config)
    span = trace.get_tracer(TRACER_NAME).start_span(f"chat {name}", context=parent)
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(span, PROVIDER, name)
    return span


def start_tool_span(name: str, call_id: str, parent: Any) -> Any:
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"execute_tool {name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", name)
    span.set_attribute("gen_ai.tool.call.id", call_id)
    return span


def usage_of(value: Any) -> tuple[dict[str, int], SpanUsage]:
    value = value if isinstance(value, dict) else {}
    raw = {
        "input_tokens": number_or_zero(
            value.get("inputTokens", value.get("input_tokens"))
        ),
        "output_tokens": number_or_zero(
            value.get("outputTokens", value.get("output_tokens"))
        ),
        "total_tokens": number_or_zero(
            value.get("totalTokens", value.get("total_tokens"))
        ),
    }
    read = number_or_zero(
        value.get("cacheReadInputTokens", value.get("cache_read_input_tokens"))
    )
    write = number_or_zero(
        value.get("cacheWriteInputTokens", value.get("cache_write_input_tokens"))
    )
    if "cacheReadInputTokens" in value or "cache_read_input_tokens" in value:
        raw["cache_read_input_tokens"] = read
    if "cacheWriteInputTokens" in value or "cache_write_input_tokens" in value:
        raw["cache_creation_input_tokens"] = write
    return raw, SpanUsage(
        input=raw["input_tokens"] + read + write,
        output=raw["output_tokens"],
        cache_read=read,
        cache_creation=write,
    )


def finish_reason(reason: str) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "limit_output_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
        "content_filtered": "content_filter",
    }.get(reason.lower(), reason)


def finish_model_span(
    span: Any, config: AiConfigRep, usage: SpanUsage, reason: str | None
) -> None:
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_id(config))
    if reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason(reason)])
    set_usage_span_attributes(span, usage)
    span.set_status(StatusCode.OK)
    span.end()


def finish_root_span(span: Any, config: AiConfigRep, usage: SpanUsage) -> None:
    if span is not None:
        span.set_attribute("gen_ai.response.model", model_id(config))
        set_usage_span_attributes(span, usage)


def succeed_span(span: Any) -> None:
    if span is not None:
        span.set_status(StatusCode.OK)
        span.end()


def mark_ok(span: Any) -> None:
    if span is not None:
        span.set_status(StatusCode.OK)


def fail_span(span: Any, error: BaseException, ended: set[int] | None = None) -> None:
    if span is None:
        return
    span.record_exception(error)
    span.set_status(StatusCode.ERROR, str(error))
    if ended is None:
        span.end()
    else:
        from launchdarkly_ai_server import end_span_once

        end_span_once(span, ended)


def content_parts(content: Any) -> list[SpanMessagePart]:
    if not isinstance(content, list):
        return []
    parts: list[SpanMessagePart] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(SpanMessagePart(type="text", content=str(block["text"])))
        elif "toolUse" in block:
            use = block["toolUse"]
            parts.append(
                SpanMessagePart(
                    type="tool_call",
                    id=use.get("toolUseId"),
                    name=str(use.get("name", "")),
                    arguments=use.get("input"),
                )
            )
        elif "toolResult" in block:
            result = block["toolResult"]
            parts.append(
                SpanMessagePart(
                    type="tool_call_response",
                    id=result.get("toolUseId"),
                    result=result.get("content"),
                )
            )
    return parts


def span_messages(messages: list[dict[str, Any]]) -> list[SpanMessage]:
    return [
        SpanMessage(
            role=str(message.get("role", "")),
            parts=content_parts(message.get("content")),
        )
        for message in messages
    ]
