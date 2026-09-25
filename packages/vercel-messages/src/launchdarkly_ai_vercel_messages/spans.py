from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanUsage,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_usage_span_attributes,
)

TRACER_NAME = "@launchdarkly/ai-vercel-messages"


def model_name(config: AiConfigRep) -> str:
    return str(config.get("model", {}).get("name") or "")


def serving_provider(config: AiConfigRep) -> str:
    return str(config.get("provider", {}).get("name") or "vercel").lower()


def start_root_span(config: AiConfigRep, variables: dict[str, Any]) -> Any:
    span = trace.get_tracer(TRACER_NAME).start_span("invoke_agent")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    set_model_identity_attributes(span, serving_provider(config), model_name(config))
    set_ld_span_attributes(span, variables)
    return span


def parent_context_of(span: Any) -> Any:
    return trace.set_span_in_context(span) if span is not None else None


def start_model_span(config: AiConfigRep, parent: Any) -> Any:
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"chat {model_name(config)}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(span, serving_provider(config), model_name(config))
    return span


def start_tool_span(name: str, call_id: str, parent: Any) -> Any:
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"execute_tool {name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", name)
    span.set_attribute("gen_ai.tool.call.id", call_id)
    return span


def finish_root_span(span: Any, response_model: str, usage: SpanUsage) -> None:
    span.set_attribute("gen_ai.response.model", response_model)
    set_usage_span_attributes(span, usage)


def finish_model_span(span: Any, response_model: str, usage: SpanUsage) -> None:
    span.set_attribute("gen_ai.response.model", response_model)
    set_usage_span_attributes(span, usage)


def succeed_span(span: Any) -> None:
    span.set_status(StatusCode.OK)
    span.end()


def mark_ok(span: Any) -> None:
    span.set_status(StatusCode.OK)


def fail_span(span: Any, error: BaseException, tracker: set[int] | None = None) -> None:
    span.record_exception(error)
    span.set_status(StatusCode.ERROR, str(error))
    if tracker is None or id(span) not in tracker:
        if tracker is not None:
            tracker.add(id(span))
        span.end()
