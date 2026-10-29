"""Span construction for the OpenCode handler.

``config().stream()`` is ``chunk`` then ``done``. The handler stream is the
AG-UI text and tool subset. This module opens the ``invoke_agent`` root, plus
the ``chat`` and ``execute_tool`` constructors the shared parity checks call.
A run starts the root.
"""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanUsage,
    end_span_once,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_usage_span_attributes,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised when the API package is absent
    _HAS_OTEL = False

TRACER_NAME = "@launchdarkly/ai-opencode-agents"


def model_name(config: AiConfigRep) -> str:
    if not isinstance(config, dict):
        return ""
    model = config.get("model")
    if not isinstance(model, dict):
        return ""
    name = model.get("name")
    return name if isinstance(name, str) else ""


def provider_name(config: AiConfigRep) -> str:
    """Who served the model, from the variation, or ``opencode`` when it is blank."""
    if isinstance(config, dict):
        provider = config.get("provider")
        if isinstance(provider, dict):
            name = provider.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip().lower()
    return "opencode"


def start_root_span(
    config: AiConfigRep, variables: dict[str, Any], *, capture_content: bool = False
) -> Any:
    """Opens the ``invoke_agent`` root and returns it, or ``None`` when OTel is absent."""
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span("invoke_agent")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    set_model_identity_attributes(span, provider_name(config), model_name(config))
    set_ld_span_attributes(span, variables, capture_content=capture_content)
    return span


def parent_context_of(span: Any) -> Any:
    """The context a child span should be parented to."""
    if not _HAS_OTEL or span is None:
        return None
    return trace.set_span_in_context(span)


def start_model_span(config: AiConfigRep, parent: Any) -> Any:
    """Opens one ``chat {model}`` span for one model turn."""
    if not _HAS_OTEL:
        return None
    name = model_name(config)
    span = trace.get_tracer(TRACER_NAME).start_span(f"chat {name}", context=parent)
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(span, provider_name(config), name)
    return span


def start_tool_span(tool_name: str, tool_call_id: str, parent: Any) -> Any:
    """Opens one ``execute_tool {name}`` span for one tool call."""
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"execute_tool {tool_name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("gen_ai.tool.call.id", tool_call_id)
    return span


def finish_root_span(span: Any, config: AiConfigRep, usage: SpanUsage) -> None:
    """Writes the requested model and token totals onto the root. Does not end it."""
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    set_usage_span_attributes(span, usage)


def finish_model_span(
    span: Any,
    config: AiConfigRep,
    usage: SpanUsage,
    finish_reason: str | None = None,
) -> None:
    """Ends one ``chat`` span successfully."""
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    set_usage_span_attributes(span, usage)
    span.set_status(SpanStatusCode.OK)
    span.end()


def succeed_span(span: Any) -> None:
    """Marks a span OK and ends it."""
    if span is None:
        return
    span.set_status(SpanStatusCode.OK)
    span.end()


def mark_ok(span: Any) -> None:
    """Marks a span OK without ending it. The streaming ``finally`` owns the end."""
    if span is None:
        return
    span.set_status(SpanStatusCode.OK)


def fail_span(span: Any, error: BaseException, tracker: set[int] | None = None) -> None:
    """Records the exception, sets ERROR, and ends the span."""
    if span is None:
        return
    span.record_exception(error)
    span.set_status(SpanStatusCode.ERROR, str(error))
    if tracker is not None:
        end_span_once(span, tracker)
    else:
        span.end()


#: A call id (tool spans) or message id (model spans) to its still-open span.
Spans = dict[str, Any]


def close_open_spans(tools: Spans, models: Spans, error: BaseException) -> None:
    """Fails every ``execute_tool`` and ``chat`` span still open.

    For the path where the run raises mid-stream: the reducer's own closing
    logic never got to run for a call or a message that was still in flight,
    so whatever is left in either dict would otherwise export as a span that
    never ended.
    """
    for span in list(tools.values()):
        fail_span(span, error)
    tools.clear()
    for span in list(models.values()):
        fail_span(span, error)
    models.clear()


# The parity suite's body extractor is indentation-based. A signature that
# wraps, then returns to column 0 for ``) -> None:``, is read as the whole
# body, and that body no longer contains ``abandoned=True``. Keep this
# definition on one physical line. E501 is ignored; ruff format is not.
# fmt: off
def abandon_open_spans(tools: Spans, models: Spans, ended: set[int], cancelled: bool = False) -> None:
# fmt: on
    """Ends every ``execute_tool`` and ``chat`` span still open.

    A consumer who stopped reading leaves each span UNSET and marked
    ``launchdarkly.stream.abandoned``. A ``CancelledError`` (a timeout, or
    ``task.cancel()``) passes *cancelled*: ``end_span_once`` then writes
    ``launchdarkly.run.cancelled`` and does not write the abandoned
    attribute. *cancelled* wins when both are set.
    """
    for span in list(tools.values()):
        end_span_once(span, ended, abandoned=True, cancelled=cancelled)
    tools.clear()
    for span in list(models.values()):
        end_span_once(span, ended, abandoned=True, cancelled=cancelled)
    models.clear()
