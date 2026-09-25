"""Span construction for the Google ADK agents handler.

The shape is ``invoke_agent`` root, one ``chat {model}`` child per model turn, one
``execute_tool {name}`` child per tool call. Tool spans are siblings of the ``chat`` span:
both take the root's context. ``gen_ai.system`` is the framework (``google_adk``).
``gen_ai.provider.name`` is who served the model. Google-family providers normalize to
``gcp.gemini``.
"""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanUsage,
    end_span_once,
    number_or_zero,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_usage_span_attributes,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    SpanStatusCode = None  # type: ignore[assignment,misc]
    _HAS_OTEL = False

TRACER_NAME = "@launchdarkly/ai-google-adk-agents"

_GOOGLE_PROVIDERS = frozenset(
    {"google", "gemini", "vertex", "google-genai", "gcp.gemini"}
)


def model_name(config: AiConfigRep) -> str:
    return str((config.get("model") or {}).get("name") or "")


def serving_provider(config: AiConfigRep) -> str:
    """Who served the model, for ``gen_ai.provider.name``.

    Google, Gemini, and Vertex are one provider in the semconv enum. Every other
    configured name is passed through lower-cased. An empty name stays on Gemini,
    which is this handler's default transport.
    """
    name = str((config.get("provider") or {}).get("name") or "").lower()
    if not name or name in _GOOGLE_PROVIDERS:
        return "gcp.gemini"
    return name


def start_root_span(config: AiConfigRep, variables: dict[str, Any]) -> Any:
    """Opens the ``invoke_agent`` root. It is the only span with LaunchDarkly identity."""
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span("invoke_agent")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    set_model_identity_attributes(
        span,
        serving_provider(config),
        model_name(config),
        legacy_system="google_adk",
    )
    set_ld_span_attributes(span, variables)
    return span


def parent_context_of(span: Any) -> Any:
    if not _HAS_OTEL or span is None:
        return None
    return trace.set_span_in_context(span)


def start_model_span(config: AiConfigRep, parent: Any) -> Any:
    """Opens one ``chat {model}`` span. Children do not repeat LaunchDarkly identity."""
    if not _HAS_OTEL:
        return None
    name = model_name(config)
    span = trace.get_tracer(TRACER_NAME).start_span(f"chat {name}", context=parent)
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(
        span, serving_provider(config), name, legacy_system="google_adk"
    )
    return span


def start_tool_span(tool_name: str, tool_call_id: str, parent: Any) -> Any:
    """Opens one ``execute_tool {name}`` span."""
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"execute_tool {tool_name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("gen_ai.tool.call.id", tool_call_id)
    return span


def span_usage_from_counts(input_tokens: int, output_tokens: int) -> SpanUsage:
    return SpanUsage(input=input_tokens, output=output_tokens)


def finish_root_span(span: Any, config: AiConfigRep, usage: SpanUsage) -> None:
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    set_usage_span_attributes(span, usage)


def finish_model_span(
    span: Any,
    config: AiConfigRep,
    usage: SpanUsage | None = None,
) -> None:
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    set_usage_span_attributes(span, usage or SpanUsage())
    if SpanStatusCode is not None:
        span.set_status(SpanStatusCode.OK)
    span.end()


def succeed_span(span: Any) -> None:
    if span is None:
        return
    if SpanStatusCode is not None:
        span.set_status(SpanStatusCode.OK)
    span.end()


def mark_ok(span: Any) -> None:
    if span is None or SpanStatusCode is None:
        return
    span.set_status(SpanStatusCode.OK)


def fail_span(span: Any, error: BaseException, tracker: set[int] | None = None) -> None:
    if span is None:
        return
    span.record_exception(error)
    if SpanStatusCode is not None:
        span.set_status(SpanStatusCode.ERROR, str(error))
    if tracker is not None:
        end_span_once(span, tracker)
    else:
        span.end()


def abandon_open_spans(spans: list[Any], ended: set[int] | None = None) -> None:
    """Ends spans still open when a consumer walks away. Does not record a failure."""
    tracker = ended if ended is not None else set()
    for span in spans:
        end_span_once(span, tracker, abandoned=True)


def usage_counts(metadata: Any) -> tuple[int, int, int]:
    """ADK ``usage_metadata`` as ``(input, output, total)``. Missing metadata is zeros."""
    if metadata is None:
        return 0, 0, 0
    prompt = _field(metadata, "prompt_token_count", "promptTokenCount")
    candidates = _field(metadata, "candidates_token_count", "candidatesTokenCount")
    total = _field(metadata, "total_token_count", "totalTokenCount")
    if total == 0 and (prompt or candidates):
        total = prompt + candidates
    return prompt, candidates, total


def _field(metadata: Any, *keys: str) -> int:
    for key in keys:
        if isinstance(metadata, dict):
            if key in metadata:
                return number_or_zero(metadata[key])
            continue
        if hasattr(metadata, key):
            return number_or_zero(getattr(metadata, key))
    return 0
