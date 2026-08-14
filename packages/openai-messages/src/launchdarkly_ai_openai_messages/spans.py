"""Span construction for the OpenAI messages handler.

Separate from ``handler.py`` so the span shape is readable on its own, and so the tool loop reads as
the tool loop rather than as span bookkeeping with a provider call in the middle.

The shape is ``invoke_agent`` root, one ``chat {model}`` child per model turn, one
``execute_tool {name}`` child per tool call. Tool spans are siblings of the ``chat`` span, not
children of it: both take the same parent context, which is the root's. See TELEMETRY-CONTRACT.md
section 1.
"""

from __future__ import annotations

import json
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanMessage,
    SpanMessagePart,
    SpanUsage,
    ToolDefinitionInput,
    number_or_zero,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_output_content_attributes,
    set_usage_span_attributes,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised by the no-OTel install path
    _HAS_OTEL = False

TRACER_NAME = "@launchdarkly/ai-openai-messages"

#: OpenAI serves every model behind this handler, so the provider name is a constant.
PROVIDER = "openai"


def tool_arguments(raw: Any) -> Any:
    """The object a Responses tool call denotes, given the JSON string the API sends.

    ``function_call.arguments`` arrives as an opaque JSON string, while every other handler puts a
    parsed object on a ``tool_call`` part. Passing the string through left the content carriers
    encoding it a second time, so an OpenAI span described the same call differently from an
    Anthropic one. See TELEMETRY-CONTRACT.md section 12: arguments hold the object the provider
    means, not the encoding it chose.

    The handler parses this same string to call the tool, so the object is the shape the provider
    means. A string that does not parse comes back verbatim rather than raising: a truncated stream
    is worth reporting as it arrived, and raising inside the telemetry path would end a run the
    provider has already billed.
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def model_name(config: AiConfigRep) -> str:
    return str(config.get("model", {}).get("name", ""))


def _attr(obj: Any, name: str) -> Any:
    """Reads a field off a provider object or a plain dict, whichever the caller holds.

    The Responses API hands back objects; input items built by the handler and by the tool loop are
    plain dicts, and both shapes reach these converters.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# ─── Span starts ─────────────────────────────────────────────────────────────


def start_root_span(config: AiConfigRep, variables: dict[str, Any]) -> Any:
    """Opens the ``invoke_agent`` root and returns it, or ``None`` when OTel is absent.

    The root is the only span carrying ``launchdarkly.*`` and the ``feature_flag`` event, so it is
    the span a config-scoped query finds. Child spans must not carry them.
    """
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span("invoke_agent")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    set_model_identity_attributes(span, PROVIDER, model_name(config))
    set_ld_span_attributes(span, variables)
    return span


def parent_context_of(span: Any) -> Any:
    """The context a child span should be parented to.

    Explicit rather than a bare current context: the current context only carries this span while a
    context manager has attached it, and these handlers open a plain span rather than an active one,
    so a host app that installs its own tracer provider would otherwise get a flat trace.
    """
    if not _HAS_OTEL or span is None:
        return None
    return trace.set_span_in_context(span)


def start_model_span(config: AiConfigRep, parent: Any) -> Any:
    """Opens one ``chat {model}`` span for one model turn.

    Named after the *requested* model, like every other handler's ``chat`` span. The value written
    to ``gen_ai.response.model`` when the turn finishes is the model that actually answered; see
    :func:`finish_model_span` and TELEMETRY-CONTRACT.md section 2a.
    """
    if not _HAS_OTEL:
        return None
    name = model_name(config)
    span = trace.get_tracer(TRACER_NAME).start_span(f"chat {name}", context=parent)
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(span, PROVIDER, name)
    return span


def start_tool_span(tool_name: str, call_id: str, parent: Any) -> Any:
    """Opens one ``execute_tool {name}`` span for one tool call."""
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"execute_tool {tool_name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("gen_ai.tool.call.id", call_id)
    return span


# ─── Span finishes ───────────────────────────────────────────────────────────


def finish_root_span(span: Any, response_model: str, run_usage: SpanUsage) -> None:
    """Writes the run-level identity and token totals onto the root.

    ``response_model`` is the model that answered, not the one requested: OpenAI resolves an alias
    such as ``gpt-4o`` to a dated snapshot, and the ``chat`` children already report the real value.
    A root copying ``config.model.name`` would contradict its own children. The caller supplies the
    fallback to the requested name; see TELEMETRY-CONTRACT.md section 2a.
    """
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", response_model)
    set_usage_span_attributes(span, run_usage)


def finish_model_span(
    span: Any,
    response_model: str,
    usage: SpanUsage,
    finish_reason: str | None = None,
) -> None:
    """Ends one ``chat`` span successfully. *finish_reason* arrives already derived."""
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", response_model)
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    set_usage_span_attributes(span, usage)
    span.set_status(SpanStatusCode.OK)
    span.end()


def succeed_span(span: Any) -> None:
    """Marks a span OK and ends it, for spans with nothing else to report."""
    if span is None:
        return
    span.set_status(SpanStatusCode.OK)
    span.end()


def mark_ok(span: Any) -> None:
    """Marks a span OK without ending it.

    The streaming path needs this: its ``finally`` owns every end, through ``end_span_once``, so a
    success tail that ended the span itself would end it twice.
    """
    if span is None:
        return
    span.set_status(SpanStatusCode.OK)


def fail_span(span: Any, error: BaseException, tracker: set[int] | None = None) -> None:
    """Records the exception, sets ERROR, and ends the span.

    *tracker* is passed only from the streaming path, where a ``finally`` may race this to the same
    span; elsewhere there is exactly one end and the tracker is unnecessary.
    """
    if span is None:
        return
    span.record_exception(error)
    span.set_status(SpanStatusCode.ERROR, str(error))
    if tracker is not None:
        from launchdarkly_ai_server import end_span_once

        end_span_once(span, tracker)
    else:
        span.end()


# ─── Usage ───────────────────────────────────────────────────────────────────


def to_span_usage(usage: Any) -> SpanUsage:
    """This turn's usage as a ``SpanUsage``, with OpenAI's cache rule applied.

    OpenAI reports cached tokens *within* the input total (a subset), so unlike Anthropic they are
    not added on top: ``cached_tokens`` is surfaced as ``cache_read`` for cross-handler parity only.
    OpenAI has no cache-creation concept, so that count is always 0.
    """
    details = _attr(usage, "input_tokens_details")
    return SpanUsage(
        input=number_or_zero(_attr(usage, "input_tokens")),
        output=number_or_zero(_attr(usage, "output_tokens")),
        cache_read=number_or_zero(_attr(details, "cached_tokens")),
        cache_creation=0,
    )


# ─── Finish reasons ──────────────────────────────────────────────────────────


def finish_reason_of(response: Any) -> str | None:
    """Maps a Responses result onto semconv's ``finish_reasons`` vocabulary.

    The Responses API has no per-message finish reason of its own: it reports a run ``status`` plus,
    on an incomplete run, a machine-readable cause. The two OpenAI handlers do not use the shared
    mapping table at all; see TELEMETRY-CONTRACT.md section 5a.

    The function-call check comes first on purpose. A live seven-turn capture put status
    ``completed`` on every turn, including the six that stopped to call a tool, so status alone made
    the attribute worthless.
    """
    output = _attr(response, "output") or []
    if any(_attr(item, "type") == "function_call" for item in output):
        return "tool_calls"
    status = _attr(response, "status")
    if status == "incomplete":
        details = _attr(response, "incomplete_details")
        reason = _attr(details, "reason") if details is not None else None
        return "length" if reason == "max_output_tokens" else "content_filter"
    if status == "completed":
        return "stop"
    return None


# ─── Provider shapes as span shapes ──────────────────────────────────────────


def split_input_messages(items: list[Any]) -> tuple[str | None, list[SpanMessage]]:
    """Splits the Responses input list into system instructions and conversation turns.

    The system message is lifted out so it lands on ``gen_ai.system_instructions`` rather than being
    buried mid-conversation; ``set_input_content_attributes`` puts it back as message 0 of the flat
    carrier, which has no separate slot for it.
    """
    system: list[str] = []
    messages: list[SpanMessage] = []

    for raw in items:
        role = _attr(raw, "role")
        if role in ("system", "developer"):
            system.append(str(_attr(raw, "content") or ""))
            continue

        item_type = _attr(raw, "type")
        if item_type == "function_call_output":
            call_id = _attr(raw, "call_id")
            messages.append(
                SpanMessage(
                    role="tool",
                    parts=[
                        SpanMessagePart(
                            type="tool_call_response",
                            id=call_id if isinstance(call_id, str) else None,
                            result=_attr(raw, "output"),
                        )
                    ],
                )
            )
            continue
        if item_type == "function_call":
            call_id = _attr(raw, "call_id")
            messages.append(
                SpanMessage(
                    role="assistant",
                    parts=[
                        SpanMessagePart(
                            type="tool_call",
                            id=call_id if isinstance(call_id, str) else None,
                            name=str(_attr(raw, "name") or ""),
                            arguments=tool_arguments(_attr(raw, "arguments")),
                        )
                    ],
                )
            )
            continue

        content = _attr(raw, "content")
        text = content if isinstance(content, str) else json.dumps(content)
        messages.append(
            SpanMessage(
                role=role if isinstance(role, str) else "user",
                parts=[SpanMessagePart(type="text", content=text)],
            )
        )

    return ("\n".join(system) if system else None, messages)


def output_item_parts(item: Any) -> list[SpanMessagePart]:
    """Converts one Responses output item into canonical span message parts."""
    item_type = _attr(item, "type")
    if item_type == "function_call":
        call_id = _attr(item, "call_id")
        return [
            SpanMessagePart(
                type="tool_call",
                id=call_id if isinstance(call_id, str) else None,
                name=str(_attr(item, "name") or ""),
                arguments=tool_arguments(_attr(item, "arguments")),
            )
        ]
    if item_type == "reasoning":
        summary = _attr(item, "summary")
        text = (
            "\n".join(str(_attr(entry, "text") or "") for entry in summary)
            if isinstance(summary, list)
            else ""
        )
        return [SpanMessagePart(type="reasoning", content=text)] if text else []

    content = _attr(item, "content")
    if not isinstance(content, list):
        return []
    return [
        SpanMessagePart(type="text", content=str(_attr(block, "text") or ""))
        for block in content
        if _attr(block, "type") == "output_text"
    ]


def set_response_output_content(span: Any, capture: bool, response: Any) -> None:
    """Records what the model produced on this turn, gated on *capture*."""
    if not capture:
        return
    finish_reason = finish_reason_of(response)
    output = _attr(response, "output") or []
    messages = [
        SpanMessage(
            role=str(_attr(item, "role") or "assistant"),
            parts=output_item_parts(item),
            finish_reason=finish_reason,
        )
        for item in output
    ]
    set_output_content_attributes(span, capture, messages)


def to_tool_definitions(tools: list[dict[str, Any]]) -> list[ToolDefinitionInput]:
    """The catalog as sent, so the span reports what the model could actually call."""
    return [
        ToolDefinitionInput(
            name=str(t.get("name", "")),
            description=t.get("description"),
            parameters=t.get("parameters"),
        )
        for t in tools
    ]
