"""Span construction for the OpenAI Agents handler.

Separate from ``handler.py`` so the span shape is readable on its own. The shape is
``invoke_agent`` root, one ``chat {model}`` child per model turn, one ``execute_tool {name}``
child per tool call. Tool spans are siblings of the ``chat`` span, not children of it: both take
the same parent context, which is the root's. See TELEMETRY-CONTRACT.md section 1.

``gen_ai.response.model`` is the *requested* name everywhere on this handler, on both the root and
every ``chat`` span. The openai-agents SDK never hands back a resolved model name the way the
Responses API does for ``openai-messages``, and TELEMETRY-CONTRACT.md section 2a is explicit that
this handler must not invent that behaviour.

Finish reasons are derived, not mapped, because the Responses API this handler sits on has no
per-message ``finish_reason`` field. See :func:`derive_finish_reason` for the important limitation
this port has relative to TELEMETRY-CONTRACT.md section 5a: only two of its four steps are
reachable through the ``openai-agents`` Python SDK's public ``ModelResponse``.
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
    set_usage_span_attributes,
    text_message,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised by the no-OTel install path
    _HAS_OTEL = False

TRACER_NAME = "@launchdarkly/ai-openai-agents"

PROVIDER = "openai"


def tool_arguments(raw: Any) -> Any:
    """The object an Agents tool call denotes, given the JSON string the SDK carries.

    ``function_call.arguments`` and ``context.tool_arguments`` arrive as an opaque JSON string, while
    every other handler puts a parsed object on a ``tool_call`` part. Passing the string through left
    the content carriers encoding it a second time, so an OpenAI span described the same call
    differently from an Anthropic one. See TELEMETRY-CONTRACT.md section 12: arguments hold the
    object the provider means, not the encoding it chose.

    A string that does not parse comes back verbatim rather than raising: a truncated stream is worth
    reporting as it arrived, and raising inside the telemetry path would end a run the provider has
    already billed.
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def model_name(config: AiConfigRep) -> str:
    return str(config.get("model", {}).get("name", ""))


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

    Explicit rather than a bare current context: these handlers open a plain span rather than an
    active one, so there is no ambient span for a bare ``context.active()`` to inherit.
    """
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
    set_model_identity_attributes(span, PROVIDER, name)
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


# ─── Span finishes ───────────────────────────────────────────────────────────


def finish_root_span(span: Any, config: AiConfigRep, usage: SpanUsage) -> None:
    """Writes the run-level identity and token totals onto the root.

    ``gen_ai.response.model`` is the requested name here: this handler never resolves an answering
    model. See TELEMETRY-CONTRACT.md section 2a.
    """
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
    # A list because one response may hold several choices; the Responses API returns one.
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
    """Marks a span OK without ending it. The streaming path's ``finally`` owns every end."""
    if span is None:
        return
    span.set_status(SpanStatusCode.OK)


def fail_span(span: Any, error: BaseException, tracker: set[int] | None = None) -> None:
    """Records the exception, sets ERROR, and ends the span.

    *tracker* is passed only from the streaming path, where a ``finally`` may race this to the
    same span; elsewhere there is exactly one end and the tracker is unnecessary.
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


# ─── Usage ────────────────────────────────────────────────────────────────────


def to_span_usage(usage: Any) -> SpanUsage:
    """One turn's ``agents.Usage`` (or the run-level equivalent) as a ``SpanUsage``.

    OpenAI reports cached tokens *inside* the input total, unlike Anthropic's separate buckets, so
    nothing is added on top here: TELEMETRY-CONTRACT.md section 8 puts ``openai-agents`` in the
    "pass through" row. OpenAI has no cache-creation concept, so that figure is always 0.

    Accepts ``None`` so a turn (or an error path) that never reported usage still yields all-zero
    numbers rather than raising.
    """
    if usage is None:
        return SpanUsage()
    details = _attr(usage, "input_tokens_details")
    cached = _attr(details, "cached_tokens") if details is not None else None
    return SpanUsage(
        input=number_or_zero(_attr(usage, "input_tokens")),
        output=number_or_zero(_attr(usage, "output_tokens")),
        cache_read=number_or_zero(cached),
        cache_creation=0,
    )


# ─── Finish reasons ───────────────────────────────────────────────────────────


def derive_finish_reason(output: list[Any] | None) -> str | None:
    """Derives a semconv finish reason from one turn's output items.

    TELEMETRY-CONTRACT.md section 5a specifies a four-step precedence for the Responses API:

    1. Any output item is a function call → ``tool_calls``.
    2. Otherwise, the response status is ``incomplete`` → ``length`` or ``content_filter``,
       depending on ``incomplete_details.reason``.
    3. Otherwise, the response status is ``completed`` → ``stop``.
    4. Otherwise, no attribute.

    Only step 1 and a version of step 3 are implemented. The openai-agents Python SDK's public
    ``Model.get_response`` return type, ``agents.items.ModelResponse``, carries only ``output``,
    ``usage``, ``response_id`` and ``request_id``. Unlike the TypeScript ``@openai/agents``
    package, whose ``ModelResponse.providerData`` holds the entire raw OpenAI response (status and
    ``incomplete_details`` included), the Python SDK's ``OpenAIResponsesModel.get_response``
    discards the raw response's ``status`` before constructing ``ModelResponse``, on both the
    blocking and the streaming path. No wrapper built on top of the public ``Model`` or
    ``ModelProvider`` interfaces, nor a ``RunHooks`` callback, can recover it, because the loss
    happens inside ``OpenAIResponsesModel`` itself, one layer below anything this handler can
    intercept without reimplementing the model call.

    So step 2 cannot be implemented without either forking the vendor SDK or re-issuing the
    Responses API call ourselves, and the latter would violate "do not change what the model is
    sent." Treated as an open question in the report rather than guessed at: a turn that is cut off
    by length or moderation reports ``stop`` here, identically to one that finished normally,
    which is wrong but is the closest available approximation given the data this SDK exposes.
    """
    items = output or []
    if any(_attr(item, "type") == "function_call" for item in items):
        return "tool_calls"
    if items:
        return "stop"
    return None


# ─── Provider shapes as span shapes ──────────────────────────────────────────


def _attr(obj: Any, name: str) -> Any:
    """Reads a field off a provider object or a plain dict, whichever the caller holds.

    Request-side items are typically plain dicts (``TResponseInputItem`` is a ``TypedDict``);
    response-side items are typically pydantic models. Both reach these converters.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _text_of_content_blocks(content: Any) -> str:
    """Flattens a Responses API ``content`` list (or a bare string) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        block_type = _attr(block, "type")
        if block_type in ("input_text", "output_text", "text"):
            parts.append(str(_attr(block, "text") or ""))
    return "".join(parts)


def _reasoning_text(item: Any) -> str:
    """A reasoning item's summary, joined. Falls back to ``content`` when ``summary`` is absent."""
    summary = _attr(item, "summary")
    if isinstance(summary, list) and summary:
        return "\n".join(str(_attr(entry, "text") or "") for entry in summary)
    return _text_of_content_blocks(_attr(item, "content"))


def item_parts(item: Any) -> list[SpanMessagePart]:
    """Converts one Responses API item (request or response side) into canonical span parts.

    Item kinds a span has no part for are dropped rather than emitted malformed.
    """
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
    if item_type == "function_call_output":
        call_id = _attr(item, "call_id")
        return [
            SpanMessagePart(
                type="tool_call_response",
                id=call_id if isinstance(call_id, str) else None,
                result=_attr(item, "output"),
            )
        ]
    if item_type == "reasoning":
        text = _reasoning_text(item)
        return [SpanMessagePart(type="reasoning", content=text)] if text else []

    content = _attr(item, "content")
    if isinstance(content, str):
        return [SpanMessagePart(type="text", content=content)] if content else []
    text = _text_of_content_blocks(content)
    return [SpanMessagePart(type="text", content=text)] if text else []


def to_request_span_messages(input_items: Any) -> list[SpanMessage]:
    """A model turn's input, whether the SDK passed a bare string or a list of items."""
    if isinstance(input_items, str):
        return [text_message("user", input_items)] if input_items else []
    if not isinstance(input_items, list):
        return []
    messages: list[SpanMessage] = []
    for item in input_items:
        role = _attr(item, "role")
        if not isinstance(role, str):
            role = (
                "tool" if _attr(item, "type") == "function_call_output" else "assistant"
            )
        messages.append(SpanMessage(role=role, parts=item_parts(item)))
    return messages


def to_response_span_messages(
    output_items: list[Any] | None, finish_reason: str | None
) -> list[SpanMessage]:
    """One turn's output items as canonical span messages.

    ``finish_reason`` is attached only to the last message, mirroring the shape of a single
    Responses API turn: one logical assistant turn, whatever it took to produce it.
    """
    items = output_items or []
    messages: list[SpanMessage] = []
    for index, item in enumerate(items):
        role = _attr(item, "role")
        if not isinstance(role, str):
            role = "assistant"
        is_last = index == len(items) - 1
        messages.append(
            SpanMessage(
                role=role,
                parts=item_parts(item),
                finish_reason=finish_reason if is_last else None,
            )
        )
    return messages


def to_tool_definitions(tools: list[Any]) -> list[ToolDefinitionInput]:
    """The catalog as the ``Agent`` actually holds it, so the span reports what the model could
    actually call.

    Filtered to ``FunctionTool``-shaped entries (``name`` / ``description`` /
    ``params_json_schema``): the Agents SDK also allows hosted and computer tools this handler
    never builds, and those have no JSON Schema to report.
    """
    definitions: list[ToolDefinitionInput] = []
    for tool in tools:
        params = _attr(tool, "params_json_schema")
        name = _attr(tool, "name")
        if not isinstance(name, str):
            continue
        definitions.append(
            ToolDefinitionInput(
                name=name,
                description=_attr(tool, "description"),
                parameters=params,
            )
        )
    return definitions
