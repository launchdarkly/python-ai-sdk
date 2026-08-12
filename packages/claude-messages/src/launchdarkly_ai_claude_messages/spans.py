"""Span construction for the Claude messages handler.

Separate from ``handler.py`` so the span shape is readable on its own, and so the tool loop reads as
the tool loop rather than as span bookkeeping with a provider call in the middle.

The shape is ``invoke_agent`` root, one ``chat {model}`` child per model turn, one
``execute_tool {name}`` child per tool call. Tool spans are siblings of the ``chat`` span, not
children of it: both take the same parent context, which is the root's. See TELEMETRY-CONTRACT.md
section 1.
"""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanMessage,
    SpanMessagePart,
    SpanUsage,
    ToolDefinitionInput,
    add_cached_tokens_to_input,
    number_or_zero,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_usage_span_attributes,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised by the no-OTel install path
    _HAS_OTEL = False

TRACER_NAME = "@launchdarkly/ai-claude-messages"

#: Anthropic serves every model behind this handler, so the provider name is a constant.
PROVIDER = "anthropic"


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

    Explicit rather than a bare current context: the current context only carries this span while a
    context manager has attached it, and these handlers open a plain span rather than an active one,
    so a host app that installs its own tracer provider would otherwise get a flat trace.
    """
    if not _HAS_OTEL or span is None:
        return None
    # `set_span_in_context` with no context argument reads the current one, which is what the
    # TypeScript SDK's `trace.setSpan(context.active(), span)` does.
    return trace.set_span_in_context(span)


def start_model_span(config: AiConfigRep, parent: Any) -> Any:
    """Opens one ``chat {model}`` span for one model turn.

    The semantic conventions name an inference span ``{operation} {model}``, so the model belongs in
    the name and not only in ``gen_ai.request.model``. A bare ``chat`` aggregates more neatly but
    tells a reader nothing about which model ran, which matters most in exactly the case this span
    exists for: a multi-turn run that switches models partway through.
    """
    if not _HAS_OTEL:
        return None
    name = model_name(config)
    span = trace.get_tracer(TRACER_NAME).start_span(f"chat {name}", context=parent)
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(span, PROVIDER, name)
    return span


def start_tool_span(tool_name: str, tool_use_id: str, parent: Any) -> Any:
    """Opens one ``execute_tool {name}`` span for one tool call."""
    if not _HAS_OTEL:
        return None
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"execute_tool {tool_name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("gen_ai.tool.call.id", tool_use_id)
    return span


# ─── Span finishes ───────────────────────────────────────────────────────────


def finish_root_span(span: Any, config: AiConfigRep, raw_usage: dict[str, Any]) -> None:
    """Writes the run-level identity and token totals onto the root.

    The per-turn ``chat`` children carry the same attributes for their own turn, but the root is the
    only span a config-scoped query finds, so leaving the totals off it means such a query returns
    nothing at all: summing the children requires having already found them.

    ``gen_ai.response.model`` is the requested name here. Anthropic does not resolve an alias to a
    different snapshot, so there is no other value to report. See TELEMETRY-CONTRACT.md section 2a.
    """
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    set_usage_span_attributes(span, add_cached_tokens_to_input(raw_usage))


def finish_model_span(
    span: Any,
    config: AiConfigRep,
    raw_usage: dict[str, Any],
    finish_reason: str | None = None,
) -> None:
    """Ends one ``chat`` span successfully. *finish_reason* arrives already mapped."""
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    # A list because one response may hold several choices; Anthropic returns one.
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    set_usage_span_attributes(span, add_cached_tokens_to_input(raw_usage))
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
    success tail that ended the span itself would end it twice. Ending twice is silently ignored by
    the OTel SDK but recorded as a diagnostic error, and would also hide a genuine leak.
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


# ─── Provider shapes as span shapes ──────────────────────────────────────────


def to_span_parts(content: Any) -> list[SpanMessagePart]:
    """Converts Anthropic content blocks into canonical span parts.

    A plain string is one text part. Unknown block types are dropped rather than guessed at.
    """
    if isinstance(content, str):
        return [SpanMessagePart(type="text", content=content)]
    if not isinstance(content, list):
        return []

    parts: list[SpanMessagePart] = []
    for block in content:
        block_type = _attr(block, "type")
        if block_type == "text":
            parts.append(
                SpanMessagePart(type="text", content=str(_attr(block, "text") or ""))
            )
        elif block_type == "thinking":
            parts.append(
                SpanMessagePart(
                    type="reasoning", content=str(_attr(block, "thinking") or "")
                )
            )
        elif block_type == "tool_use":
            block_id = _attr(block, "id")
            parts.append(
                SpanMessagePart(
                    type="tool_call",
                    id=block_id if isinstance(block_id, str) else None,
                    name=str(_attr(block, "name") or ""),
                    arguments=_attr(block, "input"),
                )
            )
        elif block_type == "tool_result":
            use_id = _attr(block, "tool_use_id")
            parts.append(
                SpanMessagePart(
                    type="tool_call_response",
                    id=use_id if isinstance(use_id, str) else None,
                    result=_attr(block, "content"),
                )
            )
    return parts


def to_span_messages(messages: list[dict[str, Any]]) -> list[SpanMessage]:
    return [
        SpanMessage(role=str(m.get("role", "")), parts=to_span_parts(m.get("content")))
        for m in messages
    ]


def to_tool_definitions(tools: list[dict[str, Any]]) -> list[ToolDefinitionInput]:
    """The catalog as sent, so the span reports what the model could actually call."""
    return [
        ToolDefinitionInput(
            name=str(t.get("name", "")),
            description=t.get("description"),
            parameters=t.get("input_schema"),
        )
        for t in tools
    ]


def _attr(obj: Any, name: str) -> Any:
    """Reads a field off a provider object or a plain dict, whichever the caller holds.

    The provider SDK hands back objects; the tool loop appends dicts to the same conversation, and
    both reach these converters.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# ─── The run accumulator, in Anthropic's own field names ─────────────────────


class RawRunUsage:
    """The run's accumulated usage, in Anthropic's own field names.

    Kept unfolded, with the cache figures beside the input total rather than added into it, so
    ``parse_usage`` can fold once and derive the breakdown. Pre-folding here would hide the cache
    detail from callers, and :func:`add_cached_tokens_to_input` would then count it twice.

    That is also why this is not the client package's ``RunUsage``: that one accumulates
    ``SpanUsage``, whose ``input`` is already cache-inclusive, and handing one back as this handler's
    return value would double-count the cache downstream. Named differently on purpose, because the
    two differ in exactly the way that matters.

    Created by the caller that owns the root span rather than by the tool loop, so a loop that
    raises still leaves the run's spend somewhere the root can read it.
    """

    #: Present on every total, because a completed turn always has a base count to report.
    _BASE_FIELDS = ("input_tokens", "output_tokens")
    #: Carried only once a turn actually reports one. See :meth:`add_turn`.
    _CACHE_FIELDS = ("cache_read_input_tokens", "cache_creation_input_tokens")

    def __init__(self) -> None:
        self.total: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
        self._turns = 0

    @property
    def reported(self) -> bool:
        """Whether any turn reported usage.

        Separates "no call completed" from "a call completed and reported zero". Only the second may
        reach a span: all-zero attributes assert the run cost nothing, which a run whose first call
        died mid-flight cannot claim.
        """
        return self._turns > 0

    def add_turn(self, raw_usage: dict[str, Any]) -> None:
        self._turns += 1
        for key in self._BASE_FIELDS:
            self.total[key] += number_or_zero(raw_usage.get(key))
        # A cache field appears in the total only once some turn reported one. Seeding them at zero
        # undid `raw_usage_of`, which omits the fields Anthropic did not send: the returned bag then
        # always looked cache-aware, so `parse_usage` emitted an input_details breakdown of zeros for
        # a model with no prompt caching at all. Reporting a zero cache read is a claim, and this
        # accumulator has no grounds to make it.
        for key in self._CACHE_FIELDS:
            if key in raw_usage:
                self.total[key] = number_or_zero(self.total.get(key)) + number_or_zero(
                    raw_usage.get(key)
                )


def raw_usage_of(usage: Any) -> dict[str, Any]:
    """Anthropic's usage object as a plain dict, tolerating an absent or partial one.

    Read through :func:`number_or_zero` at the point of use rather than coerced here, so an unknown
    cache spelling the provider adds later still reaches ``parse_usage``.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    out: dict[str, Any] = {}
    for field in fields:
        value = getattr(usage, field, None)
        if value is not None:
            out[field] = value
    return out


def span_usage_of(raw_usage: dict[str, Any]) -> SpanUsage:
    """This turn's usage as a ``SpanUsage``, with Anthropic's cache rule applied."""
    return add_cached_tokens_to_input(raw_usage)
