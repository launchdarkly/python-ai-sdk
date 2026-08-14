"""Span construction for the LangChain messages handler.

Separate from ``handler.py`` so the span shape is readable on its own, and so the tool loop reads as
the tool loop rather than as span bookkeeping with a provider call in the middle.

The shape is ``invoke_agent`` root, one ``chat {model}`` child per model turn, one
``execute_tool {name}`` child per tool call. Tool spans are siblings of the ``chat`` span, not
children of it: both take the same parent context, which is the root's. See TELEMETRY-CONTRACT.md
section 1.

Two provider keys, two different values, on purpose. ``gen_ai.system`` is the literal string
``langchain`` on every span this package opens, because that is what the handler shipped before
the span hierarchy landed. ``gen_ai.provider.name`` names *who served the model*, and semconv's
enum has no ``langchain`` member, so it follows :func:`serving_provider` instead: whichever chat
model class the handler actually instantiates. See TELEMETRY-CONTRACT.md section 9.
"""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanUsage,
    ToolDefinitionInput,
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

TRACER_NAME = "@launchdarkly/ai-langchain-messages"

#: The literal value ``gen_ai.system`` carries on every span this package opens. Not derived from
#: the configured provider: TypeScript's LangChain handlers keep this constant regardless of which
#: model actually served the call, because LangChain (the framework) is what shipped the span, not
#: Anthropic or OpenAI (the provider). See TELEMETRY-CONTRACT.md section 9.
LEGACY_SYSTEM = "langchain"


def model_name(config: AiConfigRep) -> str:
    return str(config.get("model", {}).get("name", ""))


def serving_provider(config: AiConfigRep) -> str:
    """The provider that actually serves the model.

    ``gen_ai.provider.name`` names who served the request, and its semconv enum has no
    ``langchain`` member, because LangChain is the framework, not the provider. This mirrors the
    choice the handler's model-resolution logic makes: ``ChatAnthropic`` for a configured provider
    of ``"anthropic"``, ``ChatOpenAI`` for everything else, including Bedrock, Azure, Cohere, a
    typo, or an unset value. Not a passthrough of the configured name. See TELEMETRY-CONTRACT.md
    section 9.
    """
    provider = str((config.get("provider") or {}).get("name") or "").lower()
    return "anthropic" if provider == "anthropic" else "openai"


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
    set_model_identity_attributes(
        span, serving_provider(config), model_name(config), LEGACY_SYSTEM
    )
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
    set_model_identity_attributes(span, serving_provider(config), name, LEGACY_SYSTEM)
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


def finish_root_span(span: Any, config: AiConfigRep, run_usage: SpanUsage) -> None:
    """Writes the run-level identity and token totals onto the root.

    ``gen_ai.response.model`` is the requested name here, unlike ``openai-messages``: LangChain does
    not hand this handler a resolved model id to report instead. See TELEMETRY-CONTRACT.md section
    2a.

    The per-turn ``chat`` children carry the same attributes for their own turn, but the root is the
    only span a config-scoped query finds, so leaving the totals off it means such a query returns
    nothing at all: summing the children requires having already found them.
    """
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    set_usage_span_attributes(span, run_usage)


def finish_model_span(
    span: Any,
    config: AiConfigRep,
    usage: SpanUsage,
    finish_reasons: list[str] | None = None,
) -> None:
    """Ends one ``chat`` span successfully.

    *usage* is the caller's responsibility to default to zeros when LangChain reported nothing: a
    ``chat`` span always writes the complete attribute set, unlike the root, which withholds it when
    no turn ever reported. See TELEMETRY-CONTRACT.md section 8.

    *finish_reasons* arrives already mapped through :func:`lang_chain_finish_reasons`.
    """
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    # A list because one response may hold several choices; the providers used here return one.
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
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


def to_tool_definitions(tools: list[dict[str, Any]]) -> list[ToolDefinitionInput]:
    """The catalog as bound to the model, so the span reports what it could actually call.

    *tools* are this package's own OpenAI-function-style tool dicts (``{"type": "function",
    "function": {...}}``), not the AI Config's tool type: see ``_build_tools`` in ``handler.py``.
    """
    return [
        ToolDefinitionInput(
            name=str(t.get("function", {}).get("name", "")),
            description=t.get("function", {}).get("description"),
            parameters=t.get("function", {}).get("parameters"),
        )
        for t in tools
    ]
