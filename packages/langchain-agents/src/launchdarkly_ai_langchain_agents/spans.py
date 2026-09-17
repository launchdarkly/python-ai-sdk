"""Span construction for the LangChain agents handler.

Separate from ``handler.py`` so the span shape is readable on its own, and so the agent-invocation
code reads as agent invocation rather than as span bookkeeping around a LangGraph call.

The shape is ``invoke_agent`` root, one ``chat {model}`` child per model turn, one
``execute_tool {name}`` child per tool call. Tool spans are siblings of the ``chat`` span, not
children of it: both take the same parent context, which is the root's. See TELEMETRY-CONTRACT.md
section 1.

Unlike the Claude and OpenAI handlers, this one does not drive the provider call itself: LangGraph's
``create_react_agent`` does, and the only hook into its lifecycle is LangChain's callback protocol.
:func:`build_span_callbacks` is therefore the center of this module: it returns a
``BaseCallbackHandler`` that opens and closes ``chat`` and ``execute_tool`` spans as LangChain
dispatches ``on_chat_model_start`` / ``on_llm_end`` / ``on_tool_start`` / ``on_tool_end`` events,
keyed by the callback ``run_id`` so concurrent tool calls do not collide.
"""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    RunUsage,
    SpanUsage,
    ToolDefinitionInput,
    create_run_usage,
    end_span_once,
    end_unfinished_spans,
    lang_chain_finish_reasons,
    lang_chain_span_messages,
    lang_chain_span_usage,
    set_input_content_attributes,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    set_usage_span_attributes,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised by the no-OTel install path
    _HAS_OTEL = False

try:
    from langchain_core.callbacks import AsyncCallbackHandler

    _HAS_LANGCHAIN_CORE = True
except ImportError:  # pragma: no cover
    AsyncCallbackHandler = object  # type: ignore[assignment,misc]
    _HAS_LANGCHAIN_CORE = False

TRACER_NAME = "@launchdarkly/ai-langchain-agents"


def model_name(config: AiConfigRep) -> str:
    return str(config.get("model", {}).get("name", ""))


def serving_provider(config: AiConfigRep) -> str:
    """The provider that actually serves the model.

    ``gen_ai.provider.name`` names who served the request, and its semconv enum has no
    ``langchain`` member: LangChain is the framework, not the provider. This mirrors the choice
    ``_make_default_chat_model`` makes, so the attribute agrees with the client that is really
    used. It is a binary choice, not a passthrough of the configured name: the configured name
    lower-cased if it equals ``anthropic``, otherwise ``openai``, no matter what else the config
    names (Bedrock, Azure, Cohere, a typo, or nothing at all).
    """
    provider = ((config.get("provider") or {}).get("name") or "").lower()
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
        span, serving_provider(config), model_name(config), legacy_system="langchain"
    )
    set_ld_span_attributes(span, variables)
    return span


def parent_context_of(span: Any) -> Any:
    """The context a child span should be parented to.

    Explicit rather than a bare current context: the current context only carries this span while
    a context manager has attached it, and these handlers open a plain span rather than an active
    one, so a host app that installs its own tracer provider would otherwise get a flat trace.
    """
    if not _HAS_OTEL or span is None:
        return None
    return trace.set_span_in_context(span)


def start_model_span(config: AiConfigRep, parent: Any) -> Any:
    """Opens one ``chat {model}`` span for one model turn.

    The semantic conventions name an inference span ``{operation} {model}``, so the model belongs
    in the name and not only in ``gen_ai.request.model``. A bare ``chat`` aggregates more neatly
    but tells a reader nothing about which model ran, which matters most in exactly the case this
    span exists for: a multi-turn run that switches models partway through.
    """
    if not _HAS_OTEL:
        return None
    name = model_name(config)
    span = trace.get_tracer(TRACER_NAME).start_span(f"chat {name}", context=parent)
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(
        span, serving_provider(config), name, legacy_system="langchain"
    )
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

    ``gen_ai.response.model`` is the requested name here, same as on a ``chat`` span: neither
    LangChain handler resolves an alias to a different snapshot. See TELEMETRY-CONTRACT.md
    section 2a.
    """
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    set_usage_span_attributes(span, run_usage)


def finish_model_span(
    span: Any,
    config: AiConfigRep,
    raw_usage: dict[str, Any] | None,
    finish_reasons: list[str] | None = None,
) -> None:
    """Ends one ``chat`` span successfully. *finish_reasons* arrives already mapped."""
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
    # A span always carries the complete usage attribute set, zeros included, unlike the run
    # accumulator: an absent attribute drops the span from every query that groups on usage.
    set_usage_span_attributes(span, lang_chain_span_usage(raw_usage) or SpanUsage())
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
        end_span_once(span, tracker)
    else:
        span.end()


# ─── LangChain result shapes as span shapes ──────────────────────────────────


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def generated_messages(output: Any) -> list[Any]:
    """The generated messages of an ``LLMResult``, which is where a turn's real output lives."""
    generations = _get(output, "generations")
    if not isinstance(generations, list):
        return []
    flat = [item for group in generations for item in group]
    return [msg for msg in (_get(gen, "message") for gen in flat) if msg is not None]


def extract_llm_usage(output: Any) -> dict[str, Any]:
    """LangChain's ``LLMResult`` carries usage either on each generation's message
    (``usage_metadata``) or, for some providers, in ``llm_output.token_usage``. Prefer the former,
    fall back to the latter.
    """
    generations = _get(output, "generations")
    flat = (
        [item for group in generations for item in group]
        if isinstance(generations, list)
        else []
    )
    for gen in flat:
        message = _get(gen, "message")
        usage_metadata = _get(message, "usage_metadata") if message else None
        if usage_metadata:
            return dict(usage_metadata)

    llm_output = _get(output, "llm_output") or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage")
    if token_usage:
        # `or` would read a real 0 as missing, and with both counts at zero the bag came back all
        # None, which lang_chain_span_usage reports as "the provider said nothing". A turn that
        # completed and reported zero is not the same as a turn that reported nothing: only the
        # second may leave the root without usage attributes.
        return {
            "input_tokens": _first_present(
                token_usage, "prompt_tokens", "input_tokens"
            ),
            "output_tokens": _first_present(
                token_usage, "completion_tokens", "output_tokens"
            ),
        }
    return {}


def _first_present(bag: dict[str, Any], *keys: str) -> Any:
    """The first key actually present, so a reported 0 is kept rather than skipped."""
    for key in keys:
        if key in bag:
            return bag[key]
    return None


def to_tool_definitions(config_tools: dict[str, Any]) -> list[ToolDefinitionInput]:
    """The catalog handed to the agent, so a ``chat`` span reports what the model could call."""
    return [
        ToolDefinitionInput(
            name=tool.get("name", name),
            description=tool.get("description"),
            parameters=tool.get("parameters"),
        )
        for name, tool in config_tools.items()
    ]


# ─── The callback bridge ──────────────────────────────────────────────────────


class SpanCallbackHandler(AsyncCallbackHandler):
    """Maps LangChain's async callback lifecycle onto the ``chat`` / ``execute_tool`` spans.

    Spans are keyed by the callback ``run_id`` (as ``str``) so concurrent tool calls, or a run that
    somehow issues concurrent model calls, do not collide.
    """

    def __init__(
        self,
        config: AiConfigRep,
        parent_context: Any,
        capture_content: bool,
        tool_definitions: list[ToolDefinitionInput],
        run_usage: RunUsage,
    ) -> None:
        self._config = config
        self._parent_context = parent_context
        self._capture_content = capture_content
        self._tool_definitions = tool_definitions
        self.run_usage = run_usage
        self.model_spans: dict[str, Any] = {}
        self.tool_spans: dict[str, Any] = {}

    def _start_model(self, run_id: Any, messages: Any = None) -> None:
        key = str(run_id)
        if key in self.model_spans:
            return
        span = start_model_span(self._config, self._parent_context)
        # Tracked before the content write, for the same reason as on_tool_start: serialising the
        # conversation can raise, and a span created but never inserted is unreachable by every
        # cleanup path there is.
        self.model_spans[key] = span
        if self._capture_content:
            # `on_chat_model_start` hands over `list[list[BaseMessage]]`, one list per generation.
            # The agent graph sends a single list; flattening keeps a multi-generation caller from
            # losing turns.
            flat: list[Any] = []
            if isinstance(messages, list):
                for group in messages:
                    if isinstance(group, list):
                        flat.extend(group)
            system_instructions, turn_messages = lang_chain_span_messages(flat)
            set_input_content_attributes(
                span,
                self._capture_content,
                system_instructions=system_instructions,
                messages=turn_messages,
                tool_definitions=self._tool_definitions,
            )

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._start_model(run_id, messages)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._start_model(run_id)

    async def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        key = str(run_id)
        span = self.model_spans.pop(key, None)
        if span is None:
            return
        # Popped, so close_open_spans can no longer reach this span: whatever happens next, this
        # method has to end it. Serialising conversation content raises on anything that is not
        # JSON-serialisable, and a raise here would otherwise leak the span with nothing tracking it.
        # Before the content write, because the provider has already billed these tokens: a content
        # failure is our problem and must not report the run as having spent less than it did.
        turn_usage_raw = extract_llm_usage(response)
        self.run_usage.add(lang_chain_span_usage(turn_usage_raw))
        try:
            if self._capture_content:
                set_output_content_attributes(
                    span,
                    self._capture_content,
                    lang_chain_span_messages(generated_messages(response))[1],
                )
            finish_model_span(
                span, self._config, turn_usage_raw, lang_chain_finish_reasons(response)
            )
        except Exception as exc:
            fail_span(span, exc)
            raise

    async def on_llm_error(
        self, error: BaseException, *, run_id: Any, **kwargs: Any
    ) -> None:
        key = str(run_id)
        span = self.model_spans.pop(key, None)
        if span is None:
            return
        fail_span(span, error)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        name: str | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = name or (serialized or {}).get("name") or "tool"
        span = start_tool_span(
            tool_name, tool_call_id or str(run_id), self._parent_context
        )
        # Tracked before the content write, not after. Serialising the arguments can raise, and a
        # span created but never inserted is unreachable by every cleanup path there is.
        self.tool_spans[str(run_id)] = span
        set_tool_call_content_attributes(
            span, self._capture_content, arguments=kwargs.get("inputs") or input_str
        )

    async def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        key = str(run_id)
        span = self.tool_spans.pop(key, None)
        if span is None:
            return
        # Same reason as on_llm_end: once popped, ending it is this method's job alone.
        try:
            set_tool_call_content_attributes(
                span, self._capture_content, result=_tool_result_text(output)
            )
            succeed_span(span)
        except Exception as exc:
            fail_span(span, exc)
            raise

    async def on_tool_error(
        self, error: BaseException, *, run_id: Any, **kwargs: Any
    ) -> None:
        key = str(run_id)
        span = self.tool_spans.pop(key, None)
        if span is None:
            return
        fail_span(span, error)

    def close_open_spans(self, error: BaseException) -> None:
        """Fails every span this run opened but never closed, for the caller's failure path."""
        for span in self.model_spans.values():
            fail_span(span, error)
        for span in self.tool_spans.values():
            fail_span(span, error)
        self.model_spans.clear()
        self.tool_spans.clear()

    def abandon_open_spans(self, ended: set[int], cancelled: bool = False) -> None:
        """Ends every span this run has open, for stream abandonment.

        Unlike :meth:`close_open_spans`, nothing failed: a consumer stopping early is normal.
        ``end_span_once`` leaves each span at ``UNSET`` and marks ``launchdarkly.stream.abandoned``,
        rather than recording a synthetic exception and setting ``ERROR``, which would make an early
        consumer stop indistinguishable from a provider failure in a trace.
        """
        for span in self.model_spans.values():
            end_span_once(span, ended, abandoned=True, cancelled=cancelled)
        for span in self.tool_spans.values():
            end_span_once(span, ended, abandoned=True, cancelled=cancelled)
        self.model_spans.clear()
        self.tool_spans.clear()

    def cancel_open_spans(self) -> None:
        """Ends every span this run has open, for a cancelled blocking call.

        ``asyncio.CancelledError`` is a ``BaseException``, so the blocking handler's
        ``except Exception`` never sees it and its ``finally`` calls this instead of
        :meth:`close_open_spans`. Uses :func:`end_unfinished_spans` so a chat or tool span left open
        by cancellation agrees with the root: marked ``launchdarkly.run.cancelled`` and left at
        ``UNSET``, rather than the ``launchdarkly.stream.abandoned`` a stopped stream gets.
        """
        end_unfinished_spans(*self.model_spans.values(), *self.tool_spans.values())
        self.model_spans.clear()
        self.tool_spans.clear()


def _tool_result_text(output: Any) -> Any:
    """A ``ToolMessage`` carries the result in ``content``; anything else passes through."""
    content = _get(output, "content")
    return content if content is not None else output


class SpanCallbacks:
    """The bundle a handler call site needs: callbacks to hand LangChain, the run's accumulated
    usage, and a way to close whatever spans a failure or an abandonment left open.

    A plain wrapper, not the callback handler itself, so the no-OTel / no-``langchain-core``
    fallback can satisfy the same shape with an empty callback list rather than a special case at
    every call site.
    """

    def __init__(
        self, callbacks: list[Any], run_usage: RunUsage, handler: Any = None
    ) -> None:
        self.callbacks = callbacks
        self.run_usage = run_usage
        self._handler = handler

    def close_open_spans(self, error: BaseException) -> None:
        if self._handler is not None:
            self._handler.close_open_spans(error)

    def abandon_open_spans(self, ended: set[int], cancelled: bool = False) -> None:
        if self._handler is not None:
            # Forwarded, not dropped. The wrapper taking the flag and discarding it would leave the
            # tool spans of a cancelled run saying abandoned while their root said cancelled.
            self._handler.abandon_open_spans(ended, cancelled=cancelled)

    def cancel_open_spans(self) -> None:
        if self._handler is not None:
            self._handler.cancel_open_spans()


def build_span_callbacks(
    config: AiConfigRep,
    parent_context: Any,
    capture_content: bool = False,
    tool_definitions: list[ToolDefinitionInput] | None = None,
) -> SpanCallbacks:
    """Builds a LangChain callback handler that maps the agent's LLM and tool lifecycle onto OTel
    spans: one ``chat`` child span per model turn, one ``execute_tool`` child span per tool call.

    Returns a :class:`SpanCallbacks` exposing ``.callbacks`` (to pass as
    ``config={"callbacks": ...}``), ``.run_usage`` (the run's accumulated :class:`SpanUsage`), and
    ``.close_open_spans(error)`` (for the caller's failure and abandonment paths, where a span this
    callback opened may never see its matching end event).
    """
    if not _HAS_OTEL or not _HAS_LANGCHAIN_CORE:
        return SpanCallbacks([], create_run_usage())
    run_usage = create_run_usage()
    handler = SpanCallbackHandler(
        config, parent_context, capture_content, tool_definitions or [], run_usage
    )
    return SpanCallbacks([handler], run_usage, handler)
