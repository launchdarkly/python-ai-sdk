"""Span construction for the Claude agents handler.

Separate from ``handler.py`` for the same reason as the ``claude-messages`` package: the span shape
reads on its own, and the message-stream loop that drives it reads as a message loop rather than as
span bookkeeping with the SDK call in the middle.

The shape is ``invoke_agent`` root, one ``chat {model}`` child per model turn, one
``execute_tool {name}`` child per tool call. Tool spans are siblings of the ``chat`` span, not
children of it: both take the same parent context, which is the root's. See TELEMETRY-CONTRACT.md
section 1.

This handler differs from ``claude-messages`` in one structural way: ``query()`` reports no request
boundaries, and the Agent SDK's own message stream is the only source of truth for what happened.
:class:`InferenceSpans` derives a `chat` span per model turn from that stream (grouped on the
Anthropic response id, read off ``AssistantMessage.message_id``), the same way
``@launchdarkly/ai-claude-agents``'s ``InferenceSpans`` groups on ``request_id``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from launchdarkly_ai_server import (
    AiConfigRep,
    SpanMessage,
    SpanMessagePart,
    ToolDefinitionInput,
    add_cached_tokens_to_input,
    end_span_once,
    number_or_zero,
    set_input_content_attributes,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_output_content_attributes,
    set_tool_definition_attributes,
    set_usage_span_attributes,
    to_semconv_finish_reason,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised by the no-OTel install path
    _HAS_OTEL = False

TRACER_NAME = "@launchdarkly/ai-claude-agents"

#: Anthropic serves every model behind this handler, so the provider name is a constant.
PROVIDER = "anthropic"

TOOL_MCP_NAME = "tool-mcp"
MCP_TOOL_PREFIX = f"mcp__{TOOL_MCP_NAME}__"


def model_name(config: AiConfigRep) -> str:
    return str(config.get("model", {}).get("name", ""))


def tool_display_name(provider_name: str) -> str:
    """The name the model saw, with the MCP wrapper this handler adds stripped back off."""
    if provider_name.startswith(MCP_TOOL_PREFIX):
        return provider_name[len(MCP_TOOL_PREFIX) :]
    return provider_name


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

    The Agent SDK's result message reports usage cumulatively for the whole run, which is exactly
    what the root wants, and the root is the only span carrying ``launchdarkly.*`` and the
    ``feature_flag`` event, so it is the span a config-scoped query finds.

    ``gen_ai.response.model`` is the *requested* name here, unlike on a ``chat`` span. See
    TELEMETRY-CONTRACT.md section 2a: this handler is one of only two where the root and a `chat`
    span disagree.
    """
    if span is None:
        return
    span.set_attribute("gen_ai.response.model", model_name(config))
    set_usage_span_attributes(span, add_cached_tokens_to_input(raw_usage))


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


def record_conversation_id(span: Any, message: Any) -> None:
    """Copies the CLI's session id onto the root span as ``gen_ai.conversation.id``.

    It is the only key LaunchDarkly's trace view groups a conversation on, and the ``init`` system
    message is where this side first learns it. The ``chat`` and ``execute_tool`` children read the
    same id off their own message and hook input, so one run does not split into several
    conversations. Set once: the id does not change within a run.
    """
    if (
        span is None
        or not isinstance(message, SystemMessage)
        or message.subtype != "init"
    ):
        return
    session_id = message.data.get("session_id")
    if session_id:
        span.set_attribute("gen_ai.conversation.id", session_id)


def record_native_tools(
    span: Any, message: Any, capture: bool, catalog: ToolCatalog
) -> None:
    """Widens the root's tool catalog once the CLI names the tools it brought itself.

    The root's catalog is written before the run starts, so a run that dies before ``init`` still
    reports the tools it was configured with. ``init`` is the first place the CLI's own tools become
    visible, and rewriting the one attribute keeps the root and the ``chat`` spans from describing
    the same catalog differently.
    """
    if not isinstance(message, SystemMessage) or message.subtype != "init":
        return
    if catalog.widen(message.data.get("tools")) and span is not None:
        set_tool_definition_attributes(span, capture, catalog.current)


def marks_local_work(message: Any) -> bool:
    """Whether a non-assistant message marks the end of local work, and so the start of the window
    that will produce the next response.

    Only two kinds do. A ``UserMessage`` carries the tool results the next call is being sent, and a
    ``SystemMessage`` ``init`` opens the session. Everything else is progress reporting *during* a
    call.
    """
    if isinstance(message, UserMessage):
        return True
    return isinstance(message, SystemMessage) and message.subtype == "init"


# ─── Provider shapes as span shapes ──────────────────────────────────────────


def _attr(obj: Any, name: str) -> Any:
    """Reads a field off a provider object or a plain dict, whichever the caller holds."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def to_span_parts(content: Any) -> list[SpanMessagePart]:
    """Converts one assistant or user message's content blocks into canonical span parts.

    Structural for dict-shaped blocks, typed for the Agent SDK's own dataclasses. Block kinds a span
    has no part for (images, documents) are dropped rather than emitted malformed.
    """
    if isinstance(content, str):
        return [SpanMessagePart(type="text", content=content)] if content else []
    if not isinstance(content, list):
        return []

    parts: list[SpanMessagePart] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(SpanMessagePart(type="text", content=block.text or ""))
        elif isinstance(block, ThinkingBlock):
            parts.append(
                SpanMessagePart(type="reasoning", content=block.thinking or "")
            )
        elif isinstance(block, ToolUseBlock):
            parts.append(
                SpanMessagePart(
                    type="tool_call",
                    id=block.id,
                    name=block.name or "",
                    arguments=block.input,
                )
            )
        elif isinstance(block, ToolResultBlock):
            parts.append(
                SpanMessagePart(
                    type="tool_call_response",
                    id=block.tool_use_id,
                    result=block.content,
                )
            )
        elif isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                parts.append(
                    SpanMessagePart(type="text", content=str(block.get("text") or ""))
                )
            elif block_type == "thinking":
                parts.append(
                    SpanMessagePart(
                        type="reasoning", content=str(block.get("thinking") or "")
                    )
                )
            elif block_type == "tool_use":
                block_id = block.get("id")
                parts.append(
                    SpanMessagePart(
                        type="tool_call",
                        id=block_id if isinstance(block_id, str) else None,
                        name=str(block.get("name") or ""),
                        arguments=block.get("input"),
                    )
                )
            elif block_type == "tool_result":
                use_id = block.get("tool_use_id")
                parts.append(
                    SpanMessagePart(
                        type="tool_call_response",
                        id=use_id if isinstance(use_id, str) else None,
                        result=block.get("content"),
                    )
                )
    return parts


class ToolCatalog:
    """The tools the model could call, widened as the run reports more.

    The AI Config's own tools are known before the run starts. The ones Claude Code brings itself
    (Read, Bash, and the rest) are announced only in the ``init`` message, and only by name; their
    schemas stay in that process. A name with no ``parameters`` says "this tool was offered, its
    schema is not ours to state", which describes the run better than omitting a tool the model could
    see.

    Every entry is keyed on the name the model saw, which is also the name that tool's
    ``execute_tool`` span carries. ``native_names`` maps an AI Config key to the provider tool name it
    stands for, for the config's native tools; absent entries are user-defined tools, catalogued under
    their own name.

    Shared between the root and the ``chat`` spans so the two cannot disagree about the same
    attribute.
    """

    def __init__(
        self,
        config_tools: dict[str, Any] | None,
        native_names: dict[str, str],
    ) -> None:
        self._definitions: list[ToolDefinitionInput] = []
        for key, tool in (config_tools or {}).items():
            name = native_names.get(key) or tool.get("name") or key
            self._definitions.append(
                ToolDefinitionInput(
                    name=name,
                    description=tool.get("description"),
                    parameters=tool.get("parameters"),
                )
            )
        self._named: set[str] = {d.name for d in self._definitions}

    @property
    def current(self) -> list[ToolDefinitionInput]:
        """A copy, so widening the catalog later cannot alter a span already written from it."""
        return list(self._definitions)

    def widen(self, names: Any) -> bool:
        """Absorbs the ``init`` message's tool list.

        Reports whether anything was added, so the caller rewrites the root's attribute only when
        there is something new to say. Compared on display names: the CLI lists an AI Config tool
        under its MCP name, which would otherwise be added a second time alongside the entry that
        already has its schema.
        """
        if not isinstance(names, list):
            return False
        added = False
        for name in names:
            if not isinstance(name, str):
                continue
            display = tool_display_name(name)
            if display in self._named:
                continue
            self._named.add(display)
            self._definitions.append(ToolDefinitionInput(name=display))
            added = True
        return added


@dataclass
class Opening:
    system_instructions: str | None
    messages: list[SpanMessage]


@dataclass
class _Inference:
    request_id: str | None
    parent_tool_use_id: str | None
    subagent_type: str | None
    model: str
    session_id: str | None
    usage: dict[str, Any]
    finish_reason: str | None
    start_time: int
    end_time: int
    parts: list[SpanMessagePart]
    input_messages: list[SpanMessage]


class InferenceSpans:
    """Emits the run's ``chat`` spans, one per API call.

    ``query()`` reports no request boundaries, and an ``AssistantMessage`` is not one: the CLI emits
    one message per content block of a response, so a single API call surfaces as several messages
    that share a ``message_id`` (the Anthropic response id) and repeat the same usage bag, with tool
    executions interleaved between them. So the response id is the unit, accumulated across the whole
    run rather than only while consecutive, because the messages of one response are not adjacent in
    the stream.

    Spans are built at :meth:`finish` rather than as messages arrive, so a response is described
    once, in full. :meth:`finish` must therefore run even when the run throws, since the exporter
    only receives ended spans.
    """

    def __init__(
        self,
        config: AiConfigRep,
        parent_context: Any,
        capture_content: bool,
        catalog: ToolCatalog,
        opening: Opening,
    ) -> None:
        self._config = config
        self._parent = parent_context
        self._capture = capture_content
        self._catalog = catalog
        self._opening = opening
        self._boundary = time.time_ns()
        self._inferences: list[_Inference] = []
        self._by_request_id: dict[str, _Inference] = {}
        # One conversation per agent thread, keyed by parent_tool_use_id (None = main thread). A
        # subagent's model calls arrive on the same stream as the main thread's, interleaved, so a
        # single list would hand a main-thread call an input containing turns from a conversation it
        # was never part of.
        self._threads: dict[str | None, list[SpanMessage]] = {}
        self._totals: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self._turns = 0

    def _thread_for(self, parent_tool_use_id: str | None) -> list[SpanMessage]:
        thread = self._threads.get(parent_tool_use_id)
        if thread is None:
            thread = list(self._opening.messages) if parent_tool_use_id is None else []
            self._threads[parent_tool_use_id] = thread
        return thread

    def record(self, message: Any) -> None:
        """Feed every message of the run, in order."""
        if isinstance(message, AssistantMessage):
            self._absorb(message)
            return
        if isinstance(message, UserMessage):
            self._absorb_user_turn(message)
        if marks_local_work(message):
            self._boundary = time.time_ns()

    def _absorb_user_turn(self, message: UserMessage) -> None:
        """Adds a user turn (tool results, and any context the CLI injected) to the conversation."""
        parts = to_span_parts(message.content)
        if not parts:
            return
        self._thread_for(message.parent_tool_use_id).append(
            SpanMessage(role="user", parts=parts)
        )

    def finish(self) -> None:
        """Emits a span per response and clears the accumulator. Idempotent."""
        if not self._inferences:
            return
        pending = self._inferences
        self._inferences = []
        self._by_request_id.clear()
        for inference in pending:
            self._emit(inference)

    def _absorb(self, message: AssistantMessage) -> None:
        request_id = message.message_id
        known = self._by_request_id.get(request_id) if request_id else None
        if known is not None:
            # Another block of a response already recorded. Its usage is the same bag repeated.
            known.parts.extend(to_span_parts(message.content))
            if message.stop_reason:
                known.finish_reason = to_semconv_finish_reason(message.stop_reason)
            return

        now = time.time_ns()
        parent_tool_use_id = message.parent_tool_use_id
        thread = self._thread_for(parent_tool_use_id)
        parts = to_span_parts(message.content)
        inference = _Inference(
            request_id=request_id,
            parent_tool_use_id=parent_tool_use_id,
            subagent_type=getattr(message, "subagent_type", None),
            model=message.model or model_name(self._config),
            session_id=message.session_id,
            usage=dict(message.usage or {}),
            finish_reason=to_semconv_finish_reason(message.stop_reason),
            start_time=self._boundary,
            end_time=now,
            parts=parts,
            # Which turns this call was sent, captured before the reply joins them below.
            input_messages=list(thread),
        )
        self._inferences.append(inference)
        if request_id:
            self._by_request_id[request_id] = inference
        thread.append(SpanMessage(role="assistant", parts=inference.parts))

        self._turns += 1
        for key in self._totals:
            self._totals[key] += number_or_zero(inference.usage.get(key))
        self._boundary = now

    @property
    def run_usage(self) -> dict[str, Any]:
        """The run's spend so far, for the paths where the CLI never reported its own total.

        ``reported`` is false only when no response was ever absorbed. Writing an all-zero total in
        that case would assert the run cost nothing, which a run that died before its first response
        cannot honestly claim.
        """
        return {"total": dict(self._totals), "reported": self._turns > 0}

    def _emit(self, inference: _Inference) -> None:
        if not _HAS_OTEL:
            return
        span = trace.get_tracer(TRACER_NAME).start_span(
            f"chat {inference.model}",
            context=self._parent,
            start_time=inference.start_time,
        )
        span.set_attribute("gen_ai.operation.name", "chat")
        set_model_identity_attributes(span, PROVIDER, inference.model)
        # The model the turn actually used, read off the streamed inference. Not
        # config.model.name: this is one of only two handlers where the two may disagree. See
        # TELEMETRY-CONTRACT.md section 2a.
        span.set_attribute("gen_ai.response.model", inference.model)
        if inference.request_id:
            span.set_attribute("gen_ai.response.id", inference.request_id)
        if inference.session_id:
            span.set_attribute("gen_ai.conversation.id", inference.session_id)
        # Absent in practice: measured against Agent SDK 0.3.220, stop_reason and stop_details are
        # both null on every assistant message. Written only when the SDK populates it — never
        # synthesised from the presence of a tool-use block.
        if inference.finish_reason:
            span.set_attribute(
                "gen_ai.response.finish_reasons", [inference.finish_reason]
            )
        set_usage_span_attributes(span, add_cached_tokens_to_input(inference.usage))
        if inference.subagent_type:
            span.set_attribute("gen_ai.agent.name", inference.subagent_type)

        main_thread = inference.parent_tool_use_id is None
        set_input_content_attributes(
            span,
            self._capture,
            # Both only on the main thread. A subagent runs under its own agent definition's prompt
            # and its own subset of tools, neither of which this side is told.
            system_instructions=(
                self._opening.system_instructions if main_thread else None
            ),
            messages=inference.input_messages,
            tool_definitions=self._catalog.current if main_thread else None,
        )
        set_output_content_attributes(
            span,
            self._capture,
            [
                SpanMessage(
                    role="assistant",
                    parts=inference.parts,
                    finish_reason=inference.finish_reason,
                )
            ],
        )
        span.set_status(SpanStatusCode.OK)
        span.end(end_time=inference.end_time)
