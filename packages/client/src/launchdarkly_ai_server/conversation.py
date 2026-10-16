"""Caller-supplied ``gen_ai.conversation.id`` and judge evaluation span events.

A dedicated OTel context key, not W3C baggage: the id must not leak onto outbound
provider HTTP calls. A multi-tenant process binds a different id per request.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from contextlib import aclosing, asynccontextmanager, contextmanager
from dataclasses import dataclass
from time import time_ns
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from opentelemetry.trace import Span

if TYPE_CHECKING:
    # ``opentelemetry-sdk`` is an optional extra (``[otel]``), so it must not be imported at
    # runtime — this module has to work on an api-only install. Subclassing under
    # ``TYPE_CHECKING`` still gets the interface checked, which a ``cast`` at the registration
    # site would not.
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessorBase
else:
    _SpanProcessorBase = object

GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
LD_EVALUATION_RUN_ID = "launchdarkly.evaluation_run.id"
LD_EVALUATION_RUN_ROW_INDEX = "launchdarkly.evaluation_run.row_index"

# Same namespace ai-evaluator uses for conversationId on a result row.
# Changing it re-keys every offline evaluation conversation.
_EVALUATION_CONVERSATION_NAMESPACE = uuid.UUID("6f0c2a4e-3b6d-4d0e-9c1a-2e6b7f8a9d01")

_CONV_KEY = otel_context.create_key("launchdarkly.gen_ai.conversation.id")
_EVAL_KEY = otel_context.create_key("launchdarkly.judge.evaluation")
_EVAL_ROW_KEY = otel_context.create_key("launchdarkly.evaluation_run.row")
_OBSERVER_KEY = otel_context.create_key("launchdarkly.evaluation_run.observer")


@dataclass
class _JudgeEvalCapture:
    name: str
    released: bool = False
    span: Any = None
    pending_end: Callable[[], None] | None = None


# Every tracer this SDK creates is named "@launchdarkly/ai-<package>". The processor is registered
# on the *global* provider, so without this gate it stamps a caller-supplied id onto every span in
# the process — Postgres queries, inbound HTTP server spans, and the outbound provider call itself
# — and replaces ``end`` on any third-party ``invoke_agent`` span that starts during a judge run.
_LD_TRACER_PREFIX = "@launchdarkly/"


def _read_attribute(span: Any, key: str) -> Any:
    attrs = getattr(span, "attributes", None)
    if attrs is None:
        return None
    return attrs.get(key)


def set_conversation_id_if_absent(span: Any, conversation: str) -> None:
    """Write ``gen_ai.conversation.id`` only when the span does not already carry one."""
    if not conversation or span is None:
        return
    existing = _read_attribute(span, GEN_AI_CONVERSATION_ID)
    if isinstance(existing, str) and existing:
        return
    span.set_attribute(GEN_AI_CONVERSATION_ID, conversation)


def _is_launchdarkly_span(span: Any) -> bool:
    """True only when the span came from one of this SDK's own tracers.

    Deliberately conservative: an unrecognisable scope means "not ours", so an id is never sprayed
    across unrelated telemetry and a judge run never delays a third-party ``invoke_agent`` span.
    The companion test asserts LD spans *are* stamped, so a rename of the scope attribute fails
    the suite loudly rather than silently disabling the feature.
    """
    scope_name = getattr(getattr(span, "instrumentation_scope", None), "name", None)
    return isinstance(scope_name, str) and scope_name.startswith(_LD_TRACER_PREFIX)


def _conversation_id_from(ctx: otel_context.Context | None) -> str | None:
    if ctx is None:
        return None
    value = otel_context.get_value(_CONV_KEY, ctx)
    return value if isinstance(value, str) and value else None


def conversation_id_for_evaluation_row(run_id: str, row_index: int | None) -> str:
    """UUID5 of ``{run_id}:{row_index}`` in the ai-evaluator namespace.

    Row index ``0`` is the suffix ``:0``. A missing index is row 0, matching
    the workers that persist a missing index that way.
    """
    index = 0 if row_index is None else int(row_index)
    return str(uuid.uuid5(_EVALUATION_CONVERSATION_NAMESPACE, f"{run_id}:{index}"))


def _evaluation_row_attributes(
    run_id: str, row_index: int | None
) -> dict[str, str | int]:
    index = 0 if row_index is None else int(row_index)
    return {
        LD_EVALUATION_RUN_ID: str(run_id),
        LD_EVALUATION_RUN_ROW_INDEX: index,
    }


def _evaluation_row_from(ctx: otel_context.Context | None) -> dict[str, Any] | None:
    value = otel_context.get_value(_EVAL_ROW_KEY, ctx) if ctx is not None else None
    if value is None:
        value = otel_context.get_value(_EVAL_ROW_KEY)
    if not isinstance(value, dict) or not value:
        return None
    return value


@contextmanager
def bind_evaluation_row(run_id: str, row_index: int | None) -> Iterator[None]:
    """Bind one offline-evaluation row onto SDK spans started in this block.

    Sets ``gen_ai.conversation.id`` to :func:`conversation_id_for_evaluation_row`
    and ``launchdarkly.evaluation_run.id`` / ``.row_index``. Does not set
    ``launchdarkly.run.id``; that key is the AI Config tracking id.

    An outer :func:`conversation_id` is shadowed until the block exits. The
    evaluation row id is the join key for this row.
    """
    index = 0 if row_index is None else int(row_index)
    ctx = otel_context.set_value(
        _CONV_KEY, conversation_id_for_evaluation_row(run_id, index)
    )
    ctx = otel_context.set_value(
        _EVAL_ROW_KEY, _evaluation_row_attributes(run_id, index), ctx
    )
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


@dataclass
class HandlerSpanObserver:
    """Counts SDK spans started while :func:`observe_handler_spans` is bound."""

    count: int = 0


@contextmanager
def observe_handler_spans() -> Iterator[HandlerSpanObserver]:
    """Count the SDK spans a handler opens inside this block.

    The offline runner describes a plain-function handler's call itself, and
    must not when a provider handler already traced it. Counting relies on
    :class:`ConversationIdSpanProcessor`, which ``init_client`` registers.
    """
    observer = HandlerSpanObserver()
    token = otel_context.attach(otel_context.set_value(_OBSERVER_KEY, observer))
    try:
        yield observer
    finally:
        otel_context.detach(token)


def record_evaluation_result(
    span: Any, name: str, score: float, explanation: str | None = None
) -> None:
    """Write ``gen_ai.evaluation.result`` with no verdict and no score label.

    ``explanation`` is the criterion reason. Offline evaluation always records
    it when the reason is non-empty: that text already travels on the criterion
    event. This does not add a prompt, a completion, or a tool payload.
    """
    _record_evaluation(span, name, score, explanation)


def _record_evaluation(
    span: Any, name: str, score: float, explanation: str | None = None
) -> None:
    """Write the judge score as a ``gen_ai.evaluation.result`` event plus mirrored attributes.

    ``explanation`` is passed only when the judge's own handler captures content — it is model
    prose about the user's conversation, so it follows the same gate as every other content
    attribute. The reasoning always reaches the caller in ``judge_results``.
    """
    if span is None or not span.is_recording():
        return
    attrs: dict[str, Any] = {
        "gen_ai.evaluation.name": name,
        "gen_ai.evaluation.score.value": score,
    }
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = explanation
    span.add_event("gen_ai.evaluation.result", attrs)
    for key, value in attrs.items():
        span.set_attribute(key, value)


def _delay_invoke_agent_end(span: Any, capture: _JudgeEvalCapture) -> None:
    original_end = span.end
    ended = False

    def wrapped_end(*args: Any, **kwargs: Any) -> None:
        nonlocal ended
        if ended:
            return
        if capture.released:
            ended = True
            original_end(*args, **kwargs)
            return
        capture.span = span
        # Freeze the end time at the handler's call. Replaying a no-arg end() later would let the
        # SDK stamp time_ns() at release, inflating the judge span by the tracking and parsing
        # work that runs between the handler ending the span and the score being recorded.
        if not args and "end_time" not in kwargs:
            kwargs = {**kwargs, "end_time": time_ns()}

        def pending() -> None:
            nonlocal ended
            if ended:
                return
            ended = True
            original_end(*args, **kwargs)

        capture.pending_end = pending

    span.end = wrapped_end


@contextmanager
def conversation_id(conversation: str | None) -> Iterator[None]:
    """Bind a caller-supplied conversation id for the duration of the ``with`` block.

    Every span the SDK creates while this is bound receives ``gen_ai.conversation.id``,
    provided :class:`ConversationIdSpanProcessor` is registered — which ``init_client()``
    does when OTel is installed. An empty, whitespace, or ``None`` id is treated as unbound —
    the natural call site is an optional value such as ``conversation_id(request.thread_id)``,
    which must degrade to an unstamped trace rather than raise.
    """
    trimmed = conversation.strip() if isinstance(conversation, str) else ""
    if not trimmed:
        yield
        return
    token = otel_context.attach(otel_context.set_value(_CONV_KEY, trimmed))
    try:
        yield
    finally:
        otel_context.detach(token)


RecordEvaluation = Callable[..., None]


async def _stream_with_bound_id(
    generator: AsyncGenerator[Any, None], conversation: str
) -> AsyncGenerator[Any, None]:
    """Bind ``conversation`` for the whole iteration of ``generator``.

    Attached once and detached once, deliberately. Re-attaching per step would have to detach per
    step too, and a streaming handler normally holds a span current *across* a ``yield`` — so that
    detach would reset the contextvar past the handler's own live ``start_as_current_span`` and
    reparent every span it opens after resuming. Binding for the whole iteration leaves the
    generator's own context untouched, so span parenting matches an unbound stream exactly.

    The generator is closed before the detach so its ``__exit__`` unwinds inside our scope, keeping
    contextvar tokens in LIFO order.

    Trade-off: the id is also bound while the consumer's body runs between chunks, so spans the
    consumer opens mid-stream are stamped too. That is the conversation the consumer is streaming.
    """
    token = otel_context.attach(otel_context.set_value(_CONV_KEY, conversation))
    try:
        async with aclosing(generator):
            async for item in generator:
                yield item
    finally:
        otel_context.detach(token)


def bind_conversation_id(
    generator: AsyncGenerator[Any, None],
) -> AsyncGenerator[Any, None]:
    """Pin the currently-bound conversation id onto ``generator`` at call time.

    An ``async def`` with ``yield`` does not start its body until the first ``__anext__``, which
    for a streaming caller is normally after the :func:`conversation_id` block has exited. Reading
    the id here — eagerly, before any iteration — is what lets a caller bind, hand the generator
    off, and iterate it later.
    """
    conversation = _conversation_id_from(otel_context.get_current())
    if not conversation:
        return generator
    return _stream_with_bound_id(generator, conversation)


async def bind_span_context(
    generator: AsyncGenerator[Any, None],
    ctx: otel_context.Context,
) -> AsyncGenerator[Any, None]:
    """Re-enter ``ctx`` around every step of ``generator``.

    Sibling of :func:`bind_conversation_id`, which deliberately carries only the conversation id
    and leaves span parenting alone. A generator body suspends at each ``yield``, so the context
    has to be re-applied on every ``__anext__`` — wrapping the body once is not enough.
    """
    try:
        while True:
            token = otel_context.attach(ctx)
            try:
                item = await generator.__anext__()
            except StopAsyncIteration:
                return
            finally:
                otel_context.detach(token)
            yield item
    finally:
        token = otel_context.attach(ctx)
        try:
            await generator.aclose()
        finally:
            otel_context.detach(token)


@asynccontextmanager
async def with_judge_evaluation(name: str) -> AsyncIterator[RecordEvaluation]:
    """Hold the judge ``invoke_agent`` span open until ``record`` runs.

    ``execute_and_track`` returns after the handler has already called ``span.end()``,
    so without this delay the evaluation event would be dropped.
    """
    capture = _JudgeEvalCapture(name=name)

    def record(score: float, explanation: str | None = None) -> None:
        if capture.span is not None:
            _record_evaluation(capture.span, capture.name, score, explanation)

    token = otel_context.attach(otel_context.set_value(_EVAL_KEY, capture))
    try:
        yield record
    finally:
        capture.released = True
        try:
            if capture.pending_end is not None:
                capture.pending_end()
        finally:
            # Detach even if ending the span raises (a user span processor's on_end can throw);
            # otherwise the judge capture stays bound in this task's context for good.
            otel_context.detach(token)


class ConversationIdSpanProcessor(_SpanProcessorBase):
    """Stamps conversation and evaluation-row attributes; delays judge ``invoke_agent`` end.

    Structurally a ``SpanProcessor``; the base is only real to a type checker so that an
    api-only install (no ``[otel]`` extra) still imports this module.
    ``init_client()`` registers it ahead of ``BatchSpanProcessor``.
    """

    def on_start(
        self, span: Span, parent_context: otel_context.Context | None = None
    ) -> None:
        ctx = (
            parent_context if parent_context is not None else otel_context.get_current()
        )
        conv = _conversation_id_from(ctx) or _conversation_id_from(
            otel_context.get_current()
        )
        if conv and _is_launchdarkly_span(span):
            set_conversation_id_if_absent(span, conv)

        observer = otel_context.get_value(_OBSERVER_KEY, ctx)
        if isinstance(observer, HandlerSpanObserver) and _is_launchdarkly_span(span):
            observer.count += 1

        row = _evaluation_row_from(ctx)
        if row and _is_launchdarkly_span(span):
            for key, value in row.items():
                if not isinstance(key, str) or _read_attribute(span, key) is not None:
                    continue
                if isinstance(value, str):
                    span.set_attribute(key, value)
                elif isinstance(value, int) and not isinstance(value, bool):
                    span.set_attribute(key, value)

        capture = otel_context.get_value(_EVAL_KEY, ctx)
        if capture is None:
            capture = otel_context.get_value(_EVAL_KEY)
        # Same LD-scope gate as conversation-id stamping: wrapping ``end`` on a foreign
        # ``invoke_agent`` (OpenAI Agents, LangChain, …) would overwrite ``pending_end``,
        # leak the judge root, and attach ``gen_ai.evaluation.result`` to the wrong span.
        if (
            isinstance(capture, _JudgeEvalCapture)
            and getattr(span, "name", None) == "invoke_agent"
            and _is_launchdarkly_span(span)
        ):
            _delay_invoke_agent_end(span, capture)

    def on_end(self, span: Any) -> None:
        return None

    def _on_ending(self, span: Any) -> None:
        # Python SDK calls ``_on_ending`` from ``span.end()``; JS uses ``onEnd``.
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        return True
