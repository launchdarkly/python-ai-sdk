"""Caller-supplied ``gen_ai.conversation.id`` and judge evaluation span events.

A dedicated OTel context key, not W3C baggage: the id must not leak onto outbound
provider HTTP calls. A multi-tenant process binds a different id per request.
"""

from __future__ import annotations

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

_CONV_KEY = otel_context.create_key("launchdarkly.gen_ai.conversation.id")
_EVAL_KEY = otel_context.create_key("launchdarkly.judge.evaluation")


@dataclass
class _JudgeEvalCapture:
    name: str
    released: bool = False
    span: Any = None
    pending_end: Callable[[], None] | None = None


# Every tracer this SDK creates is named "@launchdarkly/ai-<package>". The processor is registered
# on the *global* provider, so without this gate it stamps a caller-supplied id onto every span in
# the process — Postgres queries, inbound HTTP server spans, and the outbound provider call itself.
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
    across unrelated telemetry. The companion test asserts LD spans *are* stamped, so a rename of
    the scope attribute fails the suite loudly rather than silently disabling the feature.
    """
    scope_name = getattr(getattr(span, "instrumentation_scope", None), "name", None)
    return isinstance(scope_name, str) and scope_name.startswith(_LD_TRACER_PREFIX)


def _conversation_id_from(ctx: otel_context.Context | None) -> str | None:
    if ctx is None:
        return None
    value = otel_context.get_value(_CONV_KEY, ctx)
    return value if isinstance(value, str) and value else None


def _record_evaluation(span: Any, name: str, score: float) -> None:
    """Write the judge score as a ``gen_ai.evaluation.result`` event plus mirrored attributes.

    The judge's free-text reasoning is deliberately NOT exported. It is model prose about the
    user's conversation, i.e. content, and AGENTS.md restricts content attributes to callers who
    pass ``capture_content=True`` — a handler-factory option this layer has no access to. The
    reasoning is still returned to the caller in ``judge_results[key].response``; only the
    telemetry copy is dropped. Exporting it needs its own opt-in.
    """
    if span is None or not span.is_recording():
        return
    attrs: dict[str, Any] = {
        "gen_ai.evaluation.name": name,
        "gen_ai.evaluation.score.value": score,
    }
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


RecordEvaluation = Callable[[float], None]


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


@asynccontextmanager
async def with_judge_evaluation(name: str) -> AsyncIterator[RecordEvaluation]:
    """Hold the judge ``invoke_agent`` span open until ``record`` runs.

    ``execute_and_track`` returns after the handler has already called ``span.end()``,
    so without this delay the evaluation event would be dropped.
    """
    capture = _JudgeEvalCapture(name=name)

    def record(score: float) -> None:
        if capture.span is not None:
            _record_evaluation(capture.span, capture.name, score)

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
    """Stamps ``gen_ai.conversation.id`` write-if-absent; delays judge ``invoke_agent`` end.

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

        capture = otel_context.get_value(_EVAL_KEY, ctx)
        if capture is None:
            capture = otel_context.get_value(_EVAL_KEY)
        if (
            isinstance(capture, _JudgeEvalCapture)
            and getattr(span, "name", None) == "invoke_agent"
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
