from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from launchdarkly_ai_server.conversation import (
    GEN_AI_CONVERSATION_ID,
    ConversationIdSpanProcessor,
    conversation_id,
    set_conversation_id_if_absent,
    with_judge_evaluation,
)

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(ConversationIdSpanProcessor())
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
_tracer = _provider.get_tracer("@launchdarkly/ai-server")


@pytest.fixture(autouse=True)
def _reset_exporter() -> Iterator[None]:
    _exporter.clear()
    yield
    _exporter.clear()


def finished() -> list[ReadableSpan]:
    return list(_exporter.get_finished_spans())


class TestConversationId:
    def test_stamps_id_on_every_span_in_scope(self) -> None:
        with conversation_id("thread-123"):
            root = _tracer.start_span("invoke_agent")
            child = _tracer.start_span(
                "chat gpt-4o", context=trace.set_span_in_context(root)
            )
            child.end()
            root.end()
        spans = finished()
        assert len(spans) == 2
        for span in spans:
            assert span.attributes[GEN_AI_CONVERSATION_ID] == "thread-123"

    def test_writes_nothing_when_unbound(self) -> None:
        span = _tracer.start_span("invoke_agent")
        span.end()
        assert GEN_AI_CONVERSATION_ID not in finished()[0].attributes

    def test_whitespace_id_is_unbound(self) -> None:
        with conversation_id("   "):
            span = _tracer.start_span("invoke_agent")
            span.end()
        assert GEN_AI_CONVERSATION_ID not in finished()[0].attributes

    def test_does_not_invent_an_id_from_the_trace_id(self) -> None:
        span = _tracer.start_span("invoke_agent")
        span.end()
        recorded = finished()[0]
        assert GEN_AI_CONVERSATION_ID not in recorded.attributes
        assert recorded.context.trace_id != 0


class TestSetConversationIdIfAbsent:
    def test_leaves_caller_id_in_place(self) -> None:
        with conversation_id("caller-id"):
            span = _tracer.start_span("invoke_agent")
            set_conversation_id_if_absent(span, "sess-abc")
            span.end()
        assert finished()[0].attributes[GEN_AI_CONVERSATION_ID] == "caller-id"

    def test_writes_session_id_when_unbound(self) -> None:
        span = _tracer.start_span("invoke_agent")
        set_conversation_id_if_absent(span, "sess-abc")
        span.end()
        assert finished()[0].attributes[GEN_AI_CONVERSATION_ID] == "sess-abc"


class TestJudgeEvaluation:
    async def test_puts_evaluation_event_on_invoke_agent(self) -> None:
        async with with_judge_evaluation("relevance-judge") as record:
            with _tracer.start_as_current_span("invoke_agent") as span:
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
            record(0.91, "on topic")
        span = next(s for s in finished() if s.name == "invoke_agent")
        assert span.attributes["gen_ai.evaluation.name"] == "relevance-judge"
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.91
        assert span.attributes["gen_ai.evaluation.explanation"] == "on topic"
        event = next(e for e in span.events if e.name == "gen_ai.evaluation.result")
        assert event.attributes["gen_ai.evaluation.name"] == "relevance-judge"
        assert event.attributes["gen_ai.evaluation.score.value"] == 0.91
        assert event.attributes["gen_ai.evaluation.explanation"] == "on topic"
        assert "gen_ai.evaluation.score.label" not in event.attributes

class TestProcessorScope:
    """The processor is registered on the *global* provider, so it sees every span in the process.

    Stamping a caller-supplied conversation id onto unrelated telemetry — Postgres queries, inbound
    HTTP server spans, and the outbound provider call itself — is both semantically wrong and the
    leak the module docstring says this design avoids.
    """

    def test_stamps_spans_from_launchdarkly_tracers(self) -> None:
        tracer = _provider.get_tracer("@launchdarkly/ai-claude-messages")
        with conversation_id("thread-123"):
            span = tracer.start_span("invoke_agent")
            span.end()
        assert finished()[0].attributes[GEN_AI_CONVERSATION_ID] == "thread-123"

    def test_does_not_stamp_third_party_instrumentation_spans(self) -> None:
        http = _provider.get_tracer("opentelemetry.instrumentation.httpx")
        db = _provider.get_tracer("opentelemetry.instrumentation.psycopg")
        with conversation_id("thread-123"):
            for tracer, name in ((http, "POST /v1/messages"), (db, "SELECT users")):
                span = tracer.start_span(name)
                span.end()
        for span in finished():
            assert (span.attributes or {}).get(GEN_AI_CONVERSATION_ID) is None, (
                span.name
            )
