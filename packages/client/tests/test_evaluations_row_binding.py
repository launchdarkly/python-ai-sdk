"""The evaluations module's row binding, built on the core stamping primitives."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from launchdarkly_ai_server.conversation import (
    GEN_AI_CONVERSATION_ID,
    ConversationIdSpanProcessor,
    conversation_id,
)
from launchdarkly_ai_server.evaluations.spans import (
    LD_EVALUATION_RUN_ID,
    LD_EVALUATION_RUN_ROW_INDEX,
    bind_evaluation_row,
    conversation_id_for_evaluation_row,
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


class TestEvaluationRowBinding:
    """Offline rows share ai-evaluator's UUID5 and must not reuse launchdarkly.run.id."""

    def test_uuid5_matches_the_published_namespace(self) -> None:
        assert (
            conversation_id_for_evaluation_row("run-id", 3)
            == "faa03308-8fe7-5805-8c07-f82a3dbdb3f1"
        )
        assert (
            conversation_id_for_evaluation_row("run-id", 0)
            == "8b90044e-ac6d-5af1-88a3-b75fbb9c8268"
        )
        assert conversation_id_for_evaluation_row("run-id", None) == (
            conversation_id_for_evaluation_row("run-id", 0)
        )

    def test_stamps_sdk_spans_and_not_run_id(self) -> None:
        with bind_evaluation_row("run-id", 3):
            span = _tracer.start_span("invoke_agent")
            span.end()
        recorded = finished()[0]
        assert recorded.attributes is not None
        assert (
            recorded.attributes[GEN_AI_CONVERSATION_ID]
            == "faa03308-8fe7-5805-8c07-f82a3dbdb3f1"
        )
        assert recorded.attributes[LD_EVALUATION_RUN_ID] == "run-id"
        assert recorded.attributes[LD_EVALUATION_RUN_ROW_INDEX] == 3
        assert "launchdarkly.run.id" not in recorded.attributes

    def test_does_not_stamp_third_party_spans(self) -> None:
        foreign = _provider.get_tracer("opentelemetry.instrumentation.httpx")
        with bind_evaluation_row("run-id", 3):
            span = foreign.start_span("POST")
            span.end()
        attributes = finished()[0].attributes or {}
        assert GEN_AI_CONVERSATION_ID not in attributes
        assert LD_EVALUATION_RUN_ID not in attributes
        assert LD_EVALUATION_RUN_ROW_INDEX not in attributes

    def test_row_id_shadows_an_outer_conversation_id(self) -> None:
        with conversation_id("thread-123"):
            with bind_evaluation_row("run-id", 0):
                span = _tracer.start_span("chat gpt-4o")
                span.end()
        assert (
            finished()[0].attributes[GEN_AI_CONVERSATION_ID]
            == "8b90044e-ac6d-5af1-88a3-b75fbb9c8268"
        )
