"""Offline evaluation spans join the customer project on the existing exporter.

Reference: TESTING.md §8.10.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._once import Once

from launchdarkly_ai_server.conversation import (
    GEN_AI_CONVERSATION_ID,
    LD_EVALUATION_RUN_ID,
    LD_EVALUATION_RUN_ROW_INDEX,
    ConversationIdSpanProcessor,
)
from launchdarkly_ai_server.evaluations import HttpResponse
from launchdarkly_ai_server.experimental.evaluations import (
    EvalTool,
    Judge,
    init_evaluations,
)

_RUN_ID = "run-id"
_ROW_INDEX = 3
_CONVERSATION_ID = "faa03308-8fe7-5805-8c07-f82a3dbdb3f1"
_TOOL_ARGUMENTS = '{"order_id": "A1"}'
_TOOL_RESULT = "shipped"


@pytest.fixture
def stub_sdk_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setenv("LD_SDK_KEY", "sdk-key")
    client = MagicMock()
    client.flush = AsyncMock()
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.get_client",
        MagicMock(side_effect=RuntimeError("client not initialized")),
    )
    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.module.init_client",
        AsyncMock(return_value=client),
    )
    return client


@pytest.fixture
def span_export() -> Iterator[tuple[InMemorySpanExporter, list[str]]]:
    """Install the SDK processor on a provider whose project id is the SDK key."""
    saved = otel_trace._TRACER_PROVIDER
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "python-sdk", "highlight.project_id": "sdk-key"}
        )
    )
    provider.add_span_processor(ConversationIdSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    flushed: list[str] = []
    original_flush = provider.force_flush

    def spy(timeout_millis: int = 30000) -> bool:
        flushed.append("traces")
        return original_flush(timeout_millis)

    provider.force_flush = spy  # type: ignore[method-assign]
    try:
        yield exporter, flushed
    finally:
        provider.shutdown()
        once = Once()
        otel_trace._TRACER_PROVIDER = saved
        if saved is not None:
            once.do_once(lambda: None)
        otel_trace._TRACER_PROVIDER_SET_ONCE = once


def response(status: int, body: dict[str, Any] | None = None) -> HttpResponse:
    return HttpResponse(
        status=status, body=json.dumps(body) if body is not None else ""
    )


class SequencedTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse:
        index = len(self.requests)
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": json.loads(body) if body else None,
                "timeout": timeout,
            }
        )
        if index >= len(self.responses):
            raise AssertionError(f"unexpected request: {method} {url}")
        return self.responses[index]


def one_row_transport() -> SequencedTransport:
    return SequencedTransport(
        [
            response(200, {"id": "dataset-id", "name": "golden"}),
            response(
                200,
                {
                    "items": [
                        {
                            "rowIndex": _ROW_INDEX,
                            "input": "Where is order A1?",
                            "expectedOutput": "shipped",
                            "variables": {},
                        }
                    ],
                    "totalCount": 1,
                    "_links": {"self": {"href": "https://api.test/current"}},
                },
            ),
            response(201, {"id": "evaluation-id", "name": "support-qa", "version": 3}),
            response(
                201,
                {
                    "id": _RUN_ID,
                    "evaluationId": "evaluation-id",
                    "state": "PENDING",
                },
            ),
            response(
                200,
                {"statusCounts": {"total": 1, "passed": 1, "error": 0, "pending": 0}},
            ),
        ]
    )


def patch_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        assert key == "$ld:ai:judge:accuracy"
        return {
            "config": {
                "provider": {"name": "OpenAI"},
                "model": {"name": "gpt-4o"},
                "instructions": "Judge {{response_to_evaluate}}",
            },
            "meta": {"variationKey": "default", "version": 12},
        }

    monkeypatch.setattr(
        "launchdarkly_ai_server.evaluations.runner.extract_variation",
        fake_extract_variation,
    )


def forbid_new_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run() must not construct an OTLP exporter")

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", boom)


def assert_no_verdict(attributes: Any) -> None:
    keys = set(attributes or {})
    assert "gen_ai.evaluation.score.label" not in keys
    assert "verdict" not in keys
    assert "passed" not in keys
    assert "successDirection" not in keys
    assert "launchdarkly.run.id" not in keys


def assert_row_attributes(attributes: Any) -> None:
    assert attributes[GEN_AI_CONVERSATION_ID] == _CONVERSATION_ID
    assert attributes[LD_EVALUATION_RUN_ID] == _RUN_ID
    assert attributes[LD_EVALUATION_RUN_ROW_INDEX] == _ROW_INDEX
    assert_no_verdict(attributes)


@pytest.mark.asyncio
async def test_plain_function_exports_evaluate_span_on_the_sdk_key_provider(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    span_export: tuple[InMemorySpanExporter, list[str]],
) -> None:
    exporter, flushed = span_export
    forbid_new_exporter(monkeypatch)
    patch_judge(monkeypatch)

    async def client_flush() -> None:
        flushed.append("client")

    stub_sdk_client.flush = client_flush
    evals = init_evaluations(
        project_key="proj",
        api_key="token",
        sdk_key="sdk-key",
        transport=one_row_transport(),
    )

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        assert "__ld" not in variables
        if "Judge" in config.get("instructions", ""):
            return {
                "output": '{"score": 0.9, "reasoning": "Correct and concise."}',
            }
        return {"output": "shipped", "usage": {"input_tokens": 10, "output_tokens": 4}}

    result = await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["evaluate $ld:ai:judge:accuracy"]
    evaluate = spans[0]
    assert evaluate.resource is not None
    assert evaluate.resource.attributes["highlight.project_id"] == "sdk-key"
    assert_row_attributes(evaluate.attributes)
    assert evaluate.attributes["gen_ai.operation.name"] == "evaluate"
    event = next(
        item for item in evaluate.events if item.name == "gen_ai.evaluation.result"
    )
    assert event.attributes is not None
    assert event.attributes["gen_ai.evaluation.name"] == "$ld:ai:judge:accuracy"
    assert event.attributes["gen_ai.evaluation.score.value"] == 0.9
    assert event.attributes["gen_ai.evaluation.explanation"] == "Correct and concise."
    assert_no_verdict(event.attributes)
    assert "gen_ai.input.messages" not in (evaluate.attributes or {})
    assert "gen_ai.output.messages" not in (evaluate.attributes or {})

    payloads = [call.args[2] for call in stub_sdk_client.track.call_args_list]
    names = [call.args[0] for call in stub_sdk_client.track.call_args_list]
    assert names == [
        "$ld:ai:offline-evals:generation",
        "$ld:ai:offline-evals:criterion",
    ]
    for payload in payloads:
        assert "tool_calls" not in payload
        assert "input" not in payload
        assert "variables" not in payload
        assert "metadata" not in payload
        assert "verdict" not in payload
        assert "passed" not in payload
    generation = payloads[0]
    assert generation["output"] == "shipped"
    assert "message" not in generation
    criterion = payloads[1]
    assert criterion["score"] == 0.9
    assert criterion["reason"] == "Correct and concise."
    assert flushed == ["client", "traces"]


@pytest.mark.asyncio
async def test_handler_spans_carry_the_row_and_the_tool_call(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    span_export: tuple[InMemorySpanExporter, list[str]],
) -> None:
    exporter, _flushed = span_export
    forbid_new_exporter(monkeypatch)
    patch_judge(monkeypatch)
    evals = init_evaluations(
        project_key="proj",
        api_key="token",
        sdk_key="sdk-key",
        transport=one_row_transport(),
    )
    tracer = otel_trace.get_tracer("@launchdarkly/ai-server")
    foreign = otel_trace.get_tracer("opentelemetry.instrumentation.httpx")

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        assert "__ld" not in variables
        if "Judge" in config.get("instructions", ""):
            with tracer.start_as_current_span("invoke_agent"):
                with tracer.start_as_current_span("chat gpt-4o"):
                    pass
            return {"output": '{"score": 0.9, "reasoning": "Correct and concise."}'}
        with tracer.start_as_current_span("invoke_agent"):
            with tracer.start_as_current_span("chat gpt-4o"):
                pass
            with tracer.start_as_current_span("execute_tool lookup_order") as tool:
                tool.set_attribute("gen_ai.operation.name", "execute_tool")
                tool.set_attribute("gen_ai.tool.name", "lookup_order")
                tool.set_attribute("gen_ai.tool.call.id", "call-1")
                tool.set_attribute("gen_ai.tool.call.arguments", _TOOL_ARGUMENTS)
                tool.set_attribute("gen_ai.tool.call.result", _TOOL_RESULT)
                tool_handlers["lookup_order"]({"order_id": "A1"})
            http = foreign.start_span("POST /v1/chat")
            http.end()
        return {"output": "done"}

    await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        tools=[
            EvalTool(
                key="lookup_order",
                implementation=lambda args: _TOOL_RESULT,
                schema={"type": "object"},
                description="Look up an order",
            )
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    spans = list(exporter.get_finished_spans())
    by_name: dict[str, list[Any]] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)
    assert set(by_name) >= {
        "invoke_agent",
        "chat gpt-4o",
        "execute_tool lookup_order",
        "evaluate $ld:ai:judge:accuracy",
        "POST /v1/chat",
    }
    for name in (
        "invoke_agent",
        "chat gpt-4o",
        "execute_tool lookup_order",
        "evaluate $ld:ai:judge:accuracy",
    ):
        assert by_name[name]
        for span in by_name[name]:
            assert_row_attributes(span.attributes)
    tool = by_name["execute_tool lookup_order"][0]
    assert tool.attributes is not None
    assert tool.attributes["gen_ai.tool.name"] == "lookup_order"
    assert tool.attributes["gen_ai.tool.call.arguments"] == _TOOL_ARGUMENTS
    assert tool.attributes["gen_ai.tool.call.result"] == _TOOL_RESULT
    http_attributes = by_name["POST /v1/chat"][0].attributes or {}
    assert GEN_AI_CONVERSATION_ID not in http_attributes
    assert LD_EVALUATION_RUN_ID not in http_attributes
    evaluate = by_name["evaluate $ld:ai:judge:accuracy"][0]
    assert any(event.name == "gen_ai.evaluation.result" for event in evaluate.events)
    generation = next(
        call.args[2]
        for call in stub_sdk_client.track.call_args_list
        if call.args[0] == "$ld:ai:offline-evals:generation"
    )
    assert "tool_calls" not in generation
    assert _TOOL_ARGUMENTS not in json.dumps(generation)
    assert _TOOL_RESULT not in json.dumps(generation)


@pytest.mark.asyncio
async def test_errored_judge_span_has_no_result_event(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    span_export: tuple[InMemorySpanExporter, list[str]],
) -> None:
    exporter, _flushed = span_export
    patch_judge(monkeypatch)
    evals = init_evaluations(
        project_key="proj",
        api_key="token",
        sdk_key="sdk-key",
        transport=one_row_transport(),
    )

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            return {"output": "not json"}
        return {"output": "shipped"}

    await evals.run(
        key="support-qa",
        dataset="golden",
        handler=handler,
        generation={"provider": "OpenAI", "model": "gpt-4o"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    evaluate = exporter.get_finished_spans()[0]
    assert evaluate.name == "evaluate $ld:ai:judge:accuracy"
    assert evaluate.status.status_code == StatusCode.ERROR
    assert evaluate.attributes is not None
    assert evaluate.attributes["error.type"] == "invalid_judge_output"
    assert_row_attributes(evaluate.attributes)
    assert not any(
        event.name == "gen_ai.evaluation.result" for event in evaluate.events
    )
    criterion = next(
        call.args[2]
        for call in stub_sdk_client.track.call_args_list
        if call.args[0] == "$ld:ai:offline-evals:criterion"
    )
    assert criterion["status"] == "ERROR"
    assert "verdict" not in criterion
    assert "score" not in criterion
