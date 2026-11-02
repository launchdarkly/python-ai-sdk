"""Offline evaluation spans join the customer project on the existing exporter."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from time import time_ns
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

from launchdarkly_ai_server import (
    EvalTool,
    Judge,
    SDKOptions,
    configure,
    create_handler,
    init_evaluations,
)
from launchdarkly_ai_server.conversation import (
    GEN_AI_CONVERSATION_ID,
    ConversationIdSpanProcessor,
)
from launchdarkly_ai_server.evaluations.api import HttpResponse
from launchdarkly_ai_server.evaluations.spans import (
    LD_EVALUATION_RUN_ID,
    LD_EVALUATION_RUN_ROW_INDEX,
    record_chat,
    record_evaluation_result,
    record_tool_calls,
    start_row_span,
)
from launchdarkly_ai_server.lifecycle import (
    _reset_for_testing,
    effective_capture_content,
)
from launchdarkly_ai_server.trajectory import ToolInvocation

_RUN_ID = "run-id"
_ROW_INDEX = 3
_CONVERSATION_ID = "faa03308-8fe7-5805-8c07-f82a3dbdb3f1"
_TOOL_ARGUMENTS = '{"order_id": "A1"}'
_TOOL_RESULT = "shipped"
_TOOL_ARGUMENTS_RENDERED = '{"order_id":"A1"}'


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
    yield from _install_provider(sdk_processor=True)


@pytest.fixture
def foreign_span_export() -> Iterator[tuple[InMemorySpanExporter, list[str]]]:
    """A provider someone else registered, without the SDK processor."""
    yield from _install_provider(sdk_processor=False)


def _install_provider(
    *, sdk_processor: bool
) -> Iterator[tuple[InMemorySpanExporter, list[str]]]:
    saved = otel_trace._TRACER_PROVIDER
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "python-sdk",
                "launchdarkly.project_id": "sdk-key",
            }
        )
    )
    if sdk_processor:
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


def patch_judge(
    monkeypatch: pytest.MonkeyPatch, prompt: dict[str, Any] | None = None
) -> None:
    async def fake_extract_variation(
        key: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        assert key == "$ld:ai:judge:accuracy"
        body = prompt or {"instructions": "Judge {{response_to_evaluate}}"}
        mode = "messages" if "messages" in body else "agent"
        return {
            "config": {
                "provider": {"name": "OpenAI"},
                "model": {"name": "gpt-4o"},
                **body,
            },
            "meta": {"variationKey": "default", "version": 12, "mode": mode},
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


def spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list[Any]]:
    by_name: dict[str, list[Any]] = {}
    for span in exporter.get_finished_spans():
        by_name.setdefault(span.name, []).append(span)
    return by_name


def parent_of(span: Any) -> int | None:
    return span.parent.span_id if span.parent is not None else None


def declare(fn: Callable[..., Any], mode: str = "agent") -> Any:
    """A plain function with the identity ``run()`` now requires for routing.

    The wrapper opens no span of its own, so the runner still describes the call.
    """
    return create_handler(("OpenAI", mode), fn)


def lookup_order_tool() -> EvalTool:
    return EvalTool(
        key="lookup_order",
        implementation=lambda args: _TOOL_RESULT,
        schema={"type": "object"},
        description="Look up an order",
    )


@pytest.fixture
def capture_on() -> Iterator[None]:
    """Transcript-shape cases pin capture on, independent of the process default."""
    _reset_for_testing()
    configure(options=SDKOptions(capture_content=True))
    yield
    _reset_for_testing()


@pytest.mark.asyncio
@pytest.mark.usefixtures("capture_on")
async def test_plain_function_row_is_one_trace_with_the_exchange_tools_and_judge(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    span_export: tuple[InMemorySpanExporter, list[str]],
) -> None:
    exporter, flushed = span_export
    forbid_new_exporter(monkeypatch)
    patch_judge(
        monkeypatch,
        {
            "messages": [
                {"role": "system", "content": "Judge {{response_to_evaluate}}"},
                {"role": "user", "content": "Transcript:\n{{message_history}}"},
            ]
        },
    )

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
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        assert "__ld" not in variables
        if config.get("messages"):
            return {
                "output": '{"score": 0.9, "reasoning": "Correct and concise."}',
                "usage": {"input_tokens": 30, "output_tokens": 9},
            }
        assert tool_handlers["lookup_order"]({"order_id": "A1"}) == _TOOL_RESULT
        return {"output": "shipped", "usage": {"input_tokens": 10, "output_tokens": 4}}

    result = await evals.run(
        key="support-qa",
        dataset="golden",
        handlers=[declare(handler), declare(handler, "messages")],
        tools=[lookup_order_tool()],
        generation={
            "provider": "OpenAI",
            "model": "gpt-4o",
            "mode": "agent",
            "instructions": "Answer about {{input}}",
        },
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    assert result.passed is True
    spans = exporter.get_finished_spans()
    assert {span.context.trace_id for span in spans} == {spans[0].context.trace_id}
    by_name = spans_by_name(exporter)
    assert sorted(by_name) == [
        "chat gpt-4o",
        "evaluate $ld:ai:judge:accuracy",
        "execute_tool lookup_order",
        "launchdarkly.dataset_row",
    ]
    for span in spans:
        assert span.resource is not None
        assert span.resource.attributes["launchdarkly.project_id"] == "sdk-key"
        assert_row_attributes(span.attributes)
        assert span.status.status_code == StatusCode.OK

    (row,) = by_name["launchdarkly.dataset_row"]
    assert row.parent is None
    assert "gen_ai.operation.name" not in (row.attributes or {})
    (evaluate,) = by_name["evaluate $ld:ai:judge:accuracy"]
    (tool,) = by_name["execute_tool lookup_order"]
    generation_chat, judge_chat = sorted(
        by_name["chat gpt-4o"], key=lambda span: span.start_time
    )
    for child in (generation_chat, tool, evaluate):
        assert parent_of(child) == row.context.span_id
    assert parent_of(judge_chat) == evaluate.context.span_id

    attributes = generation_chat.attributes or {}
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.provider.name"] == "openai"
    assert attributes["gen_ai.request.model"] == "gpt-4o"
    assert attributes["gen_ai.usage.input_tokens"] == 10
    assert attributes["gen_ai.usage.output_tokens"] == 4
    assert json.loads(str(attributes["gen_ai.system_instructions"])) == [
        {"type": "text", "content": "Answer about Where is order A1?"}
    ]
    assert json.loads(str(attributes["gen_ai.input.messages"])) == [
        {"role": "user", "parts": [{"type": "text", "content": "Where is order A1?"}]}
    ]
    assert json.loads(str(attributes["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "lookup_order",
                    "arguments": _TOOL_ARGUMENTS_RENDERED,
                },
                {"type": "text", "content": "shipped"},
            ],
        }
    ]
    assert json.loads(str(attributes["gen_ai.tool.definitions"]))[0]["name"] == (
        "lookup_order"
    )

    tool_attributes = tool.attributes or {}
    assert tool_attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_attributes["gen_ai.tool.name"] == "lookup_order"
    assert tool_attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool_attributes["gen_ai.tool.call.arguments"] == _TOOL_ARGUMENTS_RENDERED
    assert tool_attributes["gen_ai.tool.call.result"] == _TOOL_RESULT

    evaluate_attributes = evaluate.attributes or {}
    assert evaluate_attributes["gen_ai.operation.name"] == "evaluate"
    assert evaluate_attributes["gen_ai.request.model"] == "gpt-4o"
    assert "gen_ai.input.messages" not in evaluate_attributes
    event = next(
        item for item in evaluate.events if item.name == "gen_ai.evaluation.result"
    )
    assert event.attributes is not None
    assert event.attributes["gen_ai.evaluation.name"] == "$ld:ai:judge:accuracy"
    assert event.attributes["gen_ai.evaluation.score.value"] == 0.9
    assert event.attributes["gen_ai.evaluation.explanation"] == "Correct and concise."
    assert_no_verdict(event.attributes)

    judge_attributes = judge_chat.attributes or {}
    assert judge_attributes["gen_ai.operation.name"] == "chat"
    assert json.loads(str(judge_attributes["gen_ai.system_instructions"])) == [
        {"type": "text", "content": "Judge shipped"}
    ]
    history_turn, judged_turn = json.loads(
        str(judge_attributes["gen_ai.input.messages"])
    )
    assert history_turn["role"] == "user"
    (history_part,) = history_turn["parts"]
    history = history_part["content"]
    assert history.startswith("Transcript:\n")
    assert "{{message_history}}" not in history
    for fragment in ("Where is order A1?", "lookup_order", _TOOL_RESULT, "shipped"):
        assert fragment in history
    assert judged_turn == {
        "role": "user",
        "parts": [{"type": "text", "content": "shipped"}],
    }
    assert json.loads(str(judge_attributes["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": '{"score": 0.9, "reasoning": "Correct and concise."}',
                }
            ],
        }
    ]
    assert judge_attributes["gen_ai.usage.input_tokens"] == 30

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
        assert _TOOL_ARGUMENTS_RENDERED not in json.dumps(payload)
        assert "Where is order A1?" not in json.dumps(payload)
    generation = payloads[0]
    assert generation["output"] == "shipped"
    assert "message" not in generation
    criterion = payloads[1]
    assert criterion["score"] == 0.9
    assert criterion["reason"] == "Correct and concise."
    assert flushed == ["client", "traces"]

    # Each span is timed to the call it describes, so its window sits inside
    # its parent's. A canned handler makes these windows microseconds wide;
    # a duration floor would assert the machine's speed, containment asserts
    # that the span wrapped the call.
    for span in spans:
        assert span.end_time > span.start_time
    for child, parent in (
        (generation_chat, row),
        (tool, row),
        (evaluate, row),
        (judge_chat, evaluate),
        # The tool ran inside the handler, so its window is inside the
        # generation chat's even though the span is a sibling of it.
        (tool, generation_chat),
    ):
        assert parent.start_time <= child.start_time
        assert child.end_time <= parent.end_time


@pytest.mark.asyncio
@pytest.mark.usefixtures("capture_on")
async def test_plain_function_that_raises_records_the_failed_generation(
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
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        tool_handlers["lookup_order"]({"order_id": "A1"})
        raise RuntimeError("model refused")

    calls = 0

    await evals.run(
        key="support-qa",
        dataset="golden",
        handler=declare(handler),
        tools=[lookup_order_tool()],
        generation={"provider": "OpenAI", "model": "gpt-4o", "mode": "agent"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    by_name = spans_by_name(exporter)
    (row,) = by_name["launchdarkly.dataset_row"]
    assert row.status.status_code == StatusCode.ERROR
    (chat,) = by_name["chat gpt-4o"]
    assert chat.status.status_code == StatusCode.ERROR
    assert chat.status.description == "model refused"
    attributes = chat.attributes or {}
    assert attributes["error.type"] == "RuntimeError"
    assert "gen_ai.output.messages" not in attributes
    assert "gen_ai.input.messages" in attributes
    (tool,) = by_name["execute_tool lookup_order"]
    assert (tool.attributes or {})["gen_ai.tool.call.result"] == _TOOL_RESULT
    assert sorted(by_name) == [
        "chat gpt-4o",
        "execute_tool lookup_order",
        "launchdarkly.dataset_row",
    ]
    for child in (chat, tool):
        assert parent_of(child) == row.context.span_id
        assert child.context.trace_id == row.context.trace_id
    assert calls == 1


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
        history: list[dict[str, Any]] | None = None,
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
        handler=declare(handler),
        tools=[
            EvalTool(
                key="lookup_order",
                implementation=lambda args: _TOOL_RESULT,
                schema={"type": "object"},
                description="Look up an order",
            )
        ],
        generation={"provider": "OpenAI", "model": "gpt-4o", "mode": "agent"},
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
    # The handler traced itself, so the runner adds no chat or tool span.
    assert len(by_name["chat gpt-4o"]) == 2
    assert len(by_name["execute_tool lookup_order"]) == 1
    (row,) = by_name["launchdarkly.dataset_row"]
    assert {
        span.context.trace_id for span in spans if span.name != "POST /v1/chat"
    } == {row.context.trace_id}
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
async def test_a_foreign_provider_gets_no_second_copy_of_handler_spans(
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_client: MagicMock,
    foreign_span_export: tuple[InMemorySpanExporter, list[str]],
) -> None:
    # Without the SDK processor the span count stays 0 even though the handler
    # traced itself, so the runner must not write its own chat spans too.
    exporter, _flushed = foreign_span_export
    patch_judge(monkeypatch)
    evals = init_evaluations(
        project_key="proj",
        api_key="token",
        sdk_key="sdk-key",
        transport=one_row_transport(),
    )
    tracer = otel_trace.get_tracer("@launchdarkly/ai-server")

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Callable[..., Any]],
        variables: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with tracer.start_as_current_span("invoke_agent"):
            with tracer.start_as_current_span("chat gpt-4o"):
                pass
        if "Judge" in config.get("instructions", ""):
            return {"output": '{"score": 0.9, "reasoning": "ok"}'}
        return {"output": "done"}

    await evals.run(
        key="support-qa",
        dataset="golden",
        handler=declare(handler),
        generation={"provider": "OpenAI", "model": "gpt-4o", "mode": "agent"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    by_name = spans_by_name(exporter)
    # One from the generation and one from the judge, each written once.
    assert len(by_name["chat gpt-4o"]) == 2
    assert len(by_name["invoke_agent"]) == 2


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
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if "Judge" in config.get("instructions", ""):
            return {"output": "not json"}
        return {"output": "shipped"}

    await evals.run(
        key="support-qa",
        dataset="golden",
        handler=declare(handler),
        generation={"provider": "OpenAI", "model": "gpt-4o", "mode": "agent"},
        criteria=[Judge(key="$ld:ai:judge:accuracy")],
    )

    (evaluate,) = spans_by_name(exporter)["evaluate $ld:ai:judge:accuracy"]
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


_CONTENT_KEYS = (
    "gen_ai.system_instructions",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.definitions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "gen_ai.prompt.0.content",
    "gen_ai.completion.0.content",
)


def _write_captured_row(*, capture: bool | None = None) -> None:
    started = time_ns()
    row = start_row_span("run-id", 0)
    chat_capture = {} if capture is None else {"capture": capture}
    record_chat(
        row,
        config={
            "provider": {"name": "OpenAI"},
            "model": {"name": "gpt-4o"},
            "instructions": "Answer about {{topic}}",
            "messages": [{"role": "user", "content": "earlier {{topic}}"}],
            "tools": {
                "lookup": {
                    "name": "lookup",
                    "description": "look up a place",
                    "parameters": {"type": "object"},
                }
            },
        },
        user_input="Where is Paris?",
        variables={"topic": "capitals"},
        started_ns=started,
        ended_ns=started + 1_000,
        result={
            "output": "France",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        },
        tool_calls=[
            ToolInvocation(
                name="lookup",
                arguments='{"q":"Paris"}',
                result="France",
                started_ns=started,
                ended_ns=started + 100,
            )
        ],
        **chat_capture,
    )
    record_tool_calls(
        row,
        [
            ToolInvocation(
                name="lookup",
                arguments='{"q":"Paris"}',
                result="France",
                started_ns=started,
                ended_ns=started + 100,
            )
        ],
        **chat_capture,
    )
    evaluate = otel_trace.get_tracer("@launchdarkly/ai-server").start_span(
        "evaluate accuracy"
    )
    record_evaluation_result(
        evaluate, "accuracy", 0.9, "matches the capital", **chat_capture
    )
    evaluate.end()
    row.end()


def test_offline_spans_follow_the_capture_content_gate(
    span_export: tuple[InMemorySpanExporter, list[str]],
) -> None:
    """Evaluation spans export the transcript by default and omit it when capture is off.

    An explicit false on the call removes it while the process default is on. An explicit
    true on the call still writes it when the process default is false.
    """
    exporter, _flushed = span_export
    _reset_for_testing()
    try:
        assert SDKOptions().capture_content is True
        assert effective_capture_content(None) is True

        _write_captured_row()
        by_name = spans_by_name(exporter)
        chat_attributes = by_name["chat gpt-4o"][0].attributes or {}
        assert "Answer about capitals" in str(
            chat_attributes["gen_ai.system_instructions"]
        )
        rendered_input = str(chat_attributes["gen_ai.input.messages"])
        assert "Where is Paris?" in rendered_input
        assert "earlier capitals" in rendered_input
        assert "France" in str(chat_attributes["gen_ai.output.messages"])
        assert "lookup" in str(chat_attributes["gen_ai.tool.definitions"])
        tool_attributes = by_name["execute_tool lookup"][0].attributes or {}
        assert tool_attributes["gen_ai.tool.call.arguments"] == '{"q":"Paris"}'
        assert tool_attributes["gen_ai.tool.call.result"] == "France"
        explanation = next(
            event.attributes
            for event in by_name["evaluate accuracy"][0].events
            if event.name == "gen_ai.evaluation.result"
        )
        assert explanation is not None
        assert explanation["gen_ai.evaluation.explanation"] == "matches the capital"
        assert explanation["gen_ai.evaluation.name"] == "accuracy"
        assert explanation["gen_ai.evaluation.score.value"] == 0.9

        exporter.clear()
        _write_captured_row(capture=False)
        by_name = spans_by_name(exporter)
        for attributes in (
            by_name["chat gpt-4o"][0].attributes or {},
            by_name["execute_tool lookup"][0].attributes or {},
        ):
            for key in _CONTENT_KEYS:
                assert key not in attributes
        stripped = next(
            event.attributes
            for event in by_name["evaluate accuracy"][0].events
            if event.name == "gen_ai.evaluation.result"
        )
        assert stripped is not None
        assert "gen_ai.evaluation.explanation" not in stripped
        assert stripped["gen_ai.evaluation.name"] == "accuracy"
        assert stripped["gen_ai.evaluation.score.value"] == 0.9

        exporter.clear()
        configure(options=SDKOptions(capture_content=False))
        assert effective_capture_content(None) is False
        _write_captured_row()
        by_name = spans_by_name(exporter)
        for attributes in (
            by_name["chat gpt-4o"][0].attributes or {},
            by_name["execute_tool lookup"][0].attributes or {},
        ):
            for key in _CONTENT_KEYS:
                assert key not in attributes
        result = next(
            event.attributes
            for event in by_name["evaluate accuracy"][0].events
            if event.name == "gen_ai.evaluation.result"
        )
        assert result is not None
        assert "gen_ai.evaluation.explanation" not in result
        assert result["gen_ai.evaluation.name"] == "accuracy"
        assert result["gen_ai.evaluation.score.value"] == 0.9
        assert not any(
            "Paris" in str(value)
            for value in (by_name["chat gpt-4o"][0].attributes or {}).values()
        )

        exporter.clear()
        _write_captured_row(capture=True)
        kept = spans_by_name(exporter)["chat gpt-4o"][0].attributes or {}
        assert "Where is Paris?" in str(kept["gen_ai.input.messages"])
    finally:
        _reset_for_testing()
