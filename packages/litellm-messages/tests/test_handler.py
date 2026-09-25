"""Contract tests for the future LiteLLM messages package.

These imports are intentionally unresolved at the test-first checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_litellm_messages.handler as handler_mod
from launchdarkly_ai_litellm_messages import (
    create_litellm_messages_handler,
    litellm_messages,
)

CONFIG: dict[str, Any] = {
    "provider": {"name": "Anthropic"},
    "model": {"name": "anthropic/claude-sonnet-4"},
    "instructions": "Help {{name}}.",
}


def _message(
    content: str | None = "done",
    *,
    tool_calls: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _response(
    content: str | None = "done",
    *,
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 3,
    completion_tokens: int = 2,
) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_message(content, tool_calls=tool_calls))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        model="resolved-by-litellm",
    )


def _tool_call(name: str, call_id: str, arguments: str) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def _events(stream: Any) -> list[dict[str, Any]]:
    return [event async for event in stream]


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class RecordedSpan:
    def __init__(self, name: str, context: Any = None) -> None:
        self.name = name
        self.context = context
        self.attributes: dict[str, Any] = {}
        self.exceptions: list[BaseException] = []
        self.ended = 0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Any = None) -> None:
        pass

    def set_status(self, code: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, error: BaseException) -> None:
        self.exceptions.append(error)

    def end(self) -> None:
        self.ended += 1


class SpanRecorder:
    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    def get_tracer(self, name: str) -> SpanRecorder:
        return self

    def start_span(self, name: str, context: Any = None) -> RecordedSpan:
        span = RecordedSpan(name, context)
        self.spans.append(span)
        return span

    def set_span_in_context(self, span: RecordedSpan) -> Any:
        return ("context-of", span)


def _chunk(
    *,
    text: str | None = None,
    tool_calls: list[Any] | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> Any:
    usage = None
    if prompt_tokens is not None or completion_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=tool_calls or [])
            )
        ],
        usage=usage,
    )


class TestFactoryAndTransport:
    def test_wildcard_metadata_and_capture_content(self) -> None:
        handler = create_litellm_messages_handler(capture_content=True)
        assert handler.provides_for == ("*", "messages")
        assert handler.capture_content is True
        assert create_litellm_messages_handler().capture_content is False

    async def test_calls_in_process_litellm_acompletion(self) -> None:
        completion = AsyncMock(return_value=_response())
        handler = create_litellm_messages_handler(completion=completion)
        await handler(CONFIG, "question", {}, {"name": "Ada"})
        completion.assert_awaited_once()
        assert completion.call_args.kwargs["model"] == "anthropic/claude-sonnet-4"

    async def test_never_constructs_a_provider_direct_client(self) -> None:
        completion = AsyncMock(return_value=_response())
        forbidden = MagicMock(side_effect=AssertionError("provider-direct path"))
        with (
            patch.dict(
                "sys.modules",
                {
                    "anthropic": SimpleNamespace(AsyncAnthropic=forbidden),
                    "openai": SimpleNamespace(AsyncOpenAI=forbidden),
                    "boto3": SimpleNamespace(client=forbidden),
                },
            ),
        ):
            await create_litellm_messages_handler(completion=completion)(
                CONFIG, "question"
            )
        forbidden.assert_not_called()

    async def test_evaluated_model_and_owned_parameter_precedence(self) -> None:
        completion = AsyncMock(return_value=_response())
        config = {
            **CONFIG,
            "model": {
                "name": "proxy/team-alias",
                "parameters": {
                    "temperature": 0.4,
                    "model": "wrong",
                    "messages": ["wrong"],
                    "tools": ["wrong"],
                    "stream": True,
                    "response_format": {"type": "wrong"},
                },
            },
        }
        await create_litellm_messages_handler(completion=completion)(config, "question")
        request = completion.call_args.kwargs
        assert request["model"] == "proxy/team-alias"
        assert request["temperature"] == 0.4
        assert request["messages"] != ["wrong"]
        assert request.get("tools") != ["wrong"]
        assert request["stream"] is False
        assert "response_format" not in request


class TestPromptMapping:
    async def test_messages_take_precedence_over_instructions(self) -> None:
        completion = AsyncMock(return_value=_response())
        config = {
            **CONFIG,
            "instructions": "ignored",
            "messages": [
                {"role": "system", "content": "System {{name}}"},
                {"role": "assistant", "content": "prior"},
            ],
        }
        await create_litellm_messages_handler(completion=completion)(
            config, "new", {}, {"name": "Ada"}
        )
        messages = completion.call_args.kwargs["messages"]
        assert messages == [
            {"role": "system", "content": "System Ada"},
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "new"},
        ]
        assert "ignored" not in str(messages)

    async def test_last_user_config_message_prevents_duplicate_input(self) -> None:
        completion = AsyncMock(return_value=_response())
        config = {
            **CONFIG,
            "messages": [{"role": "user", "content": "{{user_input}}"}],
        }
        await create_litellm_messages_handler(completion=completion)(
            config, "hello", {}, {"user_input": "hello"}
        )
        messages = completion.call_args.kwargs["messages"]
        assert messages == [{"role": "user", "content": "hello"}]

    async def test_history_precedes_prompt_and_maps_multimodal_blocks(self) -> None:
        completion = AsyncMock(return_value=_response())
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "old answer"},
        ]
        await create_litellm_messages_handler(completion=completion)(
            CONFIG, "new question", {}, {"name": "Ada"}, history
        )
        messages = completion.call_args.kwargs["messages"]
        assert messages[1]["content"] == [
            {"type": "text", "text": "look"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            },
        ]
        assert messages[-1] == {"role": "user", "content": "new question"}


class TestToolsAndUsage:
    async def test_filters_tools_without_callable_handlers(self) -> None:
        completion = AsyncMock(return_value=_response())
        config = {
            **CONFIG,
            "tools": {
                "kept": {"description": "yes", "parameters": {"type": "object"}},
                "missing": {"description": "no", "parameters": {}},
                "not_callable": {"description": "no", "parameters": {}},
            },
        }
        await create_litellm_messages_handler(completion=completion)(
            config,
            "q",
            {"kept": AsyncMock(), "not_callable": object()},
        )
        assert completion.call_args.kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "kept",
                    "description": "yes",
                    "parameters": {"type": "object"},
                },
            }
        ]

    async def test_single_tool_call_appends_assistant_and_result(self) -> None:
        call = _tool_call("weather", "call-1", '{"city":"Austin"}')
        completion = AsyncMock(
            side_effect=[_response(None, tool_calls=[call]), _response("sunny")]
        )
        tool = AsyncMock(return_value={"forecast": "sunny"})
        config = {
            **CONFIG,
            "tools": {"weather": {"description": "weather", "parameters": {}}},
        }
        result = await create_litellm_messages_handler(completion=completion)(
            config, "q", {"weather": tool}
        )
        tool.assert_awaited_once_with({"city": "Austin"})
        follow_up = completion.call_args_list[1].kwargs["messages"]
        assert follow_up[-2]["role"] == "assistant"
        assert follow_up[-2]["tool_calls"][0]["id"] == "call-1"
        assert follow_up[-1] == {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps({"forecast": "sunny"}),
        }
        assert result["output"] == "sunny"

    async def test_parallel_and_consecutive_calls_all_execute(self) -> None:
        calls = [
            _tool_call("a", "a1", '{"x":1}'),
            _tool_call("b", "b1", '{"y":2}'),
        ]
        third = _tool_call("a", "a2", '{"x":3}')
        completion = AsyncMock(
            side_effect=[
                _response(None, tool_calls=calls, prompt_tokens=2, completion_tokens=1),
                _response(
                    None, tool_calls=[third], prompt_tokens=3, completion_tokens=1
                ),
                _response("done", prompt_tokens=5, completion_tokens=2),
            ]
        )
        a = AsyncMock(side_effect=["A1", "A2"])
        b = AsyncMock(return_value="B1")
        config = {
            **CONFIG,
            "tools": {
                "a": {"description": "a", "parameters": {}},
                "b": {"description": "b", "parameters": {}},
            },
        }
        result = await create_litellm_messages_handler(completion=completion)(
            config, "q", {"a": a, "b": b}
        )
        assert a.await_args_list[0].args == ({"x": 1},)
        b.assert_awaited_once_with({"y": 2})
        assert a.await_args_list[1].args == ({"x": 3},)
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 4}


class TestStructuredOutput:
    async def test_response_format_is_handler_owned_and_json_is_best_effort(
        self,
    ) -> None:
        completion = AsyncMock(return_value=_response('{"answer": 42}'))
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        result = await create_litellm_messages_handler(completion=completion)(
            {**CONFIG, "outputFormat": schema}, "q"
        )
        assert completion.call_args.kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "output", "strict": False, "schema": schema},
        }
        assert result["output"] == {"answer": 42}

    async def test_response_format_remains_on_final_tool_turn(self) -> None:
        call = _tool_call("lookup", "c1", "{}")
        completion = AsyncMock(
            side_effect=[_response(None, tool_calls=[call]), _response('{"ok":true}')]
        )
        config = {
            **CONFIG,
            "outputFormat": {"type": "object"},
            "tools": {"lookup": {"description": "x", "parameters": {}}},
        }
        await create_litellm_messages_handler(completion=completion)(
            config, "q", {"lookup": AsyncMock(return_value="ok")}
        )
        assert "response_format" in completion.call_args_list[0].kwargs
        assert "response_format" in completion.call_args_list[1].kwargs


class TestStreaming:
    async def test_streaming_ignores_output_format(self) -> None:
        stream = FakeStream([_chunk(text='{"answer":42}')])
        completion = AsyncMock(return_value=stream)
        events = await _events(
            await create_litellm_messages_handler(completion=completion).stream(
                {**CONFIG, "outputFormat": {"type": "object"}}, "q"
            )
        )
        assert "response_format" not in completion.call_args.kwargs
        assert events[-1]["output"] == '{"answer":42}'

    async def test_text_chunks_usage_and_single_done(self) -> None:
        stream = FakeStream(
            [
                _chunk(text="Hel"),
                _chunk(text="lo"),
                _chunk(prompt_tokens=4, completion_tokens=2),
            ]
        )
        completion = AsyncMock(return_value=stream)
        handler = create_litellm_messages_handler(completion=completion)
        events = await _events(await handler.stream(CONFIG, "q"))
        assert [e["text"] for e in events if e["type"] == "chunk"] == ["Hel", "lo"]
        assert events[-1] == {
            "type": "done",
            "output": "Hello",
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }
        assert sum(e["type"] == "done" for e in events) == 1
        request = completion.call_args.kwargs
        assert request["stream"] is True
        assert request["stream_options"] == {"include_usage": True}

    async def test_fragmented_tool_call_is_reassembled_between_turns(self) -> None:
        first = FakeStream(
            [
                _chunk(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call-",
                            function=SimpleNamespace(
                                name="weath", arguments='{"city":'
                            ),
                        )
                    ]
                ),
                _chunk(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="1",
                            function=SimpleNamespace(name="er", arguments='"Oslo"}'),
                        )
                    ]
                ),
                _chunk(prompt_tokens=2, completion_tokens=1),
            ]
        )
        second = FakeStream(
            [_chunk(text="cold"), _chunk(prompt_tokens=3, completion_tokens=1)]
        )
        completion = AsyncMock(side_effect=[first, second])
        weather = AsyncMock(return_value="cold")
        config = {
            **CONFIG,
            "tools": {"weather": {"description": "x", "parameters": {}}},
        }
        events = await _events(
            await create_litellm_messages_handler(completion=completion).stream(
                config, "q", {"weather": weather}
            )
        )
        weather.assert_awaited_once_with({"city": "Oslo"})
        assert events[-1]["usage"] == {"input_tokens": 5, "output_tokens": 2}

    async def test_abandonment_closes_provider_stream(self) -> None:
        stream = FakeStream([_chunk(text="one"), _chunk(text="two")])
        completion = AsyncMock(return_value=stream)
        gen = await create_litellm_messages_handler(completion=completion).stream(
            CONFIG, "q"
        )
        async for _ in gen:
            break
        await gen.aclose()
        assert stream.closed is True

    async def test_stream_error_closes_transport_and_propagates(self) -> None:
        class BrokenStream(FakeStream):
            async def _iterate(self) -> AsyncIterator[Any]:
                yield _chunk(text="partial")
                raise RuntimeError("stream failed")

        stream = BrokenStream([])
        completion = AsyncMock(return_value=stream)
        with pytest.raises(RuntimeError, match="stream failed"):
            await _events(
                await create_litellm_messages_handler(completion=completion).stream(
                    CONFIG, "q"
                )
            )
        assert stream.closed is True


class TestTelemetryAndConvenience:
    async def test_nested_span_tree_identity_usage_and_parenting(self) -> None:
        call = _tool_call("lookup", "call-1", "{}")
        completion = AsyncMock(
            side_effect=[_response(None, tool_calls=[call]), _response("done")]
        )
        recorder = SpanRecorder()
        config = {
            **CONFIG,
            "tools": {"lookup": {"description": "lookup", "parameters": {}}},
        }
        with patch.object(handler_mod, "trace", recorder):
            await create_litellm_messages_handler(completion=completion)(
                config, "q", {"lookup": AsyncMock(return_value="found")}
            )

        assert [span.name for span in recorder.spans] == [
            "invoke_agent",
            "chat anthropic/claude-sonnet-4",
            "execute_tool lookup",
            "chat anthropic/claude-sonnet-4",
        ]
        root = recorder.spans[0]
        for child in recorder.spans[1:]:
            assert child.context == ("context-of", root)
        for span in recorder.spans[:2] + recorder.spans[3:]:
            assert span.attributes["gen_ai.provider.name"] == "anthropic"
            assert span.attributes["gen_ai.system"] == "litellm"
            assert span.ended == 1
        assert root.attributes["gen_ai.usage.input_tokens"] == 6
        assert recorder.spans[2].attributes["gen_ai.tool.call.id"] == "call-1"
        assert recorder.spans[2].ended == 1

    async def test_model_and_root_spans_fail_and_end_once(self) -> None:
        recorder = SpanRecorder()
        error = RuntimeError("unavailable")
        with (
            patch.object(handler_mod, "trace", recorder),
            pytest.raises(RuntimeError, match="unavailable"),
        ):
            await create_litellm_messages_handler(
                completion=AsyncMock(side_effect=error)
            )(CONFIG, "q")
        assert [span.name for span in recorder.spans] == [
            "invoke_agent",
            "chat anthropic/claude-sonnet-4",
        ]
        assert all(span.exceptions == [error] for span in recorder.spans)
        assert all(span.ended == 1 for span in recorder.spans)

    async def test_identity_and_content_are_gated(self) -> None:
        completion = AsyncMock(return_value=_response("secret"))
        span = MagicMock()
        trace = MagicMock()
        trace.get_tracer.return_value.start_span.return_value = span
        variables = {
            "__ld": {
                "configKey": "cfg",
                "variationKey": "v1",
                "runId": "run",
            },
            "ldContext": {"kind": "user", "key": "ada"},
        }
        with patch.object(handler_mod, "trace", trace):
            await create_litellm_messages_handler(completion=completion)(
                CONFIG, "private prompt", {}, variables
            )
        attributes = {
            call.args[0]: call.args[1] for call in span.set_attribute.call_args_list
        }
        assert attributes["launchdarkly.config.key"] == "cfg"
        assert attributes["gen_ai.request.model"] == "anthropic/claude-sonnet-4"
        assert not any(key.startswith("gen_ai.prompt.") for key in attributes)

    async def test_capture_content_records_prompt_and_completion(self) -> None:
        completion = AsyncMock(return_value=_response("answer"))
        span = MagicMock()
        trace = MagicMock()
        trace.get_tracer.return_value.start_span.return_value = span
        with patch.object(handler_mod, "trace", trace):
            await create_litellm_messages_handler(
                completion=completion, capture_content=True
            )(CONFIG, "question")
        written = str(span.set_attribute.call_args_list)
        assert "question" in written
        assert "answer" in written

    async def test_stream_capture_content_records_prompt_and_completion_on_root(
        self,
    ) -> None:
        stream = FakeStream([_chunk(text="streamed")])
        completion = AsyncMock(return_value=stream)
        span = MagicMock()
        trace = MagicMock()
        trace.get_tracer.return_value.start_span.return_value = span
        with patch.object(handler_mod, "trace", trace):
            events = await _events(
                await create_litellm_messages_handler(
                    completion=completion, capture_content=True
                ).stream(CONFIG, "question")
            )
        written = str(span.set_attribute.call_args_list)
        assert "question" in written
        assert "streamed" in written
        assert events[-1]["type"] == "done"

    def test_convenience_wrapper_forwards_all_public_arguments(self) -> None:
        instance = MagicMock()
        instance.invoke.return_value = "result"
        config = MagicMock(return_value=instance)
        with patch.object(handler_mod, "config", config):
            result = litellm_messages(
                "config-key",
                "hello",
                {"kind": "user", "key": "u"},
                variables={"name": "Ada"},
                capture_content=True,
            )
        assert result == "result"
        assert config.call_args.kwargs["key"] == "config-key"
        assert config.call_args.kwargs["handler"].provides_for == ("*", "messages")
        assert config.call_args.kwargs["handler"].capture_content is True
        assert "capture_content" not in config.call_args.kwargs
        instance.invoke.assert_called_once_with(
            "hello", {"kind": "user", "key": "u"}, {"name": "Ada"}
        )
