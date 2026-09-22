"""Test-first contract for the wildcard Vercel AI SDK messages adapter.

These tests intentionally precede ``launchdarkly_ai_vercel_messages`` production
modules. They use only in-memory ``ai`` doubles and must never open a network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_vercel_messages.handler as handler_mod
from launchdarkly_ai_vercel_messages import create_vercel_messages_handler

CONFIG: dict[str, Any] = {
    "model": {"name": "anthropic/claude-sonnet-4"},
    "provider": {"name": "Anthropic"},
    "instructions": "Help {{name}}.",
}


class TextDelta:
    def __init__(self, chunk: str) -> None:
        self.chunk = chunk


class FakeStream:
    """Mirrors ``ai.Stream``, including the ``message`` whose pending tool calls
    decide whether the adapter has to run another turn."""

    def __init__(
        self,
        events: list[Any] | None = None,
        *,
        text: str = "answer",
        input_tokens: int = 10,
        output_tokens: int = 4,
        output: Any = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.events = list(events or [TextDelta(text)])
        self.text = text
        self.output = output if output is not None else text
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.message = SimpleNamespace(
            role="assistant",
            text=text,
            tool_calls=list(tool_calls or []),
            usage=self.usage,
        )
        self.entered = 0
        self.exited = 0
        self.consumed = 0

    async def __aenter__(self) -> FakeStream:
        self.entered += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited += 1

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        if not self.events:
            raise StopAsyncIteration
        self.consumed += 1
        return self.events.pop(0)


@pytest.fixture
def ai_runtime() -> MagicMock:
    """Patch every Vercel runtime entry point with an in-memory implementation."""
    runtime = MagicMock()
    runtime.events.TextDelta = TextDelta
    runtime.get_model = MagicMock(return_value=object())
    runtime.system_message = MagicMock(
        side_effect=lambda content: {"role": "system", "content": content}
    )
    runtime.user_message = MagicMock(
        side_effect=lambda *content: {
            "role": "user",
            "content": content[0] if len(content) == 1 else list(content),
        }
    )
    runtime.assistant_message = MagicMock(
        side_effect=lambda *content: {
            "role": "assistant",
            "content": content[0] if len(content) == 1 else list(content),
        }
    )
    runtime.file_part = MagicMock(
        side_effect=lambda value, media_type=None: {
            "type": "file",
            "value": value,
            "media_type": media_type,
        }
    )
    runtime.tool = MagicMock(
        side_effect=lambda fn=None, **kwargs: (
            SimpleNamespace(execute=fn, **kwargs) if fn is not None else kwargs
        )
    )
    runtime.InferenceRequestParams = MagicMock(return_value=object())
    runtime.stream = MagicMock(return_value=FakeStream())
    with patch.object(handler_mod, "ai", runtime):
        yield runtime


def _stream_kwargs(runtime: MagicMock) -> dict[str, Any]:
    return runtime.stream.call_args.kwargs


class TestFactoryAndModelResolution:
    def test_advertises_wildcard_messages(self) -> None:
        assert create_vercel_messages_handler().provides_for == ("*", "messages")

    @pytest.mark.asyncio
    async def test_gateway_model_id_is_passed_unchanged_to_get_model(
        self, ai_runtime: MagicMock
    ) -> None:
        await create_vercel_messages_handler()(CONFIG, "hello", {}, {"name": "Ada"})
        ai_runtime.get_model.assert_called_once_with("anthropic/claude-sonnet-4")

    @pytest.mark.asyncio
    async def test_builds_gateway_creator_model_id(self, ai_runtime: MagicMock) -> None:
        config = {
            **CONFIG,
            "model": {"name": "grok-4.5"},
            "provider": {"name": "xAI"},
        }
        await create_vercel_messages_handler()(config, "hello")
        ai_runtime.get_model.assert_called_once_with("spacexai/grok-4.5")

    @pytest.mark.asyncio
    async def test_converts_dotted_creator_prefix(self, ai_runtime: MagicMock) -> None:
        config = {
            **CONFIG,
            "model": {"name": "openai.gpt-5.6-sol"},
            "provider": {"name": "OpenAI"},
        }
        await create_vercel_messages_handler()(config, "hello")
        ai_runtime.get_model.assert_called_once_with("openai/gpt-5.6-sol")

    @pytest.mark.asyncio
    async def test_does_not_construct_a_provider_native_client(
        self, ai_runtime: MagicMock
    ) -> None:
        forbidden = {
            "openai": MagicMock(),
            "anthropic": MagicMock(),
            "boto3": MagicMock(),
            "google.generativeai": MagicMock(),
        }
        with patch.dict("sys.modules", forbidden):
            await create_vercel_messages_handler()(CONFIG, "hello")
        for module in forbidden.values():
            assert not module.mock_calls

    @pytest.mark.asyncio
    async def test_injected_model_instance_bypasses_get_model(
        self, ai_runtime: MagicMock
    ) -> None:
        model = object()
        await create_vercel_messages_handler(model=model)(CONFIG, "hello")
        ai_runtime.get_model.assert_not_called()
        assert _stream_kwargs(ai_runtime)["model"] is model

    @pytest.mark.asyncio
    async def test_sync_factory_receives_evaluated_config_once(
        self, ai_runtime: MagicMock
    ) -> None:
        model = object()
        factory = MagicMock(return_value=model)
        await create_vercel_messages_handler(model=factory)(CONFIG, "hello")
        factory.assert_called_once_with(CONFIG)
        assert _stream_kwargs(ai_runtime)["model"] is model

    @pytest.mark.asyncio
    async def test_async_factory_is_supported(self, ai_runtime: MagicMock) -> None:
        model = object()
        factory = AsyncMock(return_value=model)
        await create_vercel_messages_handler(model=factory)(CONFIG, "hello")
        factory.assert_awaited_once_with(CONFIG)
        assert _stream_kwargs(ai_runtime)["model"] is model

    @pytest.mark.asyncio
    async def test_handler_instances_keep_model_sources_isolated(
        self, ai_runtime: MagicMock
    ) -> None:
        first, second = object(), object()
        await create_vercel_messages_handler(model=first)(CONFIG, "one")
        await create_vercel_messages_handler(model=second)(CONFIG, "two")
        assert [call.kwargs["model"] for call in ai_runtime.stream.call_args_list] == [
            first,
            second,
        ]


class TestOwnedParameters:
    @pytest.mark.asyncio
    async def test_forwards_generation_settings_and_strips_owned_fields(
        self, ai_runtime: MagicMock
    ) -> None:
        owned = {
            "model": "evil",
            "messages": [],
            "prompt": "evil",
            "system": "evil",
            "tools": ["evil"],
            "stream": False,
            "output": "evil",
            "outputFormat": {},
            "output_format": {},
            "stopWhen": "evil",
            "stop_when": "evil",
            "maxSteps": 999,
            "max_steps": 999,
            "apiKey": "secret",
            "api_key": "secret",
            "baseURL": "https://evil.invalid",
            "base_url": "https://evil.invalid",
        }
        config = {
            **CONFIG,
            "model": {
                "name": CONFIG["model"]["name"],
                "parameters": {"temperature": 0.2, "top_p": 0.8, **owned},
            },
        }
        await create_vercel_messages_handler()(config, "hello")
        kwargs = _stream_kwargs(ai_runtime)
        assert kwargs["params"] is ai_runtime.InferenceRequestParams.return_value
        ai_runtime.TemperatureSamplerParams.assert_called_once_with(temperature=0.2)
        ai_runtime.TopPSamplerParams.assert_called_once_with(top_p=0.8)
        constructor_args = repr(ai_runtime.InferenceRequestParams.call_args)
        for key in owned:
            assert key not in constructor_args


class TestMessagesAndHistory:
    @pytest.mark.asyncio
    async def test_uses_ai_stream_and_native_message_constructors(
        self, ai_runtime: MagicMock
    ) -> None:
        await create_vercel_messages_handler()(CONFIG, "hello", {}, {"name": "Ada"})
        ai_runtime.stream.assert_called_once()
        ai_runtime.system_message.assert_called_once_with("Help Ada.")
        ai_runtime.user_message.assert_called_with("hello")

    @pytest.mark.asyncio
    async def test_config_history_and_input_are_ordered(
        self, ai_runtime: MagicMock
    ) -> None:
        config = {
            **CONFIG,
            "instructions": None,
            "messages": [{"role": "user", "content": "config turn"}],
        }
        history = [{"role": "assistant", "content": "history turn"}]
        await create_vercel_messages_handler()(config, "latest", {}, {}, history)
        messages = _stream_kwargs(ai_runtime)["messages"]
        assert [message["content"] for message in messages] == [
            "config turn",
            "history turn",
            "latest",
        ]

    @pytest.mark.asyncio
    async def test_multimodal_history_uses_file_part(
        self, ai_runtime: MagicMock
    ) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "YWJj",
                        },
                    },
                ],
            }
        ]
        await create_vercel_messages_handler()(CONFIG, None, {}, {}, history)
        ai_runtime.file_part.assert_called()
        args = ai_runtime.file_part.call_args
        assert "image/png" in str(args)
        assert "b'abc'" in str(args)


class TestToolsAndOutput:
    @pytest.mark.asyncio
    async def test_only_tools_with_callable_handlers_are_offered(
        self, ai_runtime: MagicMock
    ) -> None:
        config = {
            **CONFIG,
            "tools": {
                "weather": {
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {}}},
                },
                "missing": {"description": "No implementation", "parameters": {}},
            },
        }
        weather = AsyncMock(return_value="sunny")
        await create_vercel_messages_handler()(config, "hello", {"weather": weather})
        tools = _stream_kwargs(ai_runtime)["tools"]
        assert [tool.name for tool in tools] == ["weather"]
        assert tools[0].description == "Get weather"
        assert tools[0].input_schema == config["tools"]["weather"]["parameters"]

    @pytest.mark.asyncio
    async def test_runtime_tool_executes_registered_callable(
        self, ai_runtime: MagicMock
    ) -> None:
        config = {
            **CONFIG,
            "tools": {"weather": {"description": "Get weather", "parameters": {}}},
        }
        weather = AsyncMock(return_value="sunny")
        await create_vercel_messages_handler()(config, "hello", {"weather": weather})
        tool = _stream_kwargs(ai_runtime)["tools"][0]
        assert await tool.execute(city="Oakland") == "sunny"
        weather.assert_awaited_once_with({"city": "Oakland"})

    @pytest.mark.asyncio
    async def test_executes_tool_calls_and_continues_until_a_final_answer(
        self, ai_runtime: MagicMock
    ) -> None:
        config = {
            **CONFIG,
            "tools": {"weather": {"description": "Get weather", "parameters": {}}},
        }
        call = SimpleNamespace(
            tool_call_id="tc-1", tool_name="weather", tool_args={"city": "Oakland"}
        )
        ai_runtime.stream.side_effect = [
            FakeStream(text="", tool_calls=[call], input_tokens=10, output_tokens=4),
            FakeStream(text="sunny in Oakland", input_tokens=7, output_tokens=2),
        ]
        weather = AsyncMock(return_value="sunny")

        result = await create_vercel_messages_handler()(
            config, "hello", {"weather": weather}
        )

        weather.assert_awaited_once_with({"city": "Oakland"})
        assert result["output"] == "sunny in Oakland"
        assert result["usage"] == {"input_tokens": 17, "output_tokens": 6}
        assert ai_runtime.stream.call_count == 2
        ai_runtime.tool_result_part.assert_called_once_with(
            "tc-1", tool_name="weather", result="sunny"
        )

    @pytest.mark.asyncio
    async def test_reports_a_failing_tool_back_to_the_model(
        self, ai_runtime: MagicMock
    ) -> None:
        config = {
            **CONFIG,
            "tools": {"weather": {"description": "Get weather", "parameters": {}}},
        }
        call = SimpleNamespace(tool_call_id="tc-1", tool_name="weather", tool_args={})
        ai_runtime.stream.side_effect = [
            FakeStream(text="", tool_calls=[call]),
            FakeStream(text="could not check"),
        ]
        weather = AsyncMock(side_effect=RuntimeError("upstream down"))

        result = await create_vercel_messages_handler()(
            config, "hello", {"weather": weather}
        )

        assert result["output"] == "could not check"
        ai_runtime.tool_result_part.assert_called_once_with(
            "tc-1", tool_name="weather", result="upstream down", is_error=True
        )

    @pytest.mark.asyncio
    async def test_stops_a_runaway_tool_loop(self, ai_runtime: MagicMock) -> None:
        config = {
            **CONFIG,
            "tools": {"weather": {"description": "Get weather", "parameters": {}}},
        }
        call = SimpleNamespace(tool_call_id="tc-1", tool_name="weather", tool_args={})
        ai_runtime.stream.side_effect = lambda **_: FakeStream(
            text="", tool_calls=[call]
        )

        with pytest.raises(RuntimeError, match="did not reach a final response"):
            await create_vercel_messages_handler()(
                config, "hello", {"weather": AsyncMock(return_value="sunny")}
            )

    @pytest.mark.asyncio
    async def test_structured_output_uses_output_type_and_serializes_result(
        self, ai_runtime: MagicMock
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        ai_runtime.stream.return_value = FakeStream(text="", output={"answer": "yes"})
        result = await create_vercel_messages_handler()(
            {**CONFIG, "outputFormat": schema}, "question"
        )
        assert _stream_kwargs(ai_runtime)["output_type"] is not None
        assert json.loads(result["output"]) == {"answer": "yes"}

    @pytest.mark.asyncio
    async def test_streaming_ignores_output_format(self, ai_runtime: MagicMock) -> None:
        handler = create_vercel_messages_handler()
        events = [
            event
            async for event in await handler.stream(
                {**CONFIG, "outputFormat": {"type": "object"}}, "question"
            )
        ]
        assert "output_type" not in _stream_kwargs(ai_runtime)
        assert events[-1]["type"] == "done"


class TestUsageAndStreaming:
    @pytest.mark.asyncio
    async def test_normalizes_usage_after_blocking_stream_completion(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime.stream.return_value = FakeStream(input_tokens=12, output_tokens=7)
        result = await create_vercel_messages_handler()(CONFIG, "hello")
        assert result["usage"] == {"input_tokens": 12, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_stream_forwards_deltas_and_emits_one_done(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime.stream.return_value = FakeStream(
            [TextDelta("hel"), TextDelta("lo")], text="hello"
        )
        events = [
            event
            async for event in await create_vercel_messages_handler().stream(
                CONFIG, "hello"
            )
        ]
        assert [event["text"] for event in events if event["type"] == "chunk"] == [
            "hel",
            "lo",
        ]
        assert sum(event["type"] == "done" for event in events) == 1
        assert events[-1]["output"] == "hello"

    @pytest.mark.asyncio
    async def test_early_exit_closes_async_stream_context(
        self, ai_runtime: MagicMock
    ) -> None:
        provider_stream = FakeStream([TextDelta("one"), TextDelta("two")])
        ai_runtime.stream.return_value = provider_stream
        stream = await create_vercel_messages_handler().stream(CONFIG, "hello")
        async for _event in stream:
            break
        await stream.aclose()
        assert provider_stream.exited == 1
        assert provider_stream.consumed == 1

    @pytest.mark.asyncio
    async def test_runtime_telemetry_is_not_enabled(
        self, ai_runtime: MagicMock
    ) -> None:
        await create_vercel_messages_handler()(CONFIG, "hello")
        kwargs = _stream_kwargs(ai_runtime)
        assert not kwargs.get("experimental_telemetry")
        assert not kwargs.get("telemetry")

    @pytest.mark.asyncio
    async def test_span_identity_uses_provider_and_preserves_gateway_model(
        self, ai_runtime: MagicMock
    ) -> None:
        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        with patch.object(handler_mod.trace, "get_tracer", return_value=tracer):
            await create_vercel_messages_handler()(CONFIG, "hello")
        attributes = {
            call.args[0]: call.args[1] for call in span.set_attribute.call_args_list
        }
        assert attributes["gen_ai.system"] == "anthropic"
        assert attributes["gen_ai.provider.name"] == "anthropic"
        assert attributes["gen_ai.request.model"] == "anthropic/claude-sonnet-4"
