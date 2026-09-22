"""Test-first contract for the wildcard native Vercel ``ai.Agent`` adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

import launchdarkly_ai_vercel_agents.graph as graph_mod
import launchdarkly_ai_vercel_agents.handler as handler_mod
from launchdarkly_ai_vercel_agents import (
    create_vercel_agents_handler,
    vercel_graph,
)

CONFIG: dict[str, Any] = {
    "model": {
        "name": "anthropic/claude-sonnet-4",
        "parameters": {"temperature": 0.2},
    },
    "provider": {"name": "Anthropic"},
    "instructions": "Be concise.",
}


class TextDelta:
    def __init__(self, chunk: str) -> None:
        self.chunk = chunk


class AgentStream:
    """Mirrors ``ai.agents.AgentStream``.

    The real type exposes ``output`` and per-message usage, and has no ``text``
    or aggregate ``usage``; a double carrying those would hide a handler that
    reads them and silently returns an empty response.
    """

    def __init__(
        self,
        events: list[Any] | None = None,
        *,
        text: str = "answer",
        input_tokens: int = 8,
        output_tokens: int = 3,
        turns: int = 1,
    ) -> None:
        self.events = list(events or [TextDelta(text)])
        self.output = text
        self.messages = [
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=input_tokens, output_tokens=output_tokens
                )
            )
            for _ in range(turns)
        ]
        self.entered = 0
        self.exited = 0
        self.consumed = 0

    async def __aenter__(self) -> AgentStream:
        self.entered += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited += 1

    def __aiter__(self) -> AgentStream:
        return self

    async def __anext__(self) -> Any:
        if not self.events:
            raise StopAsyncIteration
        self.consumed += 1
        return self.events.pop(0)


@pytest.fixture
def ai_runtime() -> MagicMock:
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
    agent = MagicMock()
    agent.run = MagicMock(return_value=AgentStream())
    runtime.Agent = MagicMock(return_value=agent)
    with patch.object(handler_mod, "ai", runtime):
        yield runtime


class TestNativeAgent:
    def test_advertises_wildcard_agent(self) -> None:
        assert create_vercel_agents_handler().provides_for == ("*", "agent")

    @pytest.mark.asyncio
    async def test_constructs_native_ai_agent_and_runs_it(
        self, ai_runtime: MagicMock
    ) -> None:
        await create_vercel_agents_handler()(CONFIG, "hello")
        ai_runtime.Agent.assert_called_once()
        agent = ai_runtime.Agent.return_value
        agent.run.assert_called_once()
        assert agent.run.call_args.kwargs["model"] is ai_runtime.get_model.return_value

    @pytest.mark.asyncio
    async def test_does_not_fall_back_to_messages_stream(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime.stream = MagicMock()
        await create_vercel_agents_handler()(CONFIG, "hello")
        ai_runtime.Agent.assert_called_once()
        ai_runtime.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_gateway_model_id(self, ai_runtime: MagicMock) -> None:
        await create_vercel_agents_handler()(CONFIG, "hello")
        ai_runtime.get_model.assert_called_once_with("anthropic/claude-sonnet-4")

    @pytest.mark.asyncio
    async def test_builds_gateway_creator_model_id(self, ai_runtime: MagicMock) -> None:
        config = {
            **CONFIG,
            "model": {"name": "grok-4.5"},
            "provider": {"name": "xAI"},
        }
        await create_vercel_agents_handler()(config, "hello")
        ai_runtime.get_model.assert_called_once_with("spacexai/grok-4.5")

    @pytest.mark.asyncio
    async def test_converts_dotted_creator_prefix(self, ai_runtime: MagicMock) -> None:
        config = {
            **CONFIG,
            "model": {"name": "openai.gpt-5.6-sol"},
            "provider": {"name": "OpenAI"},
        }
        await create_vercel_agents_handler()(config, "hello")
        ai_runtime.get_model.assert_called_once_with("openai/gpt-5.6-sol")

    @pytest.mark.asyncio
    async def test_model_factory_receives_config_once(
        self, ai_runtime: MagicMock
    ) -> None:
        model = object()
        factory = AsyncMock(return_value=model)
        await create_vercel_agents_handler(model=factory)(CONFIG, "hello")
        factory.assert_awaited_once_with(CONFIG)
        assert ai_runtime.Agent.return_value.run.call_args.kwargs["model"] is model

    @pytest.mark.asyncio
    async def test_system_history_and_input_use_native_messages(
        self, ai_runtime: MagicMock
    ) -> None:
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"},
        ]
        await create_vercel_agents_handler()(CONFIG, "latest", {}, {}, history)
        messages = ai_runtime.Agent.return_value.run.call_args.kwargs["messages"]
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert messages[-1]["content"] == "latest"

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
                            "type": "url",
                            "url": "https://example.invalid/image.png",
                            "media_type": "image/png",
                        },
                    },
                ],
            }
        ]
        await create_vercel_agents_handler()(CONFIG, None, {}, {}, history)
        ai_runtime.file_part.assert_called_once()
        assert "example.invalid/image.png" in str(ai_runtime.file_part.call_args)

    @pytest.mark.asyncio
    async def test_only_callable_tools_reach_agent(self, ai_runtime: MagicMock) -> None:
        config = {
            **CONFIG,
            "tools": {
                "lookup": {
                    "description": "Look up a value",
                    "parameters": {"type": "object"},
                },
                "missing": {"description": "No implementation", "parameters": {}},
            },
        }
        lookup = AsyncMock(return_value="found")
        await create_vercel_agents_handler()(config, "hello", {"lookup": lookup})
        tools = ai_runtime.Agent.call_args.kwargs["tools"]
        assert [tool.name for tool in tools] == ["lookup"]
        tool_span = MagicMock()
        with patch.object(
            handler_mod, "start_tool_span", return_value=tool_span
        ) as start_tool_span:
            assert await tools[0].execute(key="x") == "found"
        start_tool_span.assert_called_once()
        assert start_tool_span.call_args.args[0] == "lookup"
        tool_span.end.assert_called_once()
        lookup.assert_awaited_once_with({"key": "x"})

    @pytest.mark.asyncio
    async def test_agent_receives_model_settings_as_request_params(
        self, ai_runtime: MagicMock
    ) -> None:
        await create_vercel_agents_handler()(CONFIG, "hello")
        run_kwargs = ai_runtime.Agent.return_value.run.call_args.kwargs
        assert run_kwargs["params"] is ai_runtime.InferenceRequestParams.return_value
        ai_runtime.TemperatureSamplerParams.assert_called_once_with(temperature=0.2)

    @pytest.mark.asyncio
    async def test_blocking_result_normalizes_text_and_usage(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime.Agent.return_value.run.return_value = AgentStream(
            text="done", input_tokens=21, output_tokens=5
        )
        result = await create_vercel_agents_handler()(CONFIG, "hello")
        assert result == {
            "output": "done",
            "usage": {"input_tokens": 21, "output_tokens": 5},
        }

    @pytest.mark.asyncio
    async def test_sums_usage_across_every_turn_of_a_tool_loop(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime.Agent.return_value.run.return_value = AgentStream(
            text="done", input_tokens=21, output_tokens=5, turns=3
        )
        result = await create_vercel_agents_handler()(CONFIG, "hello")
        assert result["usage"] == {"input_tokens": 63, "output_tokens": 15}

    @pytest.mark.asyncio
    async def test_structured_output_is_requested_and_serialized(
        self, ai_runtime: MagicMock
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        class Answer(BaseModel):
            answer: str

        stream = AgentStream(text="")
        stream.output = Answer(answer="yes")
        ai_runtime.Agent.return_value.run.return_value = stream

        result = await create_vercel_agents_handler()(
            {**CONFIG, "outputFormat": schema}, "question"
        )

        run_kwargs = ai_runtime.Agent.return_value.run.call_args.kwargs
        assert run_kwargs["output_type"] is not None
        assert result["output"] == '{"answer":"yes"}'

    @pytest.mark.asyncio
    async def test_stream_forwards_text_and_closes_context(
        self, ai_runtime: MagicMock
    ) -> None:
        provider_stream = AgentStream([TextDelta("a"), TextDelta("b")], text="ab")
        ai_runtime.Agent.return_value.run.return_value = provider_stream
        events = [
            event
            async for event in await create_vercel_agents_handler().stream(
                CONFIG, "hello"
            )
        ]
        assert [event.get("text") for event in events[:-1]] == ["a", "b"]
        assert events[-1]["type"] == "done"
        assert provider_stream.exited == 1

    @pytest.mark.asyncio
    async def test_early_stream_exit_closes_context(
        self, ai_runtime: MagicMock
    ) -> None:
        provider_stream = AgentStream([TextDelta("a"), TextDelta("b")])
        ai_runtime.Agent.return_value.run.return_value = provider_stream
        stream = await create_vercel_agents_handler().stream(CONFIG, "hello")
        async for _event in stream:
            break
        await stream.aclose()
        assert provider_stream.exited == 1
        assert provider_stream.consumed == 1


class TestGraphWrapper:
    def test_prewires_exactly_one_wildcard_handler(self) -> None:
        graph_instance = object()
        with (
            patch.object(graph_mod, "graph", return_value=graph_instance) as graph_fn,
            patch.object(
                graph_mod,
                "create_vercel_agents_handler",
                wraps=create_vercel_agents_handler,
            ) as factory,
        ):
            assert vercel_graph("graph-key") is graph_instance
        factory.assert_called_once()
        handlers = graph_fn.call_args.kwargs["handlers"]
        assert len(handlers) == 1
        assert handlers[0].provides_for == ("*", "agent")

    def test_forwards_model_and_capture_content_to_factory(self) -> None:
        model = object()
        with (
            patch.object(graph_mod, "graph", return_value=object()),
            patch.object(
                graph_mod,
                "create_vercel_agents_handler",
                return_value=MagicMock(provides_for=("*", "agent")),
            ) as factory,
        ):
            vercel_graph(
                "graph-key",
                model=model,
                capture_content=True,
                context={"kind": "user", "key": "u"},
            )
        factory.assert_called_once_with(model=model, capture_content=True)

    def test_caller_cannot_override_handlers(self) -> None:
        with patch.object(graph_mod, "graph", return_value=object()) as graph_fn:
            vercel_graph("graph-key", handlers=[MagicMock()])
        handlers = graph_fn.call_args.kwargs["handlers"]
        assert len(handlers) == 1
        assert handlers[0].provides_for == ("*", "agent")
