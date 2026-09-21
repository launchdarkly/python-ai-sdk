"""Contract tests for the future OpenAI Agents SDK adapter over LiteLLM."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_litellm_agents.handler as handler_mod
from launchdarkly_ai_litellm_agents import (
    create_litellm_agents_handler,
    litellm_agents,
)

CONFIG: dict[str, Any] = {
    "provider": {"name": "Anthropic"},
    "model": {
        "name": "anthropic/claude-sonnet-4",
        "parameters": {"temperature": 0.2},
    },
    "instructions": "Help {{name}}.",
}


class FakeAgent:
    created: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name", "agent")
        self.model = kwargs["model"]
        self.instructions = kwargs.get("instructions")
        self.tools = kwargs.get("tools", [])
        type(self).created.append(kwargs)


class FakeFunctionTool:
    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs["name"]
        self.description = kwargs.get("description")
        self.params_json_schema = kwargs.get("params_json_schema")
        self.on_invoke_tool = kwargs["on_invoke_tool"]


class FakeStreamedResult:
    def __init__(self, events: list[Any], output: str = "done") -> None:
        self._events = events
        self.final_output = output
        self.cancelled = False
        self.context_wrapper = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=8, output_tokens=3)
        )

    async def stream_events(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    def cancel(self, mode: str = "immediate") -> None:
        self.cancelled = True


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


def _agents_module(
    *,
    run: Any | None = None,
    run_streamed: Any | None = None,
) -> Any:
    module = SimpleNamespace()
    module.Agent = FakeAgent
    module.FunctionTool = FakeFunctionTool
    module.Runner = SimpleNamespace(
        run=run
        or AsyncMock(
            return_value=SimpleNamespace(
                final_output="done",
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5, output_tokens=2)
                ),
            )
        ),
        run_streamed=run_streamed or MagicMock(return_value=FakeStreamedResult([])),
    )
    return module


def _patch_agents(module: Any) -> Any:
    real_import = __import__
    return patch(
        "importlib.import_module",
        side_effect=lambda name: module if name == "agents" else real_import(name),
    )


class TestFactoryAndModelBinding:
    def setup_method(self) -> None:
        FakeAgent.created.clear()

    def test_wildcard_metadata_and_capture_content(self) -> None:
        handler = create_litellm_agents_handler(capture_content=True)
        assert handler.provides_for == ("*", "agent")
        assert handler.capture_content is True
        assert create_litellm_agents_handler().capture_content is False

    async def test_evaluated_model_constructs_agents_sdk_litellm_model(self) -> None:
        agents = _agents_module()
        litellm_model = MagicMock(return_value=object())
        with (
            _patch_agents(agents),
            patch.object(handler_mod, "LitellmModel", litellm_model),
        ):
            await create_litellm_agents_handler()(CONFIG, "q", {}, {"name": "Ada"})
        litellm_model.assert_called_once()
        assert litellm_model.call_args.kwargs["model"] == "anthropic/claude-sonnet-4"
        assert FakeAgent.created[0]["model"] is litellm_model.return_value

    async def test_model_factory_is_evaluated_per_call_and_instance_scoped(
        self,
    ) -> None:
        agents = _agents_module()
        first_factory = MagicMock(return_value=object())
        second_factory = MagicMock(return_value=object())
        first = create_litellm_agents_handler(model_factory=first_factory)
        second = create_litellm_agents_handler(model_factory=second_factory)
        with _patch_agents(agents):
            await first(CONFIG, "one")
            await second({**CONFIG, "model": {"name": "proxy-alias"}}, "two")
        first_factory.assert_called_once_with(
            "anthropic/claude-sonnet-4", {"temperature": 0.2}
        )
        second_factory.assert_called_once_with("proxy-alias", {})
        assert first_factory.return_value is not second_factory.return_value

    async def test_injected_model_is_bound_to_agent(self) -> None:
        agents = _agents_module()
        model = object()
        with _patch_agents(agents):
            await create_litellm_agents_handler(model_factory=lambda *_: model)(
                CONFIG, "q"
            )
        assert FakeAgent.created[0]["model"] is model

    async def test_no_default_openai_model_or_client_fallback(self) -> None:
        agents = _agents_module()
        model = object()
        default_openai = MagicMock(side_effect=AssertionError("OpenAI fallback"))
        with (
            _patch_agents(agents),
            patch.dict(
                "sys.modules",
                {
                    "openai": SimpleNamespace(AsyncOpenAI=default_openai),
                    "agents.models.openai_chatcompletions": SimpleNamespace(
                        OpenAIChatCompletionsModel=default_openai
                    ),
                },
            ),
        ):
            await create_litellm_agents_handler(model_factory=lambda *_: model)(
                CONFIG, "q"
            )
        default_openai.assert_not_called()
        assert FakeAgent.created[0]["model"] is model


class TestAgentToolsAndRun:
    def setup_method(self) -> None:
        FakeAgent.created.clear()

    async def test_only_callable_backed_tools_are_bound(self) -> None:
        agents = _agents_module()
        config = {
            **CONFIG,
            "tools": {
                "weather": {
                    "description": "forecast",
                    "parameters": {"type": "object"},
                },
                "missing": {"description": "excluded", "parameters": {}},
            },
        }
        weather = AsyncMock(return_value="sunny")
        with _patch_agents(agents):
            await create_litellm_agents_handler(model_factory=lambda *_: object())(
                config, "q", {"weather": weather}
            )
        tools = FakeAgent.created[0]["tools"]
        assert [tool.name for tool in tools] == ["weather"]
        assert tools[0].description == "forecast"
        assert tools[0].params_json_schema == {"type": "object"}
        await tools[0].on_invoke_tool(SimpleNamespace(), '{"city":"Austin"}')
        weather.assert_awaited_once_with({"city": "Austin"})

    async def test_runner_receives_agent_prompt_and_max_turns(self) -> None:
        run = AsyncMock(
            return_value=SimpleNamespace(
                final_output="answer",
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=11, output_tokens=4)
                ),
            )
        )
        agents = _agents_module(run=run)
        config = {
            **CONFIG,
            "model": {**CONFIG["model"], "parameters": {"max_turns": 7}},
        }
        with _patch_agents(agents):
            result = await create_litellm_agents_handler(
                model_factory=lambda *_: object()
            )(config, "question", {}, {"name": "Ada"})
        run.assert_awaited_once()
        assert run.call_args.args[1] == "question"
        assert run.call_args.kwargs["max_turns"] == 7
        assert FakeAgent.created[0]["instructions"] == "Help Ada."
        assert result == {
            "output": "answer",
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }

    async def test_history_uses_agents_sdk_structured_input(self) -> None:
        run = AsyncMock(
            return_value=SimpleNamespace(
                final_output="answer",
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1)
                ),
            )
        )
        agents = _agents_module(run=run)
        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "prior"},
        ]
        with _patch_agents(agents):
            await create_litellm_agents_handler(model_factory=lambda *_: object())(
                CONFIG, "describe", {}, {"name": "Ada"}, history
            )
        prompt = run.call_args.args[1]
        assert isinstance(prompt, list)
        assert prompt[0]["content"] == [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,abc",
            }
        ]
        assert prompt[1]["content"] == [{"type": "output_text", "text": "prior"}]
        assert prompt[-1]["content"] == "describe"

    async def test_output_format_is_bound_as_agent_output_type(self) -> None:
        agents = _agents_module()
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        with _patch_agents(agents):
            result = await create_litellm_agents_handler(
                model_factory=lambda *_: object()
            )({**CONFIG, "outputFormat": schema}, "q")
        assert FakeAgent.created[0]["output_type"]["schema"] == schema
        assert result["output"] == "done"


class TestLifecycleAndTelemetry:
    def setup_method(self) -> None:
        FakeAgent.created.clear()

    async def test_run_error_propagates_without_openai_retry(self) -> None:
        run = AsyncMock(side_effect=RuntimeError("LiteLLM unavailable"))
        agents = _agents_module(run=run)
        with (
            _patch_agents(agents),
            pytest.raises(RuntimeError, match="LiteLLM unavailable"),
        ):
            await create_litellm_agents_handler(model_factory=lambda *_: object())(
                CONFIG, "q"
            )
        run.assert_awaited_once()

    async def test_stream_forwards_deltas_usage_and_done(self) -> None:
        events = [
            SimpleNamespace(
                type="raw_response_event",
                data=SimpleNamespace(type="response.output_text.delta", delta="Hel"),
            ),
            SimpleNamespace(
                type="raw_response_event",
                data=SimpleNamespace(type="response.output_text.delta", delta="lo"),
            ),
        ]
        streamed = FakeStreamedResult(events, "Hello")
        agents = _agents_module(run_streamed=MagicMock(return_value=streamed))
        with _patch_agents(agents):
            handler = create_litellm_agents_handler(model_factory=lambda *_: object())
            output = [event async for event in await handler.stream(CONFIG, "q")]
        assert [event["text"] for event in output[:-1]] == ["Hel", "lo"]
        assert output[-1] == {
            "type": "done",
            "output": "Hello",
            "usage": {"input_tokens": 8, "output_tokens": 3},
        }

    async def test_abandoned_stream_cancels_agents_runner(self) -> None:
        events = [
            SimpleNamespace(
                type="raw_response_event",
                data=SimpleNamespace(type="response.output_text.delta", delta="one"),
            ),
            SimpleNamespace(
                type="raw_response_event",
                data=SimpleNamespace(type="response.output_text.delta", delta="two"),
            ),
        ]
        streamed = FakeStreamedResult(events)
        agents = _agents_module(run_streamed=MagicMock(return_value=streamed))
        with _patch_agents(agents):
            gen = await create_litellm_agents_handler(
                model_factory=lambda *_: object()
            ).stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert streamed.cancelled is True

    async def test_root_identity_and_content_capture_gating(self) -> None:
        agents = _agents_module()
        span = MagicMock()
        trace = MagicMock()
        trace.get_tracer.return_value.start_span.return_value = span
        variables = {
            "__ld": {"configKey": "cfg", "variationKey": "v", "runId": "run"},
            "name": "Ada",
        }
        with _patch_agents(agents), patch.object(handler_mod, "trace", trace):
            await create_litellm_agents_handler(model_factory=lambda *_: object())(
                CONFIG, "private", {}, variables
            )
        attributes = {
            call.args[0]: call.args[1] for call in span.set_attribute.call_args_list
        }
        assert attributes["launchdarkly.config.key"] == "cfg"
        assert attributes["gen_ai.request.model"] == "anthropic/claude-sonnet-4"
        assert not any(key.startswith("gen_ai.prompt.") for key in attributes)

    async def test_hooks_emit_nested_model_and_tool_spans(self) -> None:
        async def run(
            agent: Any,
            prompt: Any,
            *,
            max_turns: int,
            hooks: Any,
        ) -> Any:
            await hooks.on_llm_start(
                SimpleNamespace(), agent, agent.instructions, prompt
            )
            context = SimpleNamespace(
                tool_call_id="call-1",
                tool_name="weather",
                tool_arguments='{"city":"Oslo"}',
            )
            tool = SimpleNamespace(name="weather")
            await hooks.on_tool_start(context, agent, tool)
            await hooks.on_tool_end(context, agent, tool, "cold")
            await hooks.on_llm_end(
                SimpleNamespace(),
                agent,
                SimpleNamespace(
                    output=["done"],
                    usage=SimpleNamespace(input_tokens=5, output_tokens=2),
                ),
            )
            return SimpleNamespace(
                final_output="done",
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5, output_tokens=2)
                ),
            )

        recorder = SpanRecorder()
        agents = _agents_module(run=AsyncMock(side_effect=run))
        with (
            _patch_agents(agents),
            patch.object(handler_mod, "trace", recorder),
        ):
            await create_litellm_agents_handler(model_factory=lambda *_: object())(
                CONFIG, "q"
            )

        assert [span.name for span in recorder.spans] == [
            "invoke_agent",
            "chat anthropic/claude-sonnet-4",
            "execute_tool weather",
        ]
        root = recorder.spans[0]
        assert all(span.context == ("context-of", root) for span in recorder.spans[1:])
        for span in recorder.spans[:2]:
            assert span.attributes["gen_ai.provider.name"] == "anthropic"
            assert span.attributes["gen_ai.system"] == "litellm"
        assert recorder.spans[1].attributes["gen_ai.usage.input_tokens"] == 5
        assert recorder.spans[2].attributes["gen_ai.tool.call.id"] == "call-1"
        assert all(span.ended == 1 for span in recorder.spans)

    async def test_open_model_and_root_fail_exactly_once(self) -> None:
        error = RuntimeError("run failed")

        async def run(
            agent: Any,
            prompt: Any,
            *,
            max_turns: int,
            hooks: Any,
        ) -> Any:
            await hooks.on_llm_start(
                SimpleNamespace(), agent, agent.instructions, prompt
            )
            raise error

        recorder = SpanRecorder()
        agents = _agents_module(run=AsyncMock(side_effect=run))
        with (
            _patch_agents(agents),
            patch.object(handler_mod, "trace", recorder),
            pytest.raises(RuntimeError, match="run failed"),
        ):
            await create_litellm_agents_handler(model_factory=lambda *_: object())(
                CONFIG, "q"
            )
        assert [span.name for span in recorder.spans] == [
            "invoke_agent",
            "chat anthropic/claude-sonnet-4",
        ]
        assert all(span.exceptions == [error] for span in recorder.spans)
        assert all(span.ended == 1 for span in recorder.spans)


class TestConvenienceWrapper:
    def test_forwards_handler_variables_and_capture_content(self) -> None:
        instance = MagicMock()
        instance.invoke.return_value = "result"
        config = MagicMock(return_value=instance)
        with patch.object(handler_mod, "config", config):
            result = litellm_agents(
                "agent-key",
                "hello",
                {"kind": "user", "key": "u"},
                variables={"name": "Ada"},
                capture_content=True,
            )
        assert result == "result"
        configured_handler = config.call_args.kwargs["handler"]
        assert configured_handler.provides_for == ("*", "agent")
        assert configured_handler.capture_content is True
        assert "capture_content" not in config.call_args.kwargs
        instance.invoke.assert_called_once_with(
            "hello", {"kind": "user", "key": "u"}, {"name": "Ada"}
        )
