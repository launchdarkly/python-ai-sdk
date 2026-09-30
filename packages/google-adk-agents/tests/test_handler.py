"""Google ADK agents handler tests.

Reference: TESTING.md §1 and §2.x Google ADK, Appendix A.15.
The ADK runtime is mocked. These tests must not open a network connection.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_google_adk_agents.handler as handler_mod
import launchdarkly_ai_google_adk_agents.spans as spans_mod
from launchdarkly_ai_google_adk_agents.handler import (
    create_google_adk_agents_handler,
    history_contents,
)

BASE_CONFIG: dict[str, Any] = {
    "model": {"name": "gemini-2.5-flash"},
    "provider": {"name": "Google"},
    "instructions": "Be helpful.",
}


def _usage(prompt: int = 10, candidates: int = 5, total: int = 15) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
    )


def _event(
    text: str = "answer",
    *,
    partial: bool = False,
    usage: Any = None,
    final: bool = True,
) -> SimpleNamespace:
    event = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text, inline_data=None)]),
        partial=partial,
        usage_metadata=usage if usage is not None else _usage(),
    )
    event.is_final_response = lambda: final and not partial
    return event


def _message_text(message: Any) -> str:
    parts = message.parts if hasattr(message, "parts") else message["parts"]
    part = parts[0]
    if hasattr(part, "text"):
        return part.text
    return part["text"]


class _FakeFunctionTool:
    created: ClassVar[list[_FakeFunctionTool]] = []

    def __init__(self, func: Any, **kwargs: Any) -> None:
        self.func = func
        self.name = kwargs.get("name", getattr(func, "__name__", ""))
        self.description = kwargs.get("description")
        self.parameters = kwargs.get("parameters")
        _FakeFunctionTool.created.append(self)


class _FakeRunner:
    last: ClassVar[_FakeRunner | None] = None
    events: ClassVar[list[Any]] = []
    explode: ClassVar[BaseException | None] = None

    def __init__(
        self,
        agent: Any = None,
        *,
        app_name: str | None = None,
        plugins: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.agent = agent
        self.app_name = app_name
        self.plugins = list(plugins or [])
        self.kwargs = kwargs
        session = SimpleNamespace(id="sess-1")
        self.session_service = SimpleNamespace(
            create_session=AsyncMock(return_value=session),
            append_event=AsyncMock(),
        )
        self.run_calls: list[dict[str, Any]] = []
        _FakeRunner.last = self

    async def run_async(self, **kwargs: Any) -> Any:
        self.run_calls.append(kwargs)
        if _FakeRunner.explode is not None:
            raise _FakeRunner.explode
        plugin = self.plugins[0] if self.plugins else None
        if plugin is not None and hasattr(plugin, "after_model_callback"):
            response = SimpleNamespace(
                content=_FakeRunner.events[-1].content if _FakeRunner.events else None,
                usage_metadata=_usage(),
                partial=False,
            )
            result = plugin.after_model_callback(
                callback_context=SimpleNamespace(),
                llm_response=response,
            )
            if inspect.isawaitable(result):
                await result
        for event in _FakeRunner.events:
            yield event


@pytest.fixture(autouse=True)
def _adk_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeFunctionTool.created = []
    _FakeRunner.last = None
    _FakeRunner.events = [_event()]
    _FakeRunner.explode = None
    monkeypatch.setattr(handler_mod, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(
        handler_mod,
        "LlmAgent",
        MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw)),
    )
    monkeypatch.setattr(handler_mod, "FunctionTool", _FakeFunctionTool)
    monkeypatch.setattr(
        handler_mod, "Gemini", MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    )
    monkeypatch.setattr(
        handler_mod,
        "LiteLlm",
        MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw)),
    )
    monkeypatch.setattr(
        spans_mod, "start_root_span", MagicMock(return_value=MagicMock(name="root"))
    )
    monkeypatch.setattr(
        spans_mod, "start_model_span", MagicMock(return_value=MagicMock(name="model"))
    )
    monkeypatch.setattr(
        spans_mod, "start_tool_span", MagicMock(return_value=MagicMock(name="tool"))
    )
    monkeypatch.setattr(spans_mod, "finish_model_span", MagicMock())
    monkeypatch.setattr(spans_mod, "finish_root_span", MagicMock())
    monkeypatch.setattr(spans_mod, "fail_span", MagicMock())


def _gemini_client_kwargs() -> dict[str, Any]:
    kwargs = handler_mod.Gemini.call_args.kwargs
    return dict(kwargs.get("client_kwargs") or {})


class TestFactory:
    def test_provides_for_is_wildcard_agent(self) -> None:
        handler = create_google_adk_agents_handler()
        assert handler.provides_for == ("*", "agent")

    def test_capture_content_defaults_false(self) -> None:
        handler = create_google_adk_agents_handler()
        assert handler.capture_content is False

    def test_capture_content_can_be_enabled(self) -> None:
        handler = create_google_adk_agents_handler(capture_content=True)
        assert handler.capture_content is True

    def test_factory_returns_independent_handlers(self) -> None:
        first = create_google_adk_agents_handler()
        second = create_google_adk_agents_handler()
        assert first is not second


class TestAuth:
    async def test_default_transport_is_gemini_without_vertex(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        assert handler_mod.Gemini.call_args.kwargs["model"] == "gemini-2.5-flash"
        assert _gemini_client_kwargs().get("vertexai") is not True
        handler_mod.LiteLlm.assert_not_called()

    async def test_explicit_api_key_is_forwarded_only_in_gemini_mode(self) -> None:
        await create_google_adk_agents_handler(api_key="gemini-key")(
            BASE_CONFIG, "hello"
        )
        client_kwargs = _gemini_client_kwargs()
        assert client_kwargs["api_key"] == "gemini-key"
        assert client_kwargs.get("vertexai") is not True

    async def test_vertex_opt_in_sets_project_and_location_without_api_key(
        self,
    ) -> None:
        await create_google_adk_agents_handler(
            use_vertexai=True,
            project="ld-proj",
            location="us-central1",
            api_key="should-not-be-used",
        )(BASE_CONFIG, "hello")
        client_kwargs = _gemini_client_kwargs()
        assert client_kwargs["vertexai"] is True
        assert client_kwargs["project"] == "ld-proj"
        assert client_kwargs["location"] == "us-central1"
        assert "api_key" not in client_kwargs

    async def test_vertex_reads_project_and_location_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
        await create_google_adk_agents_handler(use_vertexai=True)(BASE_CONFIG, "hello")
        client_kwargs = _gemini_client_kwargs()
        assert client_kwargs["project"] == "env-proj"
        assert client_kwargs["location"] == "global"

    def test_vertex_without_project_or_location_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        with pytest.raises(ValueError, match="project"):
            create_google_adk_agents_handler(use_vertexai=True)

    async def test_injected_model_skips_gemini_and_litellm(self) -> None:
        model = object()
        await create_google_adk_agents_handler(model=model)(BASE_CONFIG, "hello")
        assert handler_mod.LlmAgent.call_args.kwargs["model"] is model
        handler_mod.Gemini.assert_not_called()
        handler_mod.LiteLlm.assert_not_called()

    async def test_model_factory_receives_the_config(self) -> None:
        seen: list[dict[str, Any]] = []

        def factory(config: dict[str, Any]) -> object:
            seen.append(config)
            return "built-model"

        await create_google_adk_agents_handler(model=factory)(BASE_CONFIG, "hello")
        assert seen == [BASE_CONFIG]
        assert handler_mod.LlmAgent.call_args.kwargs["model"] == "built-model"

    async def test_non_gemini_provider_uses_litellm_model_id(self) -> None:
        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "instructions": "Be helpful.",
        }
        await create_google_adk_agents_handler()(config, "hello")
        assert handler_mod.LiteLlm.call_args.kwargs["model"] == "openai/gpt-4o"
        handler_mod.Gemini.assert_not_called()

    async def test_slashed_model_name_is_passed_through_to_litellm(self) -> None:
        config = {
            "model": {"name": "anthropic/claude-sonnet-4"},
            "provider": {"name": "Anthropic"},
            "instructions": "Be helpful.",
        }
        await create_google_adk_agents_handler()(config, "hello")
        assert (
            handler_mod.LiteLlm.call_args.kwargs["model"] == "anthropic/claude-sonnet-4"
        )


class TestPromptAndRun:
    async def test_instructions_are_templated_onto_the_agent(self) -> None:
        config = {**BASE_CONFIG, "instructions": "Hello {{name}} {{missing}}"}
        await create_google_adk_agents_handler()(
            config, "question", None, {"name": "Ada"}
        )
        assert (
            handler_mod.LlmAgent.call_args.kwargs["instruction"]
            == "Hello Ada {{missing}}"
        )

    async def test_instructions_win_over_messages(self) -> None:
        config = {
            **BASE_CONFIG,
            "instructions": "from-instructions",
            "messages": [{"role": "system", "content": "from-messages"}],
        }
        await create_google_adk_agents_handler()(config, "question")
        assert (
            handler_mod.LlmAgent.call_args.kwargs["instruction"] == "from-instructions"
        )

    async def test_user_input_is_the_new_message_text(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        runner = _FakeRunner.last
        assert runner is not None
        assert _message_text(runner.run_calls[0]["new_message"]) == "hello"

    async def test_missing_user_input_sends_an_empty_text_part(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, None)
        runner = _FakeRunner.last
        assert runner is not None
        assert _message_text(runner.run_calls[0]["new_message"]) == ""

    async def test_runner_receives_exactly_one_plugin(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        runner = _FakeRunner.last
        assert runner is not None
        assert len(runner.plugins) == 1

    async def test_final_event_usage_metadata_is_returned(self) -> None:
        _FakeRunner.events = [_event("the answer", usage=_usage(11, 7, 18))]
        result = await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        assert result["output"] == "the answer"
        assert result["usage"] == {"input": 11, "output": 7, "total": 18}

    async def test_missing_usage_is_zeros(self) -> None:
        _FakeRunner.events = [_event("ok", usage=None)]
        _FakeRunner.events[0].usage_metadata = None
        result = await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        assert result["usage"] == {"input": 0, "output": 0, "total": 0}

    async def test_partial_events_are_not_double_counted(self) -> None:
        _FakeRunner.events = [
            _event("hel", partial=True, usage=_usage(3, 1, 4), final=False),
            _event("hello", usage=_usage(3, 2, 5)),
        ]
        result = await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        assert result["output"] == "hello"
        assert result["usage"]["total"] == 5

    async def test_blocking_call_sets_output_schema(self) -> None:
        config = {
            **BASE_CONFIG,
            "outputFormat": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        }
        await create_google_adk_agents_handler()(config, "hello")
        assert handler_mod.LlmAgent.call_args.kwargs.get("output_schema") is not None

    async def test_history_is_seeded_before_run(self) -> None:
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ack"},
        ]
        await create_google_adk_agents_handler()(
            BASE_CONFIG, "now", None, None, history
        )
        runner = _FakeRunner.last
        assert runner is not None
        assert runner.session_service.append_event.await_count == 2
        assert _message_text(runner.run_calls[0]["new_message"]) == "now"


class TestHistoryContents:
    def test_image_history_uses_inline_data_bytes(self) -> None:
        contents = history_contents(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",
                            },
                        },
                    ],
                }
            ]
        )
        image = contents[0].parts[1].inline_data
        assert image.mime_type == "image/png"
        assert image.data == b"hello"


class TestTools:
    async def test_only_tools_with_handlers_are_forwarded(self) -> None:
        config = {
            **BASE_CONFIG,
            "tools": {
                "lookup": {
                    "name": "lookup",
                    "description": "find",
                    "parameters": {"type": "object"},
                },
                "other": {
                    "name": "other",
                    "description": "nope",
                    "parameters": {"type": "object"},
                },
                "extra": {
                    "name": "extra",
                    "description": "also",
                    "parameters": {"type": "object"},
                },
            },
        }

        def lookup(q: str) -> str:
            return f"found {q}"

        def extra() -> str:
            return "extra"

        await create_google_adk_agents_handler()(
            config,
            "hello",
            {"lookup": lookup, "extra": extra},
        )
        names = {tool.name for tool in _FakeFunctionTool.created}
        assert names == {"lookup", "extra"}
        lookup_tool = next(
            tool for tool in _FakeFunctionTool.created if tool.name == "lookup"
        )
        assert lookup_tool.description == "find"
        assert lookup_tool.func("ada") == "found ada"

    async def test_tool_error_is_returned_to_the_model(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        plugin = _FakeRunner.last.plugins[0]
        result = plugin.on_tool_error_callback(
            tool=SimpleNamespace(name="lookup"),
            tool_args={"q": "x"},
            tool_context=SimpleNamespace(),
            error=RuntimeError("tool broke"),
        )
        if inspect.isawaitable(result):
            result = await result
        assert isinstance(result, dict)
        assert "tool broke" in str(result)
        spans_mod.fail_span.assert_called()

    async def test_mapping_handler_receives_one_argument_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _StrictTool:
            def __init__(
                self, func: Any, *, require_confirmation: bool = False
            ) -> None:
                self.func = func
                self.name = getattr(func, "__name__", "")

        monkeypatch.setattr(handler_mod, "FunctionTool", _StrictTool)
        config = {
            **BASE_CONFIG,
            "tools": {
                "prefs": {
                    "name": "get_user_preferences",
                    "description": "prefs",
                    "parameters": {"type": "object"},
                }
            },
        }

        def get_preferences(args: dict[str, Any]) -> str:
            return str(args["user_id"])

        await create_google_adk_agents_handler()(
            config, "hello", {"get_user_preferences": get_preferences}
        )
        agent_tools = handler_mod.LlmAgent.call_args.kwargs["tools"]
        built = agent_tools[0]
        assert built.name == "get_user_preferences"
        assert built.func(user_id="ada") == "ada"

    async def test_native_tool_is_not_wrapped_as_a_function(self) -> None:
        from launchdarkly_ai_server import NATIVE_TOOL_KEY, NativeTool

        tracked: list[str] = []

        def stub() -> None:
            tracked.append("google_search")

        setattr(stub, NATIVE_TOOL_KEY, NativeTool("google_search"))
        config = {
            **BASE_CONFIG,
            "tools": {
                "search": {
                    "name": "google_search",
                    "description": "web",
                    "parameters": {"type": "object"},
                }
            },
        }
        await create_google_adk_agents_handler()(
            config, "hello", {"google_search": stub}
        )
        assert _FakeFunctionTool.created == []
        plugin = _FakeRunner.last.plugins[0]
        started = plugin.before_tool_callback(
            tool=SimpleNamespace(name="google_search"),
            tool_args={},
            tool_context=SimpleNamespace(function_call_id="call-1"),
        )
        if inspect.isawaitable(started):
            await started
        assert tracked == ["google_search"]

    async def test_output_schema_marks_properties_required(self) -> None:
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        await create_google_adk_agents_handler()(
            {**BASE_CONFIG, "outputFormat": schema}, "hello"
        )
        passed = handler_mod.LlmAgent.call_args.kwargs["output_schema"]
        assert passed["required"] == ["ok"]
        assert passed["additionalProperties"] is False
        assert "required" not in schema


class TestTelemetry:
    async def test_model_turn_uses_span_helpers(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        spans_mod.start_root_span.assert_called()
        spans_mod.start_model_span.assert_called()
        spans_mod.finish_model_span.assert_called()

    async def test_tool_callback_uses_start_tool_span(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        plugin = _FakeRunner.last.plugins[0]
        started = plugin.before_tool_callback(
            tool=SimpleNamespace(name="lookup"),
            tool_args={"q": "x"},
            tool_context=SimpleNamespace(),
        )
        if inspect.isawaitable(started):
            await started
        finished = plugin.after_tool_callback(
            tool=SimpleNamespace(name="lookup"),
            tool_args={"q": "x"},
            tool_context=SimpleNamespace(),
            result={"ok": True},
        )
        if inspect.isawaitable(finished):
            await finished
        spans_mod.start_tool_span.assert_called()
        assert spans_mod.start_tool_span.call_args.args[0] == "lookup"

    async def test_overlapping_calls_keep_distinct_ids(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        plugin = _FakeRunner.last.plugins[0]
        for call_id in ("call-1", "call-2"):
            started = plugin.before_tool_callback(
                tool=SimpleNamespace(name="lookup"),
                tool_args={"q": call_id},
                tool_context=SimpleNamespace(function_call_id=call_id),
            )
            if inspect.isawaitable(started):
                await started
        ids = [call.args[1] for call in spans_mod.start_tool_span.call_args_list]
        assert ids == ["call-1", "call-2"]

    async def test_capture_content_writes_messages_and_tool_arguments(self) -> None:
        span = MagicMock()
        span.attributes = {}
        span.set_attribute = lambda key, value: span.attributes.__setitem__(key, value)
        spans_mod.start_model_span.return_value = span
        spans_mod.start_tool_span.return_value = span
        await create_google_adk_agents_handler(capture_content=True)(
            BASE_CONFIG, "hello"
        )
        assert "gen_ai.input.messages" in span.attributes
        plugin = _FakeRunner.last.plugins[0]
        started = plugin.before_tool_callback(
            tool=SimpleNamespace(name="lookup"),
            tool_args={"q": "ada"},
            tool_context=SimpleNamespace(function_call_id="call-1"),
        )
        if inspect.isawaitable(started):
            await started
        finished = plugin.after_tool_callback(
            tool=SimpleNamespace(name="lookup"),
            tool_args={"q": "ada"},
            tool_context=SimpleNamespace(function_call_id="call-1"),
            result="found",
        )
        if inspect.isawaitable(finished):
            await finished
        assert "gen_ai.tool.call.arguments" in span.attributes

    async def test_runner_error_fails_and_ends_the_root_span(self) -> None:
        _FakeRunner.explode = RuntimeError("runner broke")
        with pytest.raises(RuntimeError, match="runner broke"):
            await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        spans_mod.fail_span.assert_called()

    async def test_abandon_open_spans_does_not_fail_them(self) -> None:
        await create_google_adk_agents_handler()(BASE_CONFIG, "hello")
        plugin = _FakeRunner.last.plugins[0]
        spans_mod.fail_span.reset_mock()
        plugin.abandon_open_spans(set())
        spans_mod.fail_span.assert_not_called()


class TestStreaming:
    async def test_stream_yields_chunks_then_one_done(self) -> None:
        _FakeRunner.events = [
            _event("hel", partial=True, final=False),
            _event("hello", usage=_usage(4, 2, 6)),
        ]
        handler = create_google_adk_agents_handler()
        events = [event async for event in await handler.stream(BASE_CONFIG, "hello")]
        assert events[0] == {"type": "chunk", "text": "hel"}
        assert events[-1]["type"] == "done"
        assert events[-1]["output"] == "hello"
        assert events[-1]["usage"] == {"input": 4, "output": 2, "total": 6}
        assert sum(1 for event in events if event["type"] == "done") == 1

    async def test_streaming_ignores_output_format(self) -> None:
        config = {**BASE_CONFIG, "outputFormat": {"type": "object"}}
        handler = create_google_adk_agents_handler()
        async for _ in await handler.stream(config, "hello"):
            pass
        assert handler_mod.LlmAgent.call_args.kwargs.get("output_schema") is None

    async def test_early_close_abandons_tool_spans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        abandoned: list[bool] = []
        plugin_cls = handler_mod.LaunchDarklyTelemetryPlugin

        class _SpyPlugin(plugin_cls):
            def abandon_open_spans(self, *args: Any, **kwargs: Any) -> None:
                abandoned.append(True)
                super().abandon_open_spans(*args, **kwargs)

        monkeypatch.setattr(handler_mod, "LaunchDarklyTelemetryPlugin", _SpyPlugin)
        handler = create_google_adk_agents_handler()
        generator = await handler.stream(BASE_CONFIG, "hello")
        await generator.__anext__()
        spans_mod.fail_span.reset_mock()
        await generator.aclose()
        assert abandoned == [True]
        spans_mod.fail_span.assert_not_called()
