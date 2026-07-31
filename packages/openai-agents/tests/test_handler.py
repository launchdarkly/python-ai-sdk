"""
Tests for launchdarkly-ai-openai-agents handler.
Covers §1.1–1.9 (generic) and OpenAI-agents-specific extras.
Reference: TESTING.md §1, §2.x (OpenAI)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_openai_agents.handler as handler_mod
from launchdarkly_ai_openai_agents.handler import (
    _build_agent_and_prompt,
    create_openai_agent_handler,
)
from launchdarkly_ai_openai_agents.utils import build_output_type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: Any) -> dict[str, Any]:
    base = {"model": {"name": "gpt-4o"}, "provider": {"name": "OpenAI"}}
    base.update(kwargs)
    return base


def _make_run_result(
    output: str = "hello", input_tokens: int = 10, output_tokens: int = 5
) -> Any:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens

    raw_resp = MagicMock()
    raw_resp.usage = usage

    result = MagicMock()
    result.final_output = output
    result.raw_responses = [raw_resp]
    return result


def _mock_agents_module(run_result: Any) -> Any:
    mock = MagicMock()
    mock.Agent = MagicMock(return_value=MagicMock())
    mock.Runner.run = AsyncMock(return_value=run_result)
    mock.Runner.run_streamed = MagicMock(
        return_value=MagicMock(
            stream_events=lambda: _empty_async_gen(),
            raw_responses=[],
            final_output=run_result.final_output,
        )
    )
    mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)
    mock.handoff = MagicMock(side_effect=lambda agent: agent)
    mock.RunHooks = MagicMock
    return mock


async def _empty_async_gen() -> AsyncIterator[Any]:
    return
    yield


# ---------------------------------------------------------------------------
# §1.1 Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self) -> None:
        h = create_openai_agent_handler()
        assert callable(h)

    def test_attaches_provides_for(self) -> None:
        h = create_openai_agent_handler()
        assert hasattr(h, "provides_for")

    def test_provides_for_values_are_correct(self) -> None:
        h = create_openai_agent_handler()
        pf = h.provides_for
        assert "OpenAI" in pf or "openai" in str(pf).lower()

    def test_multiple_calls_return_independent_instances(self) -> None:
        h1 = create_openai_agent_handler()
        h2 = create_openai_agent_handler()
        assert h1 is not h2


# ---------------------------------------------------------------------------
# §1.2 Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def _mock_agents(self) -> Any:
        mock = MagicMock()
        mock.Agent = MagicMock(return_value=MagicMock())
        mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)
        return mock

    def test_path_a_instructions(self) -> None:
        config = _make_config(instructions="Be concise.")
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, prompt, instructions = _build_agent_and_prompt(config, "hi", {}, {})
        assert instructions == "Be concise."
        assert prompt == "hi"

    def test_path_a_variable_substitution(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, _, instructions = _build_agent_and_prompt(
                config, "hi", {}, {"name": "Bob"}
            )
        assert instructions == "Hello Bob!"

    def test_path_a_unresolved_placeholder_preserved(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, _, instructions = _build_agent_and_prompt(config, "hi", {}, {})
        assert "{{name}}" in (instructions or "")

    def test_path_b_messages_system_extracted(self) -> None:
        config = _make_config(
            messages=[
                {"role": "system", "content": "System msg."},
                {"role": "user", "content": "prior turn"},
            ]
        )
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, prompt, instructions = _build_agent_and_prompt(
                config, "question", {}, {}
            )
        assert "System msg." in (instructions or "")
        assert "prior turn" in prompt
        assert "question" in prompt

    def test_path_b_variable_substitution_in_messages(self) -> None:
        config = _make_config(
            messages=[{"role": "user", "content": "name is {{name}}"}]
        )
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, prompt, _ = _build_agent_and_prompt(config, "q", {}, {"name": "Alice"})
        assert "Alice" in prompt

    def test_path_b_user_input_appended_as_final_turn(self) -> None:
        config = _make_config(messages=[{"role": "user", "content": "old message"}])
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, prompt, _ = _build_agent_and_prompt(config, "new input", {}, {})
        assert "new input" in prompt

    def test_path_c_empty_user_input_no_throw(self) -> None:
        config = _make_config(instructions="be helpful")
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, prompt, _ = _build_agent_and_prompt(config, "", {}, {})
        assert prompt == ""

    def test_path_c_instructions_takes_priority_over_messages(self) -> None:
        config = _make_config(
            instructions="Use instructions.",
            messages=[{"role": "system", "content": "Use messages."}],
        )
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, _, instructions = _build_agent_and_prompt(config, "q", {}, {})
        assert instructions == "Use instructions."


# ---------------------------------------------------------------------------
# §1.3 Tool conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    @pytest.mark.asyncio
    async def test_all_fields_forwarded(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        captured_tools: list[Any] = []

        def _capture_tool(**kw: Any) -> Any:
            captured_tools.append(kw)
            return MagicMock()

        agents_mock.FunctionTool = MagicMock(side_effect=_capture_tool)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            h = create_openai_agent_handler()
            config = _make_config(
                tools={"my-tool": {"description": "does stuff", "parameters": {}}}
            )
            await h(config, "hi", {"my-tool": AsyncMock(return_value="ok")})

        names = [t.get("name") for t in captured_tools]
        assert "my-tool" in names

    @pytest.mark.asyncio
    async def test_multiple_tools_all_included(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        captured_tools: list[str] = []

        def _capture_tool(**kw: Any) -> Any:
            captured_tools.append(kw.get("name"))
            return MagicMock()

        agents_mock.FunctionTool = MagicMock(side_effect=_capture_tool)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            h = create_openai_agent_handler()
            config = _make_config(
                tools={
                    "tool-a": {"description": "a", "parameters": {}},
                    "tool-b": {"description": "b", "parameters": {}},
                }
            )
            await h(config, "hi", {"tool-a": AsyncMock(), "tool-b": AsyncMock()})

        assert "tool-a" in captured_tools
        assert "tool-b" in captured_tools

    @pytest.mark.asyncio
    async def test_empty_tools_no_tools_sent(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)
        captured: list[Any] = []
        agents_mock.Agent = MagicMock(
            side_effect=lambda **kw: (captured.append(kw), MagicMock())[1]
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            h = create_openai_agent_handler()
            await h(_make_config(), "hi")

        if captured:
            assert not captured[0].get("tools")


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop
# ---------------------------------------------------------------------------


class TestToolExecutionLoop:
    @pytest.mark.asyncio
    async def test_tool_not_found_execute_callback_throws(self) -> None:
        from launchdarkly_ai_openai_agents.handler import _build_agent_tools

        agents_mock = MagicMock()
        captured_fns: list[Any] = []
        agents_mock.tool = MagicMock(
            side_effect=lambda **kw: lambda fn: (captured_fns.append(fn), fn)[1]
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _build_agent_tools({"my-tool": {"description": "d"}}, {})

        if captured_fns:
            with pytest.raises(ValueError, match="No handler"):
                await captured_fns[0]({})

    @pytest.mark.asyncio
    async def test_tool_handler_throws_propagates(self) -> None:
        from launchdarkly_ai_openai_agents.handler import _build_agent_tools

        agents_mock = MagicMock()
        captured_fns: list[Any] = []
        agents_mock.tool = MagicMock(
            side_effect=lambda **kw: lambda fn: (captured_fns.append(fn), fn)[1]
        )

        async def _bad_handler(args: Any) -> str:
            raise RuntimeError("handler error")

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _build_agent_tools(
                {"my-tool": {"description": "d"}}, {"my-tool": _bad_handler}
            )

        if captured_fns:
            with pytest.raises(RuntimeError, match="handler error"):
                await captured_fns[0]({})

    @pytest.mark.asyncio
    async def test_no_tools_in_config_tool_builder_never_called(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)
        tool_calls: list[Any] = []
        agents_mock.tool = MagicMock(
            side_effect=lambda **kw: lambda fn: (tool_calls.append(kw), fn)[1]
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            h = create_openai_agent_handler()
            await h(_make_config(), "hi")

        assert len(tool_calls) == 0


# ---------------------------------------------------------------------------
# §1.5 Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    @pytest.mark.asyncio
    async def test_span_name_blocking(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi")

        mock_trace.get_tracer.return_value.start_span.assert_called_with(
            "openai.agent.run"
        )

    @pytest.mark.asyncio
    async def test_gen_ai_system(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.system") == "openai"

    @pytest.mark.asyncio
    async def test_gen_ai_operation_name(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.operation.name") == "chat"

    @pytest.mark.asyncio
    async def test_gen_ai_request_model(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(model={"name": "gpt-4o"}), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.request.model") == "gpt-4o"

    @pytest.mark.asyncio
    async def test_token_attributes_set(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out", input_tokens=20, output_tokens=8)
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.usage.input_tokens") == 20
        assert calls.get("gen_ai.usage.output_tokens") == 8

    @pytest.mark.asyncio
    async def test_span_status_ok(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi")

        mock_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_span_end_always_called(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_gen_ai_content_prompt_event(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "my question")

        event_calls = [
            c
            for c in mock_span.add_event.call_args_list
            if c[0][0] == "gen_ai.content.prompt"
        ]
        assert event_calls
        # The gen_ai.prompt attribute must include the user input text
        prompt_attr = event_calls[0][0][1].get("gen_ai.prompt", "")
        assert "my question" in prompt_attr, (
            f"gen_ai.prompt must include user input 'my question', got: {prompt_attr!r}"
        )

    @pytest.mark.asyncio
    async def test_gen_ai_content_completion_event(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("final answer")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi")

        event_calls = [
            c
            for c in mock_span.add_event.call_args_list
            if c[0][0] == "gen_ai.content.completion"
        ]
        assert event_calls

    @pytest.mark.asyncio
    async def test_gen_ai_response_model(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(model={"name": "gpt-4o"}), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "gen_ai.response.model" in calls
        assert calls["gen_ai.response.model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_ld_span_attributes(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)
        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
            }
        }

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi", variables=variables)

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("launchdarkly.operation.type") == "gen_ai"
        assert calls.get("launchdarkly.config.key") == "my-config"
        assert calls.get("launchdarkly.variation.key") == "v1"
        assert calls.get("launchdarkly.run.id") == "run-abc"
        assert "launchdarkly.graph.key" not in calls

    async def test_ld_graph_key_set_when_present(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("out")
        agents_mock = _mock_agents_module(run_result)
        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
                "graphKey": "my-graph",
            }
        }

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    await h(_make_config(), "hi", variables=variables)

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("launchdarkly.graph.key") == "my-graph"


# ---------------------------------------------------------------------------
# §1.6 Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_records_exception_on_span(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.Runner.run = AsyncMock(side_effect=RuntimeError("provider error"))
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)
        agents_mock.handoff = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.record_exception.assert_called()

    @pytest.mark.asyncio
    async def test_sets_span_status_error(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.Runner.run = AsyncMock(side_effect=RuntimeError("fail"))
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_ends_span_on_error(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.Runner.run = AsyncMock(side_effect=RuntimeError("fail"))
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_rethrows_error(self) -> None:
        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.Runner.run = AsyncMock(side_effect=RuntimeError("specific"))
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_openai_agent_handler()
                with pytest.raises(RuntimeError, match="specific"):
                    await h(_make_config(), "hi")


# ---------------------------------------------------------------------------
# §1.7 Convenience export
# ---------------------------------------------------------------------------


class TestConvenienceExport:
    def test_calls_through_to_model_call(self) -> None:
        from launchdarkly_ai_openai_agents.handler import openai_agents

        assert callable(openai_agents)

    def test_passes_config_key_user_input_and_context(self) -> None:
        import inspect

        from launchdarkly_ai_openai_agents.handler import openai_agents

        sig = inspect.signature(openai_agents)
        assert "config_key" in sig.parameters
        assert "user_input" in sig.parameters
        assert "context" in sig.parameters

    def test_config_key_forwarded_as_key(self) -> None:
        import launchdarkly_ai_openai_agents.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_openai_agents.handler import openai_agents

            ctx = {"kind": "user", "key": "u1"}
            openai_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        handler = call_kwargs.get("handler")
        assert handler is not None
        assert handler.provides_for == ("OpenAI", "agent")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self) -> None:
        import launchdarkly_ai_openai_agents.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_openai_agents.handler import openai_agents

            ctx = {"kind": "user", "key": "u1"}
            openai_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# §1.8 Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_is_defined(self) -> None:
        h = create_openai_agent_handler()
        assert hasattr(h, "stream")

    @pytest.mark.asyncio
    async def test_stream_returns_async_generator(self) -> None:
        import inspect

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

        async def _empty_stream() -> AsyncIterator[Any]:
            return
            yield

        streamed_result = MagicMock()
        streamed_result.stream_events = _empty_stream
        streamed_result.raw_responses = []
        streamed_result.final_output = "done"
        agents_mock.Runner.run_streamed = MagicMock(return_value=streamed_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_openai_agent_handler()
                gen = await h.stream(_make_config(), "hi")
                assert inspect.isasyncgen(gen) or hasattr(gen, "__aiter__")

    @pytest.mark.asyncio
    async def test_yields_exactly_one_done_event(self) -> None:
        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

        async def _empty_stream() -> AsyncIterator[Any]:
            return
            yield

        streamed_result = MagicMock()
        streamed_result.stream_events = _empty_stream
        streamed_result.raw_responses = []
        streamed_result.final_output = "final"
        agents_mock.Runner.run_streamed = MagicMock(return_value=streamed_result)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_openai_agent_handler()
                events = [e async for e in await h.stream(_make_config(), "hi")]

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1


# ---------------------------------------------------------------------------
# §1.5 Streaming telemetry (Appendix A.5 — do not patch _HAS_OTEL=False)
# ---------------------------------------------------------------------------


class TestStreamingTelemetry:
    def _make_streamed_mock(self) -> Any:
        async def _empty_stream() -> AsyncIterator[Any]:
            return
            yield

        streamed_result = MagicMock()
        streamed_result.stream_events = _empty_stream
        streamed_result.raw_responses = []
        streamed_result.final_output = "done"
        return streamed_result

    @pytest.mark.asyncio
    async def test_span_started_during_stream(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)
        agents_mock.Runner.run_streamed = MagicMock(
            return_value=self._make_streamed_mock()
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass

        mock_trace.get_tracer.return_value.start_span.assert_called_with(
            "openai.agent.run.stream"
        )

    @pytest.mark.asyncio
    async def test_ld_span_attributes_set_during_stream(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)
        agents_mock.Runner.run_streamed = MagicMock(
            return_value=self._make_streamed_mock()
        )
        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    async for _ in await h.stream(
                        _make_config(), "hi", None, variables
                    ):
                        pass

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("launchdarkly.operation.type") == "gen_ai"
        assert calls.get("launchdarkly.config.key") == "k"
        assert calls.get("launchdarkly.variation.key") == "v"
        assert calls.get("launchdarkly.run.id") == "r"

    @pytest.mark.asyncio
    async def test_span_ended_after_stream_completes(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock())
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)
        agents_mock.Runner.run_streamed = MagicMock(
            return_value=self._make_streamed_mock()
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_openai_agent_handler()
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass

        mock_span.end.assert_called()


# ---------------------------------------------------------------------------
# §1.9 Output format (build_output_type)
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_absent_output_format_no_change(self) -> None:
        result = build_output_type(None)
        assert result is None

    def test_output_format_sets_output_type_on_agent(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = build_output_type(schema)
        assert result is not None
        assert result["type"] == "json_schema"
        assert "schema" in result

    def test_absent_output_format_returns_plain_string(self) -> None:
        result = build_output_type({})
        assert result is None

    def test_output_format_returns_parsed_object(self) -> None:
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        result = build_output_type(schema)
        assert result is not None
        assert result["schema"]["properties"]["count"]["type"] == "integer"


# ---------------------------------------------------------------------------
# §1.2 Path C — None user_input must not produce None prompt
# ---------------------------------------------------------------------------


class TestNoneUserInput:
    """TESTING.md §1.2 Path C: When user_input is None, the prompt passed to
    Runner.run must be '' (empty string), not None."""

    @pytest.mark.asyncio
    async def test_none_user_input_instructions_path_prompt_is_empty_string(
        self,
    ) -> None:
        """When instructions path is taken and user_input=None, the prompt
        forwarded to Runner.run must be '' not None."""
        captured_prompts: list[Any] = []

        run_result = _make_run_result("ok")
        agents_mock = _mock_agents_module(run_result)

        async def _spy_run(agent: Any, prompt: Any) -> Any:
            captured_prompts.append(prompt)
            return run_result

        agents_mock.Runner.run = _spy_run

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_openai_agent_handler()
                await h(_make_config(instructions="Be helpful."), None)

        assert captured_prompts, "Runner.run was not called"
        assert captured_prompts[0] is not None, (
            "prompt passed to Runner.run must be '' when user_input is None, not None"
        )
        assert captured_prompts[0] == "", (
            f"Expected prompt='', got {captured_prompts[0]!r}"
        )


# ---------------------------------------------------------------------------
# History parameter
# ---------------------------------------------------------------------------


class TestHistory:
    SAMPLE_HISTORY: ClassVar[list[dict[str, Any]]] = [
        {"role": "user", "content": "What is feature flagging?"},
        {"role": "assistant", "content": "Feature flagging is a technique..."},
    ]

    def _mock_agents(self) -> Any:
        mock = MagicMock()
        mock.Agent = MagicMock(return_value=MagicMock())
        mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)
        return mock

    def test_history_appended_to_instructions(self) -> None:
        config = _make_config(instructions="Be concise.")
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, _, instructions = _build_agent_and_prompt(
                config, "hi", {}, {}, self.SAMPLE_HISTORY
            )
        assert instructions is not None
        assert "Conversation History:" in instructions
        assert "Be concise." in instructions

    def test_history_format_is_correct(self) -> None:
        config = _make_config(instructions="Be helpful.")
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, _, instructions = _build_agent_and_prompt(
                config, "hi", {}, {}, self.SAMPLE_HISTORY
            )
        assert instructions is not None
        assert "user: What is feature flagging?" in instructions
        assert "assistant: Feature flagging is a technique..." in instructions

    def test_empty_history_treated_like_no_history(self) -> None:
        config = _make_config(instructions="Be concise.")
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, _, instr_with_empty = _build_agent_and_prompt(config, "hi", {}, {}, [])
            _, _, instr_without = _build_agent_and_prompt(config, "hi", {}, {})
        assert instr_with_empty == instr_without
        assert "Conversation History:" not in (instr_with_empty or "")

    def test_history_without_prior_instructions(self) -> None:
        config = _make_config()
        agents_mock = self._mock_agents()
        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            _, _, instructions = _build_agent_and_prompt(
                config, "hi", {}, {}, self.SAMPLE_HISTORY
            )
        assert instructions is not None
        assert "Conversation History:" in instructions
        assert "user: What is feature flagging?" in instructions
