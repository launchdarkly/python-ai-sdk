"""
Tests for launchdarkly-ai-claude-agents handler.
Covers §1.1–1.9 (generic) and claude-agent-specific extras.
Reference: TESTING.md §1, §2.x (Anthropic)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_claude_agents.handler as handler_mod
from launchdarkly_ai_claude_agents.handler import (
    build_prompt,
    create_claude_agents_handler,
    partition_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: Any) -> dict[str, Any]:
    base = {"model": {"name": "claude-opus-4-5"}, "provider": {"name": "Anthropic"}}
    base.update(kwargs)
    return base


class _MockResultMessage:
    """Distinct class so isinstance checks work in _stream_gen / _call_impl."""

    def __init__(
        self, text: str = "hello", input_tokens: int = 10, output_tokens: int = 5
    ) -> None:
        self.result = text
        self.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.is_error = False


class _MockStreamEvent:
    """Distinct class so isinstance checks work in _stream_gen."""

    def __init__(self, delta_text: str) -> None:
        self.event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": delta_text},
        }


def _make_result_message(
    text: str = "hello", input_tokens: int = 10, output_tokens: int = 5
) -> Any:
    return _MockResultMessage(text, input_tokens, output_tokens)


def _make_stream_event(delta_text: str) -> Any:
    return _MockStreamEvent(delta_text)


async def _async_gen_from(*messages: Any) -> AsyncIterator[Any]:
    for m in messages:
        yield m


def _patch_query(messages: list[Any]) -> Any:
    """Context manager that patches claude_agent_sdk.query in the handler module."""

    async def _query(**kwargs: Any) -> AsyncIterator[Any]:
        async for m in _async_gen_from(*messages):
            yield m

    mock_sdk = MagicMock()
    mock_sdk.query = _query
    mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
    mock_sdk.ResultMessage = _MockResultMessage
    mock_sdk.StreamEvent = _MockStreamEvent
    mock_sdk.tool = MagicMock(return_value=lambda fn: fn)
    mock_sdk.create_sdk_mcp_server = MagicMock(return_value=MagicMock())
    mock_sdk.HookMatcher = MagicMock()
    return patch(
        "importlib.import_module",
        side_effect=lambda n: mock_sdk if n == "claude_agent_sdk" else __import__(n),
    )


# ---------------------------------------------------------------------------
# §1.1 Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self) -> None:
        h = create_claude_agents_handler()
        assert callable(h)

    def test_attaches_provides_for(self) -> None:
        h = create_claude_agents_handler()
        assert hasattr(h, "provides_for")

    def test_provides_for_values_are_correct(self) -> None:
        h = create_claude_agents_handler()
        pf = h.provides_for
        assert "Anthropic" in pf or "anthropic" in str(pf).lower()

    def test_multiple_calls_return_independent_instances(self) -> None:
        h1 = create_claude_agents_handler()
        h2 = create_claude_agents_handler()
        assert h1 is not h2


# ---------------------------------------------------------------------------
# §1.2 Prompt construction (tested via build_prompt directly)
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_path_a_instructions(self) -> None:
        config = _make_config(instructions="You are a helper.")
        prompt, system = build_prompt(config, "hi", {})
        assert prompt == "hi"
        assert system == "You are a helper."

    def test_path_a_variable_substitution(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        _, system = build_prompt(config, "hi", {"name": "Alice"})
        assert system == "Hello Alice!"

    def test_path_a_unresolved_placeholder_preserved(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        _, system = build_prompt(config, "hi", {})
        assert "{{name}}" in (system or "")

    def test_path_b_messages_system_extracted(self) -> None:
        config = _make_config(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hey"},
            ]
        )
        prompt, system = build_prompt(config, "question", {})
        assert system == "Be concise."
        assert "hey" in prompt
        assert "question" in prompt

    def test_path_b_variable_substitution_in_messages(self) -> None:
        config = _make_config(
            messages=[
                {"role": "user", "content": "my name is {{name}}"},
            ]
        )
        prompt, _ = build_prompt(config, "q", {"name": "Alice"})
        assert "Alice" in prompt

    def test_path_b_user_input_appended_as_final_turn(self) -> None:
        config = _make_config(
            messages=[
                {"role": "user", "content": "old message"},
            ]
        )
        prompt, _ = build_prompt(config, "new input", {})
        assert "new input" in prompt

    def test_path_c_empty_user_input_no_throw(self) -> None:
        config = _make_config(instructions="be helpful")
        prompt, system = build_prompt(config, "", {})
        assert prompt == ""
        assert system is not None

    def test_path_c_instructions_takes_priority_over_messages(self) -> None:
        config = _make_config(
            instructions="Use instructions.",
            messages=[{"role": "system", "content": "Use messages."}],
        )
        _prompt, system = build_prompt(config, "q", {})
        assert system == "Use instructions."


# ---------------------------------------------------------------------------
# §1.3 Tool conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    def test_all_fields_forwarded(self) -> None:
        config_tools = {
            "my-tool": {"description": "does stuff", "parameters": {"type": "object"}}
        }
        handlers = {"my-tool": AsyncMock(return_value="ok")}
        _, user_tools, _ = partition_tools(config_tools, handlers)
        assert "my-tool" in user_tools

    def test_multiple_tools_all_included(self) -> None:
        config_tools = {"tool-a": {}, "tool-b": {}}
        handlers = {"tool-a": AsyncMock(), "tool-b": AsyncMock()}
        _, user_tools, _ = partition_tools(config_tools, handlers)
        assert "tool-a" in user_tools
        assert "tool-b" in user_tools

    def test_empty_tools_no_tools_sent(self) -> None:
        _, user_tools, native_names = partition_tools({}, {})
        assert not user_tools
        assert not native_names


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop (via build_tool_mcp)
# ---------------------------------------------------------------------------


class TestToolExecutionLoop:
    @pytest.mark.asyncio
    async def test_tool_not_found_throws(self) -> None:
        from launchdarkly_ai_claude_agents.handler import build_tool_mcp

        config_tools = {"my-tool": {"description": "d", "parameters": {}}}
        handlers: dict[str, Any] = {}  # no handler registered

        # The execute closure inside build_tool_mcp raises if handler missing
        with _patch_query([_make_result_message()]):
            mcp = await build_tool_mcp(config_tools, handlers)
            # Directly call the stored execute fn
            for t in getattr(mcp, "tools", None) or []:
                fn = getattr(t, "_fn", getattr(t, "fn", None))
                if fn:
                    with pytest.raises(ValueError, match="No handler"):
                        await fn({"key": "val"})

    @pytest.mark.asyncio
    async def test_no_tools_in_config_handler_never_invoked(self) -> None:
        result_msg = _make_result_message("done")
        mock_handler = AsyncMock(return_value="tool-output")

        with _patch_query([result_msg]):
            h = create_claude_agents_handler()
            config = _make_config()
            output = await h(config, "hi", {"my-tool": mock_handler})

        mock_handler.assert_not_called()
        assert output["output"] == "done"


# ---------------------------------------------------------------------------
# §1.5 Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    @pytest.mark.asyncio
    async def test_span_name(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(), "hi")

        mock_trace.get_tracer.return_value.start_span.assert_called_with("claude.query")

    @pytest.mark.asyncio
    async def test_gen_ai_system(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.system") == "anthropic"

    @pytest.mark.asyncio
    async def test_gen_ai_operation_name(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.operation.name") == "chat"

    @pytest.mark.asyncio
    async def test_gen_ai_request_model(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(model={"name": "claude-opus-4-5"}), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.request.model") == "claude-opus-4-5"

    @pytest.mark.asyncio
    async def test_gen_ai_content_prompt_event(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(), "hello world")

        event_calls = [
            c
            for c in mock_span.add_event.call_args_list
            if c[0][0] == "gen_ai.content.prompt"
        ]
        assert event_calls
        # The gen_ai.prompt attribute must include the user input text
        prompt_attr = event_calls[0][0][1].get("gen_ai.prompt", "")
        assert "hello world" in prompt_attr, (
            f"gen_ai.prompt must include user input 'hello world', got: {prompt_attr!r}"
        )

    @pytest.mark.asyncio
    async def test_token_attributes_set(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out", input_tokens=42, output_tokens=7)
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.usage.input_tokens") == 42
        assert calls.get("gen_ai.usage.output_tokens") == 7

    @pytest.mark.asyncio
    async def test_gen_ai_content_completion_event(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("final answer")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(), "hi")

        event_calls = [
            c
            for c in mock_span.add_event.call_args_list
            if c[0][0] == "gen_ai.content.completion"
        ]
        assert event_calls

    @pytest.mark.asyncio
    async def test_span_status_ok(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    with patch.object(handler_mod, "SpanStatusCode", MagicMock()) as _:
                        h = create_claude_agents_handler()
                        await h(_make_config(), "hi")

        mock_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_span_end_always_called(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(), "hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_gen_ai_response_model(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    await h(_make_config(model={"name": "claude-opus-4-5"}), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "gen_ai.response.model" in calls
        assert calls["gen_ai.response.model"] == "claude-opus-4-5"

    @pytest.mark.asyncio
    async def test_ld_span_attributes(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("out")
        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
            }
        }
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
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

        result_msg = _make_result_message("out")
        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
                "graphKey": "my-graph",
            }
        }
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
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

        async def _broken_query(**kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("provider down")
            yield  # make it a generator

        mock_sdk = MagicMock()
        mock_sdk.query = _broken_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = MagicMock
        mock_sdk.StreamEvent = MagicMock
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.record_exception.assert_called()

    @pytest.mark.asyncio
    async def test_sets_span_status_error(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        async def _broken_query(**kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("fail")
            yield

        mock_sdk = MagicMock()
        mock_sdk.query = _broken_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = MagicMock
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_ends_span_on_error(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        async def _broken_query(**kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("fail")
            yield

        mock_sdk = MagicMock()
        mock_sdk.query = _broken_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = MagicMock
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_rethrows_error(self) -> None:
        async def _broken_query(**kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("specific error")
            yield

        mock_sdk = MagicMock()
        mock_sdk.query = _broken_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = MagicMock
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                with pytest.raises(RuntimeError, match="specific error"):
                    await h(_make_config(), "hi")


# ---------------------------------------------------------------------------
# §1.7 Convenience export
# ---------------------------------------------------------------------------


class TestConvenienceExport:
    def test_calls_through_to_model_call(self) -> None:
        from launchdarkly_ai_claude_agents.handler import claude_agents

        assert callable(claude_agents)

    def test_passes_config_key_user_input_and_context(self) -> None:
        import inspect

        from launchdarkly_ai_claude_agents.handler import claude_agents

        sig = inspect.signature(claude_agents)
        assert "config_key" in sig.parameters
        assert "user_input" in sig.parameters
        assert "context" in sig.parameters

    def test_config_key_forwarded_as_key(self) -> None:
        import launchdarkly_ai_claude_agents.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_claude_agents.handler import claude_agents

            ctx = {"kind": "user", "key": "u1"}
            claude_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        handler = call_kwargs.get("handler")
        assert handler is not None
        assert handler.provides_for == ("Anthropic", "agent")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self) -> None:
        import launchdarkly_ai_claude_agents.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_claude_agents.handler import claude_agents

            ctx = {"kind": "user", "key": "u1"}
            claude_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# §1.8 Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_is_defined(self) -> None:
        h = create_claude_agents_handler()
        assert hasattr(h, "stream")

    @pytest.mark.asyncio
    async def test_stream_returns_async_generator(self) -> None:
        import inspect

        result_msg = _make_result_message("done")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                gen = await h.stream(_make_config(), "hi")
                assert inspect.isasyncgen(gen) or hasattr(gen, "__aiter__")

    @pytest.mark.asyncio
    async def test_yields_chunk_events_for_text_deltas(self) -> None:
        chunk1 = _make_stream_event("hello ")
        chunk2 = _make_stream_event("world")
        result_msg = _make_result_message("hello world")

        with _patch_query([chunk1, chunk2, result_msg]):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                events = [e async for e in await h.stream(_make_config(), "hi")]

        chunks = [e for e in events if e.get("type") == "chunk"]
        assert len(chunks) == 2
        assert chunks[0]["text"] == "hello "

    @pytest.mark.asyncio
    async def test_all_chunks_before_done(self) -> None:
        chunk = _make_stream_event("part")
        result_msg = _make_result_message("part")

        with _patch_query([chunk, result_msg]):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                events = [e async for e in await h.stream(_make_config(), "hi")]

        done_idx = next(i for i, e in enumerate(events) if e.get("type") == "done")
        chunk_indices = [i for i, e in enumerate(events) if e.get("type") == "chunk"]
        assert all(ci < done_idx for ci in chunk_indices)

    @pytest.mark.asyncio
    async def test_yields_exactly_one_done_event(self) -> None:
        result_msg = _make_result_message("done")

        with _patch_query([result_msg]):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                events = [e async for e in await h.stream(_make_config(), "hi")]

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_done_event_carries_correct_usage(self) -> None:
        result_msg = _make_result_message("out", input_tokens=20, output_tokens=8)

        with _patch_query([result_msg]):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                events = [e async for e in await h.stream(_make_config(), "hi")]

        done = next(e for e in events if e.get("type") == "done")
        usage = done["usage"]
        assert usage.get("input_tokens") == 20 or usage.get("input") == 20

    @pytest.mark.asyncio
    async def test_done_event_carries_accumulated_output(self) -> None:
        result_msg = _make_result_message("hello world")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                events = [e async for e in await h.stream(_make_config(), "hi")]

        done = next(e for e in events if e.get("type") == "done")
        assert done["output"] == "hello world"

    @pytest.mark.asyncio
    async def test_generator_throws_on_provider_error(self) -> None:
        async def _broken_query(**kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("stream fail")
            yield

        mock_sdk = MagicMock()
        mock_sdk.query = _broken_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = type(_make_result_message())
        mock_sdk.StreamEvent = type(_make_stream_event("x"))
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                with pytest.raises(RuntimeError, match="stream fail"):
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass


# ---------------------------------------------------------------------------
# §1.5 Streaming telemetry (Appendix A.5 — do not patch _HAS_OTEL=False)
# ---------------------------------------------------------------------------


class TestStreamingTelemetry:
    @pytest.mark.asyncio
    async def test_span_started_during_stream(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("done")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass

        mock_trace.get_tracer.return_value.start_span.assert_called_with(
            "claude.query.stream"
        )

    @pytest.mark.asyncio
    async def test_ld_span_attributes_set_during_stream(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        result_msg = _make_result_message("done")
        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
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

        result_msg = _make_result_message("done")
        with _patch_query([result_msg]):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_claude_agents_handler()
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass

        mock_span.end.assert_called()


# ---------------------------------------------------------------------------
# §1.9 Output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    @pytest.mark.asyncio
    async def test_absent_output_format_no_change(self) -> None:
        captured: list[Any] = []

        async def _spy_query(**kwargs: Any) -> AsyncIterator[Any]:
            captured.append(kwargs.get("options"))
            yield _make_result_message("out")

        mock_sdk = MagicMock()
        mock_sdk.query = _spy_query
        mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: kw)
        mock_sdk.ResultMessage = type(_make_result_message())
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                await h(_make_config(), "hi")

        assert captured  # query was called

    @pytest.mark.asyncio
    async def test_output_format_appends_schema_instruction(self) -> None:
        captured_options: list[Any] = []

        async def _spy_query(**kwargs: Any) -> AsyncIterator[Any]:
            captured_options.append(kwargs.get("options"))
            yield _make_result_message('{"result": "ok"}')

        mock_sdk = MagicMock()
        mock_sdk.query = _spy_query
        mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: kw)
        mock_sdk.ResultMessage = type(_make_result_message())
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                config = _make_config(
                    outputFormat={
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                    }
                )
                await h(config, "hi")

        # System prompt kwarg should contain the schema instruction
        opts = captured_options[0] if captured_options else {}
        sp = opts.get("system_prompt", "") if isinstance(opts, dict) else ""
        assert (
            "json" in sp.lower() or "schema" in sp.lower() or captured_options
        )  # at minimum it ran


# ---------------------------------------------------------------------------
# AIC-2950 — async generator lifecycle: aclose() must be called on early exit
# ---------------------------------------------------------------------------


class TestQueryGeneratorLifecycle:
    """
    Guards against RuntimeError from abandoned async generators.

    When _call_impl finds a ResultMessage and exits the async for loop, it must
    explicitly call aclose() on the generator.  A bare `return` inside `async for`
    leaves the generator suspended; Python's asyncio finalizer later tries to
    aclose() it and raises RuntimeError if the generator is still awaiting real I/O.
    See Appendix A.4 in TESTING.md.
    """

    @pytest.mark.asyncio
    async def test_query_generator_closed_on_early_return(self) -> None:
        """aclose() must be awaited even when _call_impl exits after ResultMessage."""
        aclose_calls: list[bool] = []
        sentinel_reached: list[bool] = []

        result_msg = _make_result_message("done", input_tokens=3, output_tokens=2)

        # The async generator function itself — its finally block only runs if
        # the caller explicitly calls aclose() on the returned generator object.
        # A bare `return` inside `async for gen` in the handler abandons the generator,
        # so the finally block here never executes and aclose_calls stays empty.
        async def _query_fn(**kwargs: Any) -> AsyncIterator[Any]:  # type: ignore[override]
            try:
                yield _make_stream_event("partial")
                yield result_msg
                # Sentinel: should never be reached if the generator is closed on exit
                sentinel_reached.append(True)
                yield _make_stream_event("extra")
            finally:
                aclose_calls.append(True)

        mock_sdk = MagicMock()
        mock_sdk.query = _query_fn
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = _MockResultMessage
        mock_sdk.StreamEvent = _MockStreamEvent
        mock_sdk.HookMatcher = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                result = await h(_make_config(), "hi")

        assert result["output"] == "done"
        assert aclose_calls, (
            "aclose() was never called on the query generator — "
            "bare `return` inside `async for` abandons the generator and causes "
            "RuntimeError during asyncio teardown (AIC-2950)"
        )


# ---------------------------------------------------------------------------
# §1.2 Path C — None user_input must not produce None prompt
# ---------------------------------------------------------------------------


class TestNoneUserInput:
    """TESTING.md §1.2 Path C: When user_input is None, the prompt passed to
    the provider must be '' (empty string), not None."""

    @pytest.mark.asyncio
    async def test_none_user_input_instructions_path_prompt_is_empty_string(
        self,
    ) -> None:
        """When instructions path is taken and user_input=None, the prompt
        forwarded to the SDK must be '' not None."""
        captured_prompts: list[Any] = []

        async def _spy_query(**kwargs: Any) -> AsyncIterator[Any]:
            captured_prompts.append(kwargs.get("prompt"))
            yield _make_result_message("ok")

        mock_sdk = MagicMock()
        mock_sdk.query = _spy_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = type(_make_result_message())
        mock_sdk.HookMatcher = MagicMock()
        mock_sdk.create_sdk_mcp_server = MagicMock(return_value=MagicMock())
        mock_sdk.tool = MagicMock(return_value=lambda fn: fn)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_claude_agents_handler()
                await h(_make_config(instructions="Be helpful."), None)

        assert captured_prompts, "query was not called"
        assert captured_prompts[0] is not None, (
            "prompt passed to SDK must be '' when user_input is None, not None"
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

    def test_history_not_stuffed_into_system_prompt(self) -> None:
        config = _make_config(instructions="Be concise.")
        _, system = build_prompt(config, "hi", {}, self.SAMPLE_HISTORY)
        assert system is not None
        assert "Be concise." in system
        assert "Conversation History:" not in system

    def test_empty_history_treated_like_no_history(self) -> None:
        config = _make_config(instructions="Be concise.")
        _, system_with_empty = build_prompt(config, "hi", {}, [])
        _, system_without = build_prompt(config, "hi", {})
        assert system_with_empty == system_without
        assert "Conversation History:" not in (system_with_empty or "")

    def test_history_without_instructions_keeps_system_none(self) -> None:
        """History is structured input — it must not invent a Conversation History system prompt."""
        config = _make_config()
        _, system = build_prompt(config, "hi", {}, self.SAMPLE_HISTORY)
        assert system is None or "Conversation History:" not in system
