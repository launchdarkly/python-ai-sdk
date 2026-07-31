"""
Tests for launchdarkly-ai-claude-messages handler.
Covers §1.1–1.9 (generic handler tests).
Reference: TESTING.md §1
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake anthropic response helpers
# ---------------------------------------------------------------------------


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(name: str, id: str = "tu1", input: dict | None = None) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.id = id
    b.input = input or {}
    return b


def _anthropic_response(
    content: list[Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    r = MagicMock()
    r.content = content
    r.stop_reason = stop_reason
    r.usage = MagicMock()
    r.usage.input_tokens = input_tokens
    r.usage.output_tokens = output_tokens
    return r


CONFIG = {
    "model": {"name": "claude-3-sonnet-20240229"},
    "provider": {"name": "Anthropic"},
    "instructions": "Be helpful.",
}


@pytest.fixture
def mock_anthropic(mocker):
    """Patches anthropic.AsyncAnthropic so no real network call is made."""
    mock_client = MagicMock()
    response = _anthropic_response([_text_block("Hello World")])
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=response)
    mocker.patch("anthropic.AsyncAnthropic", return_value=mock_client)
    return mock_client


# ---------------------------------------------------------------------------
# §1.1 Factory function and metadata
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        assert callable(h)

    def test_attaches_provides_for(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        assert h.provides_for is not None

    def test_provides_for_values_are_correct(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        assert h.provides_for == ("Anthropic", "messages")

    def test_multiple_calls_return_independent_instances(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h1 = create_claude_messages_handler()
        h2 = create_claude_messages_handler()
        assert h1 is not h2


# ---------------------------------------------------------------------------
# §1.2 Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    async def test_path_a_instructions(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        await h(CONFIG, "hi", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "Be helpful."

    async def test_path_a_variable_substitution(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {**CONFIG, "instructions": "Hello {{name}}"}
        h = create_claude_messages_handler()
        await h(config, "q", {}, {"name": "Alice"})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Hello Alice"

    async def test_path_a_unresolved_placeholder_preserved(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {**CONFIG, "instructions": "Hello {{missing}}"}
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert "{{missing}}" in call_kwargs.get("system", "")

    async def test_path_b_messages_system_extracted(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hello"},
            ],
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "System prompt"

    async def test_path_b_conversation_history_order(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
        }
        h = create_claude_messages_handler()
        await h(config, "Third", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[-1]["content"] == "Third"

    async def test_path_b_variable_substitution_in_messages(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "messages": [{"role": "user", "content": "Hello {{name}}"}],
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {"name": "Bob"})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "Hello Bob"

    async def test_path_b_user_input_appended_as_final_turn(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "messages": [{"role": "assistant", "content": "Hi"}],
        }
        h = create_claude_messages_handler()
        await h(config, "final-user-input", {}, {})
        msgs = mock_anthropic.messages.create.call_args.kwargs["messages"]
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "final-user-input"

    async def test_path_c_empty_user_input_no_throw(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        await h(CONFIG, "", {}, {})  # must not raise

    async def test_path_c_undefined_user_input_no_throw(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        await h(CONFIG, None, {}, {})  # must not raise  # type: ignore[arg-type]

    async def test_path_b_variable_substitution_in_system_message(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "messages": [{"role": "system", "content": "Hello {{name}}"}],
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {"name": "World"})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "Hello World"

    async def test_path_c_both_instructions_and_messages_messages_wins(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            **CONFIG,  # instructions = "Be helpful."
            "messages": [
                {"role": "system", "content": "from-messages"},
            ],
        }
        # When both present, messages-mode handler sends the messages array
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        # messages take priority — system comes from messages, not instructions
        assert call_kwargs.get("system") == "from-messages"


# ---------------------------------------------------------------------------
# §1.3 Tool conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    async def test_all_fields_forwarded(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            **CONFIG,
            "tools": {
                "search": {
                    "name": "search",
                    "type": "function",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            },
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        tools = call_kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search the web"
        assert "properties" in tools[0]["input_schema"]

    async def test_multiple_tools_all_included(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            **CONFIG,
            "tools": {
                "t1": {"name": "t1", "type": "function", "parameters": {}},
                "t2": {"name": "t2", "type": "function", "parameters": {}},
            },
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert len(call_kwargs.get("tools", [])) == 2

    async def test_empty_tools_no_tools_sent(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        await h(CONFIG, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs or call_kwargs.get("tools") == []

    async def test_custom_parameters_passthrough(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            **CONFIG,
            "tools": {
                "t1": {
                    "name": "t1",
                    "type": "function",
                    "parameters": {
                        "type": "object",
                        "properties": {"custom": {"type": "number"}},
                    },
                }
            },
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        tools = mock_anthropic.messages.create.call_args.kwargs.get("tools", [])
        assert "custom" in tools[0]["input_schema"].get("properties", {})


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop
# ---------------------------------------------------------------------------


class TestToolExecutionLoop:
    async def test_single_tool_call_then_done(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        tool_resp = _anthropic_response(
            [_tool_use_block("search", id="tu1")], stop_reason="tool_use"
        )
        final_resp = _anthropic_response([_text_block("result after tool")])
        mock_anthropic.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])
        tool_fn = AsyncMock(return_value="search result")
        h = create_claude_messages_handler()
        config = {
            **CONFIG,
            "tools": {
                "search": {"name": "search", "type": "function", "parameters": {}}
            },
        }
        result = await h(config, "q", {"search": tool_fn}, {})
        assert mock_anthropic.messages.create.call_count == 2
        tool_fn.assert_called_once()
        assert "after tool" in result["output"]

    async def test_tool_not_found_throws(self, mock_anthropic: MagicMock) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        tool_resp = _anthropic_response(
            [_tool_use_block("unknown_tool")], stop_reason="tool_use"
        )
        mock_anthropic.messages.create = AsyncMock(return_value=tool_resp)
        h = create_claude_messages_handler()
        config = {
            **CONFIG,
            "tools": {"other": {"name": "other", "type": "function", "parameters": {}}},
        }
        with pytest.raises(Exception, match="No handler"):
            await h(config, "q", {}, {})

    async def test_tool_handler_throws_propagates(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        tool_resp = _anthropic_response([_tool_use_block("t1")], stop_reason="tool_use")
        mock_anthropic.messages.create = AsyncMock(return_value=tool_resp)
        fn = AsyncMock(side_effect=RuntimeError("tool failed"))
        h = create_claude_messages_handler()
        config = {
            **CONFIG,
            "tools": {"t1": {"name": "t1", "type": "function", "parameters": {}}},
        }
        with pytest.raises(RuntimeError, match="tool failed"):
            await h(config, "q", {"t1": fn}, {})

    async def test_no_tools_in_config_handler_never_invoked(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        tool_fn = AsyncMock()
        h = create_claude_messages_handler()
        await h(CONFIG, "q", {"t1": tool_fn}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs or not call_kwargs.get("tools")
        tool_fn.assert_not_called()

    async def test_multiple_consecutive_tool_calls(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        resp1 = _anthropic_response(
            [_tool_use_block("t1", "id1")], stop_reason="tool_use"
        )
        resp2 = _anthropic_response(
            [_tool_use_block("t2", "id2")], stop_reason="tool_use"
        )
        resp3 = _anthropic_response([_text_block("final")])
        mock_anthropic.messages.create = AsyncMock(side_effect=[resp1, resp2, resp3])
        fn1 = AsyncMock(return_value="r1")
        fn2 = AsyncMock(return_value="r2")
        cfg = {
            **CONFIG,
            "tools": {
                "t1": {"name": "t1", "type": "function", "parameters": {}},
                "t2": {"name": "t2", "type": "function", "parameters": {}},
            },
        }
        h = create_claude_messages_handler()
        result = await h(cfg, "q", {"t1": fn1, "t2": fn2}, {})
        fn1.assert_called_once()
        fn2.assert_called_once()
        assert result["output"] == "final"


# ---------------------------------------------------------------------------
# §1.5 Telemetry
# ---------------------------------------------------------------------------


def _make_tracer_patch(mock_span: MagicMock) -> Any:
    """Creates a patched trace module targeting the handler's imported `trace`."""
    mock_tracer = MagicMock()
    mock_tracer.start_span = MagicMock(return_value=mock_span)
    mock_trace_mod = MagicMock()
    mock_trace_mod.get_tracer = MagicMock(return_value=mock_tracer)
    return mock_trace_mod, mock_tracer


class TestTelemetry:
    async def test_span_name(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, mock_tracer = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        mock_tracer.start_span.assert_called_with("claude.messages")

    async def test_gen_ai_system(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.system") == "anthropic"

    async def test_gen_ai_request_model(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.request.model") == CONFIG["model"]["name"]

    async def test_token_attributes_set(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "gen_ai.usage.input_tokens" in attrs
        assert "gen_ai.usage.output_tokens" in attrs

    async def test_span_status_ok(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        from opentelemetry.trace import StatusCode

        mock_span.set_status.assert_called_with(StatusCode.OK)

    async def test_span_end_always_called(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        mock_span.end.assert_called_once()

    async def test_gen_ai_operation_name(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.operation.name") == "chat"

    async def test_gen_ai_content_prompt_event(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "user input", {}, {})
        event_names = [c[0][0] for c in mock_span.add_event.call_args_list]
        assert "gen_ai.content.prompt" in event_names

    async def test_gen_ai_content_completion_event(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        event_names = [c[0][0] for c in mock_span.add_event.call_args_list]
        assert "gen_ai.content.completion" in event_names

    async def test_total_tokens_attribute(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "gen_ai.usage.total_tokens" in attrs
        assert attrs["gen_ai.usage.total_tokens"] == attrs.get(
            "gen_ai.usage.input_tokens", 0
        ) + attrs.get("gen_ai.usage.output_tokens", 0)

    async def test_gen_ai_response_model(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, {})
        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "gen_ai.response.model" in attrs
        assert attrs["gen_ai.response.model"] == CONFIG["model"]["name"]

    async def test_ld_span_attributes(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
            }
        }
        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, variables)
        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("launchdarkly.operation.type") == "gen_ai"
        assert attrs.get("launchdarkly.config.key") == "my-config"
        assert attrs.get("launchdarkly.variation.key") == "v1"
        assert attrs.get("launchdarkly.run.id") == "run-abc"
        assert "launchdarkly.graph.key" not in attrs

    async def test_ld_graph_key_set_when_present(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
                "graphKey": "my-graph",
            }
        }
        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            await h(CONFIG, "q", {}, variables)
        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("launchdarkly.graph.key") == "my-graph"


# ---------------------------------------------------------------------------
# §1.6 Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_records_exception_on_span(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=RuntimeError("api error")
        )
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            with pytest.raises(RuntimeError):
                await h(CONFIG, "q", {}, {})
        mock_span.record_exception.assert_called_once()

    async def test_sets_span_status_error(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=RuntimeError("api error")
        )
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            with pytest.raises(RuntimeError):
                await h(CONFIG, "q", {}, {})
        from opentelemetry.trace import StatusCode

        status_calls = [c[0][0] for c in mock_span.set_status.call_args_list]
        assert StatusCode.ERROR in status_calls

    async def test_ends_span_on_error(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=RuntimeError("api error")
        )
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod

        with patch.object(handler_mod, "trace", mock_trace_mod):
            from launchdarkly_ai_claude_messages import create_claude_messages_handler

            h = create_claude_messages_handler()
            with pytest.raises(RuntimeError):
                await h(CONFIG, "q", {}, {})
        mock_span.end.assert_called_once()

    async def test_rethrows_error(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.messages.create = AsyncMock(side_effect=RuntimeError("rethrown"))
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        with pytest.raises(RuntimeError, match="rethrown"):
            await h(CONFIG, "q", {}, {})


# ---------------------------------------------------------------------------
# §1.9 Structured output (outputFormat)
# ---------------------------------------------------------------------------


class TestOutputFormat:
    async def test_absent_output_format_no_change(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        await h(CONFIG, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        system = call_kwargs.get("system", "")
        assert "schema" not in system.lower()

    async def test_output_format_appends_schema_instruction_to_system(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            **CONFIG,
            "outputFormat": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        system = call_kwargs.get("system", "")
        assert "schema" in system.lower() or "JSON" in system

    async def test_output_format_with_messages_system_appended(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "messages": [{"role": "system", "content": "base-system"}],
            "outputFormat": {"type": "object"},
        }
        h = create_claude_messages_handler()
        await h(config, "q", {}, {})
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        system = call_kwargs.get("system", "")
        assert "base-system" in system
        assert "JSON" in system or "schema" in system.lower()


# ---------------------------------------------------------------------------
# §1.7 Convenience export
# ---------------------------------------------------------------------------


class TestConvenienceExport:
    def test_calls_through_to_model_call(self, mock_anthropic: MagicMock) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_claude_messages.handler import claude_messages

            ctx = {"kind": "user", "key": "u1"}
            claude_messages("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        handler = call_kwargs.get("handler")
        assert handler is not None
        assert handler.provides_for == ("Anthropic", "messages")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_claude_messages.handler import claude_messages

            ctx = {"kind": "user", "key": "u1"}
            claude_messages("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# §1.8 Streaming
# ---------------------------------------------------------------------------


def _make_stream_event(text: str) -> MagicMock:
    e = MagicMock()
    e.type = "content_block_delta"
    e.delta = MagicMock()
    e.delta.type = "text_delta"
    e.delta.text = text
    return e


def _make_stream_context(
    chunks: list[str], input_tok: int = 5, output_tok: int = 3
) -> Any:
    """Returns a mock anthropic stream context manager."""
    events = [_make_stream_event(c) for c in chunks]
    final_msg = MagicMock()
    final_msg.stop_reason = "end_turn"
    final_msg.usage = MagicMock(input_tokens=input_tok, output_tokens=output_tok)
    final_msg.content = []

    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[Any]:
            return self._iter()

        async def _iter(self) -> AsyncIterator[Any]:
            for e in events:
                yield e

        async def get_final_message(self) -> Any:
            return final_msg

    @asynccontextmanager
    async def _ctx_mgr() -> AsyncGenerator[Any, None]:
        yield _FakeStream()

    stream_mgr = _ctx_mgr()
    return stream_mgr, final_msg


class TestStreaming:
    def _patch_stream(
        self,
        mock_anthropic: MagicMock,
        chunks: list[str],
        input_tok: int = 5,
        output_tok: int = 3,
    ) -> None:
        ctx, _ = _make_stream_context(chunks, input_tok, output_tok)
        mock_anthropic.messages.stream = MagicMock(return_value=ctx)

    async def test_stream_defined_and_async_generator(
        self, mock_anthropic: MagicMock
    ) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["hi"])
        with patch.object(handler_mod, "_HAS_OTEL", False):
            h = create_claude_messages_handler()
        assert h.has_stream
        gen = await h.stream(CONFIG, "q")
        assert hasattr(gen, "__aiter__")

    async def test_yields_chunk_events_for_text_deltas(
        self, mock_anthropic: MagicMock
    ) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["hello ", "world"])
        with patch.object(handler_mod, "_HAS_OTEL", False):
            h = create_claude_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        chunks = [e for e in events if e.get("type") == "chunk"]
        assert len(chunks) == 2
        assert chunks[0]["text"] == "hello "
        assert chunks[1]["text"] == "world"

    async def test_all_chunks_before_done(self, mock_anthropic: MagicMock) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["a", "b"])
        with patch.object(handler_mod, "_HAS_OTEL", False):
            h = create_claude_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        done_idx = next(i for i, e in enumerate(events) if e.get("type") == "done")
        for e in events[done_idx + 1 :]:
            assert e.get("type") != "chunk"

    async def test_yields_exactly_one_done_event(
        self, mock_anthropic: MagicMock
    ) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["x"])
        with patch.object(handler_mod, "_HAS_OTEL", False):
            h = create_claude_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    async def test_done_event_carries_usage(self, mock_anthropic: MagicMock) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["text"], input_tok=7, output_tok=3)
        with patch.object(handler_mod, "_HAS_OTEL", False):
            h = create_claude_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        done = next(e for e in events if e.get("type") == "done")
        assert done["usage"]["input_tokens"] == 7
        assert done["usage"]["output_tokens"] == 3

    async def test_done_event_carries_accumulated_output(
        self, mock_anthropic: MagicMock
    ) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["hello ", "world"])
        with patch.object(handler_mod, "_HAS_OTEL", False):
            h = create_claude_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        done = next(e for e in events if e.get("type") == "done")
        assert done["output"] == "hello world"

    async def test_generator_throws_on_provider_error(
        self, mock_anthropic: MagicMock
    ) -> None:
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        @asynccontextmanager
        async def _bad_ctx() -> AsyncGenerator[Any, None]:
            raise RuntimeError("stream error")
            yield  # make it a generator

        mock_anthropic.messages.stream = MagicMock(return_value=_bad_ctx())
        with patch.object(handler_mod, "_HAS_OTEL", False):
            h = create_claude_messages_handler()
            with pytest.raises(RuntimeError, match="stream error"):
                async for _ in await h.stream(CONFIG, "q"):
                    pass


# ---------------------------------------------------------------------------
# §1.2 Path C — None user_input must not produce None content
# ---------------------------------------------------------------------------


class TestNoneUserInput:
    """TESTING.md §1.2 Path C: When user_input is None, the user-role message
    content sent to the provider must be '' not None."""

    async def test_none_user_input_instructions_path_no_none_content(
        self, mock_anthropic: MagicMock
    ) -> None:
        """When instructions path is taken and user_input=None, no message in
        the API call may have content=None."""
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        captured: list[Any] = []

        async def _capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return _anthropic_response([_text_block("ok")])

        mock_anthropic.messages.create = _capture
        h = create_claude_messages_handler()
        await h(CONFIG, None, {}, {})

        assert captured, "messages.create was not called"
        msgs = captured[0].get("messages", [])
        for msg in msgs:
            content = msg.get("content") if isinstance(msg, dict) else None
            assert content is not None, (
                f"Message with role '{msg.get('role')}' has content=None; "
                "must be '' when user_input is None"
            )


# ---------------------------------------------------------------------------
# §1.10 MAX_STEPS cap
# ---------------------------------------------------------------------------


class TestMaxStepsCap:
    """TESTING.md §1.10: The tool loop must break with an error after MAX_STEPS (5) iterations."""

    def _tool_use_response(self, id: str = "tu1") -> MagicMock:
        return _anthropic_response(
            content=[_tool_use_block("myTool", id=id, input={})],
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
        )

    async def test_invoke_throws_after_max_steps(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        tool_resp = self._tool_use_response()
        mock_anthropic.messages.create = AsyncMock(return_value=tool_resp)

        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_claude_messages_handler()
        with pytest.raises(RuntimeError, match="maximum number of steps"):
            await h(cfg, "q", {"myTool": lambda _: "result"})

    async def test_invoke_succeeds_at_exactly_max_steps(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        tool_resp = self._tool_use_response()
        final_resp = _anthropic_response([_text_block("Done")])
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                tool_resp,
                tool_resp,
                tool_resp,
                tool_resp,
                tool_resp,
                tool_resp,
                tool_resp,
                tool_resp,
                tool_resp,
                tool_resp,
                final_resp,
            ]
        )

        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_claude_messages_handler()
        result = await h(cfg, "q", {"myTool": lambda _: "result"})
        assert result["output"] == "Done"

    async def test_stream_throws_after_max_steps(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        def _make_stream_cm(tool_name: str = "myTool") -> Any:
            @asynccontextmanager
            async def _ctx() -> AsyncGenerator:
                async def _iter() -> AsyncGenerator:
                    yield MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(type="text_delta", text=""),
                    )

                mock_s = MagicMock()
                mock_s.__aiter__ = lambda _: _iter().__aiter__()
                final = _anthropic_response(
                    content=[_tool_use_block(tool_name, id="tu1", input={})],
                    stop_reason="tool_use",
                )
                mock_s.get_final_message = AsyncMock(return_value=final)
                yield mock_s

            return _ctx()

        mock_anthropic.messages.stream = MagicMock(
            side_effect=lambda **_: _make_stream_cm()
        )

        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_claude_messages_handler()
        with pytest.raises(RuntimeError, match="maximum number of steps"):
            async for _ in await h.stream(cfg, "q", {"myTool": lambda _: "result"}):
                pass


# ---------------------------------------------------------------------------
# §1.5 Streaming telemetry (Appendix A.5 — do not patch _HAS_OTEL=False)
# ---------------------------------------------------------------------------


class TestStreamingTelemetry:
    def _patch_stream(
        self,
        mock_anthropic: MagicMock,
        chunks: list[str],
        input_tok: int = 5,
        output_tok: int = 3,
    ) -> None:
        ctx, _ = _make_stream_context(chunks, input_tok, output_tok)
        mock_anthropic.messages.stream = MagicMock(return_value=ctx)

    async def test_span_started_during_stream(self, mock_anthropic: MagicMock) -> None:
        mock_span = MagicMock()
        mock_trace_mod, mock_tracer = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["hi"])
        with patch.object(handler_mod, "trace", mock_trace_mod):
            h = create_claude_messages_handler()
            async for _ in await h.stream(CONFIG, "q"):
                pass
        mock_tracer.start_span.assert_called_with("claude.messages.stream")

    async def test_ld_span_attributes_set_during_stream(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        self._patch_stream(mock_anthropic, ["hi"])
        with patch.object(handler_mod, "trace", mock_trace_mod):
            h = create_claude_messages_handler()
            async for _ in await h.stream(CONFIG, "q", None, variables):
                pass
        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("launchdarkly.operation.type") == "gen_ai"
        assert attrs.get("launchdarkly.config.key") == "k"
        assert attrs.get("launchdarkly.variation.key") == "v"
        assert attrs.get("launchdarkly.run.id") == "r"

    async def test_span_ended_after_stream_completes(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_span = MagicMock()
        mock_trace_mod, _ = _make_tracer_patch(mock_span)
        import launchdarkly_ai_claude_messages.handler as handler_mod
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        self._patch_stream(mock_anthropic, ["hi"])
        with patch.object(handler_mod, "trace", mock_trace_mod):
            h = create_claude_messages_handler()
            async for _ in await h.stream(CONFIG, "q"):
                pass
        mock_span.end.assert_called()


# ---------------------------------------------------------------------------
# History parameter
# ---------------------------------------------------------------------------


class TestHistory:
    SAMPLE_HISTORY: ClassVar[list[dict[str, Any]]] = [
        {"role": "user", "content": "What is feature flagging?"},
        {"role": "assistant", "content": "Feature flagging is a technique..."},
    ]

    async def test_history_inserted_between_config_messages_and_user_input(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        config = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
        }
        h = create_claude_messages_handler()
        await h(config, "Third", {}, {}, self.SAMPLE_HISTORY)
        msgs = mock_anthropic.messages.create.call_args.kwargs["messages"]
        assert msgs[0]["content"] == "First"
        assert msgs[1]["content"] == "Second"
        assert msgs[2]["content"] == "What is feature flagging?"
        assert msgs[3]["content"] == "Feature flagging is a technique..."
        assert msgs[-1]["content"] == "Third"

    async def test_history_with_instructions_path(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        await h(CONFIG, "my question", {}, {}, self.SAMPLE_HISTORY)
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "Be helpful."
        msgs = call_kwargs["messages"]
        assert msgs[0]["content"] == "What is feature flagging?"
        assert msgs[1]["content"] == "Feature flagging is a technique..."
        assert msgs[-1]["content"] == "my question"

    async def test_empty_history_treated_like_no_history(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        h = create_claude_messages_handler()
        await h(CONFIG, "hi", {}, {}, [])
        msgs_with_empty = mock_anthropic.messages.create.call_args.kwargs["messages"]

        mock_anthropic.messages.create.reset_mock()
        h2 = create_claude_messages_handler()
        await h2(CONFIG, "hi", {}, {})
        msgs_without = mock_anthropic.messages.create.call_args.kwargs["messages"]

        assert msgs_with_empty == msgs_without

    async def test_system_role_in_history_filtered_out(
        self, mock_anthropic: MagicMock
    ) -> None:
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        history_with_system = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are evil"},
            {"role": "assistant", "content": "Hi there"},
        ]
        h = create_claude_messages_handler()
        await h(CONFIG, "q", {}, {}, history_with_system)
        msgs = mock_anthropic.messages.create.call_args.kwargs["messages"]
        roles = [m["role"] for m in msgs]
        assert "system" not in roles
