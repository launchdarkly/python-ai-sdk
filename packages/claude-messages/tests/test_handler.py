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


class _Usage:
    """Anthropic's usage object, with only the fields Anthropic actually sets.

    A bare MagicMock would answer every cache attribute with a mock, which is not what a real
    response looks like and would let a handler read a cache field that was never reported.
    """

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int | None = None,
        cache_creation: int | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if cache_read is not None:
            self.cache_read_input_tokens = cache_read
        if cache_creation is not None:
            self.cache_creation_input_tokens = cache_creation


def _anthropic_response(
    content: list[Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int | None = None,
    cache_creation: int | None = None,
) -> MagicMock:
    r = MagicMock()
    r.content = content
    r.stop_reason = stop_reason
    r.usage = _Usage(input_tokens, output_tokens, cache_read, cache_creation)
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
# Span recording
# ---------------------------------------------------------------------------


class RecordedSpan:
    """A span that remembers what a handler did to it, so a test can assert on the whole thing."""

    def __init__(self, name: str, context: Any = None) -> None:
        self.name = name
        self.context = context
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.statuses: list[Any] = []
        self.exceptions: list[BaseException] = []
        self.ended = 0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, code: Any, description: str | None = None) -> None:
        self.statuses.append(code)

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended += 1


class SpanRecorder:
    """Stands in for the ``trace`` module inside ``spans.py`` and records every span opened.

    Replaces the old single-MagicMock approach, which could not see a span tree at all: every span
    was the same object, so a parent and its children were indistinguishable.
    """

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

    @property
    def root(self) -> RecordedSpan:
        return self.spans[0]

    def named(self, prefix: str) -> list[RecordedSpan]:
        return [s for s in self.spans if s.name.startswith(prefix)]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.spans]


def _recording() -> Any:
    """Patches the tracer that ``spans.py`` holds, and yields the recorder."""
    import launchdarkly_ai_claude_messages.spans as spans_mod

    recorder = SpanRecorder()
    return patch.object(spans_mod, "trace", recorder), recorder


def _make_tracer_patch(mock_span: MagicMock) -> Any:
    """Kept for the tests that only need to know a span was opened."""
    mock_tracer = MagicMock()
    mock_tracer.start_span = MagicMock(return_value=mock_span)
    mock_trace_mod = MagicMock()
    mock_trace_mod.get_tracer = MagicMock(return_value=mock_tracer)
    return mock_trace_mod, mock_tracer


class TestSpanTree:
    """TELEMETRY-CONTRACT.md section 1."""

    async def test_opens_a_root_span_named_invoke_agent(
        self, mock_anthropic: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert rec.root.name == "invoke_agent"
        assert rec.root.attributes["gen_ai.operation.name"] == "invoke_agent"

    async def test_emits_one_chat_child_per_model_turn(
        self, mock_anthropic: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        chats = rec.named("chat ")
        assert len(chats) == 1
        assert chats[0].name == "chat claude-3-sonnet-20240229"
        assert chats[0].attributes["gen_ai.operation.name"] == "chat"
        # Parented to the root, not to nothing.
        assert chats[0].context == ("context-of", rec.root)

    async def test_names_the_chat_span_after_the_model(
        self, mock_anthropic: MagicMock
    ) -> None:
        # The semantic conventions name an inference span `{operation} {model}`. A bare `chat` tells
        # a reader nothing about which model ran.
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        cfg = {**CONFIG, "model": {"name": "claude-opus-4"}}
        with ctx:
            await create_claude_messages_handler()(cfg, "q", {}, {})
        assert "chat claude-opus-4" in rec.names

    async def test_emits_a_chat_span_per_turn_of_a_tool_loop(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1", {"a": 1})], stop_reason="tool_use"
                ),
                _anthropic_response([_text_block("done")]),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "result"}, {}
            )
        assert len(rec.named("chat ")) == 2

    async def test_emits_an_execute_tool_span_per_tool_call(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1", {"a": 1})], stop_reason="tool_use"
                ),
                _anthropic_response([_text_block("done")]),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "result"}, {}
            )
        tools = rec.named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].name == "execute_tool myTool"
        assert tools[0].attributes["gen_ai.operation.name"] == "execute_tool"
        assert tools[0].attributes["gen_ai.tool.name"] == "myTool"
        assert tools[0].attributes["gen_ai.tool.call.id"] == "tu1"

    async def test_tool_spans_are_siblings_of_chat_not_children(
        self, mock_anthropic: MagicMock
    ) -> None:
        # Both take the root's context. See TELEMETRY-CONTRACT.md section 1.
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1")], stop_reason="tool_use"
                ),
                _anthropic_response([_text_block("done")]),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.named("execute_tool ")[0].context == ("context-of", rec.root)

    async def test_every_span_is_ended(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1")], stop_reason="tool_use"
                ),
                _anthropic_response([_text_block("done")]),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)


class TestRootSpanAttributes:
    """TELEMETRY-CONTRACT.md sections 2 and 2a."""

    async def test_writes_both_provider_keys_and_the_requested_model(
        self, mock_anthropic: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        attrs = rec.root.attributes
        assert attrs["gen_ai.system"] == "anthropic"
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.request.model"] == "claude-3-sonnet-20240229"

    async def test_response_model_is_the_requested_name(
        self, mock_anthropic: MagicMock
    ) -> None:
        # Anthropic does not resolve an alias to a different snapshot, so there is no other value
        # to report. Section 2a.
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert (
            rec.root.attributes["gen_ai.response.model"] == "claude-3-sonnet-20240229"
        )

    async def test_carries_the_launchdarkly_attributes_and_feature_flag_event(
        self, mock_anthropic: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        variables = {
            "__ld": {
                "configKey": "k",
                "variationKey": "v",
                "runId": "r",
                "graphKey": "g",
            }
        }
        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, variables)
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"
        assert attrs["launchdarkly.graph.key"] == "g"
        assert [n for n, _ in rec.root.events] == ["feature_flag"]

    async def test_child_spans_carry_no_launchdarkly_identity(
        self, mock_anthropic: MagicMock
    ) -> None:
        # The root is the only span a config-scoped query finds; children must not duplicate it.
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1")], stop_reason="tool_use"
                ),
                _anthropic_response([_text_block("done")]),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, variables
            )
        for child in rec.spans[1:]:
            assert not [k for k in child.attributes if k.startswith("launchdarkly.")]
            assert "feature_flag" not in [n for n, _ in child.events]

    async def test_carries_the_run_total_not_one_turn(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1")],
                    stop_reason="tool_use",
                    input_tokens=10,
                    output_tokens=1,
                ),
                _anthropic_response(
                    [_text_block("done")], input_tokens=20, output_tokens=2
                ),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 30
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 3
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 33


class TestChatSpanAttributes:
    """TELEMETRY-CONTRACT.md sections 3, 5 and 8."""

    async def test_writes_all_seven_usage_attributes_including_zeros(
        self, mock_anthropic: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.usage.total_tokens"] == 15
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 0
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0
        assert attrs["gen_ai.usage.prompt_tokens"] == 10
        assert attrs["gen_ai.usage.completion_tokens"] == 5

    async def test_folds_cache_tokens_into_the_input_total(
        self, mock_anthropic: MagicMock
    ) -> None:
        # Anthropic reports cache beside input, so the real input is the sum of all three. This is
        # the assertion that catches a fold in the wrong direction.
        mock_anthropic.messages.create = AsyncMock(
            return_value=_anthropic_response(
                [_text_block("hi")],
                input_tokens=3,
                output_tokens=10,
                cache_read=19971,
                cache_creation=3580,
            )
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 23554
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 19971
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 3580
        assert attrs["gen_ai.usage.total_tokens"] == 23564

    async def test_yields_raw_usage_so_parse_usage_folds_exactly_once(
        self, mock_anthropic: MagicMock
    ) -> None:
        # The returned bag keeps the cache fields unfolded. Returning a pre-folded input alongside
        # them would count the cache twice downstream.
        mock_anthropic.messages.create = AsyncMock(
            return_value=_anthropic_response(
                [_text_block("hi")],
                input_tokens=3,
                output_tokens=1,
                cache_read=100,
                cache_creation=50,
            )
        )
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        result = await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert result["usage"]["input_tokens"] == 3
        assert result["usage"]["cache_read_input_tokens"] == 100
        assert result["usage"]["cache_creation_input_tokens"] == 50

    async def test_reports_the_mapped_finish_reason(
        self, mock_anthropic: MagicMock
    ) -> None:
        # Anthropic's `end_turn` is semconv's `stop`.
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "stop"
        ]

    async def test_maps_tool_use_to_tool_calls(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1")], stop_reason="tool_use"
                ),
                _anthropic_response([_text_block("done")]),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        first = rec.named("chat ")[0]
        assert first.attributes["gen_ai.response.finish_reasons"] == ["tool_calls"]

    async def test_omits_the_finish_reason_when_the_provider_gives_none(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_anthropic.messages.create = AsyncMock(
            return_value=_anthropic_response([_text_block("hi")], stop_reason=None)
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert "gen_ai.response.finish_reasons" not in rec.named("chat ")[0].attributes

    async def test_sets_status_ok_on_a_successful_turn(
        self, mock_anthropic: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert StatusCode.OK in rec.named("chat ")[0].statuses


class TestContentCapture:
    """TELEMETRY-CONTRACT.md section 7."""

    async def test_emits_no_content_at_all_by_default(
        self, mock_anthropic: MagicMock
    ) -> None:
        # Conversation content is PII. This is the assertion worth pinning hardest.
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        for span in rec.spans:
            content_keys = [
                k
                for k in span.attributes
                if k.startswith(("gen_ai.prompt", "gen_ai.completion"))
                or k
                in (
                    "gen_ai.input.messages",
                    "gen_ai.output.messages",
                    "gen_ai.system_instructions",
                    "gen_ai.tool.definitions",
                )
            ]
            assert content_keys == []
            assert [n for n, _ in span.events if n.startswith("gen_ai.content")] == []

    async def test_puts_prompt_and_completion_on_spans_when_enabled(
        self, mock_anthropic: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler(capture_content=True)(
                CONFIG, "q", {}, {}
            )
        chat = rec.named("chat ")[0]
        assert chat.attributes["gen_ai.prompt.0.role"] == "system"
        assert chat.attributes["gen_ai.prompt.0.content"] == "Be helpful."
        assert "gen_ai.input.messages" in chat.attributes
        assert chat.attributes["gen_ai.completion.0.content"] == "Hello World"
        assert "gen_ai.output.messages" in chat.attributes

    async def test_records_the_tool_catalog_on_the_chat_span_when_enabled(
        self, mock_anthropic: MagicMock
    ) -> None:
        import json

        cfg = {
            **CONFIG,
            "tools": {"myTool": {"description": "d", "parameters": {"type": "object"}}},
        }
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler(capture_content=True)(
                cfg, "q", {"myTool": lambda _: "r"}, {}
            )
        definitions = json.loads(
            rec.named("chat ")[0].attributes["gen_ai.tool.definitions"]
        )
        assert definitions[0]["name"] == "myTool"
        assert definitions[0]["type"] == "function"

    async def test_records_tool_arguments_and_results_when_enabled(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1", {"city": "NYC"})],
                    stop_reason="tool_use",
                ),
                _anthropic_response([_text_block("done")]),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler(capture_content=True)(
                CONFIG, "q", {"myTool": lambda _: "72F"}, {}
            )
        tool = rec.named("execute_tool ")[0]
        assert tool.attributes["gen_ai.tool.call.arguments"] == '{"city": "NYC"}'
        assert tool.attributes["gen_ai.tool.call.result"] == "72F"

    async def test_still_writes_the_legacy_content_events_when_enabled(
        self, mock_anthropic: MagicMock
    ) -> None:
        # Redundant and deprecated, but every published version emitted them.
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            await create_claude_messages_handler(capture_content=True)(
                CONFIG, "q", {}, {}
            )
        names = [n for n, _ in rec.named("chat ")[0].events]
        assert "gen_ai.content.prompt" in names
        assert "gen_ai.content.completion" in names


# ---------------------------------------------------------------------------
# §1.6 Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """TELEMETRY-CONTRACT.md section 6."""

    async def test_fails_the_chat_span_when_the_provider_call_raises(
        self, mock_anthropic: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        mock_anthropic.messages.create = AsyncMock(
            side_effect=RuntimeError("api error")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert len(chat.exceptions) == 1
        assert StatusCode.ERROR in chat.statuses
        assert chat.ended == 1

    async def test_fails_the_root_span_too(self, mock_anthropic: MagicMock) -> None:
        from opentelemetry.trace import StatusCode

        mock_anthropic.messages.create = AsyncMock(
            side_effect=RuntimeError("api error")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert len(rec.root.exceptions) == 1
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_fails_the_execute_tool_span_when_a_tool_raises(
        self, mock_anthropic: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        mock_anthropic.messages.create = AsyncMock(
            return_value=_anthropic_response(
                [_tool_use_block("myTool", "tu1")], stop_reason="tool_use"
            )
        )

        def _boom(_: Any) -> Any:
            raise RuntimeError("tool exploded")

        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx, pytest.raises(RuntimeError, match="tool exploded"):
            await create_claude_messages_handler()(CONFIG, "q", {"myTool": _boom}, {})
        tool = rec.named("execute_tool ")[0]
        assert len(tool.exceptions) == 1
        assert StatusCode.ERROR in tool.statuses
        assert tool.ended == 1

    async def test_reports_the_spend_of_completed_turns_on_a_failed_run(
        self, mock_anthropic: MagicMock
    ) -> None:
        # The first turn was billed. The root is the only span a config-scoped cost query finds it on.
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _anthropic_response(
                    [_tool_use_block("myTool", "tu1")],
                    stop_reason="tool_use",
                    input_tokens=40,
                    output_tokens=7,
                ),
                RuntimeError("second turn died"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_claude_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 40
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 7

    async def test_writes_no_usage_when_no_turn_ever_reported_any(
        self, mock_anthropic: MagicMock
    ) -> None:
        # All-zero attributes would assert the run cost nothing, which a run whose first call died
        # mid-flight cannot claim. An absent attribute correctly says "unknown".
        mock_anthropic.messages.create = AsyncMock(
            side_effect=RuntimeError("died on the first call")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_claude_messages_handler()(CONFIG, "q", {}, {})
        assert "gen_ai.usage.input_tokens" not in rec.root.attributes

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
    final_msg.usage = _Usage(input_tok, output_tok)
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
    """TELEMETRY-CONTRACT.md sections 1 and 6. The streaming path emits the same tree."""

    def _patch_stream(
        self,
        mock_anthropic: MagicMock,
        chunks: list[str],
        input_tok: int = 5,
        output_tok: int = 3,
    ) -> None:
        ctx, _ = _make_stream_context(chunks, input_tok, output_tok)
        mock_anthropic.messages.stream = MagicMock(return_value=ctx)

    async def test_opens_the_same_root_span_name_as_the_blocking_path(
        self, mock_anthropic: MagicMock
    ) -> None:
        # A consumer must not be able to tell from the trace which path ran.
        self._patch_stream(mock_anthropic, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            async for _ in await create_claude_messages_handler().stream(CONFIG, "q"):
                pass
        assert rec.root.name == "invoke_agent"
        assert "chat claude-3-sonnet-20240229" in rec.names

    async def test_carries_the_launchdarkly_attributes_on_the_root(
        self, mock_anthropic: MagicMock
    ) -> None:
        self._patch_stream(mock_anthropic, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            async for _ in await create_claude_messages_handler().stream(
                CONFIG, "q", None, variables
            ):
                pass
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"

    async def test_ends_every_span_once_when_the_stream_completes(
        self, mock_anthropic: MagicMock
    ) -> None:
        self._patch_stream(mock_anthropic, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            async for _ in await create_claude_messages_handler().stream(CONFIG, "q"):
                pass
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)
        assert "launchdarkly.stream.abandoned" not in rec.root.attributes

    async def test_writes_the_run_total_to_the_root(
        self, mock_anthropic: MagicMock
    ) -> None:
        self._patch_stream(mock_anthropic, ["hi"], input_tok=11, output_tok=4)
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            async for _ in await create_claude_messages_handler().stream(CONFIG, "q"):
                pass
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 11
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 15

    async def test_an_abandoned_stream_still_ends_and_exports_every_span(
        self, mock_anthropic: MagicMock
    ) -> None:
        # A consumer that breaks out mid-stream makes the generator run `finally` without ever
        # entering `except`: GeneratorExit is a BaseException. Without the cleanup there the root is
        # never ended, so it is never exported, and the whole run vanishes from AI Config Monitoring
        # along with the feature_flag event it carries.
        self._patch_stream(mock_anthropic, ["one", "two", "three"])
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            gen = await create_claude_messages_handler().stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.ended == 1
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)

    async def test_an_abandoned_stream_is_marked_but_not_failed(
        self, mock_anthropic: MagicMock
    ) -> None:
        # Stopping early is normal, and LaunchDarkly's own metrics record neither a success nor an
        # error for it, so ERROR here would put two dashboards in disagreement about one run.
        from opentelemetry.trace import StatusCode

        self._patch_stream(mock_anthropic, ["one", "two", "three"])
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            gen = await create_claude_messages_handler().stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.attributes["launchdarkly.stream.abandoned"] is True
        assert StatusCode.ERROR not in rec.root.statuses
        assert rec.root.exceptions == []

    async def test_fails_the_spans_when_the_stream_raises(
        self, mock_anthropic: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        mock_anthropic.messages.stream = MagicMock(
            side_effect=RuntimeError("stream died")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx, pytest.raises(RuntimeError, match="stream died"):
            async for _ in await create_claude_messages_handler().stream(CONFIG, "q"):
                pass
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_emits_no_content_by_default_on_the_streaming_path(
        self, mock_anthropic: MagicMock
    ) -> None:
        self._patch_stream(mock_anthropic, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_claude_messages import create_claude_messages_handler

        with ctx:
            async for _ in await create_claude_messages_handler().stream(CONFIG, "q"):
                pass
        for span in rec.spans:
            assert [k for k in span.attributes if k.startswith("gen_ai.prompt")] == []
            assert [n for n, _ in span.events if n.startswith("gen_ai.content")] == []


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


class TestConvenienceWrapperForwardsCaptureContent:
    """`capture_content` must reach the handler, not fall through into `config()`.

    `config()` takes no such argument, so leaving it in kwargs raised TypeError: a caller asking for
    content on spans got an exception instead. Five of the six wrappers had this.
    """

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        import launchdarkly_ai_claude_messages.handler as handler_mod

        seen: dict[str, Any] = {}

        def _factory(*args: Any, capture_content: bool = False, **kw: Any) -> Any:
            seen["capture_content"] = capture_content
            return MagicMock()

        fake_config = MagicMock()
        fake_config.return_value.invoke = MagicMock(return_value="ok")
        with (
            patch.object(handler_mod, "create_claude_messages_handler", _factory),
            patch.object(handler_mod, "config", fake_config),
        ):
            handler_mod.claude_messages("k", "q", {}, **kwargs)
        seen["config_kwargs"] = fake_config.call_args.kwargs
        return seen

    def test_capture_content_reaches_the_factory(self) -> None:
        seen = self._run(capture_content=True)
        assert seen["capture_content"] is True
        # And it must not have been forwarded to config(), which does not accept it.
        assert "capture_content" not in seen["config_kwargs"]

    def test_defaults_to_off(self) -> None:
        assert self._run()["capture_content"] is False
