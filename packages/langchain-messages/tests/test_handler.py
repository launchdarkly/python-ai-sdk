"""
Tests for launchdarkly-ai-langchain-messages handler.
Covers §1.1-1.10 (generic handler tests) plus TELEMETRY-CONTRACT.md sections 1-9.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake LangChain message helpers
#
# Deliberately plain objects rather than MagicMock: MagicMock answers every attribute access with
# a fresh Mock rather than raising AttributeError, so `lang_chain_finish_reasons`'s `_get(obj, key)`
# (a `getattr(obj, key, None)`) never falls through to its default, and a finish reason silently
# stops being derivable from the mock message.
# ---------------------------------------------------------------------------


class FakeAIMessage:
    def __init__(
        self,
        content: str = "Hello",
        tool_calls: list[dict[str, Any]] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
        cache_read: int | None = None,
        cache_creation: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        usage: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if cache_read is not None or cache_creation is not None:
            details: dict[str, Any] = {}
            if cache_read is not None:
                details["cache_read"] = cache_read
            if cache_creation is not None:
                details["cache_creation"] = cache_creation
            usage["input_token_details"] = details
        self.usage_metadata = usage
        self.response_metadata = (
            {"finish_reason": finish_reason} if finish_reason else {}
        )

    def _get_type(self) -> str:
        return "ai"


def _make_llm(response_content: str = "Hello") -> MagicMock:
    """Creates a mock LangChain LLM."""
    llm = MagicMock()
    ai_msg = FakeAIMessage(response_content)
    llm.ainvoke = AsyncMock(return_value=ai_msg)
    llm.bind_tools = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.bind = MagicMock(return_value=llm)

    async def _astream(msgs: Any) -> AsyncGenerator[Any, None]:
        chunk = FakeAIMessage(response_content, input_tokens=5, output_tokens=3)
        yield chunk

    llm.astream = _astream
    return llm


CONFIG = {
    "model": {"name": "gpt-4o"},
    "provider": {"name": "OpenAI"},
    "instructions": "Be helpful.",
}


# ---------------------------------------------------------------------------
# Span recording, mirroring launchdarkly_ai_claude_messages' test approach.
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
    """Stands in for the ``trace`` module inside ``spans.py`` and records every span opened."""

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
    import launchdarkly_ai_langchain_messages.spans as spans_mod

    recorder = SpanRecorder()
    return patch.object(spans_mod, "trace", recorder), recorder


def _make_tracer_patch(mock_span: MagicMock) -> Any:
    mock_tracer = MagicMock()
    mock_tracer.start_span = MagicMock(return_value=mock_span)
    mock_trace_mod = MagicMock()
    mock_trace_mod.get_tracer = MagicMock(return_value=mock_tracer)
    return mock_trace_mod, mock_tracer


# ---------------------------------------------------------------------------
# §1.1 Factory function and metadata
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        h = create_langchain_messages_handler(llm=_make_llm())
        assert callable(h)

    def test_attaches_provides_for(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        h = create_langchain_messages_handler(llm=_make_llm())
        assert h.provides_for is not None

    def test_provides_for_is_the_wildcard_provider(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        h = create_langchain_messages_handler(llm=_make_llm())
        assert h.provides_for == ("*", "messages")

    def test_multiple_calls_return_independent_instances(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        h1 = create_langchain_messages_handler(llm=_make_llm())
        h2 = create_langchain_messages_handler(llm=_make_llm())
        assert h1 is not h2


# ---------------------------------------------------------------------------
# §1.2 Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    async def test_path_a_instructions(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "hi", {}, {})
        call_args = llm.ainvoke.call_args[0][0]
        assert any(
            "system" in str(m.__class__.__name__).lower() or "System" in str(type(m))
            for m in call_args
        )

    async def test_path_a_variable_substitution(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {**CONFIG, "instructions": "Hello {{name}}"}
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {"name": "Alice"})
        call_args = llm.ainvoke.call_args[0][0]
        contents = [getattr(m, "content", "") for m in call_args]
        assert any("Hello Alice" in str(c) for c in contents)

    async def test_path_a_unresolved_placeholder_preserved(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {**CONFIG, "instructions": "Hello {{missing}}"}
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {})
        call_args = llm.ainvoke.call_args[0][0]
        all_content = " ".join(str(getattr(m, "content", "")) for m in call_args)
        assert "{{missing}}" in all_content

    async def test_path_b_messages_system_extracted(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "system", "content": "Be a poet"}],
        }
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {})
        call_args = llm.ainvoke.call_args[0][0]
        all_content = " ".join(str(getattr(m, "content", "")) for m in call_args)
        assert "Be a poet" in all_content

    async def test_path_b_user_input_appended_as_final_turn(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "instructions": "be helpful",
        }
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "final-input", {}, {})
        messages_sent = llm.ainvoke.call_args_list[0][0][0]
        all_content = " ".join(str(getattr(m, "content", "")) for m in messages_sent)
        assert "final-input" in all_content

    async def test_path_c_empty_user_input_no_throw(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "", {}, {})

    async def test_path_b_variable_substitution_in_system_message(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "system", "content": "Hello {{name}}"}],
        }
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {"name": "World"})
        call_args = llm.ainvoke.call_args[0][0]
        all_content = " ".join(str(getattr(m, "content", "")) for m in call_args)
        assert "Hello World" in all_content

    async def test_path_c_both_instructions_and_messages_messages_wins(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            **CONFIG,
            "messages": [{"role": "system", "content": "from-messages"}],
        }
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {})
        call_args = llm.ainvoke.call_args[0][0]
        all_content = " ".join(str(getattr(m, "content", "")) for m in call_args)
        assert "from-messages" in all_content
        assert "Be helpful" not in all_content


# ---------------------------------------------------------------------------
# §1.3 Tool conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    async def test_all_fields_forwarded(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            **CONFIG,
            "tools": {
                "search": {
                    "name": "search",
                    "type": "function",
                    "description": "Search the web",
                    "parameters": {"type": "object"},
                }
            },
        }
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {})
        llm.bind_tools.assert_called_once()
        tools_arg = llm.bind_tools.call_args[0][0]
        assert any(t.get("function", {}).get("name") == "search" for t in tools_arg)

    async def test_multiple_tools_all_included(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            **CONFIG,
            "tools": {
                "t1": {"name": "t1", "type": "function", "parameters": {}},
                "t2": {"name": "t2", "type": "function", "parameters": {}},
                "t3": {"name": "t3", "type": "function", "parameters": {}},
            },
        }
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {})
        tools_arg = llm.bind_tools.call_args[0][0]
        assert len(tools_arg) == 3

    async def test_empty_tools_no_tools_sent(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "q", {}, {})
        llm.bind_tools.assert_not_called()


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop
# ---------------------------------------------------------------------------


class TestToolExecutionLoop:
    async def test_single_tool_call_then_done(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        tool_ai_msg = FakeAIMessage(
            tool_calls=[{"name": "search", "id": "tc1", "args": {"q": "test"}}]
        )
        final_ai_msg = FakeAIMessage("final answer")
        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=[tool_ai_msg, final_ai_msg])
        config = {
            **CONFIG,
            "tools": {
                "search": {"name": "search", "type": "function", "parameters": {}}
            },
        }
        h = create_langchain_messages_handler(llm=llm)
        fn = AsyncMock(return_value="result")
        result = await h(config, "q", {"search": fn}, {})
        assert llm.ainvoke.call_count == 2
        assert result["output"] == "final answer"

    async def test_tool_not_found_throws(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        tool_ai_msg = FakeAIMessage(
            tool_calls=[{"name": "unknown_tool", "id": "tc1", "args": {}}]
        )
        llm = _make_llm()
        llm.ainvoke = AsyncMock(return_value=tool_ai_msg)
        config = {
            **CONFIG,
            "tools": {"other": {"name": "other", "type": "function", "parameters": {}}},
        }
        h = create_langchain_messages_handler(llm=llm)
        with pytest.raises(Exception, match="No handler"):
            await h(config, "q", {}, {})

    async def test_tool_handler_throws_propagates(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        tool_ai_msg = FakeAIMessage(
            tool_calls=[{"name": "t1", "id": "tc1", "args": {}}]
        )
        llm = _make_llm()
        llm.ainvoke = AsyncMock(return_value=tool_ai_msg)
        config = {
            **CONFIG,
            "tools": {"t1": {"name": "t1", "type": "function", "parameters": {}}},
        }
        h = create_langchain_messages_handler(llm=llm)
        fn = AsyncMock(side_effect=RuntimeError("tool failed"))
        with pytest.raises(RuntimeError, match="tool failed"):
            await h(config, "q", {"t1": fn}, {})

    async def test_no_tools_in_config_handler_never_invoked(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        fn = AsyncMock()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "q", {"t1": fn}, {})
        llm.bind_tools.assert_not_called()
        fn.assert_not_called()

    async def test_multiple_consecutive_tool_calls(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        msg1 = FakeAIMessage(tool_calls=[{"name": "t1", "id": "tc1", "args": {"x": 1}}])
        msg2 = FakeAIMessage(tool_calls=[{"name": "t2", "id": "tc2", "args": {"y": 2}}])
        msg3 = FakeAIMessage("final")
        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=[msg1, msg2, msg3])
        cfg = {
            **CONFIG,
            "tools": {
                "t1": {"name": "t1", "type": "function", "parameters": {}},
                "t2": {"name": "t2", "type": "function", "parameters": {}},
            },
        }
        fn1 = AsyncMock(return_value="r1")
        fn2 = AsyncMock(return_value="r2")
        h = create_langchain_messages_handler(llm=llm)
        result = await h(cfg, "q", {"t1": fn1, "t2": fn2}, {})
        fn1.assert_called_once()
        fn2.assert_called_once()
        assert result["output"] == "final"


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md section 1: span tree
# ---------------------------------------------------------------------------


class TestSpanTree:
    async def test_opens_a_root_span_named_invoke_agent(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
        assert rec.root.name == "invoke_agent"
        assert rec.root.attributes["gen_ai.operation.name"] == "invoke_agent"

    async def test_emits_one_chat_child_per_model_turn(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
        chats = rec.named("chat ")
        assert len(chats) == 1
        assert chats[0].name == "chat gpt-4o"
        assert chats[0].attributes["gen_ai.operation.name"] == "chat"
        assert chats[0].context == ("context-of", rec.root)

    async def test_emits_a_chat_span_per_turn_of_a_tool_loop(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(tool_calls=[{"name": "myTool", "id": "tc1", "args": {}}]),
                FakeAIMessage("done"),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        with ctx:
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": lambda _: "result"}, {}
            )
        assert len(rec.named("chat ")) == 2

    async def test_emits_an_execute_tool_span_per_tool_call(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(tool_calls=[{"name": "myTool", "id": "tu1", "args": {}}]),
                FakeAIMessage("done"),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        with ctx:
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": lambda _: "result"}, {}
            )
        tools = rec.named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].name == "execute_tool myTool"
        assert tools[0].attributes["gen_ai.operation.name"] == "execute_tool"
        assert tools[0].attributes["gen_ai.tool.name"] == "myTool"
        assert tools[0].attributes["gen_ai.tool.call.id"] == "tu1"

    async def test_tool_spans_are_siblings_of_chat_not_children(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(tool_calls=[{"name": "myTool", "id": "tu1", "args": {}}]),
                FakeAIMessage("done"),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        with ctx:
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.named("execute_tool ")[0].context == ("context-of", rec.root)

    async def test_every_span_is_ended(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(tool_calls=[{"name": "myTool", "id": "tu1", "args": {}}]),
                FakeAIMessage("done"),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        with ctx:
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": lambda _: "r"}, {}
            )
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md sections 2, 2a and 9: root span / model identity
# ---------------------------------------------------------------------------


class TestRootSpanAttributes:
    async def test_gen_ai_system_is_the_literal_langchain(self) -> None:
        # TELEMETRY-CONTRACT.md section 9: Python used to set this to the configured provider,
        # lower-cased. TypeScript's LangChain handlers keep it the constant `langchain` regardless.
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        cfg = {**CONFIG, "provider": {"name": "Anthropic"}}
        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(cfg, "q", {}, {})
        assert rec.root.attributes["gen_ai.system"] == "langchain"

    async def test_gen_ai_provider_name_is_anthropic_only_for_anthropic(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        cfg = {**CONFIG, "provider": {"name": "Anthropic"}}
        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(cfg, "q", {}, {})
        assert rec.root.attributes["gen_ai.provider.name"] == "anthropic"

    @pytest.mark.parametrize(
        "provider_name", ["OpenAI", "Bedrock", "Azure", "Cohere", "Typo", ""]
    )
    async def test_gen_ai_provider_name_is_openai_for_everything_else(
        self, provider_name: str
    ) -> None:
        # Not a passthrough. Bedrock, Azure, Cohere, a typo and an unset value all report `openai`,
        # mirroring the chat model class the handler actually instantiates. Section 9.
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        cfg = {**CONFIG, "provider": {"name": provider_name}}
        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(cfg, "q", {}, {})
        assert rec.root.attributes["gen_ai.provider.name"] == "openai"

    async def test_writes_the_requested_model(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
        assert rec.root.attributes["gen_ai.request.model"] == "gpt-4o"

    async def test_response_model_is_the_requested_name(self) -> None:
        # LangChain does not resolve an alias to a different snapshot in this handler. Section 2a.
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
        assert rec.root.attributes["gen_ai.response.model"] == "gpt-4o"

    async def test_carries_the_launchdarkly_attributes_and_feature_flag_event(
        self,
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        variables = {
            "__ld": {
                "configKey": "k",
                "variationKey": "v",
                "runId": "r",
                "graphKey": "g",
            }
        }
        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, variables
            )
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"
        assert attrs["launchdarkly.graph.key"] == "g"
        assert [n for n, _ in rec.root.events] == ["feature_flag"]

    async def test_child_spans_carry_no_launchdarkly_identity(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(tool_calls=[{"name": "myTool", "id": "tu1", "args": {}}]),
                FakeAIMessage("done"),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": lambda _: "r"}, variables
            )
        for child in rec.spans[1:]:
            assert not [k for k in child.attributes if k.startswith("launchdarkly.")]
            assert "feature_flag" not in [n for n, _ in child.events]

    async def test_carries_the_run_total_not_one_turn(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(
                    tool_calls=[{"name": "myTool", "id": "tu1", "args": {}}],
                    input_tokens=10,
                    output_tokens=1,
                ),
                FakeAIMessage("done", input_tokens=20, output_tokens=2),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        with ctx:
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 30
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 3
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 33


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md sections 3, 5 and 8: chat span attributes
# ---------------------------------------------------------------------------


class TestChatSpanAttributes:
    async def test_writes_all_seven_usage_attributes_including_zeros(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.usage.total_tokens"] == 15
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 0
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0
        assert attrs["gen_ai.usage.prompt_tokens"] == 10
        assert attrs["gen_ai.usage.completion_tokens"] == 5

    async def test_passes_the_input_figure_through_without_folding_cache_into_it(
        self,
    ) -> None:
        # TELEMETRY-CONTRACT.md section 8: LangChain already counts cached tokens inside
        # `input_tokens`, unlike Anthropic. Adding the cache buckets on top here would double-count.
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            return_value=FakeAIMessage(
                input_tokens=23554,
                output_tokens=100,
                cache_read=19971,
                cache_creation=3580,
            )
        )
        with ctx:
            await create_langchain_messages_handler(llm=llm)(CONFIG, "q", {}, {})
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 23554
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 19971
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 3580

    async def test_reports_the_mapped_finish_reason(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage(finish_reason="stop"))
        with ctx:
            await create_langchain_messages_handler(llm=llm)(CONFIG, "q", {}, {})
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "stop"
        ]

    async def test_maps_the_anthropic_word_through_the_table(self) -> None:
        # LangChain can serve an Anthropic model, and this handler does use the mapping table,
        # unlike the two OpenAI handlers. Section 5.
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage(finish_reason="end_turn"))
        with ctx:
            await create_langchain_messages_handler(llm=llm)(CONFIG, "q", {}, {})
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "stop"
        ]

    async def test_omits_the_finish_reason_when_the_provider_gives_none(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
        assert "gen_ai.response.finish_reasons" not in rec.named("chat ")[0].attributes

    async def test_sets_status_ok_on_a_successful_turn(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
        assert rec.named("chat ")[0].statuses == [StatusCode.OK]


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md section 7: content capture
# ---------------------------------------------------------------------------


class TestContentCapture:
    async def test_emits_no_content_at_all_by_default(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(llm=_make_llm())(
                CONFIG, "q", {}, {}
            )
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

    async def test_puts_prompt_and_completion_on_spans_when_enabled(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(
                llm=_make_llm(), capture_content=True
            )(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert chat.attributes["gen_ai.prompt.0.role"] == "system"
        assert chat.attributes["gen_ai.prompt.0.content"] == "Be helpful."
        assert "gen_ai.input.messages" in chat.attributes
        assert chat.attributes["gen_ai.completion.0.content"] == "Hello"
        assert "gen_ai.output.messages" in chat.attributes

    async def test_records_the_tool_catalog_on_the_chat_span_when_enabled(self) -> None:
        import json

        cfg = {
            **CONFIG,
            "tools": {"myTool": {"description": "d", "parameters": {"type": "object"}}},
        }
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage("done"))
        with ctx:
            await create_langchain_messages_handler(llm=llm, capture_content=True)(
                cfg, "q", {"myTool": lambda _: "r"}, {}
            )
        definitions = json.loads(
            rec.named("chat ")[0].attributes["gen_ai.tool.definitions"]
        )
        assert definitions[0]["name"] == "myTool"
        assert definitions[0]["type"] == "function"

    async def test_records_tool_arguments_and_results_when_enabled(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(
                    tool_calls=[
                        {"name": "myTool", "id": "tu1", "args": {"city": "NYC"}}
                    ]
                ),
                FakeAIMessage("done"),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        with ctx:
            await create_langchain_messages_handler(llm=llm, capture_content=True)(
                cfg, "q", {"myTool": lambda _: "72F"}, {}
            )
        tool = rec.named("execute_tool ")[0]
        assert tool.attributes["gen_ai.tool.call.arguments"] == '{"city": "NYC"}'
        assert tool.attributes["gen_ai.tool.call.result"] == "72F"

    async def test_still_writes_the_legacy_content_events_when_enabled(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        with ctx:
            await create_langchain_messages_handler(
                llm=_make_llm(), capture_content=True
            )(CONFIG, "q", {}, {})
        names = [n for n, _ in rec.named("chat ")[0].events]
        assert "gen_ai.content.prompt" in names
        assert "gen_ai.content.completion" in names


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md section 6: errors
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_fails_the_chat_span_when_the_provider_call_raises(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("api error"))
        with ctx, pytest.raises(RuntimeError):
            await create_langchain_messages_handler(llm=llm)(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert len(chat.exceptions) == 1
        assert StatusCode.ERROR in chat.statuses
        assert chat.ended == 1

    async def test_fails_the_root_span_too(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("api error"))
        with ctx, pytest.raises(RuntimeError):
            await create_langchain_messages_handler(llm=llm)(CONFIG, "q", {}, {})
        assert len(rec.root.exceptions) == 1
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_fails_the_execute_tool_span_when_a_tool_raises(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            return_value=FakeAIMessage(
                tool_calls=[{"name": "myTool", "id": "tu1", "args": {}}]
            )
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}

        def _boom(_: Any) -> Any:
            raise RuntimeError("tool exploded")

        with ctx, pytest.raises(RuntimeError, match="tool exploded"):
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": _boom}, {}
            )
        tool = rec.named("execute_tool ")[0]
        assert len(tool.exceptions) == 1
        assert StatusCode.ERROR in tool.statuses
        assert tool.ended == 1

    async def test_reports_the_spend_of_completed_turns_on_a_failed_run(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[
                FakeAIMessage(
                    tool_calls=[{"name": "myTool", "id": "tu1", "args": {}}],
                    input_tokens=40,
                    output_tokens=7,
                ),
                RuntimeError("second turn died"),
            ]
        )
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        with ctx, pytest.raises(RuntimeError):
            await create_langchain_messages_handler(llm=llm)(
                cfg, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 40
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 7

    async def test_writes_no_usage_when_no_turn_ever_reported_any(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("died on the first call"))
        with ctx, pytest.raises(RuntimeError):
            await create_langchain_messages_handler(llm=llm)(CONFIG, "q", {}, {})
        assert "gen_ai.usage.input_tokens" not in rec.root.attributes

    async def test_rethrows_error(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("rethrown"))
        h = create_langchain_messages_handler(llm=llm)
        with pytest.raises(RuntimeError, match="rethrown"):
            await h(CONFIG, "q", {}, {})


# ---------------------------------------------------------------------------
# §1.9 Structured output — withStructuredOutput
# ---------------------------------------------------------------------------


class TestOutputFormat:
    async def test_absent_output_format_no_change(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "q", {}, {})
        llm.with_structured_output.assert_not_called()

    async def test_with_structured_output_called_when_set(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {**CONFIG, "outputFormat": {"type": "object"}}
        llm = _make_llm()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(
            return_value={"parsed": {"ok": True}, "raw": FakeAIMessage("")}
        )
        llm.with_structured_output = MagicMock(return_value=structured_llm)
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "q", {}, {})
        llm.with_structured_output.assert_called_once()

    async def test_returns_parsed_object_when_set(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {**CONFIG, "outputFormat": {"type": "object"}}
        llm = _make_llm()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(
            return_value={"parsed": {"result": 42}, "raw": FakeAIMessage("")}
        )
        llm.with_structured_output = MagicMock(return_value=structured_llm)
        h = create_langchain_messages_handler(llm=llm)
        result = await h(config, "q", {}, {})
        assert result["output"] == {"result": 42}

    async def test_with_structured_output_not_called_when_absent(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "q", {}, {})
        llm.with_structured_output.assert_not_called()

    async def test_does_not_throw_when_both_output_format_and_tools_set(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            **CONFIG,
            "outputFormat": {"type": "object"},
            "tools": {"t1": {"name": "t1", "type": "function", "parameters": {}}},
        }
        llm = _make_llm()
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage(""))
        h = create_langchain_messages_handler(llm=llm)
        result = await h(config, "q", {}, {})
        assert "output" in result

    async def test_token_usage_from_usage_metadata_when_structured_output(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {**CONFIG, "outputFormat": {"type": "object"}}
        llm = _make_llm()
        raw_msg = FakeAIMessage("", input_tokens=12, output_tokens=8)
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(
            return_value={"parsed": {"x": 1}, "raw": raw_msg}
        )
        llm.with_structured_output = MagicMock(return_value=structured_llm)
        h = create_langchain_messages_handler(llm=llm)
        result = await h(config, "q", {}, {})
        assert result["usage"]["input_tokens"] == 12
        assert result["usage"]["output_tokens"] == 8


# ---------------------------------------------------------------------------
# §1.7 Convenience export
# ---------------------------------------------------------------------------


class TestConvenienceExport:
    def test_calls_through_to_model_call(self) -> None:
        import launchdarkly_ai_langchain_messages.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_langchain_messages.handler import langchain_messages

            ctx = {"kind": "user", "key": "u1"}
            langchain_messages("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        handler = call_kwargs.get("handler")
        assert handler is not None
        assert handler.provides_for == ("*", "messages")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self) -> None:
        import launchdarkly_ai_langchain_messages.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_langchain_messages.handler import langchain_messages

            ctx = {"kind": "user", "key": "u1"}
            langchain_messages("my-flag", "hello", ctx)

        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# §1.8 Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def _make_streaming_llm(
        self, chunks: list[str], input_tok: int = 5, output_tok: int = 3
    ) -> MagicMock:
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)

        async def _astream(msgs: Any) -> AsyncGenerator[Any, None]:
            for c in chunks:
                yield FakeAIMessage(c, input_tokens=input_tok, output_tokens=output_tok)

        llm.astream = _astream
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage(""))
        return llm

    async def test_stream_defined_and_async_generator(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hi"])
        h = create_langchain_messages_handler(llm=llm)
        assert h.has_stream
        gen = await h.stream(CONFIG, "q")
        assert hasattr(gen, "__aiter__")

    async def test_yields_chunk_events(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hello ", "world"])
        h = create_langchain_messages_handler(llm=llm)
        events = [e async for e in await h.stream(CONFIG, "q")]
        chunks = [e for e in events if e.get("type") == "chunk"]
        assert len(chunks) == 2
        assert chunks[0]["text"] == "hello "
        assert chunks[1]["text"] == "world"

    async def test_yields_exactly_one_done_event(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["x"])
        h = create_langchain_messages_handler(llm=llm)
        events = [e async for e in await h.stream(CONFIG, "q")]
        assert len([e for e in events if e.get("type") == "done"]) == 1

    async def test_done_event_carries_accumulated_output(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hello ", "world"])
        h = create_langchain_messages_handler(llm=llm)
        events = [e async for e in await h.stream(CONFIG, "q")]
        done = next(e for e in events if e.get("type") == "done")
        assert done["output"] == "hello world"

    async def test_done_usage(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["text"], input_tok=7, output_tok=3)
        h = create_langchain_messages_handler(llm=llm)
        events = [e async for e in await h.stream(CONFIG, "q")]
        done = next(e for e in events if e.get("type") == "done")
        assert done["usage"]["input_tokens"] == 7
        assert done["usage"]["output_tokens"] == 3


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md sections 1 and 6: the streaming path emits the same tree.
# ---------------------------------------------------------------------------


class TestStreamingTelemetry:
    def _make_streaming_llm(self, chunks: list[str]) -> MagicMock:
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)

        async def _astream(msgs: Any) -> AsyncGenerator[Any, None]:
            for c in chunks:
                yield FakeAIMessage(c, input_tokens=5, output_tokens=3)

        llm.astream = _astream
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage(""))
        return llm

    async def test_opens_the_same_root_span_name_as_the_blocking_path(self) -> None:
        # A consumer must not be able to tell from the trace which path ran.
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hi"])
        with ctx:
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q"
            ):
                pass
        assert rec.root.name == "invoke_agent"
        assert "chat gpt-4o" in rec.names

    async def test_carries_the_launchdarkly_attributes_on_the_root(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hi"])
        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q", None, variables
            ):
                pass
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"

    async def test_ends_every_span_once_when_the_stream_completes(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hi"])
        with ctx:
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q"
            ):
                pass
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)
        assert "launchdarkly.stream.abandoned" not in rec.root.attributes

    async def test_writes_the_run_total_to_the_root(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hi"])
        with ctx:
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q"
            ):
                pass
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 5
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 8

    async def test_an_abandoned_stream_still_ends_and_exports_every_span(self) -> None:
        # A consumer that breaks out mid-stream makes the generator run `finally` without ever
        # entering `except`: GeneratorExit is a BaseException. Without the cleanup there the root is
        # never ended, so it is never exported, and the whole run vanishes from AI Config
        # Monitoring along with the feature_flag event it carries.
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["one", "two", "three"])
        with ctx:
            gen = await create_langchain_messages_handler(llm=llm).stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.ended == 1
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)

    async def test_an_abandoned_stream_is_marked_but_not_failed(self) -> None:
        # Stopping early is normal, and LaunchDarkly's own metrics record neither a success nor an
        # error for it, so ERROR here would put two dashboards in disagreement about one run.
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["one", "two", "three"])
        with ctx:
            gen = await create_langchain_messages_handler(llm=llm).stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.attributes["launchdarkly.stream.abandoned"] is True
        assert StatusCode.ERROR not in rec.root.statuses
        assert rec.root.exceptions == []

    async def test_fails_the_spans_when_the_stream_raises(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)

        async def _astream(msgs: Any) -> AsyncGenerator[Any, None]:
            raise RuntimeError("stream died")
            yield  # pragma: no cover - unreachable, keeps this an async generator

        llm.astream = _astream
        with ctx, pytest.raises(RuntimeError, match="stream died"):
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q"
            ):
                pass
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_emits_no_content_by_default_on_the_streaming_path(self) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_streaming_llm(["hi"])
        with ctx:
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q"
            ):
                pass
        for span in rec.spans:
            assert [k for k in span.attributes if k.startswith("gen_ai.prompt")] == []
            assert [n for n, _ in span.events if n.startswith("gen_ai.content")] == []


# ---------------------------------------------------------------------------
# §1.10 MAX_STEPS cap
# ---------------------------------------------------------------------------


class TestMaxStepsCap:
    def _make_tool_call_llm(self) -> MagicMock:
        tool_msg = FakeAIMessage(
            content="",
            tool_calls=[{"id": "tc_1", "name": "myTool", "args": {}}],
            input_tokens=1,
            output_tokens=1,
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=tool_msg)
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    async def test_invoke_throws_after_max_steps(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = self._make_tool_call_llm()
        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_langchain_messages_handler(llm=llm)
        with pytest.raises(RuntimeError, match="maximum number of steps"):
            await h(cfg, "q", {"myTool": lambda _: "result"})

    async def test_invoke_succeeds_at_exactly_max_steps(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        tool_msg = FakeAIMessage(
            content="",
            tool_calls=[{"id": "tc_1", "name": "myTool", "args": {}}],
            input_tokens=1,
            output_tokens=1,
        )
        final_msg = FakeAIMessage("Done", input_tokens=1, output_tokens=1)
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_msg] * 10 + [final_msg])
        llm.bind_tools = MagicMock(return_value=llm)

        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_langchain_messages_handler(llm=llm)
        result = await h(cfg, "q", {"myTool": lambda _: "result"})
        assert result["output"] == "Done"

    async def test_stream_throws_after_max_steps(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        async def _tool_chunk_stream(_msgs: Any) -> AsyncGenerator[Any, None]:
            yield FakeAIMessage(
                "",
                tool_calls=[{"id": "tc_1", "name": "myTool", "args": {}}],
                input_tokens=1,
                output_tokens=1,
            )

        llm = MagicMock()
        llm.astream = _tool_chunk_stream
        llm.bind_tools = MagicMock(return_value=llm)

        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_langchain_messages_handler(llm=llm)
        with pytest.raises(RuntimeError, match="maximum number of steps"):
            async for _ in await h.stream(cfg, "q", {"myTool": lambda _: "result"}):
                pass


# ---------------------------------------------------------------------------
# History parameter
# ---------------------------------------------------------------------------


class TestHistory:
    SAMPLE_HISTORY: ClassVar[list[dict[str, Any]]] = [
        {"role": "user", "content": "What is feature flagging?"},
        {"role": "assistant", "content": "Feature flagging is a technique..."},
    ]

    async def test_history_inserted_between_config_messages_and_user_input(
        self,
    ) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "system", "content": "base"}],
        }
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(config, "final question", {}, {}, self.SAMPLE_HISTORY)
        call_args = llm.ainvoke.call_args[0][0]
        contents = [str(getattr(m, "content", "")) for m in call_args]
        assert contents == [
            "base",
            "What is feature flagging?",
            "Feature flagging is a technique...",
            "final question",
        ]

    async def test_history_with_instructions_path(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "final question", {}, {}, self.SAMPLE_HISTORY)
        call_args = llm.ainvoke.call_args[0][0]
        contents = [str(getattr(m, "content", "")) for m in call_args]
        assert "What is feature flagging?" in contents
        assert "final question" in contents

    async def test_empty_history_treated_like_no_history(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "q", {}, {}, [])
        call_args = llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # system + user

    async def test_system_role_in_history_filtered_out(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        history_with_system = [
            *self.SAMPLE_HISTORY,
            {"role": "system", "content": "ignored"},
        ]
        llm = _make_llm()
        h = create_langchain_messages_handler(llm=llm)
        await h(CONFIG, "q", {}, {}, history_with_system)
        call_args = llm.ainvoke.call_args[0][0]
        contents = [str(getattr(m, "content", "")) for m in call_args]
        assert "ignored" not in contents


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md section 6: reported is not the same as reported zero
# ---------------------------------------------------------------------------


class TestStreamingUsageReported:
    """A stream whose chunks carried no usage must not mark the run as having reported.

    Adding an all-zero turn to the accumulator makes a later failure or abandonment write all-zero
    totals on the root, which asserts the run cost nothing. That is a different claim from "unknown",
    and it is the one thing the reported flag exists to prevent. The blocking path gets this for
    free, because `lang_chain_span_usage` returns None for a bag the provider never filled.
    """

    def _two_turn_llm(self, *, first_turn_usage: bool) -> MagicMock:
        """One turn that completes with a tool call, then a second turn that dies.

        The first turn is what puts something in the accumulator. Whether it carried usage is the
        variable under test, and a turn that dies mid-iteration never reaches the accumulator at
        all, which is why a single failing turn cannot exercise this.
        """
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        state = {"turn": 0}

        async def _astream(msgs: Any) -> AsyncGenerator[Any, None]:
            state["turn"] += 1
            if state["turn"] == 1:
                msg = FakeAIMessage(
                    "calling",
                    tool_calls=[{"name": "search", "id": "c1", "args": {}}],
                    input_tokens=12,
                    output_tokens=3,
                )
                if not first_turn_usage:
                    msg.usage_metadata = None
                yield msg
                return
            raise RuntimeError("second turn died")

        llm.astream = _astream
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage(""))
        return llm

    async def test_a_completed_turn_with_no_usage_does_not_mark_reported(self) -> None:
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        ctx, rec = _recording()
        llm = self._two_turn_llm(first_turn_usage=False)
        with ctx, pytest.raises(RuntimeError):
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q", {"search": lambda _: "ok"}
            ):
                pass
        # Zeros here would assert the run cost nothing. It is unknown, so nothing is written.
        assert "gen_ai.usage.input_tokens" not in rec.root.attributes

    async def test_a_completed_turn_with_usage_still_reaches_the_root(self) -> None:
        # The guard must not cost the run its real numbers.
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        ctx, rec = _recording()
        llm = self._two_turn_llm(first_turn_usage=True)
        with ctx, pytest.raises(RuntimeError):
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q", {"search": lambda _: "ok"}
            ):
                pass
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 12


class TestConvenienceWrapperForwardsCaptureContent:
    """`capture_content` must reach the handler, not fall through into `config()`.

    `config()` takes no such argument, so leaving it in kwargs raised TypeError: a caller asking for
    content on spans got an exception instead. Five of the six wrappers had this.
    """

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        import launchdarkly_ai_langchain_messages.handler as handler_mod

        seen: dict[str, Any] = {}

        def _factory(*args: Any, capture_content: bool = False, **kw: Any) -> Any:
            seen["capture_content"] = capture_content
            return MagicMock()

        fake_config = MagicMock()
        fake_config.return_value.invoke = MagicMock(return_value="ok")
        with (
            patch.object(handler_mod, "create_langchain_messages_handler", _factory),
            patch.object(handler_mod, "config", fake_config),
        ):
            handler_mod.langchain_messages("k", "q", {}, **kwargs)
        seen["config_kwargs"] = fake_config.call_args.kwargs
        return seen

    def test_capture_content_reaches_the_factory(self) -> None:
        seen = self._run(capture_content=True)
        assert seen["capture_content"] is True
        # And it must not have been forwarded to config(), which does not accept it.
        assert "capture_content" not in seen["config_kwargs"]

    def test_defaults_to_off(self) -> None:
        assert self._run()["capture_content"] is False


class TestChatSpanAndTeardownNeverLeak:
    """Two ways the run could vanish from the trace, both reachable through content serialisation."""

    @pytest.mark.asyncio
    async def test_an_unserialisable_completion_still_ends_the_chat_span(self) -> None:
        # The blocking path has no `finally`, so a raise outside the guard leaves the span open with
        # nothing able to recover it.
        from opentelemetry.trace import StatusCode

        class _Exploding:
            """Shaped like an AIMessage, but reading its content raises."""

            usage_metadata: ClassVar[dict[str, Any]] = {
                "input_tokens": 5,
                "output_tokens": 1,
            }
            tool_calls: ClassVar[list[Any]] = []
            response_metadata: ClassVar[dict[str, Any]] = {}

            @property
            def content(self) -> Any:
                raise TypeError("cannot serialise this content")

        from launchdarkly_ai_langchain_messages import (
            create_langchain_messages_handler,
        )

        ctx, rec = _recording()
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.ainvoke = AsyncMock(return_value=_Exploding())
        with ctx, pytest.raises(TypeError):
            await create_langchain_messages_handler(llm=llm, capture_content=True)(
                CONFIG, "q", {}, {}
            )
        chat = rec.named("chat ")
        assert len(chat) == 1
        assert chat[0].ended == 1, "the chat span leaked"
        assert StatusCode.ERROR in chat[0].statuses

    @pytest.mark.asyncio
    async def test_an_aclose_failure_does_not_cost_the_run_its_root_span(self) -> None:
        # aclose() runs after span teardown, so its own failure cannot take the trace with it.
        from launchdarkly_ai_langchain_messages import (
            create_langchain_messages_handler,
        )

        ctx, rec = _recording()
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)

        class _BadStream:
            def __aiter__(self) -> Any:
                return self

            async def __anext__(self) -> Any:
                return FakeAIMessage("chunk")

            async def aclose(self) -> None:
                raise RuntimeError("vendor teardown exploded")

        llm.astream = MagicMock(return_value=_BadStream())
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage(""))
        with ctx:
            gen = await create_langchain_messages_handler(llm=llm).stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.ended == 1, (
            "the root span was lost to a vendor teardown failure"
        )
        assert rec.root.attributes["launchdarkly.stream.abandoned"] is True


class TestOpenToolSpanIsNeverLeaked:
    """A BaseException while a tool runs must still close the execute_tool span.

    The streaming `finally` closed the model span and the root, but the in-flight tool span was held
    only by a local. `except Exception` does not see a `CancelledError` or a `GeneratorExit`, so a
    tool cancelled mid-flight left its span open and unexported: the trace showed a closed parent
    above a child that never arrived.
    """

    @pytest.mark.asyncio
    async def test_a_tool_cancelled_mid_flight_still_ends_its_span(self) -> None:
        import asyncio

        from launchdarkly_ai_langchain_messages import (
            create_langchain_messages_handler,
        )

        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.ainvoke = AsyncMock(return_value=FakeAIMessage("done"))

        async def _astream(msgs: Any) -> AsyncGenerator[Any, None]:
            yield FakeAIMessage(
                "",
                tool_calls=[{"name": "myTool", "id": "tu1", "args": {"a": 1}}],
                input_tokens=5,
                output_tokens=3,
            )

        llm.astream = _astream

        async def _cancelled_tool(_: Any) -> Any:
            # A BaseException, so `except Exception` in the tool loop does not see it.
            raise asyncio.CancelledError()

        ctx, rec = _recording()
        with ctx, pytest.raises(asyncio.CancelledError):
            async for _ in await create_langchain_messages_handler(llm=llm).stream(
                CONFIG, "q", {"myTool": _cancelled_tool}
            ):
                pass

        tools = rec.named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].ended == 1, "the execute_tool span leaked"
        assert tools[0].attributes["launchdarkly.stream.abandoned"] is True
        assert rec.root.ended == 1


class TestStructuredTurnChatSpanNeverLeaks:
    """The structured-output turn has no `finally`, so a raise outside its guard is unrecoverable.

    The output content write and the span finish sat outside the try that fails the chat span. A raise
    while serialising the parsed object left the span open and unexported, and dropped the turn from
    the run total even though the provider had already billed it.
    """

    @pytest.mark.asyncio
    async def test_an_unserialisable_parsed_object_still_ends_the_chat_span(
        self,
    ) -> None:
        from opentelemetry.trace import StatusCode

        from launchdarkly_ai_langchain_messages import (
            create_langchain_messages_handler,
        )

        class _Unserialisable:
            """json.dumps refuses this, which is how a real caller trips the same wire."""

        config = {**CONFIG, "outputFormat": {"type": "object"}}
        llm = _make_llm()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(
            return_value={
                "parsed": _Unserialisable(),
                "raw": FakeAIMessage("", input_tokens=17, output_tokens=4),
            }
        )
        llm.with_structured_output = MagicMock(return_value=structured_llm)

        ctx, rec = _recording()
        with ctx, pytest.raises(TypeError):
            await create_langchain_messages_handler(llm=llm, capture_content=True)(
                config, "q", {}, {}
            )

        chat = rec.named("chat ")
        assert len(chat) == 1
        assert chat[0].ended == 1, "the chat span leaked"
        assert StatusCode.ERROR in chat[0].statuses

    @pytest.mark.asyncio
    async def test_a_content_failure_does_not_lose_the_tokens_already_billed(
        self,
    ) -> None:
        from launchdarkly_ai_langchain_messages import (
            create_langchain_messages_handler,
        )

        class _Unserialisable:
            pass

        config = {**CONFIG, "outputFormat": {"type": "object"}}
        llm = _make_llm()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(
            return_value={
                "parsed": _Unserialisable(),
                "raw": FakeAIMessage("", input_tokens=17, output_tokens=4),
            }
        )
        llm.with_structured_output = MagicMock(return_value=structured_llm)

        ctx, rec = _recording()
        with ctx, pytest.raises(TypeError):
            await create_langchain_messages_handler(llm=llm, capture_content=True)(
                config, "q", {}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 17
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 4


class TestToolTurnKeepsBilledTokens:
    """The tool loop must report the tokens it spent even when serialising its content fails."""

    @pytest.mark.asyncio
    async def test_a_content_failure_does_not_lose_the_tokens_already_billed(
        self,
    ) -> None:
        from launchdarkly_ai_langchain_messages import (
            create_langchain_messages_handler,
        )

        class _Exploding:
            """Shaped like an AIMessage, with usage readable and content not."""

            usage_metadata: ClassVar[dict[str, Any]] = {
                "input_tokens": 29,
                "output_tokens": 6,
            }
            tool_calls: ClassVar[list[Any]] = []
            response_metadata: ClassVar[dict[str, Any]] = {}

            @property
            def content(self) -> Any:
                raise TypeError("cannot serialise this content")

        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.ainvoke = AsyncMock(return_value=_Exploding())

        ctx, rec = _recording()
        with ctx, pytest.raises(TypeError):
            await create_langchain_messages_handler(llm=llm, capture_content=True)(
                CONFIG, "q", {}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 29
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 6


class TestNoContentWorkWhenCaptureIsOff:
    """With capture off, nothing should serialise the output, because nothing will read it.

    set_output_content_attributes is a no-op without the flag, but json.dumps is not. Serialising a
    parsed object json.dumps refuses turned a successful run into a raised TypeError for a caller who
    had asked for no content at all.
    """

    @pytest.mark.asyncio
    async def test_an_unserialisable_parsed_object_still_returns_normally(self) -> None:
        from launchdarkly_ai_langchain_messages import (
            create_langchain_messages_handler,
        )

        class _Unserialisable:
            pass

        parsed = _Unserialisable()
        config = {**CONFIG, "outputFormat": {"type": "object"}}
        llm = _make_llm()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(
            return_value={
                "parsed": parsed,
                "raw": FakeAIMessage("", input_tokens=8, output_tokens=2),
            }
        )
        llm.with_structured_output = MagicMock(return_value=structured_llm)

        ctx, rec = _recording()
        with ctx:
            result = await create_langchain_messages_handler(llm=llm)(
                config, "q", {}, {}
            )
        assert result["output"] is parsed
        assert result["usage"] == {"input_tokens": 8, "output_tokens": 2}
        assert "gen_ai.output.messages" not in rec.root.attributes
