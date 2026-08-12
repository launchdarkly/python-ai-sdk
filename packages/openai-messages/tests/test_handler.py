"""
Tests for launchdarkly-ai-openai-messages handler.
Covers §1.1-1.9.
Reference: TESTING.md §1, TELEMETRY-CONTRACT.md
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

CONFIG = {
    "model": {"name": "gpt-4o"},
    "provider": {"name": "OpenAI"},
    "instructions": "Be helpful.",
}


def _message_item(text: str) -> MagicMock:
    item = MagicMock()
    item.type = "message"
    item.role = "assistant"
    block = MagicMock()
    block.type = "output_text"
    block.text = text
    item.content = [block]
    return item


def _function_call_item(
    name: str, call_id: str = "call_1", args: dict | None = None
) -> MagicMock:
    item = MagicMock()
    item.type = "function_call"
    item.name = name
    item.call_id = call_id
    item.arguments = json.dumps(args or {})
    return item


def _make_response(
    output_text: str = "Hello",
    tool_calls: list[Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    resp_id: str = "resp-1",
    model: str = "gpt-4o",
    status: str = "completed",
    include_output_message: bool = True,
    cache_read: int | None = None,
) -> MagicMock:
    r = MagicMock()
    r.id = resp_id
    r.model = model
    r.output_text = output_text
    r.status = status
    r.incomplete_details = None
    r.usage = MagicMock()
    r.usage.input_tokens = input_tokens
    r.usage.output_tokens = output_tokens
    if cache_read is not None:
        r.usage.input_tokens_details = MagicMock(cached_tokens=cache_read)
    else:
        r.usage.input_tokens_details = MagicMock(cached_tokens=0)
    items: list[MagicMock] = []
    for tc in tool_calls or []:
        items.append(_function_call_item(tc["name"], tc["call_id"], tc.get("args", {})))
    if not tool_calls and include_output_message and output_text:
        items.append(_message_item(output_text))
    r.output = items
    return r


CONFIG = {
    "model": {"name": "gpt-4o"},
    "provider": {"name": "OpenAI"},
    "instructions": "Be helpful.",
}


@pytest.fixture
def mock_openai(mocker):
    mock_client = MagicMock()
    mock_client.responses = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=_make_response())
    mocker.patch("openai.AsyncOpenAI", return_value=mock_client)
    return mock_client


# ---------------------------------------------------------------------------
# §1.1 Factory function and metadata
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        assert callable(h)

    def test_attaches_provides_for(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        assert h.provides_for is not None

    def test_provides_for_values_are_correct(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        assert h.provides_for == ("OpenAI", "messages")

    def test_multiple_calls_return_independent_instances(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h1 = create_openai_messages_handler()
        h2 = create_openai_messages_handler()
        assert h1 is not h2


# ---------------------------------------------------------------------------
# §1.2 Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    async def test_path_a_instructions(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        await h(CONFIG, "hi", {}, {})
        call_kwargs = mock_openai.responses.create.call_args.kwargs
        msgs = call_kwargs["input"]
        assert any(
            m.get("role") == "system" and "Be helpful" in m.get("content", "")
            for m in msgs
        )

    async def test_path_a_variable_substitution(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {**CONFIG, "instructions": "Hello {{name}}"}
        h = create_openai_messages_handler()
        await h(config, "q", {}, {"name": "Alice"})
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        assert any(m.get("content") == "Hello Alice" for m in msgs)

    async def test_path_a_unresolved_placeholder_preserved(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {**CONFIG, "instructions": "Hello {{missing}}"}
        h = create_openai_messages_handler()
        await h(config, "q", {}, {})
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        all_content = " ".join(str(m.get("content", "")) for m in msgs)
        assert "{{missing}}" in all_content

    async def test_path_b_messages_system_extracted(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [
                {"role": "system", "content": "Be a poet"},
                {"role": "user", "content": "Write something"},
            ],
        }
        h = create_openai_messages_handler()
        await h(config, "go", {}, {})
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        assert any(
            m.get("role") == "system" and "poet" in m.get("content", "") for m in msgs
        )

    async def test_path_b_variable_substitution_in_messages(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "user", "content": "Hello {{name}}"}],
        }
        h = create_openai_messages_handler()
        await h(config, "q", {}, {"name": "Bob"})
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        assert any("Hello Bob" in str(m.get("content", "")) for m in msgs)

    async def test_path_b_user_input_appended_as_final_turn(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "assistant", "content": "Hi"}],
        }
        h = create_openai_messages_handler()
        await h(config, "final", {}, {})
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        assert msgs[-1].get("content") == "final"

    async def test_path_c_empty_user_input_no_throw(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        await h(CONFIG, "", {}, {})

    async def test_path_c_undefined_user_input_no_throw(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        await h(CONFIG, None, {}, {})  # type: ignore[arg-type]

    async def test_path_b_variable_substitution_in_system_message(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "system", "content": "Hello {{name}}"}],
        }
        h = create_openai_messages_handler()
        await h(config, "q", {}, {"name": "World"})
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        assert any("Hello World" in str(m.get("content", "")) for m in msgs)

    async def test_path_c_both_instructions_and_messages_messages_wins(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {
            **CONFIG,  # has instructions
            "messages": [{"role": "system", "content": "from-messages"}],
        }
        h = create_openai_messages_handler()
        await h(config, "q", {}, {})
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        all_content = " ".join(str(m.get("content", "")) for m in msgs)
        assert "from-messages" in all_content
        assert "Be helpful" not in all_content


# ---------------------------------------------------------------------------
# §1.3 Tool conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    async def test_all_fields_forwarded(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

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
        h = create_openai_messages_handler()
        await h(config, "q", {}, {})
        kwargs = mock_openai.responses.create.call_args.kwargs
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search the web"

    async def test_multiple_tools_all_included(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {
            **CONFIG,
            "tools": {
                "t1": {"name": "t1", "type": "function", "parameters": {}},
                "t2": {"name": "t2", "type": "function", "parameters": {}},
            },
        }
        h = create_openai_messages_handler()
        await h(config, "q", {}, {})
        tools = mock_openai.responses.create.call_args.kwargs.get("tools", [])
        assert len(tools) == 2

    async def test_empty_tools_no_tools_sent(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        await h(CONFIG, "q", {}, {})
        kwargs = mock_openai.responses.create.call_args.kwargs
        assert "tools" not in kwargs or not kwargs.get("tools")


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop
# ---------------------------------------------------------------------------


class TestToolExecutionLoop:
    async def test_single_tool_call_then_done(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_resp = _make_response(tool_calls=[{"name": "search", "call_id": "c1"}])
        final_resp = _make_response(output_text="final answer")
        mock_openai.responses.create = AsyncMock(side_effect=[tool_resp, final_resp])
        tool_fn = AsyncMock(return_value="result")
        h = create_openai_messages_handler()
        config = {
            **CONFIG,
            "tools": {
                "search": {"name": "search", "type": "function", "parameters": {}}
            },
        }
        result = await h(config, "q", {"search": tool_fn}, {})
        assert mock_openai.responses.create.call_count == 2
        assert result["output"] == "final answer"

    async def test_tool_not_found_throws(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_resp = _make_response(tool_calls=[{"name": "unknown", "call_id": "c1"}])
        mock_openai.responses.create = AsyncMock(return_value=tool_resp)
        h = create_openai_messages_handler()
        config = {
            **CONFIG,
            "tools": {"other": {"name": "other", "type": "function", "parameters": {}}},
        }
        with pytest.raises(Exception, match="No handler"):
            await h(config, "q", {}, {})

    async def test_tool_handler_throws_propagates(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_resp = _make_response(tool_calls=[{"name": "t1", "call_id": "c1"}])
        mock_openai.responses.create = AsyncMock(return_value=tool_resp)
        fn = AsyncMock(side_effect=RuntimeError("tool failed"))
        h = create_openai_messages_handler()
        config = {
            **CONFIG,
            "tools": {"t1": {"name": "t1", "type": "function", "parameters": {}}},
        }
        with pytest.raises(RuntimeError, match="tool failed"):
            await h(config, "q", {"t1": fn}, {})

    async def test_no_tools_in_config_handler_never_invoked(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_fn = AsyncMock()
        h = create_openai_messages_handler()
        await h(CONFIG, "q", {"t1": tool_fn}, {})
        kwargs = mock_openai.responses.create.call_args.kwargs
        assert "tools" not in kwargs or not kwargs.get("tools")
        tool_fn.assert_not_called()

    async def test_multiple_consecutive_tool_calls(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        resp1 = _make_response(tool_calls=[{"name": "t1", "call_id": "c1"}])
        resp2 = _make_response(tool_calls=[{"name": "t2", "call_id": "c2"}])
        resp3 = _make_response(output_text="final")
        mock_openai.responses.create = AsyncMock(side_effect=[resp1, resp2, resp3])
        fn1 = AsyncMock(return_value="r1")
        fn2 = AsyncMock(return_value="r2")
        cfg = {
            **CONFIG,
            "tools": {
                "t1": {"name": "t1", "type": "function", "parameters": {}},
                "t2": {"name": "t2", "type": "function", "parameters": {}},
            },
        }
        h = create_openai_messages_handler()
        result = await h(cfg, "q", {"t1": fn1, "t2": fn2}, {})
        fn1.assert_called_once()
        fn2.assert_called_once()
        assert result["output"] == "final"


# ---------------------------------------------------------------------------
# §1.5 Telemetry — span recording
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
    import launchdarkly_ai_openai_messages.spans as spans_mod

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
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert rec.root.name == "invoke_agent"
        assert rec.root.attributes["gen_ai.operation.name"] == "invoke_agent"

    async def test_emits_one_chat_child_per_model_turn(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        chats = rec.named("chat ")
        assert len(chats) == 1
        assert chats[0].name == "chat gpt-4o"
        assert chats[0].attributes["gen_ai.operation.name"] == "chat"
        # Parented to the root, not to nothing.
        assert chats[0].context == ("context-of", rec.root)

    async def test_names_the_chat_span_after_the_requested_model(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        cfg = {**CONFIG, "model": {"name": "gpt-4o-mini"}}
        with ctx:
            await create_openai_messages_handler()(cfg, "q", {}, {})
        assert "chat gpt-4o-mini" in rec.names

    async def test_emits_a_chat_span_per_turn_of_a_tool_loop(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(tool_calls=[{"name": "myTool", "call_id": "tu1"}]),
                _make_response(output_text="done"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "result"}, {}
            )
        assert len(rec.named("chat ")) == 2

    async def test_emits_an_execute_tool_span_per_tool_call(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(tool_calls=[{"name": "myTool", "call_id": "tu1"}]),
                _make_response(output_text="done"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "result"}, {}
            )
        tools = rec.named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].name == "execute_tool myTool"
        assert tools[0].attributes["gen_ai.operation.name"] == "execute_tool"
        assert tools[0].attributes["gen_ai.tool.name"] == "myTool"
        assert tools[0].attributes["gen_ai.tool.call.id"] == "tu1"

    async def test_tool_spans_are_siblings_of_chat_not_children(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(tool_calls=[{"name": "myTool", "call_id": "tu1"}]),
                _make_response(output_text="done"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.named("execute_tool ")[0].context == ("context-of", rec.root)

    async def test_every_span_is_ended(self, mock_openai: MagicMock) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(tool_calls=[{"name": "myTool", "call_id": "tu1"}]),
                _make_response(output_text="done"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)


class TestRootSpanAttributes:
    """TELEMETRY-CONTRACT.md sections 2 and 2a."""

    async def test_writes_both_provider_keys_and_the_requested_model(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        attrs = rec.root.attributes
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-4o"

    async def test_response_model_is_the_model_that_answered(
        self, mock_openai: MagicMock
    ) -> None:
        # OpenAI resolves an alias like `gpt-4o` to a dated snapshot. openai-messages is the only
        # handler whose root reports the answering model rather than the requested one. Section 2a.
        mock_openai.responses.create = AsyncMock(
            return_value=_make_response(model="gpt-4o-2024-08-06")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert rec.root.attributes["gen_ai.response.model"] == "gpt-4o-2024-08-06"

    async def test_carries_the_launchdarkly_attributes_and_feature_flag_event(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        variables = {
            "__ld": {
                "configKey": "k",
                "variationKey": "v",
                "runId": "r",
                "graphKey": "g",
            }
        }
        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, variables)
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"
        assert attrs["launchdarkly.graph.key"] == "g"
        assert [n for n, _ in rec.root.events] == ["feature_flag"]

    async def test_child_spans_carry_no_launchdarkly_identity(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(tool_calls=[{"name": "myTool", "call_id": "tu1"}]),
                _make_response(output_text="done"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, variables
            )
        for child in rec.spans[1:]:
            assert not [k for k in child.attributes if k.startswith("launchdarkly.")]
            assert "feature_flag" not in [n for n, _ in child.events]

    async def test_carries_the_run_total_not_one_turn(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(
                    tool_calls=[{"name": "myTool", "call_id": "tu1"}],
                    input_tokens=10,
                    output_tokens=1,
                ),
                _make_response(output_text="done", input_tokens=20, output_tokens=2),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 30
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 3
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 33


class TestChatSpanAttributes:
    """TELEMETRY-CONTRACT.md sections 3, 5a and 8."""

    async def test_writes_all_seven_usage_attributes_including_zeros(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.usage.total_tokens"] == 15
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 0
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0
        assert attrs["gen_ai.usage.prompt_tokens"] == 10
        assert attrs["gen_ai.usage.completion_tokens"] == 5

    async def test_reports_cached_tokens_as_cache_read_without_folding_into_input(
        self, mock_openai: MagicMock
    ) -> None:
        # OpenAI already counts cached tokens inside input_tokens: this is the assertion that
        # catches a fold in the Anthropic direction, which would double-count.
        mock_openai.responses.create = AsyncMock(
            return_value=_make_response(input_tokens=50, output_tokens=5, cache_read=30)
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 50
        assert attrs["gen_ai.usage.total_tokens"] == 55
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 30
        # OpenAI has no cache-creation concept; still emitted, as 0, so the set is always complete.
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0

    async def test_derives_tool_calls_before_checking_status(
        self, mock_openai: MagicMock
    ) -> None:
        # A live capture put status `completed` on every turn including the ones that stopped to
        # call a tool, so the function-call check must run first. Section 5a.
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(
                    tool_calls=[{"name": "myTool", "call_id": "tu1"}],
                    status="completed",
                ),
                _make_response(output_text="done", status="completed"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        first = rec.named("chat ")[0]
        assert first.attributes["gen_ai.response.finish_reasons"] == ["tool_calls"]

    async def test_derives_stop_from_completed_status(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "stop"
        ]

    async def test_derives_length_from_incomplete_max_output_tokens(
        self, mock_openai: MagicMock
    ) -> None:
        resp = _make_response(status="incomplete", include_output_message=False)
        resp.incomplete_details = MagicMock(reason="max_output_tokens")
        mock_openai.responses.create = AsyncMock(return_value=resp)
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "length"
        ]

    async def test_derives_content_filter_from_incomplete_other_reason(
        self, mock_openai: MagicMock
    ) -> None:
        resp = _make_response(status="incomplete", include_output_message=False)
        resp.incomplete_details = MagicMock(reason="content_filter")
        mock_openai.responses.create = AsyncMock(return_value=resp)
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "content_filter"
        ]

    async def test_writes_no_finish_reason_for_an_unrecognised_status(
        self, mock_openai: MagicMock
    ) -> None:
        # No passthrough for the two OpenAI handlers: an unrecognised status drops the attribute.
        resp = _make_response(status="cancelled", include_output_message=False)
        mock_openai.responses.create = AsyncMock(return_value=resp)
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert "gen_ai.response.finish_reasons" not in rec.named("chat ")[0].attributes

    async def test_sets_status_ok_on_a_successful_turn(
        self, mock_openai: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert StatusCode.OK in rec.named("chat ")[0].statuses


class TestContentCapture:
    """TELEMETRY-CONTRACT.md section 7."""

    async def test_emits_no_content_at_all_by_default(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
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
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            return_value=_make_response(output_text="Hello World")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler(capture_content=True)(
                CONFIG, "q", {}, {}
            )
        chat = rec.named("chat ")[0]
        assert chat.attributes["gen_ai.prompt.0.role"] == "system"
        assert chat.attributes["gen_ai.prompt.0.content"] == "Be helpful."
        assert "gen_ai.input.messages" in chat.attributes
        assert chat.attributes["gen_ai.completion.0.content"] == "Hello World"
        assert "gen_ai.output.messages" in chat.attributes

    async def test_records_the_tool_catalog_on_the_chat_span_when_enabled(
        self, mock_openai: MagicMock
    ) -> None:
        cfg = {
            **CONFIG,
            "tools": {"myTool": {"description": "d", "parameters": {"type": "object"}}},
        }
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler(capture_content=True)(
                cfg, "q", {"myTool": lambda _: "r"}, {}
            )
        definitions = json.loads(
            rec.named("chat ")[0].attributes["gen_ai.tool.definitions"]
        )
        assert definitions[0]["name"] == "myTool"
        assert definitions[0]["type"] == "function"

    async def test_records_tool_arguments_and_results_when_enabled(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(
                    tool_calls=[
                        {"name": "myTool", "call_id": "tu1", "args": {"city": "NYC"}}
                    ]
                ),
                _make_response(output_text="done"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler(capture_content=True)(
                CONFIG, "q", {"myTool": lambda _: "72F"}, {}
            )
        tool = rec.named("execute_tool ")[0]
        assert tool.attributes["gen_ai.tool.call.arguments"] == '{"city": "NYC"}'
        assert tool.attributes["gen_ai.tool.call.result"] == "72F"

    async def test_still_writes_the_legacy_content_events_when_enabled(
        self, mock_openai: MagicMock
    ) -> None:
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            await create_openai_messages_handler(capture_content=True)(
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
        self, mock_openai: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        mock_openai.responses.create = AsyncMock(side_effect=RuntimeError("api error"))
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert len(chat.exceptions) == 1
        assert StatusCode.ERROR in chat.statuses
        assert chat.ended == 1

    async def test_fails_the_root_span_too(self, mock_openai: MagicMock) -> None:
        from opentelemetry.trace import StatusCode

        mock_openai.responses.create = AsyncMock(side_effect=RuntimeError("api error"))
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert len(rec.root.exceptions) == 1
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_fails_the_execute_tool_span_when_a_tool_raises(
        self, mock_openai: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        mock_openai.responses.create = AsyncMock(
            return_value=_make_response(
                tool_calls=[{"name": "myTool", "call_id": "tu1"}]
            )
        )

        def _boom(_: Any) -> Any:
            raise RuntimeError("tool exploded")

        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(RuntimeError, match="tool exploded"):
            await create_openai_messages_handler()(CONFIG, "q", {"myTool": _boom}, {})
        tool = rec.named("execute_tool ")[0]
        assert len(tool.exceptions) == 1
        assert StatusCode.ERROR in tool.statuses
        assert tool.ended == 1

    async def test_reports_the_spend_of_completed_turns_on_a_failed_run(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=[
                _make_response(
                    tool_calls=[{"name": "myTool", "call_id": "tu1"}],
                    input_tokens=40,
                    output_tokens=7,
                ),
                RuntimeError("second turn died"),
            ]
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_openai_messages_handler()(
                CONFIG, "q", {"myTool": lambda _: "r"}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 40
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 7

    async def test_writes_no_usage_when_no_turn_ever_reported_any(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.responses.create = AsyncMock(
            side_effect=RuntimeError("died on the first call")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(RuntimeError):
            await create_openai_messages_handler()(CONFIG, "q", {}, {})
        assert "gen_ai.usage.input_tokens" not in rec.root.attributes

    async def test_rethrows_error(self, mock_openai: MagicMock) -> None:
        mock_openai.responses.create = AsyncMock(side_effect=RuntimeError("rethrown"))
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        with pytest.raises(RuntimeError, match="rethrown"):
            await h(CONFIG, "q", {}, {})


# ---------------------------------------------------------------------------
# §1.9 Structured output (outputFormat)
# ---------------------------------------------------------------------------


class TestOutputFormat:
    async def test_absent_output_format_no_change(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        await h(CONFIG, "q", {}, {})
        kwargs = mock_openai.responses.create.call_args.kwargs
        assert "text" not in kwargs

    async def test_output_format_uses_text_format_json_schema(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {**CONFIG, "outputFormat": {"type": "object", "properties": {}}}
        h = create_openai_messages_handler()
        await h(config, "q", {}, {})
        kwargs = mock_openai.responses.create.call_args.kwargs
        assert "text" in kwargs
        assert kwargs["text"]["format"]["type"] == "json_schema"


# ---------------------------------------------------------------------------
# §1.7 Convenience export
# ---------------------------------------------------------------------------


class TestConvenienceExport:
    def test_calls_through_to_model_call(self, mock_openai: MagicMock) -> None:
        import launchdarkly_ai_openai_messages.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_openai_messages.handler import openai_messages

            ctx = {"kind": "user", "key": "u1"}
            openai_messages("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        handler = call_kwargs.get("handler")
        assert handler is not None
        assert handler.provides_for == ("OpenAI", "messages")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self, mock_openai: MagicMock) -> None:
        import launchdarkly_ai_openai_messages.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_openai_messages.handler import openai_messages

            ctx = {"kind": "user", "key": "u1"}
            openai_messages("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# §1.8 Streaming
# ---------------------------------------------------------------------------


def _make_openai_stream_context(
    chunks: list[str],
    input_tok: int = 5,
    output_tok: int = 3,
    output: list[Any] | None = None,
) -> Any:
    """Returns a mock OpenAI stream context manager."""
    events = []
    for c in chunks:
        e = MagicMock()
        e.type = "response.output_text.delta"
        e.delta = c
        events.append(e)

    final_resp = MagicMock()
    final_resp.output = output if output is not None else []
    final_resp.status = "completed"
    final_resp.incomplete_details = None
    final_resp.usage = MagicMock(input_tokens=input_tok, output_tokens=output_tok)
    final_resp.usage.input_tokens_details = MagicMock(cached_tokens=0)
    final_resp.id = "resp-stream"
    final_resp.model = "gpt-4o"

    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[Any]:
            return self._iter()

        async def _iter(self) -> AsyncIterator[Any]:
            for e in events:
                yield e

        async def get_final_response(self) -> Any:
            return final_resp

    @asynccontextmanager
    async def _ctx_mgr() -> AsyncGenerator[Any, None]:
        yield _FakeStream()

    return _ctx_mgr()


class TestStreaming:
    def _patch_stream(
        self,
        mock_openai: MagicMock,
        chunks: list[str],
        input_tok: int = 5,
        output_tok: int = 3,
    ) -> None:
        mock_openai.responses.stream = MagicMock(
            return_value=_make_openai_stream_context(chunks, input_tok, output_tok)
        )

    async def test_stream_defined_and_async_generator(
        self, mock_openai: MagicMock
    ) -> None:
        import launchdarkly_ai_openai_messages.spans as spans_mod
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        self._patch_stream(mock_openai, ["hi"])
        with patch.object(spans_mod, "_HAS_OTEL", False):
            h = create_openai_messages_handler()
        assert h.has_stream
        gen = await h.stream(CONFIG, "q")
        assert hasattr(gen, "__aiter__")

    async def test_yields_chunk_events(self, mock_openai: MagicMock) -> None:
        import launchdarkly_ai_openai_messages.spans as spans_mod
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        self._patch_stream(mock_openai, ["hello ", "world"])
        with patch.object(spans_mod, "_HAS_OTEL", False):
            h = create_openai_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        chunks = [e for e in events if e.get("type") == "chunk"]
        assert len(chunks) == 2
        assert chunks[0]["text"] == "hello "
        assert chunks[1]["text"] == "world"

    async def test_yields_exactly_one_done_event(self, mock_openai: MagicMock) -> None:
        import launchdarkly_ai_openai_messages.spans as spans_mod
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        self._patch_stream(mock_openai, ["x"])
        with patch.object(spans_mod, "_HAS_OTEL", False):
            h = create_openai_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    async def test_done_event_carries_usage(self, mock_openai: MagicMock) -> None:
        import launchdarkly_ai_openai_messages.spans as spans_mod
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        self._patch_stream(mock_openai, ["text"], input_tok=7, output_tok=3)
        with patch.object(spans_mod, "_HAS_OTEL", False):
            h = create_openai_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        done = next(e for e in events if e.get("type") == "done")
        assert done["usage"]["input_tokens"] == 7
        assert done["usage"]["output_tokens"] == 3

    async def test_done_event_carries_accumulated_output(
        self, mock_openai: MagicMock
    ) -> None:
        import launchdarkly_ai_openai_messages.spans as spans_mod
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        self._patch_stream(mock_openai, ["hello ", "world"])
        with patch.object(spans_mod, "_HAS_OTEL", False):
            h = create_openai_messages_handler()
            events = [e async for e in await h.stream(CONFIG, "q")]
        done = next(e for e in events if e.get("type") == "done")
        assert done["output"] == "hello world"

    async def test_generator_throws_on_provider_error(
        self, mock_openai: MagicMock
    ) -> None:
        import launchdarkly_ai_openai_messages.spans as spans_mod
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        @asynccontextmanager
        async def _bad_ctx() -> AsyncGenerator[Any, None]:
            raise RuntimeError("stream error")
            yield

        mock_openai.responses.stream = MagicMock(return_value=_bad_ctx())
        with patch.object(spans_mod, "_HAS_OTEL", False):
            h = create_openai_messages_handler()
            with pytest.raises(RuntimeError, match="stream error"):
                async for _ in await h.stream(CONFIG, "q"):
                    pass

    async def test_tools_forwarded_on_second_streaming_turn(
        self, mock_openai: MagicMock
    ) -> None:
        """§1.8 - tools must appear in stream_params on every streaming turn.

        This is a pre-existing Python-only behaviour that diverges from the TypeScript SDK (which
        does not resend tools after the first turn). It changes what the model is offered, not what
        the span reports, so this test only pins that the behaviour is unchanged by the span work.
        """
        import launchdarkly_ai_openai_messages.spans as spans_mod
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_call_item = _function_call_item("my-tool", "call-1", {"q": "x"})

        first_final = MagicMock()
        first_final.output = [tool_call_item]
        first_final.status = "completed"
        first_final.incomplete_details = None
        first_final.usage = MagicMock(input_tokens=3, output_tokens=1)
        first_final.usage.input_tokens_details = MagicMock(cached_tokens=0)
        first_final.id = "resp-first"
        first_final.model = "gpt-4o"

        class _FirstStream:
            def __aiter__(self) -> AsyncIterator[Any]:
                return self._iter()

            async def _iter(self) -> AsyncIterator[Any]:
                e = MagicMock()
                e.type = "response.output_text.delta"
                e.delta = "thinking..."
                yield e

            async def get_final_response(self) -> Any:
                return first_final

        second_final = MagicMock()
        second_final.output = []
        second_final.output_text = "done"
        second_final.status = "completed"
        second_final.incomplete_details = None
        second_final.usage = MagicMock(input_tokens=4, output_tokens=2)
        second_final.usage.input_tokens_details = MagicMock(cached_tokens=0)
        second_final.id = "resp-second"
        second_final.model = "gpt-4o"

        class _SecondStream:
            def __aiter__(self) -> AsyncIterator[Any]:
                return self._iter()

            async def _iter(self) -> AsyncIterator[Any]:
                e = MagicMock()
                e.type = "response.output_text.delta"
                e.delta = "done"
                yield e

            async def get_final_response(self) -> Any:
                return second_final

        captured_stream_calls: list[dict[str, Any]] = []
        stream_returns = [_FirstStream(), _SecondStream()]

        @asynccontextmanager
        async def _stream_ctx(**kwargs: Any) -> AsyncGenerator[Any, None]:
            captured_stream_calls.append(kwargs)
            yield stream_returns.pop(0)

        mock_openai.responses.stream = MagicMock(
            side_effect=lambda **kw: _stream_ctx(**kw)
        )

        config = {
            **CONFIG,
            "tools": {"my-tool": {"description": "does stuff", "parameters": {}}},
        }

        with patch.object(spans_mod, "_HAS_OTEL", False):
            h = create_openai_messages_handler()
            _events = [
                e
                async for e in await h.stream(
                    config, "q", {"my-tool": AsyncMock(return_value="result")}
                )
            ]

        assert len(captured_stream_calls) == 2, (
            f"Expected 2 streaming calls (initial + tool follow-up), got {len(captured_stream_calls)}"
        )
        for i, call_kwargs in enumerate(captured_stream_calls):
            assert "tools" in call_kwargs, (
                f"streaming turn {i + 1} was missing 'tools' in stream_params. "
                "Tools must be forwarded on every streaming turn."
            )


# ---------------------------------------------------------------------------
# §1.2 Path C — None user_input must not produce None content
# ---------------------------------------------------------------------------


class TestNoneUserInput:
    """TESTING.md §1.2 Path C: When user_input is None, the user-role message
    content sent to the provider must be '' not None."""

    async def test_none_user_input_instructions_path_no_none_content(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        captured: list[Any] = []

        async def _capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return _make_response()

        mock_openai.responses.create = _capture
        h = create_openai_messages_handler()
        await h(CONFIG, None, {}, {})

        assert captured, "responses.create was not called"
        input_msgs = captured[0].get("input", [])
        for msg in input_msgs:
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

    def _tool_response(self) -> MagicMock:
        return _make_response(
            tool_calls=[{"name": "myTool", "call_id": "c1", "args": {}}],
            input_tokens=1,
            output_tokens=1,
        )

    async def test_invoke_throws_after_max_steps(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_resp = self._tool_response()
        mock_openai.responses.create = AsyncMock(return_value=tool_resp)

        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_openai_messages_handler()
        with pytest.raises(RuntimeError, match="maximum number of steps"):
            await h(cfg, "q", {"myTool": lambda _: "result"})

    async def test_invoke_succeeds_at_exactly_max_steps(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_resp = self._tool_response()
        final_resp = _make_response("Done")
        mock_openai.responses.create = AsyncMock(
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
        h = create_openai_messages_handler()
        result = await h(cfg, "q", {"myTool": lambda _: "result"})
        assert result["output"] == "Done"

    async def test_stream_throws_after_max_steps(self, mock_openai: MagicMock) -> None:
        from contextlib import asynccontextmanager

        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        def _make_stream_cm() -> Any:
            @asynccontextmanager
            async def _ctx() -> AsyncGenerator:
                mock_s = MagicMock()

                async def _iter() -> AsyncGenerator:
                    yield MagicMock(type="response.output_text.delta", delta="")

                mock_s.__aiter__ = lambda _: _iter().__aiter__()
                final = _make_response(
                    tool_calls=[{"name": "myTool", "call_id": "c1", "args": {}}],
                    input_tokens=1,
                    output_tokens=1,
                )
                mock_s.get_final_response = AsyncMock(return_value=final)
                yield mock_s

            return _ctx()

        mock_openai.responses.stream = MagicMock(
            side_effect=lambda **_: _make_stream_cm()
        )

        cfg = {**CONFIG, "tools": {"myTool": {"type": "function", "parameters": {}}}}
        h = create_openai_messages_handler()
        with pytest.raises(RuntimeError, match="maximum number of steps"):
            async for _ in await h.stream(cfg, "q", {"myTool": lambda _: "result"}):
                pass


# ---------------------------------------------------------------------------
# §1.5 Streaming telemetry (do not patch _HAS_OTEL=False)
# ---------------------------------------------------------------------------


class TestStreamingTelemetry:
    """TELEMETRY-CONTRACT.md sections 1 and 6. The streaming path emits the same tree."""

    def _patch_stream(
        self,
        mock_openai: MagicMock,
        chunks: list[str],
        input_tok: int = 5,
        output_tok: int = 3,
    ) -> None:
        mock_openai.responses.stream = MagicMock(
            return_value=_make_openai_stream_context(chunks, input_tok, output_tok)
        )

    async def test_opens_the_same_root_span_name_as_the_blocking_path(
        self, mock_openai: MagicMock
    ) -> None:
        self._patch_stream(mock_openai, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            async for _ in await create_openai_messages_handler().stream(CONFIG, "q"):
                pass
        assert rec.root.name == "invoke_agent"
        assert "chat gpt-4o" in rec.names

    async def test_carries_the_launchdarkly_attributes_on_the_root(
        self, mock_openai: MagicMock
    ) -> None:
        self._patch_stream(mock_openai, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            async for _ in await create_openai_messages_handler().stream(
                CONFIG, "q", None, variables
            ):
                pass
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"

    async def test_ends_every_span_once_when_the_stream_completes(
        self, mock_openai: MagicMock
    ) -> None:
        self._patch_stream(mock_openai, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            async for _ in await create_openai_messages_handler().stream(CONFIG, "q"):
                pass
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)
        assert "launchdarkly.stream.abandoned" not in rec.root.attributes

    async def test_writes_the_run_total_to_the_root(
        self, mock_openai: MagicMock
    ) -> None:
        self._patch_stream(mock_openai, ["hi"], input_tok=11, output_tok=4)
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            async for _ in await create_openai_messages_handler().stream(CONFIG, "q"):
                pass
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 11
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 15

    async def test_an_abandoned_stream_still_ends_and_exports_every_span(
        self, mock_openai: MagicMock
    ) -> None:
        self._patch_stream(mock_openai, ["one", "two", "three"])
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            gen = await create_openai_messages_handler().stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.ended == 1
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)

    async def test_an_abandoned_stream_is_marked_but_not_failed(
        self, mock_openai: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        self._patch_stream(mock_openai, ["one", "two", "three"])
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            gen = await create_openai_messages_handler().stream(CONFIG, "q")
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.attributes["launchdarkly.stream.abandoned"] is True
        assert StatusCode.ERROR not in rec.root.statuses
        assert rec.root.exceptions == []

    async def test_fails_the_spans_when_the_stream_raises(
        self, mock_openai: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        mock_openai.responses.stream = MagicMock(
            side_effect=RuntimeError("stream died")
        )
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(RuntimeError, match="stream died"):
            async for _ in await create_openai_messages_handler().stream(CONFIG, "q"):
                pass
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_emits_no_content_by_default_on_the_streaming_path(
        self, mock_openai: MagicMock
    ) -> None:
        self._patch_stream(mock_openai, ["hi"])
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx:
            async for _ in await create_openai_messages_handler().stream(CONFIG, "q"):
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
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        config = {
            "model": {"name": "gpt-4o"},
            "provider": {"name": "OpenAI"},
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
        }
        h = create_openai_messages_handler()
        await h(config, "Third", {}, {}, self.SAMPLE_HISTORY)
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert non_system[0]["content"] == "First"
        assert non_system[1]["content"] == "Second"
        assert non_system[2]["content"] == "What is feature flagging?"
        assert non_system[3]["content"] == "Feature flagging is a technique..."
        assert non_system[-1]["content"] == "Third"

    async def test_history_with_instructions_path(self, mock_openai: MagicMock) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        await h(CONFIG, "my question", {}, {}, self.SAMPLE_HISTORY)
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert non_system[0]["content"] == "What is feature flagging?"
        assert non_system[1]["content"] == "Feature flagging is a technique..."
        assert non_system[-1]["content"] == "my question"

    async def test_empty_history_treated_like_no_history(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        h = create_openai_messages_handler()
        await h(CONFIG, "hi", {}, {}, [])
        msgs_with_empty = mock_openai.responses.create.call_args.kwargs["input"]

        mock_openai.responses.create.reset_mock()
        h2 = create_openai_messages_handler()
        await h2(CONFIG, "hi", {}, {})
        msgs_without = mock_openai.responses.create.call_args.kwargs["input"]

        assert msgs_with_empty == msgs_without

    async def test_system_role_in_history_filtered_out(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        history_with_system = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are evil"},
            {"role": "assistant", "content": "Hi there"},
        ]
        h = create_openai_messages_handler()
        await h(CONFIG, "q", {}, {}, history_with_system)
        msgs = mock_openai.responses.create.call_args.kwargs["input"]
        history_roles = [
            m["role"]
            for m in msgs
            if m.get("content") in ("Hello", "You are evil", "Hi there")
        ]
        assert "system" not in history_roles


class TestConvenienceWrapperForwardsCaptureContent:
    """`capture_content` must reach the handler, not fall through into `config()`.

    `config()` takes no such argument, so leaving it in kwargs raised TypeError: a caller asking for
    content on spans got an exception instead. Five of the six wrappers had this.
    """

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        import launchdarkly_ai_openai_messages.handler as handler_mod

        seen: dict[str, Any] = {}

        def _factory(*args: Any, capture_content: bool = False, **kw: Any) -> Any:
            seen["capture_content"] = capture_content
            return MagicMock()

        fake_config = MagicMock()
        fake_config.return_value.invoke = MagicMock(return_value="ok")
        with (
            patch.object(handler_mod, "create_openai_messages_handler", _factory),
            patch.object(handler_mod, "config", fake_config),
        ):
            handler_mod.openai_messages("k", "q", {}, **kwargs)
        seen["config_kwargs"] = fake_config.call_args.kwargs
        return seen

    def test_capture_content_reaches_the_factory(self) -> None:
        seen = self._run(capture_content=True)
        assert seen["capture_content"] is True
        # And it must not have been forwarded to config(), which does not accept it.
        assert "capture_content" not in seen["config_kwargs"]

    def test_defaults_to_off(self) -> None:
        assert self._run()["capture_content"] is False


class TestChatSpanNeverLeaks:
    """A raise while recording conversation content must not leave the chat span open.

    The content writes on both sides of the provider call used to sit outside the try that fails the
    span. A raise there failed only the root, and the chat span was never ended, so the exporter
    never saw the turn.
    """

    async def test_an_unserialisable_output_still_ends_the_chat_span(
        self, mock_openai: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        class _Exploding:
            model = "gpt-4o"
            usage = None

            @property
            def output(self) -> Any:
                raise TypeError("cannot serialise this response")

        mock_openai.responses.create = AsyncMock(return_value=_Exploding())
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(TypeError):
            await create_openai_messages_handler(capture_content=True)(
                CONFIG, "q", {}, {}
            )
        chat = rec.named("chat ")
        assert len(chat) == 1
        assert chat[0].ended == 1, "the chat span leaked"
        assert StatusCode.ERROR in chat[0].statuses

    async def test_a_content_failure_does_not_lose_the_tokens_already_billed(
        self, mock_openai: MagicMock
    ) -> None:
        # The provider has already charged for this turn. Failing to serialise its content is our
        # problem, not a reason to report the run as having spent less than it did.
        class _Exploding:
            model = "gpt-4o"
            usage = MagicMock(input_tokens=31, output_tokens=9)

            def __init__(self) -> None:
                self.usage.input_tokens_details = MagicMock(cached_tokens=0)

            @property
            def output(self) -> Any:
                raise TypeError("cannot serialise this response")

        mock_openai.responses.create = AsyncMock(return_value=_Exploding())
        ctx, rec = _recording()
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        with ctx, pytest.raises(TypeError):
            await create_openai_messages_handler(capture_content=True)(
                CONFIG, "q", {}, {}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 31
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 9


class TestOpenToolSpanIsNeverLeaked:
    """A BaseException while a tool runs must still close the execute_tool span.

    The streaming `finally` closed the model span and the root, but the in-flight tool span was held
    only by a local. `except Exception` does not see a `CancelledError` or a `GeneratorExit`, so a
    tool cancelled mid-flight left its span open and unexported: the trace showed a closed parent
    above a child that never arrived.
    """

    async def test_a_tool_cancelled_mid_flight_still_ends_its_span(
        self, mock_openai: MagicMock
    ) -> None:
        import asyncio

        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        tool_call_item = _function_call_item("my-tool", "call-1", {"q": "x"})
        final = MagicMock()
        final.output = [tool_call_item]
        final.status = "completed"
        final.incomplete_details = None
        final.usage = MagicMock(input_tokens=3, output_tokens=1)
        final.usage.input_tokens_details = MagicMock(cached_tokens=0)
        final.id = "resp-1"
        final.model = "gpt-4o"

        class _Stream:
            def __aiter__(self) -> AsyncIterator[Any]:
                return self._iter()

            async def _iter(self) -> AsyncIterator[Any]:
                e = MagicMock()
                e.type = "response.output_text.delta"
                e.delta = "thinking..."
                yield e

            async def get_final_response(self) -> Any:
                return final

        @asynccontextmanager
        async def _ctx() -> Any:
            yield _Stream()

        mock_openai.responses.stream = MagicMock(return_value=_ctx())

        async def _cancelled_tool(_: Any) -> Any:
            # A BaseException, so `except Exception` in the tool loop does not see it.
            raise asyncio.CancelledError()

        ctx, rec = _recording()
        with ctx, pytest.raises(asyncio.CancelledError):
            async for _ in await create_openai_messages_handler().stream(
                CONFIG, "q", {"my-tool": _cancelled_tool}
            ):
                pass

        tools = rec.named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].ended == 1, "the execute_tool span leaked"
        assert tools[0].attributes["launchdarkly.stream.abandoned"] is True
        assert rec.root.ended == 1


class TestStreamingChatSpanAndTokens:
    """The streaming path must fail its span and keep its tokens when content serialisation raises.

    The content write and the span finish sat after the try that fails the chat span, and the usage
    was accumulated last. A raise while serialising the response therefore left the span for `finally`
    to end as abandoned, which reads as a consumer who walked away rather than as the failure it was,
    and dropped a turn the provider had already billed.
    """

    def _exploding_stream(self, mock_openai: MagicMock) -> None:
        class _Exploding:
            model = "gpt-4o"
            status = "completed"
            incomplete_details = None
            id = "resp-1"

            def __init__(self) -> None:
                self.usage = MagicMock(input_tokens=44, output_tokens=12)
                self.usage.input_tokens_details = MagicMock(cached_tokens=0)

            @property
            def output(self) -> Any:
                raise TypeError("cannot serialise this response")

        class _Stream:
            def __aiter__(self) -> AsyncIterator[Any]:
                return self._iter()

            async def _iter(self) -> AsyncIterator[Any]:
                e = MagicMock()
                e.type = "response.output_text.delta"
                e.delta = "thinking..."
                yield e

            async def get_final_response(self) -> Any:
                return _Exploding()

        @asynccontextmanager
        async def _ctx() -> Any:
            yield _Stream()

        mock_openai.responses.stream = MagicMock(return_value=_ctx())

    async def test_the_chat_span_is_failed_not_abandoned(
        self, mock_openai: MagicMock
    ) -> None:
        from opentelemetry.trace import StatusCode

        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        self._exploding_stream(mock_openai)
        ctx, rec = _recording()
        with ctx, pytest.raises(TypeError):
            async for _ in await create_openai_messages_handler(
                capture_content=True
            ).stream(CONFIG, "q"):
                pass

        chat = rec.named("chat ")[0]
        assert chat.ended == 1
        assert StatusCode.ERROR in chat.statuses
        assert "launchdarkly.stream.abandoned" not in chat.attributes

    async def test_the_tokens_already_billed_survive(
        self, mock_openai: MagicMock
    ) -> None:
        from launchdarkly_ai_openai_messages import create_openai_messages_handler

        self._exploding_stream(mock_openai)
        ctx, rec = _recording()
        with ctx, pytest.raises(TypeError):
            async for _ in await create_openai_messages_handler(
                capture_content=True
            ).stream(CONFIG, "q"):
                pass
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 44
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 12
