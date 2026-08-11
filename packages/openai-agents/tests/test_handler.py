"""
Tests for launchdarkly-ai-openai-agents handler.

TELEMETRY-CONTRACT.md sections 1-9 for the span tree, and TESTING.md §1 for the generic handler
behaviours. Rewritten from the pre-span-work version: this handler drives the ``agents`` SDK's own
``Runner``, which owns the per-turn loop, so a test drives spans by calling the ``RunHooks``
callbacks (``on_llm_start`` / ``on_llm_end`` / ``on_tool_start`` / ``on_tool_end``) the way the real
``Runner`` would, rather than by mocking a client response directly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_openai_agents.handler as handler_mod
from launchdarkly_ai_openai_agents.handler import (
    _build_agent_and_prompt,
    _build_agent_tools,
    create_openai_agent_handler,
    openai_agents,
)
from launchdarkly_ai_openai_agents.utils import build_output_type

# ---------------------------------------------------------------------------
# Fake `agents` SDK
# ---------------------------------------------------------------------------


class FakeFunctionTool:
    def __init__(self, **kw: Any) -> None:
        self.name = kw["name"]
        self.description = kw.get("description")
        self.params_json_schema = kw.get("params_json_schema")
        self.on_invoke_tool = kw.get("on_invoke_tool")


class FakeAgent:
    def __init__(self, **kw: Any) -> None:
        self.kwargs = kw
        self.name = kw.get("name")
        self.model = kw.get("model")
        self.instructions = kw.get("instructions")
        self.tools = kw.get("tools", [])
        self.output_type = kw.get("output_type")


class FakeRunResult:
    def __init__(self, final_output: Any, run_data: Any = None) -> None:
        self.final_output = final_output
        self.run_data = run_data


async def _drive_turns(
    hooks: Any, agent: Any, prompt: str, turns: list[dict[str, Any]]
) -> None:
    """Calls the ``RunHooks`` callbacks the way the real ``Runner.run`` would, one turn at a time."""
    for turn in turns:
        await hooks.on_llm_start(MagicMock(), agent, agent.instructions, prompt)
        for call in turn.get("tool_calls", []):
            ctx = SimpleNamespace(
                tool_name=call["name"],
                tool_call_id=call["id"],
                tool_arguments=json.dumps(call.get("args", {})),
            )
            tool_obj = SimpleNamespace(name=call["name"])
            await hooks.on_tool_start(ctx, agent, tool_obj)
            if "error" in call:
                raise call["error"]
            await hooks.on_tool_end(ctx, agent, tool_obj, call.get("result", "ok"))
        if turn.get("error") is not None:
            raise turn["error"]
        response = SimpleNamespace(
            output=turn.get("output", []), usage=turn.get("usage", {})
        )
        await hooks.on_llm_end(MagicMock(), agent, response)


def _make_run(turns: list[dict[str, Any]], final_output: str = "done") -> Any:
    async def run(agent: Any, prompt: str, hooks: Any = None, **kw: Any) -> Any:
        await _drive_turns(hooks, agent, prompt, turns)
        return FakeRunResult(final_output)

    return run


class FakeStreamedResult:
    """Interleaves hook calls with delta events, the way ``RunResultStreaming`` really does.

    Breaking out of ``stream_events()`` early therefore leaves whichever turn was in flight
    without its ``on_llm_end`` (or ``on_tool_end``), exactly as an abandoned real run would.
    """

    def __init__(
        self,
        agent: Any,
        prompt: str,
        hooks: Any,
        turns: list[dict[str, Any]],
        final_output: str,
    ) -> None:
        self.agent = agent
        self.prompt = prompt
        self.hooks = hooks
        self.turns = turns
        self.final_output = final_output
        self.cancelled = False

    async def stream_events(self) -> AsyncIterator[Any]:
        for turn in self.turns:
            await self.hooks.on_llm_start(
                MagicMock(), self.agent, self.agent.instructions, self.prompt
            )
            for delta in turn.get("deltas", []):
                yield SimpleNamespace(
                    type="raw_response_event",
                    data=SimpleNamespace(
                        type="response.output_text.delta", delta=delta
                    ),
                )
            for call in turn.get("tool_calls", []):
                ctx = SimpleNamespace(
                    tool_name=call["name"],
                    tool_call_id=call["id"],
                    tool_arguments=json.dumps(call.get("args", {})),
                )
                tool_obj = SimpleNamespace(name=call["name"])
                await self.hooks.on_tool_start(ctx, self.agent, tool_obj)
                await self.hooks.on_tool_end(
                    ctx, self.agent, tool_obj, call.get("result", "ok")
                )
            if turn.get("error") is not None:
                raise turn["error"]
            response = SimpleNamespace(
                output=turn.get("output", []), usage=turn.get("usage", {})
            )
            await self.hooks.on_llm_end(MagicMock(), self.agent, response)

    def cancel(self, mode: str = "immediate") -> None:
        self.cancelled = True


def _make_run_streamed(turns: list[dict[str, Any]], final_output: str = "done") -> Any:
    def run_streamed(agent: Any, prompt: str, hooks: Any = None, **kw: Any) -> Any:
        return FakeStreamedResult(agent, prompt, hooks, turns, final_output)

    return run_streamed


def _fake_agents_module(run: Any = None, run_streamed: Any = None) -> Any:
    mod = SimpleNamespace()
    mod.FunctionTool = FakeFunctionTool
    mod.Agent = FakeAgent

    class Runner:
        pass

    Runner.run = staticmethod(run or _make_run([{"output": [], "usage": {}}]))
    Runner.run_streamed = staticmethod(
        run_streamed or _make_run_streamed([{"output": [], "usage": {}}])
    )
    mod.Runner = Runner
    return mod


def _patched_agents(agents_mod: Any) -> Any:
    return patch(
        "importlib.import_module",
        side_effect=lambda n: agents_mod if n == "agents" else __import__(n),
    )


def _make_run_result(
    output: str = "hello", input_tokens: int = 10, output_tokens: int = 5
) -> Any:
    """The pre-span-era flat mock result, kept for the restored non-telemetry tests that predate
    the ``RunHooks``-driven span work and never touch spans or usage attribution."""
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


async def _empty_async_gen() -> AsyncIterator[Any]:
    return
    yield


def _mock_agents_module(run_result: Any) -> Any:
    """The pre-span-era fully-flat ``agents`` mock: ``Runner.run``/``run_streamed`` never invoke
    ``hooks``. Kept for the restored tests that only care about prompt/tool wiring, not spans.
    """
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


def _make_config(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"model": {"name": "gpt-4o"}, "provider": {"name": "OpenAI"}}
    base.update(kwargs)
    return base


CONFIG = _make_config(instructions="Be helpful.")


def _text_output(text: str) -> list[dict[str, Any]]:
    return [{"type": "message", "role": "assistant", "content": text}]


def _tool_call_output(name: str, call_id: str, arguments: str = "{}") -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _usage(
    input_tokens: int = 10, output_tokens: int = 5, cached: int | None = None
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cached is not None:
        usage["input_tokens_details"] = {"cached_tokens": cached}
    return usage


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

    A single ``MagicMock`` cannot see a span tree at all, because every span is the same object and
    a parent and its children are indistinguishable. This keeps one object per span.
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
    import launchdarkly_ai_openai_agents.spans as spans_mod

    recorder = SpanRecorder()
    return patch.object(spans_mod, "trace", recorder), recorder


# ---------------------------------------------------------------------------
# §1.1 Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self) -> None:
        assert callable(create_openai_agent_handler())

    def test_attaches_provides_for(self) -> None:
        h = create_openai_agent_handler()
        assert h.provides_for == ("OpenAI", "agent")

    def test_multiple_calls_return_independent_instances(self) -> None:
        assert create_openai_agent_handler() is not create_openai_agent_handler()

    def test_provides_for_values_are_correct(self) -> None:
        h = create_openai_agent_handler()
        pf = h.provides_for
        assert "OpenAI" in pf or "openai" in str(pf).lower()


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

    def test_instructions_path(self) -> None:
        config = _make_config(instructions="Be concise.")
        _, prompt, instructions = _build_agent_and_prompt(config, "hi", {}, {})
        assert instructions == "Be concise."
        assert prompt == "hi"

    def test_variable_substitution(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        _, _, instructions = _build_agent_and_prompt(config, "hi", {}, {"name": "Bob"})
        assert instructions == "Hello Bob!"

    def test_messages_path_extracts_system(self) -> None:
        config = _make_config(
            messages=[
                {"role": "system", "content": "System msg."},
                {"role": "user", "content": "prior turn"},
            ]
        )
        _, prompt, instructions = _build_agent_and_prompt(config, "question", {}, {})
        assert instructions == "System msg."
        assert "prior turn" in prompt
        assert "question" in prompt

    def test_instructions_take_priority_over_messages(self) -> None:
        config = _make_config(
            instructions="From instructions.",
            messages=[{"role": "system", "content": "From messages."}],
        )
        _, _, instructions = _build_agent_and_prompt(config, "q", {}, {})
        assert instructions == "From instructions."

    def test_none_user_input_becomes_empty_string(self) -> None:
        config = _make_config(instructions="Be helpful.")
        _, prompt, _ = _build_agent_and_prompt(config, None, {}, {})
        assert prompt == ""


# ---------------------------------------------------------------------------
# §1.3 / §1.4 Tool conversion and execution
# ---------------------------------------------------------------------------


class TestToolConversion:
    def test_all_fields_forwarded(self) -> None:
        tools = _build_agent_tools(
            {
                "my-tool": {
                    "description": "does stuff",
                    "parameters": {"type": "object"},
                }
            },
            {"my-tool": AsyncMock()},
        )
        assert len(tools) == 1
        assert tools[0].name == "my-tool"
        assert tools[0].description == "does stuff"

    def test_multiple_tools_all_included(self) -> None:
        tools = _build_agent_tools(
            {"tool-a": {"description": "a"}, "tool-b": {"description": "b"}},
            {"tool-a": AsyncMock(), "tool-b": AsyncMock()},
        )
        assert {t.name for t in tools} == {"tool-a", "tool-b"}

    def test_empty_tools_no_tools_built(self) -> None:
        assert _build_agent_tools({}, {}) == []

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

    async def test_tool_not_found_throws(self) -> None:
        tools = _build_agent_tools({"my-tool": {"description": "d"}}, {})
        with pytest.raises(ValueError, match="No handler"):
            await tools[0].on_invoke_tool(MagicMock(), "{}")

    async def test_tool_handler_throws_propagates(self) -> None:
        async def _bad(args: Any) -> str:
            raise RuntimeError("handler error")

        tools = _build_agent_tools({"my-tool": {"description": "d"}}, {"my-tool": _bad})
        with pytest.raises(RuntimeError, match="handler error"):
            await tools[0].on_invoke_tool(MagicMock(), "{}")


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop (pre-span-era; restored, not telemetry)
# ---------------------------------------------------------------------------


class TestToolExecutionLoop:
    async def test_tool_not_found_execute_callback_throws(self) -> None:
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
# TELEMETRY-CONTRACT.md section 1: span tree
# ---------------------------------------------------------------------------


class TestSpanTree:
    async def test_opens_a_root_span_named_invoke_agent(self) -> None:
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("hi")}])
        )
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        assert rec.root.name == "invoke_agent"
        assert rec.root.attributes["gen_ai.operation.name"] == "invoke_agent"

    async def test_emits_one_chat_child_per_model_turn(self) -> None:
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("hi")}])
        )
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        chats = rec.named("chat ")
        assert len(chats) == 1
        assert chats[0].name == "chat gpt-4o"
        assert chats[0].attributes["gen_ai.operation.name"] == "chat"
        assert chats[0].context == ("context-of", rec.root)

    async def test_emits_a_chat_span_per_turn_of_a_tool_loop(self) -> None:
        turns = [
            {
                "output": [_tool_call_output("search", "call_1")],
                "tool_calls": [{"name": "search", "id": "call_1"}],
            },
            {"output": _text_output("done")},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(
                CONFIG, "q", {"search": AsyncMock()}, {}
            )
        assert len(rec.named("chat ")) == 2

    async def test_emits_an_execute_tool_span_per_tool_call(self) -> None:
        turns = [
            {
                "output": [_tool_call_output("search", "call_1")],
                "tool_calls": [{"name": "search", "id": "call_1"}],
            },
            {"output": _text_output("done")},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(
                CONFIG, "q", {"search": AsyncMock()}, {}
            )
        tools = rec.named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].name == "execute_tool search"
        assert tools[0].attributes["gen_ai.operation.name"] == "execute_tool"
        assert tools[0].attributes["gen_ai.tool.name"] == "search"
        assert tools[0].attributes["gen_ai.tool.call.id"] == "call_1"

    async def test_tool_spans_are_siblings_of_chat_not_children(self) -> None:
        turns = [
            {
                "output": [_tool_call_output("search", "call_1")],
                "tool_calls": [{"name": "search", "id": "call_1"}],
            },
            {"output": _text_output("done")},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(
                CONFIG, "q", {"search": AsyncMock()}, {}
            )
        assert rec.named("execute_tool ")[0].context == ("context-of", rec.root)

    async def test_every_span_is_ended(self) -> None:
        turns = [
            {
                "output": [_tool_call_output("search", "call_1")],
                "tool_calls": [{"name": "search", "id": "call_1"}],
            },
            {"output": _text_output("done")},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(
                CONFIG, "q", {"search": AsyncMock()}, {}
            )
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)

    async def test_children_carry_no_launchdarkly_attributes(self) -> None:
        turns = [
            {
                "output": [_tool_call_output("search", "call_1")],
                "tool_calls": [{"name": "search", "id": "call_1"}],
            },
            {"output": _text_output("done")},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        vs = {"__ld": {"configKey": "c", "variationKey": "v", "runId": "r"}}
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(
                CONFIG, "q", {"search": AsyncMock()}, vs
            )
        for span in rec.spans[1:]:
            assert [k for k in span.attributes if k.startswith("launchdarkly.")] == []
        assert [n for n, _ in rec.root.events] == ["feature_flag"]
        assert all(
            "feature_flag" not in [n for n, _ in s.events] for s in rec.spans[1:]
        )


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md sections 2 / 2a: root span
# ---------------------------------------------------------------------------


class TestRootSpanAttributes:
    async def test_writes_both_provider_keys_and_the_requested_model(self) -> None:
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("hi")}])
        )
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        attrs = rec.root.attributes
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-4o"

    async def test_response_model_is_the_requested_name_not_a_resolved_one(
        self,
    ) -> None:
        # Section 2a: openai-agents never resolves an answering model.
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("hi")}])
        )
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        assert rec.root.attributes["gen_ai.response.model"] == "gpt-4o"

    async def test_run_totals_on_the_root(self) -> None:
        turns = [
            {"output": _text_output("a"), "usage": _usage(10, 5)},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        attrs = rec.root.attributes
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.usage.total_tokens"] == 15

    async def test_feature_flag_event_on_root_only(self) -> None:
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("hi")}])
        )
        vs = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
                "environmentId": "env-1",
            }
        }
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, vs)
        assert rec.root.events == [
            (
                "feature_flag",
                {
                    "feature_flag.key": "my-config",
                    "feature_flag.provider.name": "LaunchDarkly",
                    "feature_flag.set.id": "env-1",
                },
            )
        ]
        assert rec.root.attributes["launchdarkly.config.key"] == "my-config"
        assert rec.root.attributes["launchdarkly.variation.key"] == "v1"
        assert rec.root.attributes["launchdarkly.run.id"] == "run-abc"


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md sections 3, 5a, 8: chat span attributes
# ---------------------------------------------------------------------------


class TestChatSpanAttributes:
    async def test_passes_input_through_without_folding_cache_into_it(self) -> None:
        # Section 8: OpenAI already counts cached tokens inside input_tokens.
        turns = [{"output": _text_output("hi"), "usage": _usage(50, 5, cached=30)}]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert chat.attributes["gen_ai.usage.input_tokens"] == 50
        assert chat.attributes["gen_ai.usage.output_tokens"] == 5
        assert chat.attributes["gen_ai.usage.total_tokens"] == 55
        assert chat.attributes["gen_ai.usage.cache_read.input_tokens"] == 30
        assert chat.attributes["gen_ai.usage.cache_creation.input_tokens"] == 0

    async def test_derives_total_from_input_plus_output_not_the_provider_total(
        self,
    ) -> None:
        turns = [{"output": _text_output("hi"), "usage": _usage(10, 5)}]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert chat.attributes["gen_ai.usage.total_tokens"] == 15

    async def test_finish_reason_tool_calls_when_output_holds_a_function_call(
        self,
    ) -> None:
        turns = [
            {
                "output": [_tool_call_output("search", "call_1")],
                "tool_calls": [{"name": "search", "id": "call_1"}],
            },
            {"output": _text_output("done")},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(
                CONFIG, "q", {"search": AsyncMock()}, {}
            )
        chats = rec.named("chat ")
        assert chats[0].attributes["gen_ai.response.finish_reasons"] == ["tool_calls"]
        assert chats[1].attributes["gen_ai.response.finish_reasons"] == ["stop"]

    async def test_omits_the_finish_reason_when_there_is_no_output(self) -> None:
        turns = [{"output": []}]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert "gen_ai.response.finish_reasons" not in chat.attributes

    async def test_fails_the_chat_span_when_the_model_turn_throws(self) -> None:
        err = RuntimeError("model down")
        turns = [{"output": [], "error": err}]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with (
            ctx,
            _patched_agents(agents_mod),
            pytest.raises(RuntimeError, match="model down"),
        ):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        chat = rec.named("chat ")[0]
        assert chat.exceptions == [err]
        from opentelemetry.trace import StatusCode

        assert StatusCode.ERROR in chat.statuses
        assert chat.ended == 1

    async def test_root_still_reports_partial_usage_on_failure(self) -> None:
        err = RuntimeError("second turn failed")
        turns = [
            {"output": _text_output("partial"), "usage": _usage(10, 5)},
        ]

        async def run(agent: Any, prompt: str, hooks: Any = None, **kw: Any) -> Any:
            await _drive_turns(hooks, agent, prompt, turns)
            await hooks.on_llm_start(MagicMock(), agent, agent.instructions, prompt)
            raise err

        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=run)
        with (
            ctx,
            _patched_agents(agents_mod),
            pytest.raises(RuntimeError, match="second turn failed"),
        ):
            await create_openai_agent_handler()(CONFIG, "q", {}, {})
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 10
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 5
        from opentelemetry.trace import StatusCode

        assert StatusCode.ERROR in rec.root.statuses


class TestToolSpanAttributes:
    async def test_fails_an_open_tool_span_when_the_run_crashes_mid_flight(
        self,
    ) -> None:
        err = RuntimeError("run boom")
        turns = [
            {
                "output": [_tool_call_output("search", "call_1")],
                "tool_calls": [{"name": "search", "id": "call_1", "error": err}],
            }
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with (
            ctx,
            _patched_agents(agents_mod),
            pytest.raises(RuntimeError, match="run boom"),
        ):
            await create_openai_agent_handler()(
                CONFIG, "q", {"search": AsyncMock()}, {}
            )
        tool_span = rec.named("execute_tool ")[0]
        from opentelemetry.trace import StatusCode

        assert StatusCode.ERROR in tool_span.statuses
        assert tool_span.ended == 1


# ---------------------------------------------------------------------------
# TELEMETRY-CONTRACT.md section 7: content capture
# ---------------------------------------------------------------------------


class TestContentCapture:
    async def test_emits_no_content_at_all_by_default(self) -> None:
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("answer")}])
        )
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler()(CONFIG, "my question", {}, {})
        for span in rec.spans:
            assert [k for k in span.attributes if k.startswith("gen_ai.prompt")] == []
            assert [
                k for k in span.attributes if k.startswith("gen_ai.completion")
            ] == []
            assert "gen_ai.input.messages" not in span.attributes
            assert "gen_ai.output.messages" not in span.attributes
            assert [n for n, _ in span.events if n.startswith("gen_ai.content")] == []

    async def test_puts_prompt_and_completion_on_spans_when_enabled(self) -> None:
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("answer")}])
        )
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler(capture_content=True)(
                CONFIG, "my question", {}, {}
            )
        assert "gen_ai.input.messages" in rec.root.attributes
        assert "gen_ai.output.messages" in rec.root.attributes
        prompt_written = json.dumps(rec.root.attributes)
        assert "my question" in prompt_written

    async def test_tool_call_arguments_and_result_gated(self) -> None:
        turns = [
            {
                "output": [_tool_call_output("search", "call_1", '{"q": "x"}')],
                "tool_calls": [
                    {
                        "name": "search",
                        "id": "call_1",
                        "args": {"q": "x"},
                        "result": "found",
                    }
                ],
            },
            {"output": _text_output("done")},
        ]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=_make_run(turns))
        with ctx, _patched_agents(agents_mod):
            await create_openai_agent_handler(capture_content=True)(
                CONFIG, "q", {"search": AsyncMock()}, {}
            )
        tool_span = rec.named("execute_tool ")[0]
        assert tool_span.attributes["gen_ai.tool.call.arguments"] == '{"q": "x"}'
        assert tool_span.attributes["gen_ai.tool.call.result"] == "found"


# ---------------------------------------------------------------------------
# §1.6 Error handling (top-level, no partial usage)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_records_exception_sets_error_ends_span_and_rethrows(self) -> None:
        err = RuntimeError("provider error")

        async def run(agent: Any, prompt: str, hooks: Any = None, **kw: Any) -> Any:
            raise err

        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=run)
        with (
            ctx,
            _patched_agents(agents_mod),
            pytest.raises(RuntimeError, match="provider error"),
        ):
            await create_openai_agent_handler()(CONFIG, "hi", {}, {})
        assert rec.root.exceptions == [err]
        from opentelemetry.trace import StatusCode

        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_usage_from_agents_exception_reaches_the_root(self) -> None:
        err = RuntimeError("max turns exceeded")
        err.run_data = SimpleNamespace(
            context_wrapper=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=20)
            )
        )

        async def run(agent: Any, prompt: str, hooks: Any = None, **kw: Any) -> Any:
            raise err

        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run=run)
        with ctx, _patched_agents(agents_mod), pytest.raises(RuntimeError):
            await create_openai_agent_handler()(CONFIG, "hi", {}, {})
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 100
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 20


# ---------------------------------------------------------------------------
# §1.7 Convenience export
# ---------------------------------------------------------------------------


class TestConvenienceExport:
    def test_calls_through_to_model_call(self) -> None:
        assert callable(openai_agents)

    def test_passes_config_key_user_input_and_context(self) -> None:
        import inspect

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
            ctx = {"kind": "user", "key": "u1"}
            openai_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        assert call_kwargs["handler"].provides_for == ("OpenAI", "agent")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self) -> None:
        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            ctx = {"kind": "user", "key": "u1"}
            openai_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_stream_is_defined(self) -> None:
        assert hasattr(create_openai_agent_handler(), "stream")

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

    async def test_yields_chunks_then_exactly_one_done_event(self) -> None:
        turns = [{"deltas": ["Hello", " world"], "output": _text_output("Hello world")}]
        agents_mod = _fake_agents_module(
            run_streamed=_make_run_streamed(turns, "Hello world")
        )
        with _patched_agents(agents_mod):
            events = [
                e
                async for e in await create_openai_agent_handler().stream(
                    CONFIG, "q", {}, {}
                )
            ]
        chunks = [e["text"] for e in events if e["type"] == "chunk"]
        assert chunks == ["Hello", " world"]
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert done[0]["output"] == "Hello world"

    async def test_opens_the_same_root_span_name_as_the_blocking_path(self) -> None:
        ctx, rec = _recording()
        turns = [{"deltas": ["hi"], "output": _text_output("hi")}]
        agents_mod = _fake_agents_module(run_streamed=_make_run_streamed(turns, "hi"))
        with ctx, _patched_agents(agents_mod):
            async for _ in await create_openai_agent_handler().stream(
                CONFIG, "q", {}, {}
            ):
                pass
        assert rec.root.name == "invoke_agent"
        assert len(rec.named("chat ")) == 1

    async def test_ends_every_span_once_when_the_stream_completes(self) -> None:
        ctx, rec = _recording()
        turns = [{"deltas": ["hi"], "output": _text_output("hi")}]
        agents_mod = _fake_agents_module(run_streamed=_make_run_streamed(turns, "hi"))
        with ctx, _patched_agents(agents_mod):
            async for _ in await create_openai_agent_handler().stream(
                CONFIG, "q", {}, {}
            ):
                pass
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)

    async def test_an_abandoned_stream_ends_every_span_but_stays_unset(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        turns = [{"deltas": ["one", "two", "three"], "output": _text_output("done")}]
        streamed_holder: dict[str, Any] = {}

        def run_streamed(agent: Any, prompt: str, hooks: Any = None, **kw: Any) -> Any:
            streamed = FakeStreamedResult(agent, prompt, hooks, turns, "done")
            streamed_holder["streamed"] = streamed
            return streamed

        agents_mod = _fake_agents_module(run_streamed=run_streamed)
        with ctx, _patched_agents(agents_mod):
            gen = await create_openai_agent_handler().stream(CONFIG, "q", {}, {})
            async for _ in gen:
                break
            await gen.aclose()

        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)
        assert rec.root.attributes["launchdarkly.stream.abandoned"] is True
        assert StatusCode.ERROR not in rec.root.statuses
        assert rec.root.exceptions == []
        # The chat span left open by the abandoned turn is also ended cleanly, not failed.
        chat = rec.named("chat ")[0]
        assert StatusCode.ERROR not in chat.statuses
        # Cancelling the vendor's own run is the other half of teardown: ending our span does not
        # stop the Runner's background task from spending more tokens.
        assert streamed_holder["streamed"].cancelled is True

    async def test_fails_the_spans_when_the_stream_raises(self) -> None:
        from opentelemetry.trace import StatusCode

        err = RuntimeError("stream died")
        turns = [{"deltas": [], "output": [], "error": err}]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run_streamed=_make_run_streamed(turns, ""))
        with (
            ctx,
            _patched_agents(agents_mod),
            pytest.raises(RuntimeError, match="stream died"),
        ):
            async for _ in await create_openai_agent_handler().stream(
                CONFIG, "q", {}, {}
            ):
                pass
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    async def test_emits_no_content_by_default_on_the_streaming_path(self) -> None:
        turns = [{"deltas": ["hi"], "output": _text_output("hi")}]
        ctx, rec = _recording()
        agents_mod = _fake_agents_module(run_streamed=_make_run_streamed(turns, "hi"))
        with ctx, _patched_agents(agents_mod):
            async for _ in await create_openai_agent_handler().stream(
                CONFIG, "q", {}, {}
            ):
                pass
        for span in rec.spans:
            assert [k for k in span.attributes if k.startswith("gen_ai.prompt")] == []
            assert [n for n, _ in span.events if n.startswith("gen_ai.content")] == []


# ---------------------------------------------------------------------------
# §1.9 Output format (build_output_type)
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_absent_output_format_no_change(self) -> None:
        assert build_output_type(None) is None

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

    async def test_output_format_returns_parsed_object_from_result(self) -> None:
        agents_mod = _fake_agents_module(
            run=_make_run([{"output": _text_output("ignored")}])
        )

        async def run(agent: Any, prompt: str, hooks: Any = None, **kw: Any) -> Any:
            await _drive_turns(
                hooks, agent, prompt, [{"output": _text_output("ignored")}]
            )
            return FakeRunResult({"score": 9})

        agents_mod.Runner.run = staticmethod(run)
        with _patched_agents(agents_mod):
            result = await create_openai_agent_handler()(
                _make_config(instructions="x", outputFormat={"type": "object"}), "hello"
            )
        assert result["output"] == {"score": 9}


# ---------------------------------------------------------------------------
# §1.2 Path C — None user_input must not produce None prompt
# ---------------------------------------------------------------------------


class TestNoneUserInput:
    """TESTING.md §1.2 Path C: When user_input is None, the prompt passed to
    Runner.run must be '' (empty string), not None."""

    async def test_none_user_input_instructions_path_prompt_is_empty_string(
        self,
    ) -> None:
        """When instructions path is taken and user_input=None, the prompt
        forwarded to Runner.run must be '' not None."""
        captured_prompts: list[Any] = []

        run_result = _make_run_result("ok")
        agents_mock = _mock_agents_module(run_result)

        # `hooks` is accepted (and ignored) here because `_call_impl` now always passes
        # `hooks=hooks` to `Runner.run` — a genuine signature change from the pre-span-work
        # handler this test predates, per the assignment's adaptation rule.
        async def _spy_run(agent: Any, prompt: Any, hooks: Any = None) -> Any:
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

    def test_history_appended_to_instructions(self) -> None:
        config = _make_config(instructions="Be concise.")
        _, _, instructions = _build_agent_and_prompt(
            config, "hi", {}, {}, self.SAMPLE_HISTORY
        )
        assert instructions is not None
        assert "Conversation History:" in instructions
        assert "Be concise." in instructions

    def test_history_format_is_correct(self) -> None:
        config = _make_config(instructions="Be helpful.")
        _, _, instructions = _build_agent_and_prompt(
            config, "hi", {}, {}, self.SAMPLE_HISTORY
        )
        assert instructions is not None
        assert "user: What is feature flagging?" in instructions
        assert "assistant: Feature flagging is a technique..." in instructions

    def test_empty_history_treated_like_no_history(self) -> None:
        config = _make_config(instructions="Be concise.")
        _, _, instr_with_empty = _build_agent_and_prompt(config, "hi", {}, {}, [])
        _, _, instr_without = _build_agent_and_prompt(config, "hi", {}, {})
        assert instr_with_empty == instr_without
        assert "Conversation History:" not in (instr_with_empty or "")

    def test_history_without_prior_instructions(self) -> None:
        config = _make_config()
        _, _, instructions = _build_agent_and_prompt(
            config, "hi", {}, {}, self.SAMPLE_HISTORY
        )
        assert instructions is not None
        assert "Conversation History:" in instructions
        assert "user: What is feature flagging?" in instructions
