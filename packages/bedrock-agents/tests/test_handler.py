"""Contract tests for the Bedrock Strands agents handler."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from launchdarkly_ai_bedrock_agents import create_bedrock_agents_handler
from launchdarkly_ai_bedrock_agents import handler as handler_module

BASE_CONFIG: dict[str, Any] = {
    "model": {
        "name": "anthropic.claude-sonnet-4-5",
        "region": "us",
        "parameters": {"max_tokens": 512, "temperature": 0.1},
        "custom": {"mustNeverLeak": "sentinel"},
    },
    "provider": {"name": "Bedrock"},
    "instructions": "Help {{name}}.",
}


def agent_result(text: str = "done") -> SimpleNamespace:
    return SimpleNamespace(
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=SimpleNamespace(
            accumulated_usage={
                "inputTokens": 11,
                "outputTokens": 4,
                "totalTokens": 15,
            }
        ),
    )


class TestFactory:
    def test_routes_as_bedrock_agent(self) -> None:
        with (
            patch("strands.models.BedrockModel"),
            patch("strands.Agent"),
        ):
            handler = create_bedrock_agents_handler()

        assert handler.provides_for == ("Bedrock", "agent")


class TestStrandsConstruction:
    async def test_constructs_bedrock_model_and_agent(self) -> None:
        model = MagicMock()
        agent = MagicMock()
        agent.invoke_async = AsyncMock(return_value=agent_result())

        with (
            patch("strands.models.BedrockModel", return_value=model) as model_cls,
            patch("strands.Agent", return_value=agent) as agent_cls,
        ):
            result = await create_bedrock_agents_handler()(
                BASE_CONFIG, "hello", {}, {"name": "Ada"}
            )

        model_options = model_cls.call_args.kwargs
        assert model_options["model_id"] == "us.anthropic.claude-sonnet-4-5"
        assert model_options["max_tokens"] == 512
        assert model_options["temperature"] == 0.1
        assert "mustNeverLeak" not in repr(model_options)
        agent_cls.assert_called_once()
        assert agent_cls.call_args.kwargs["model"] is model
        assert agent_cls.call_args.kwargs["system_prompt"] == "Help Ada."
        agent.invoke_async.assert_awaited_once_with("hello")
        assert result["output"] == "done"

    async def test_does_not_double_prepend_matching_region_prefix(self) -> None:
        agent = MagicMock()
        agent.invoke_async = AsyncMock(return_value=agent_result())
        config = {
            **BASE_CONFIG,
            "model": {
                **BASE_CONFIG["model"],
                "region": "us",
                "name": "us.anthropic.claude-sonnet-4-5",
            },
        }

        with (
            patch("strands.models.BedrockModel") as model_cls,
            patch("strands.Agent", return_value=agent),
        ):
            await create_bedrock_agents_handler()(config, "hello")

        model_id = model_cls.call_args.kwargs["model_id"]
        assert model_id == "us.anthropic.claude-sonnet-4-5"
        assert model_id != "us.us.anthropic.claude-sonnet-4-5"

    async def test_model_options_callback_exposes_strands_options(self) -> None:
        agent = MagicMock()
        agent.invoke_async = AsyncMock(return_value=agent_result())
        callback = MagicMock(
            return_value={
                "guardrail_id": "guardrail",
                "guardrail_version": "1",
                "additional_request_fields": {"thinking": {"type": "enabled"}},
            }
        )

        with (
            patch("strands.models.BedrockModel") as model_cls,
            patch("strands.Agent", return_value=agent),
        ):
            await create_bedrock_agents_handler(model_options=callback)(
                BASE_CONFIG, "hello"
            )

        callback.assert_called_once_with(BASE_CONFIG)
        options = model_cls.call_args.kwargs
        assert options["model_id"] == "us.anthropic.claude-sonnet-4-5"
        assert options["guardrail_id"] == "guardrail"
        assert options["additional_request_fields"]["thinking"]["type"] == "enabled"

    async def test_model_id_cannot_be_overridden_by_options_callback(self) -> None:
        agent = MagicMock()
        agent.invoke_async = AsyncMock(return_value=agent_result())

        with (
            patch("strands.models.BedrockModel") as model_cls,
            patch("strands.Agent", return_value=agent),
        ):
            await create_bedrock_agents_handler(
                model_options=lambda _config: {"model_id": "wrong"}
            )(BASE_CONFIG, "hello")

        assert (
            model_cls.call_args.kwargs["model_id"] == "us.anthropic.claude-sonnet-4-5"
        )


class TestClientEscapeHatch:
    async def test_preconstructed_boto3_client_is_installed_on_model(self) -> None:
        client = MagicMock(name="bedrock-runtime-client")
        model = MagicMock()
        internally_created_client = model.client
        agent = MagicMock()
        agent.invoke_async = AsyncMock(return_value=agent_result())

        with (
            patch("strands.models.BedrockModel", return_value=model) as model_cls,
            patch("strands.Agent", return_value=agent),
        ):
            await create_bedrock_agents_handler(
                client=client,
                api_key="must-not-reconfigure-client",
                region="us-east-1",
            )(BASE_CONFIG, "hello")

        assert model.client is client
        assert model.client is not internally_created_client
        assert "api_key" not in model_cls.call_args.kwargs
        assert "region_name" not in model_cls.call_args.kwargs

    async def test_boto_session_is_forwarded_without_raw_client(self) -> None:
        session = MagicMock(name="boto-session")
        agent = MagicMock()
        agent.invoke_async = AsyncMock(return_value=agent_result())

        with (
            patch("strands.models.BedrockModel") as model_cls,
            patch("strands.Agent", return_value=agent),
        ):
            await create_bedrock_agents_handler(boto_session=session)(
                BASE_CONFIG, "hello"
            )

        assert model_cls.call_args.kwargs["boto_session"] is session


class TestStreamingLifecycle:
    async def test_abandoned_stream_closes_model_tool_and_root_spans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spans: list[_RecordingSpan] = []
        monkeypatch.setattr(
            handler_module,
            "start_root_span",
            lambda *_args: _new_span(spans, "root"),
        )
        monkeypatch.setattr(
            handler_module,
            "start_model_span",
            lambda *_args: _new_span(spans, "model"),
        )
        monkeypatch.setattr(
            handler_module,
            "start_tool_span",
            lambda *_args: _new_span(spans, "tool"),
        )
        monkeypatch.setattr(handler_module, "parent_context_of", lambda _span: None)

        class FakeModel:
            client = MagicMock()

            async def stream(self, *_args: Any, **_kwargs: Any):
                yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}
                await asyncio.Event().wait()

        class FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                self.model = kwargs["model"]
                self.hook = kwargs["hooks"][0]

            async def stream_async(self, _prompt: Any):
                async for _event in self.model.stream([], None, None):
                    self.hook.before(
                        SimpleNamespace(
                            tool_use={
                                "toolUseId": "tool-1",
                                "name": "lookup",
                                "input": {},
                            }
                        )
                    )
                    yield {"data": "first"}
                    await asyncio.Event().wait()

        with (
            patch("strands.models.BedrockModel", return_value=FakeModel()),
            patch("strands.Agent", FakeAgent),
        ):
            provider_stream = await create_bedrock_agents_handler().stream(
                BASE_CONFIG, "hello"
            )
            assert await anext(provider_stream) == {"type": "chunk", "text": "first"}
            await provider_stream.aclose()

        assert [span.name for span in spans] == ["root", "model", "tool"]
        assert all(not span.is_recording() for span in spans)
        assert all(
            span.attributes["launchdarkly.stream.abandoned"] is True for span in spans
        )

    async def test_cancelled_stream_marks_model_tool_and_root_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spans: list[_RecordingSpan] = []
        monkeypatch.setattr(
            handler_module,
            "start_root_span",
            lambda *_args: _new_span(spans, "root"),
        )
        monkeypatch.setattr(
            handler_module,
            "start_model_span",
            lambda *_args: _new_span(spans, "model"),
        )
        monkeypatch.setattr(
            handler_module,
            "start_tool_span",
            lambda *_args: _new_span(spans, "tool"),
        )
        monkeypatch.setattr(handler_module, "parent_context_of", lambda _span: None)
        entered = asyncio.Event()

        class FakeModel:
            client = MagicMock()

            async def stream(self, *_args: Any, **_kwargs: Any):
                yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}
                await asyncio.Event().wait()

        class FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                self.model = kwargs["model"]
                self.hook = kwargs["hooks"][0]

            async def stream_async(self, _prompt: Any):
                async for _event in self.model.stream([], None, None):
                    self.hook.before(
                        SimpleNamespace(
                            tool_use={
                                "toolUseId": "tool-1",
                                "name": "lookup",
                                "input": {},
                            }
                        )
                    )
                    entered.set()
                    await asyncio.Event().wait()
                    yield {"data": "unreachable"}

        with (
            patch("strands.models.BedrockModel", return_value=FakeModel()),
            patch("strands.Agent", FakeAgent),
        ):
            provider_stream = await create_bedrock_agents_handler().stream(
                BASE_CONFIG, "hello"
            )
            pending = asyncio.create_task(anext(provider_stream))
            await entered.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

        assert [span.name for span in spans] == ["root", "model", "tool"]
        assert all(not span.is_recording() for span in spans)
        assert all(
            span.attributes["launchdarkly.run.cancelled"] is True for span in spans
        )


class RecordedSpan:
    def __init__(self, name: str, context: Any = None) -> None:
        self.name = name
        self.context = context
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.ended = 0

    def is_recording(self) -> bool:
        return self.ended == 0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, *_args: Any) -> None:
        pass

    def record_exception(self, _error: BaseException) -> None:
        pass

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

    def named(self, prefix: str) -> list[RecordedSpan]:
        return [span for span in self.spans if span.name.startswith(prefix)]


def _recording() -> Any:
    import launchdarkly_ai_bedrock_agents.spans as spans_mod

    recorder = SpanRecorder()
    return patch.object(spans_mod, "trace", recorder), recorder


class TestTelemetryContract:
    async def test_maps_cased_stop_reason_and_folds_cache_on_chat_span(self) -> None:
        class FakeModel:
            async def stream(self, *_args: Any, **_kwargs: Any):
                yield {
                    "contentBlockDelta": {"delta": {"text": "hi"}},
                }
                yield {
                    "metadata": {
                        "usage": {
                            "inputTokens": 3,
                            "outputTokens": 10,
                            "cacheReadInputTokens": 20,
                            "cacheWriteInputTokens": 2,
                        }
                    }
                }
                yield {"messageStop": {"stopReason": "End_Turn"}}

        class FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                self.model = kwargs["model"]

            async def invoke_async(self, prompt: Any) -> Any:
                async for _event in self.model.stream(
                    [{"role": "user", "content": [{"text": prompt}]}],
                    None,
                    "Help Ada.",
                ):
                    pass
                raise RuntimeError("provider failed after a billed turn")

        ctx, rec = _recording()
        with (
            ctx,
            patch("strands.models.BedrockModel", return_value=FakeModel()),
            patch("strands.Agent", FakeAgent),
        ):
            with pytest.raises(RuntimeError, match="billed turn"):
                await create_bedrock_agents_handler()(
                    BASE_CONFIG, "hello", {}, {"name": "Ada"}
                )

        root = rec.named("invoke_agent")[0]
        chat = rec.named("chat ")[0]
        assert chat.attributes["gen_ai.response.finish_reasons"] == ["stop"]
        assert chat.attributes["gen_ai.usage.input_tokens"] == 25
        assert root.attributes["gen_ai.usage.input_tokens"] == 25
        assert root.attributes["gen_ai.system"] == "aws.bedrock"
        assert chat.name == "chat us.anthropic.claude-sonnet-4-5"

    async def test_records_history_and_chat_content_when_capture_is_on(self) -> None:
        class FakeModel:
            async def stream(self, *_args: Any, **_kwargs: Any):
                yield {"contentBlockDelta": {"delta": {"text": "done"}}}
                yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}
                yield {"messageStop": {"stopReason": "end_turn"}}

        class FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                self.model = kwargs["model"]

            async def invoke_async(self, prompt: Any) -> Any:
                async for _event in self.model.stream(prompt, None, "Be concise."):
                    pass
                return agent_result()

        config = {
            **BASE_CONFIG,
            "instructions": None,
            "messages": [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "ok"},
            ],
        }
        ctx, rec = _recording()
        with (
            ctx,
            patch("strands.models.BedrockModel", return_value=FakeModel()),
            patch("strands.Agent", FakeAgent),
        ):
            await create_bedrock_agents_handler(capture_content=True)(config, "hello")

        root = rec.named("invoke_agent")[0]
        chat = rec.named("chat ")[0]
        assert "gen_ai.input.messages" in root.attributes
        assert "gen_ai.input.messages" in chat.attributes
        assert "gen_ai.output.messages" in chat.attributes
        assert "earlier" in root.attributes["gen_ai.input.messages"]


class _RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, *_args: Any) -> None:
        pass

    def record_exception(self, _error: BaseException) -> None:
        pass

    def end(self) -> None:
        self.ended = True

    def is_recording(self) -> bool:
        return not self.ended


def _new_span(spans: list[_RecordingSpan], name: str) -> _RecordingSpan:
    span = _RecordingSpan(name)
    spans.append(span)
    return span
