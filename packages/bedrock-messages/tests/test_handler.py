"""Contract tests for the Bedrock Converse messages handler."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from launchdarkly_ai_bedrock_messages import create_bedrock_messages_handler
from launchdarkly_ai_bedrock_messages import handler as handler_module

BASE_CONFIG: dict[str, Any] = {
    "model": {
        "name": "anthropic.claude-sonnet-4-5",
        "region": "us",
        "parameters": {"maxTokens": 256, "temperature": 0.2},
        "custom": {"mustNeverLeak": "sentinel"},
    },
    "provider": {"name": "Bedrock"},
    "instructions": "Be concise.",
}


def response(text: str = "hello") -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
    }


class TestFactoryAndModelId:
    async def test_routes_as_bedrock_messages(self) -> None:
        client = MagicMock()
        client.converse = AsyncMock(return_value=response())

        handler = create_bedrock_messages_handler(client=client)

        assert handler.provides_for == ("Bedrock", "messages")
        result = await handler(BASE_CONFIG, "hello")
        assert result["output"] == "hello"
        assert result["usage"] == {
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
        }

    @pytest.mark.parametrize(
        ("region", "name", "expected"),
        [
            ("us", "anthropic.claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-5"),
            ("us", "us.anthropic.claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-5"),
            (None, "anthropic.claude-sonnet-4-5", "anthropic.claude-sonnet-4-5"),
        ],
    )
    async def test_short_region_prefix_is_prepended_once(
        self, region: str | None, name: str, expected: str
    ) -> None:
        client = MagicMock()
        client.converse = AsyncMock(return_value=response())
        config = {
            **BASE_CONFIG,
            "model": {**BASE_CONFIG["model"], "name": name, "region": region},
        }

        await create_bedrock_messages_handler(client=client)(config, "hello")

        assert client.converse.await_args.kwargs["modelId"] == expected

    async def test_does_not_double_prepend_matching_region_prefix(self) -> None:
        client = MagicMock()
        client.converse = AsyncMock(return_value=response())
        config = {
            **BASE_CONFIG,
            "model": {
                **BASE_CONFIG["model"],
                "region": "us",
                "name": "us.anthropic.claude-sonnet-4-5",
            },
        }

        await create_bedrock_messages_handler(client=client)(config, "hello")

        model_id = client.converse.await_args.kwargs["modelId"]
        assert model_id == "us.anthropic.claude-sonnet-4-5"
        assert model_id != "us.us.anthropic.claude-sonnet-4-5"


class TestOptionsAndClientEscapeHatch:
    async def test_options_callback_exposes_converse_without_mapping_custom(
        self,
    ) -> None:
        client = MagicMock()
        client.converse = AsyncMock(return_value=response())
        options = MagicMock(
            return_value={
                "guardrailConfig": {
                    "guardrailIdentifier": "guardrail-id",
                    "guardrailVersion": "1",
                },
                "requestMetadata": {"tenant": "acme"},
            }
        )

        await create_bedrock_messages_handler(client=client, converse_options=options)(
            BASE_CONFIG, "hello"
        )

        options.assert_called_once_with(BASE_CONFIG)
        request = client.converse.await_args.kwargs
        assert request["guardrailConfig"]["guardrailIdentifier"] == "guardrail-id"
        assert request["requestMetadata"] == {"tenant": "acme"}
        assert "mustNeverLeak" not in repr(request)

    async def test_handler_critical_fields_override_options_callback(self) -> None:
        client = MagicMock()
        client.converse = AsyncMock(return_value=response())

        await create_bedrock_messages_handler(
            client=client,
            converse_options=lambda _config: {
                "modelId": "wrong",
                "messages": [],
                "system": [],
            },
        )(BASE_CONFIG, "hello")

        request = client.converse.await_args.kwargs
        assert request["modelId"] == "us.anthropic.claude-sonnet-4-5"
        assert request["messages"]
        assert request["system"] == [{"text": "Be concise."}]

    async def test_injected_async_client_is_used_and_not_closed(self) -> None:
        client = MagicMock()
        client.converse = AsyncMock(return_value=response())
        client.close = AsyncMock()

        handler = create_bedrock_messages_handler(
            client=client,
            api_key="must-not-reconfigure-client",
            region="us-east-1",
        )
        await handler(BASE_CONFIG, "hello")

        client.converse.assert_awaited_once()
        client.close.assert_not_awaited()

    async def test_injected_sync_boto3_client_runs_off_event_loop(self) -> None:
        event_loop_thread = threading.get_ident()
        called_on: list[int] = []
        client = MagicMock()

        def converse(**_kwargs: Any) -> dict[str, Any]:
            called_on.append(threading.get_ident())
            return response()

        client.converse = converse

        result = await create_bedrock_messages_handler(client=client)(
            BASE_CONFIG, "hello"
        )

        assert result["output"] == "hello"
        assert called_on and called_on[0] != event_loop_thread


class TestToolsAndStreaming:
    async def test_converse_tool_use_executes_handler_and_submits_result(self) -> None:
        client = MagicMock()
        client.converse = AsyncMock(
            side_effect=[
                {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "toolUse": {
                                        "toolUseId": "tool-1",
                                        "name": "lookup",
                                        "input": {"id": 42},
                                    }
                                }
                            ],
                        }
                    },
                    "stopReason": "tool_use",
                    "usage": {
                        "inputTokens": 4,
                        "outputTokens": 2,
                        "totalTokens": 6,
                    },
                },
                response("done"),
            ]
        )
        config = {
            **BASE_CONFIG,
            "tools": {
                "lookup": {
                    "name": "lookup",
                    "description": "Look up an item",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                }
            },
        }
        lookup = AsyncMock(return_value={"value": "found"})

        result = await create_bedrock_messages_handler(client=client)(
            config, "hello", {"lookup": lookup}
        )

        lookup.assert_awaited_once_with({"id": 42})
        second_messages = client.converse.await_args_list[1].kwargs["messages"]
        assert second_messages[-1]["content"][0]["toolResult"]["toolUseId"] == "tool-1"
        assert result["output"] == "done"

    async def test_stream_emits_text_and_terminal_usage(self) -> None:
        client = MagicMock()

        async def events():
            yield {"contentBlockDelta": {"delta": {"text": "hel"}}}
            yield {"contentBlockDelta": {"delta": {"text": "lo"}}}
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 5,
                        "outputTokens": 2,
                        "totalTokens": 7,
                    }
                }
            }

        client.converse_stream = AsyncMock(return_value={"stream": events()})
        handler = create_bedrock_messages_handler(client=client)

        stream = await handler.stream(BASE_CONFIG, "hello")
        output = [event async for event in stream]

        assert output == [
            {"type": "chunk", "text": "hel"},
            {"type": "chunk", "text": "lo"},
            {
                "type": "done",
                "output": "hello",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            },
        ]

    async def test_abandoned_stream_closes_model_and_root_spans(
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
        monkeypatch.setattr(handler_module, "parent_context_of", lambda _span: None)

        async def events():
            yield {"contentBlockDelta": {"delta": {"text": "first"}}}
            await asyncio.Event().wait()

        client = MagicMock()
        client.converse_stream = AsyncMock(return_value={"stream": events()})
        provider_stream = await create_bedrock_messages_handler(client=client).stream(
            BASE_CONFIG, "hello"
        )

        assert await anext(provider_stream) == {"type": "chunk", "text": "first"}
        await provider_stream.aclose()

        assert [span.name for span in spans] == ["root", "model"]
        assert all(not span.is_recording() for span in spans)
        assert all(
            span.attributes["launchdarkly.stream.abandoned"] is True for span in spans
        )

    async def test_cancelled_stream_tool_closes_tool_and_root_spans(
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

        async def events():
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "tool-1", "name": "lookup"}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": "{}"}},
                }
            }
            yield {"messageStop": {"stopReason": "tool_use"}}
            yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}

        entered = asyncio.Event()

        async def lookup(_args: dict[str, Any]) -> None:
            entered.set()
            await asyncio.Event().wait()

        config = {
            **BASE_CONFIG,
            "tools": {
                "lookup": {
                    "description": "lookup",
                    "parameters": {"type": "object"},
                }
            },
        }
        client = MagicMock()
        client.converse_stream = AsyncMock(return_value={"stream": events()})
        provider_stream = await create_bedrock_messages_handler(client=client).stream(
            config, "hello", {"lookup": lookup}
        )
        pending = asyncio.create_task(anext(provider_stream))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        assert [span.name for span in spans] == ["root", "model", "tool"]
        assert all(not span.is_recording() for span in spans)
        assert spans[0].attributes["launchdarkly.run.cancelled"] is True
        assert spans[2].attributes["launchdarkly.run.cancelled"] is True


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
    import launchdarkly_ai_bedrock_messages.spans as spans_mod

    recorder = SpanRecorder()
    return patch.object(spans_mod, "trace", recorder), recorder


class TestTelemetryContract:
    async def test_root_and_chat_identify_bedrock_and_resolved_model(self) -> None:
        client = MagicMock()
        client.converse = AsyncMock(return_value=response())
        ctx, rec = _recording()

        with ctx:
            await create_bedrock_messages_handler(client=client)(BASE_CONFIG, "hello")

        root = rec.named("invoke_agent")[0]
        chat = rec.named("chat ")[0]
        model_id = "us.anthropic.claude-sonnet-4-5"
        assert root.name == "invoke_agent"
        assert chat.name == f"chat {model_id}"
        for span in (root, chat):
            assert span.attributes["gen_ai.system"] == "aws.bedrock"
            assert span.attributes["gen_ai.provider.name"] == "aws.bedrock"
            assert span.attributes["gen_ai.request.model"] == model_id
            assert span.attributes["gen_ai.response.model"] == model_id
        assert chat.attributes["gen_ai.response.finish_reasons"] == ["stop"]
        assert chat.context == ("context-of", root)

    async def test_folds_cache_tokens_into_span_input_but_returns_raw_usage(
        self,
    ) -> None:
        client = MagicMock()
        client.converse = AsyncMock(
            return_value={
                **response(),
                "usage": {
                    "inputTokens": 3,
                    "outputTokens": 10,
                    "totalTokens": 13,
                    "cacheReadInputTokens": 19971,
                    "cacheWriteInputTokens": 3580,
                },
            }
        )
        ctx, rec = _recording()

        with ctx:
            result = await create_bedrock_messages_handler(client=client)(
                BASE_CONFIG, "hello"
            )

        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 23554
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 19971
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 3580
        assert attrs["gen_ai.usage.total_tokens"] == 23564
        assert attrs["gen_ai.usage.prompt_tokens"] == 23554
        assert result["usage"]["input_tokens"] == 3
        assert result["usage"]["cache_read_input_tokens"] == 19971
        assert result["usage"]["cache_creation_input_tokens"] == 3580

    async def test_stream_writes_the_same_content_carriers_as_invoke(self) -> None:
        async def events():
            yield {"contentBlockDelta": {"delta": {"text": "hello"}}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}}

        client = MagicMock()
        client.converse_stream = AsyncMock(return_value={"stream": events()})
        ctx, rec = _recording()

        with ctx:
            stream = await create_bedrock_messages_handler(
                client=client, capture_content=True
            ).stream(BASE_CONFIG, "hello")
            async for _event in stream:
                pass

        chat = rec.named("chat ")[0]
        root = rec.named("invoke_agent")[0]
        assert "gen_ai.input.messages" in chat.attributes
        assert "gen_ai.output.messages" in chat.attributes
        assert "gen_ai.output.messages" in root.attributes
        assert "gen_ai.prompt.0.role" in chat.attributes
        assert chat.attributes["gen_ai.response.finish_reasons"] == ["stop"]


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
