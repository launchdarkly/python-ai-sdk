"""Process-wide ``configure`` defaults.

``init_client`` stays the LaunchDarkly client and telemetry initializer.
``SDKOptions.capture_content`` is a default, not a ceiling: an explicit handler
override wins, including an explicit ``False`` over a configured ``True``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import (
    SDKOptions,
    configure,
    init_client,
    shutdown,
)
from launchdarkly_ai_server.lifecycle import _reset_for_testing
from launchdarkly_ai_server.types import ProviderHandler


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    _reset_for_testing()
    yield
    _reset_for_testing()


async def _fn(
    config: Any,
    user_input: str | None = None,
    tool_handlers: dict[str, Any] | None = None,
    variables: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"output": "ok", "usage": {}}


def _handler(capture_content: bool | None = None) -> ProviderHandler:
    return ProviderHandler(_fn, capture_content=capture_content)


def test_builtin_default_is_on() -> None:
    assert SDKOptions().capture_content is True
    assert _handler().capture_content is True


def test_configure_is_read_after_the_handler_is_built() -> None:
    handler = _handler()
    configure(options=SDKOptions(capture_content=False))
    assert handler.capture_content is False


def test_configure_can_be_replaced_before_an_invocation() -> None:
    configure(options=SDKOptions(capture_content=True))
    configure(options=SDKOptions(capture_content=False))
    assert _handler().capture_content is False


def test_explicit_false_overrides_a_configured_true() -> None:
    configure(options=SDKOptions(capture_content=True))
    assert _handler(False).capture_content is False
    assert _handler(True).capture_content is True
    assert _handler(None).capture_content is True


def test_assigning_capture_content_sets_the_override() -> None:
    configure(options=SDKOptions(capture_content=True))
    handler = _handler()
    handler.capture_content = False
    assert handler.capture_content is False
    handler.capture_content = True
    assert handler.capture_content is True


def test_configure_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        configure(SDKOptions())  # type: ignore[misc]


def test_configure_rejects_a_non_options_value() -> None:
    with pytest.raises(TypeError, match="SDKOptions"):
        configure(options={"capture_content": True})  # type: ignore[arg-type]


def test_sdk_options_rejects_a_non_bool() -> None:
    with pytest.raises(TypeError, match="bool"):
        SDKOptions(capture_content=1)  # type: ignore[arg-type]


def test_sdk_options_is_frozen() -> None:
    options = SDKOptions()
    with pytest.raises(FrozenInstanceError):
        options.capture_content = True  # type: ignore[misc]


async def test_a_changed_configuration_is_rejected_after_invocation() -> None:
    await _handler()({})
    with pytest.raises(RuntimeError, match="cannot change"):
        configure(options=SDKOptions(capture_content=False))


async def test_an_identical_configuration_is_allowed_after_invocation() -> None:
    configure(options=SDKOptions(capture_content=True))
    await _handler()({})
    configure(options=SDKOptions(capture_content=True))
    assert _handler().capture_content is True


async def test_streaming_latches_invocation_before_the_body_runs() -> None:
    async def _stream(*_args: Any, **_kwargs: Any) -> Any:
        yield {"type": "done", "output": "", "usage": {}}

    handler = ProviderHandler(_fn, stream_fn=_stream)
    gen = await handler.stream({})
    await gen.aclose()
    with pytest.raises(RuntimeError, match="cannot change"):
        configure(options=SDKOptions(capture_content=False))


async def test_a_missing_stream_does_not_latch() -> None:
    with pytest.raises(NotImplementedError):
        await _handler().stream({})
    configure(options=SDKOptions(capture_content=False))
    assert _handler().capture_content is False


async def test_shutdown_restores_the_builtin_default() -> None:
    configure(options=SDKOptions(capture_content=False))
    await _handler()({})
    await shutdown()
    assert _handler().capture_content is True
    configure(options=SDKOptions(capture_content=False))
    assert _handler().capture_content is False


async def test_run_judges_writes_the_explanation_from_the_builtin_default() -> None:
    """The judge explanation follows the handler's effective capture flag.

    Goes through ``run_judges``, not a copy of the ``if capture_content``
    expression, and reads ``gen_ai.evaluation.explanation`` off the span.
    """
    import random

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from launchdarkly_ai_server import run_judges
    from launchdarkly_ai_server.conversation import ConversationIdSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(ConversationIdSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("@launchdarkly/ai-server")

    async def fn(
        config: Any,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        span = tracer.start_span("invoke_agent")
        span.end()
        return {
            "output": '{"score": 0.9, "reasoning": "good"}',
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    client = MagicMock()
    client.track = MagicMock()
    client.variation = AsyncMock(
        return_value={
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "judge",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "messages",
            },
        }
    )
    lifecycle_module._set_client_for_testing(client)
    parent = {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "hi",
        "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
    }

    def explanation() -> str | None:
        spans = [
            span
            for span in exporter.get_finished_spans()
            if span.name == "invoke_agent"
        ]
        assert spans, "judge invoke_agent span was not exported"
        value = (spans[-1].attributes or {}).get("gen_ai.evaluation.explanation")
        return value if isinstance(value, str) else None

    with patch.object(random, "random", return_value=0.0):
        await run_judges(
            config=parent,
            user_context={"kind": "user", "key": "u1"},
            handler=ProviderHandler(fn, provides_for=("TestProvider", "messages")),
            user_input="q",
            llm_response="r",
            base_track_data={"runId": "x"},
        )
    assert explanation() == "good"

    exporter.clear()
    with patch.object(random, "random", return_value=0.0):
        await run_judges(
            config=parent,
            user_context={"kind": "user", "key": "u1"},
            handler=ProviderHandler(
                fn, provides_for=("TestProvider", "messages"), capture_content=False
            ),
            user_input="q",
            llm_response="r",
            base_track_data={"runId": "x"},
        )
    assert explanation() is None


async def test_init_client_does_not_carry_behavior_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stub = MagicMock()
    with (
        patch.object(lifecycle_module, "_setup_telemetry", return_value=None),
        caplog.at_level("WARNING", logger="launchdarkly_ai_server.lifecycle"),
    ):
        await init_client({"capture_content": False}, stub)
    assert _handler().capture_content is True
    assert any("capture_content" in record.getMessage() for record in caplog.records)


def test_rewrapping_keeps_an_explicit_override() -> None:
    from launchdarkly_ai_server import create_handler

    inner = create_handler(("A", "messages"), _fn, None, False)
    outer = create_handler(("B", "messages"), inner)
    assert outer.capture_content is False
    assert create_handler(("B", "messages"), inner, None, True).capture_content is True


def test_rewrapping_an_unset_handler_still_follows_configure() -> None:
    from launchdarkly_ai_server import create_handler

    outer = create_handler(("B", "messages"), create_handler(("A", "messages"), _fn))
    configure(options=SDKOptions(capture_content=False))
    assert outer.capture_content is False


def test_rewrapping_keeps_the_stream() -> None:
    from launchdarkly_ai_server import create_handler

    async def stream(*_args: Any, **_kwargs: Any) -> Any:
        yield {"type": "done", "output": "", "usage": {}}

    inner = create_handler(("A", "messages"), _fn, stream)
    assert create_handler(("B", "messages"), inner)._stream_fn is stream


def test_rewrapping_keeps_the_providers_allowlist() -> None:
    """Re-wrapping a scoped wildcard handler without passing ``providers``
    must keep the inner allowlist, not silently become unscoped."""
    from launchdarkly_ai_server import create_handler

    inner = create_handler(("*", "messages"), _fn, providers=["Bedrock"])
    outer = create_handler(("*", "messages"), inner)
    assert outer.providers == ("Bedrock",)
    overridden = create_handler(("*", "messages"), inner, providers=["Gemini"])
    assert overridden.providers == ("Gemini",)
