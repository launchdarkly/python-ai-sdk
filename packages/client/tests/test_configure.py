"""Process-wide ``configure`` defaults.

``init_client`` stays the LaunchDarkly client and telemetry initializer.
``SDKOptions.capture_content`` is a default, not a ceiling: an explicit handler
override wins, including an explicit ``False`` over a configured ``True``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

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


def test_builtin_default_is_off() -> None:
    assert _handler().capture_content is False


def test_configure_is_read_after_the_handler_is_built() -> None:
    handler = _handler()
    configure(options=SDKOptions(capture_content=True))
    assert handler.capture_content is True


def test_configure_can_be_replaced_before_an_invocation() -> None:
    configure(options=SDKOptions(capture_content=True))
    configure(options=SDKOptions(capture_content=False))
    assert _handler().capture_content is False


def test_explicit_false_overrides_a_configured_true() -> None:
    configure(options=SDKOptions(capture_content=True))
    assert _handler(False).capture_content is False
    assert _handler(True).capture_content is True
    assert _handler(None).capture_content is True


def test_judge_gate_uses_the_effective_bool() -> None:
    # The same expression judges.py uses. No second content gate.
    configure(options=SDKOptions(capture_content=True))
    reasoning = "because"
    assert (reasoning if _handler().capture_content else None) == "because"
    assert (reasoning if _handler(False).capture_content else None) is None


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
        configure(options=SDKOptions(capture_content=True))


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
        configure(options=SDKOptions(capture_content=True))


async def test_a_missing_stream_does_not_latch() -> None:
    with pytest.raises(NotImplementedError):
        await _handler().stream({})
    configure(options=SDKOptions(capture_content=True))
    assert _handler().capture_content is True


async def test_shutdown_restores_the_builtin_default() -> None:
    configure(options=SDKOptions(capture_content=True))
    await _handler()({})
    await shutdown()
    assert _handler().capture_content is False
    configure(options=SDKOptions(capture_content=True))
    assert _handler().capture_content is True


async def test_init_client_does_not_carry_behavior_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stub = MagicMock()
    with (
        patch.object(lifecycle_module, "_setup_telemetry", return_value=None),
        caplog.at_level("WARNING", logger="launchdarkly_ai_server.lifecycle"),
    ):
        await init_client({"capture_content": True}, stub)
    assert _handler().capture_content is False
    assert any("capture_content" in record.getMessage() for record in caplog.records)
