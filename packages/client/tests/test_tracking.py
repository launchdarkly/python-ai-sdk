"""
Tests for wrap_tool_handlers and the handler variables execute_and_track builds.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import NATIVE_TOOL_KEY, NativeTool, wrap_tool_handlers
from launchdarkly_ai_server.tracking import execute_and_track


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.track = MagicMock()
    return client


CONTEXT = {"kind": "user", "key": "user-1"}
TRACK_DATA = {
    "runId": "abc",
    "configKey": "my-flag",
    "variationKey": "v1",
    "version": 1,
    "modelName": "gpt-4",
    "providerName": "OpenAI",
}


class TestWrapToolHandlers:
    async def test_regular_function_is_wrapped(self) -> None:
        mock_client = _make_mock_client()
        original = MagicMock(return_value="result")
        with patch("launchdarkly_ai_server.lifecycle._client", mock_client):
            wrapped = wrap_tool_handlers({"my_tool": original}, CONTEXT, TRACK_DATA)
            await wrapped["my_tool"]("arg1")
        mock_client.track.assert_called_once()
        original.assert_called_once_with("arg1")

    async def test_tracking_event_carries_correct_metadata(self) -> None:
        mock_client = _make_mock_client()
        with patch("launchdarkly_ai_server.lifecycle._client", mock_client):
            wrapped = wrap_tool_handlers({"my_tool": MagicMock()}, CONTEXT, TRACK_DATA)
            await wrapped["my_tool"]()
        call_args = mock_client.track.call_args
        assert call_args[0][0] == "$ld:ai:tool_call"
        assert call_args[0][2]["toolKey"] == "my_tool"

    async def test_return_value_is_preserved(self) -> None:
        mock_client = _make_mock_client()
        original = MagicMock(return_value=42)
        with patch("launchdarkly_ai_server.lifecycle._client", mock_client):
            wrapped = wrap_tool_handlers({"t": original}, CONTEXT, TRACK_DATA)
            result = await wrapped["t"]()
        assert result == 42

    def test_native_tool_becomes_tracking_stub(self) -> None:
        mock_client = _make_mock_client()
        native = NativeTool("WebSearch")
        with patch("launchdarkly_ai_server.lifecycle._client", mock_client):
            wrapped = wrap_tool_handlers({"search": native}, CONTEXT, TRACK_DATA)
            wrapped["search"]()
        mock_client.track.assert_called_once()
        call_args = mock_client.track.call_args
        assert call_args[0][0] == "$ld:ai:tool_call"

    def test_native_tool_instance_preserved_on_stub(self) -> None:
        mock_client = _make_mock_client()
        native = NativeTool("WebSearch")
        with patch("launchdarkly_ai_server.lifecycle._client", mock_client):
            wrapped = wrap_tool_handlers({"search": native}, CONTEXT, TRACK_DATA)
        stub = wrapped["search"]
        assert getattr(stub, NATIVE_TOOL_KEY) is native

    def test_handoff_prefix_skips_tracking(self) -> None:
        mock_client = _make_mock_client()
        original = MagicMock(return_value=None)
        with patch("launchdarkly_ai_server.lifecycle._client", mock_client):
            wrapped = wrap_tool_handlers(
                {"__handoff_leaf": original}, CONTEXT, TRACK_DATA
            )
            # Handoff tools stay sync so stream/invoke routing can record the choice
            # with a bare call (an async wrapper would leave `chosen` empty).
            wrapped["__handoff_leaf"]()
        mock_client.track.assert_not_called()
        original.assert_called_once()

    def test_undefined_tool_handlers(self) -> None:
        result = wrap_tool_handlers(None, CONTEXT, TRACK_DATA)
        assert result == {}


class TestHandlerVariables:
    @pytest.mark.parametrize(
        ("evaluated", "expected"),
        [(None, None), ({"raw": True}, {"raw": True})],
    )
    async def test_a_caller_supplied_ld_value_never_reaches_the_handler(
        self, evaluated: Any, expected: Any
    ) -> None:
        seen: list[dict[str, Any]] = []

        async def handler(
            config: Any,
            user_input: Any,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> dict[str, Any]:
            seen.append(dict(variables or {}))
            return {"output": "ok", "usage": {}}

        lifecycle_module._set_client_for_testing(_make_mock_client())
        try:
            await execute_and_track(
                config_key="c",
                config={"model": {"name": "m"}, "provider": {"name": "P"}},
                meta={"variationKey": "v", "version": 1},
                user_context={"kind": "user", "key": "u1"},
                handler=handler,  # type: ignore[arg-type]
                variables={"__ldValue": "from-caller"},
                evaluated_value=evaluated,
            )
        finally:
            lifecycle_module._reset_for_testing()

        assert seen[0].get("__ldValue") == expected


class TestCustomHandlerCapture:
    @pytest.mark.parametrize(
        ("explicit", "default", "expected"),
        [
            (False, True, '"v1"'),
            (True, False, '{"instructions": "secret prompt"}'),
        ],
    )
    async def test_a_custom_handler_s_explicit_setting_beats_the_default(
        self, explicit: bool, default: bool, expected: str
    ) -> None:
        """A custom handler's function calls the helper without a flag; its own setting beats the process default."""
        from launchdarkly_ai_server import create_handler
        from launchdarkly_ai_server.utils import set_ld_span_attributes

        events: list[dict[str, Any]] = []

        class Span:
            def set_attribute(self, *_a: Any) -> None:
                pass

            def add_event(self, name: str, attributes: dict[str, Any]) -> None:
                events.append(attributes)

        async def fn(
            config: Any,
            user_input: Any,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> dict[str, Any]:
            set_ld_span_attributes(Span(), variables)
            return {"output": "ok", "usage": {}}

        from launchdarkly_ai_server import SDKOptions, configure

        handler = create_handler(("Custom", "messages"), fn, None, explicit)
        lifecycle_module._set_client_for_testing(_make_mock_client())
        configure(options=SDKOptions(capture_content=default))
        try:
            await execute_and_track(
                config_key="c",
                config={"model": {"name": "m"}, "provider": {"name": "Custom"}},
                meta={"variationKey": "v1", "version": 1},
                user_context={"kind": "user", "key": "u1"},
                handler=handler,
                evaluated_value={"instructions": "secret prompt"},
            )
        finally:
            lifecycle_module._reset_for_testing()

        assert events[0]["feature_flag.result.value"] == expected
