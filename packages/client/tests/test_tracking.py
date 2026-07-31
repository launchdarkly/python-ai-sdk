"""
Tests for §3.8 wrap_tool_handlers.
Reference: TESTING.md §3.8
"""

from unittest.mock import MagicMock, patch

from launchdarkly_ai_server import NATIVE_TOOL_KEY, NativeTool, wrap_tool_handlers


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
        assert call_args[0][2]["toolName"] == "my_tool"

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

    async def test_handoff_prefix_skips_tracking(self) -> None:
        mock_client = _make_mock_client()
        original = MagicMock(return_value=None)
        with patch("launchdarkly_ai_server.lifecycle._client", mock_client):
            wrapped = wrap_tool_handlers(
                {"__handoff_leaf": original}, CONTEXT, TRACK_DATA
            )
            await wrapped["__handoff_leaf"]()
        mock_client.track.assert_not_called()

    def test_undefined_tool_handlers(self) -> None:
        result = wrap_tool_handlers(None, CONTEXT, TRACK_DATA)
        assert result == {}
