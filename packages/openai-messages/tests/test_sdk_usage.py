"""Tests for TESTING.md §3.27 — openai-messages."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_openai_messages import (
    create_openai_messages_handler,
    openai_messages,
)
from launchdarkly_ai_server import SDK_INFO_CONTEXT, __version__, init_client, shutdown

CONTEXT = {"kind": "user", "key": "user-1"}
ANONYMOUS = SDK_INFO_CONTEXT


def _fake() -> MagicMock:
    client = MagicMock()
    client.track = MagicMock()
    client.variation = AsyncMock(return_value=None)
    client.flush = AsyncMock()
    client.close = AsyncMock()
    return client


async def _settle(value: Any) -> None:
    if hasattr(value, "__await__"):
        try:
            await value
        except Exception:
            return


@pytest.fixture
async def client() -> Any:
    await shutdown()
    installed = _fake()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=installed)
    installed.track.reset_mock()
    yield installed
    await shutdown()


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        ("openai-messages.openaiMessages", lambda: openai_messages("k", "q", CONTEXT)),
        ("openai-messages.createOpenAIHandler", create_openai_messages_handler),
    ],
)
async def test_helper_sends_one_usage_event(
    client: MagicMock, helper: str, call: Any
) -> None:
    try:
        await _settle(call())
    except Exception:
        pass
    calls = [
        entry
        for entry in client.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]
    assert [entry.args[2].get("helper") for entry in calls] == [helper]
    assert calls[0].args[1] == ANONYMOUS
    assert calls[0].args[2] == {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": helper,
    }
    assert calls[0].args[3] == 1
