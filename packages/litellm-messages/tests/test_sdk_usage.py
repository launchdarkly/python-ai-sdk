"""Tests for TESTING.md §3.27 — litellm-messages."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_litellm_messages import (
    create_litellm_messages_handler,
    litellm_messages,
)
from launchdarkly_ai_litellm_messages._version import PACKAGE_NAME
from launchdarkly_ai_litellm_messages._version import __version__ as PACKAGE_VERSION
from launchdarkly_ai_server import SDK_INFO_CONTEXT, __version__, init_client, shutdown

CONTEXT = {"kind": "user", "key": "user-1"}


@pytest.fixture
async def client() -> Any:
    await shutdown()
    installed = MagicMock()
    installed.variation = AsyncMock(return_value=None)
    installed.flush = AsyncMock()
    installed.close = AsyncMock()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=installed)
    installed.track.reset_mock()
    yield installed
    await shutdown()


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        (
            "litellm-messages.litellmMessages",
            lambda: litellm_messages("k", "q", CONTEXT),
        ),
        (
            "litellm-messages.createLiteLLMMessagesHandler",
            create_litellm_messages_handler,
        ),
    ],
)
async def test_helper_sends_one_usage_event(
    client: MagicMock, helper: str, call: Any
) -> None:
    try:
        value = call()
        if hasattr(value, "__await__"):
            await value
        elif hasattr(value, "invoke"):
            await value.invoke("q")
    except Exception:
        pass
    calls = [
        entry
        for entry in client.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]
    assert [entry.args[2].get("helper") for entry in calls] == [helper]
    assert calls[0].args[1] == SDK_INFO_CONTEXT
    assert calls[0].args[2] == {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": helper,
        "helperPackageName": PACKAGE_NAME,
        "helperPackageVersion": PACKAGE_VERSION,
    }
