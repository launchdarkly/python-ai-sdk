"""Tests for TESTING.md §3.27 — google-adk-agents."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_google_adk_agents import (
    create_google_adk_agents_handler,
    google_adk_agents,
    google_adk_graph,
    to_adk_agents,
)
from launchdarkly_ai_google_adk_agents._version import PACKAGE_NAME
from launchdarkly_ai_google_adk_agents._version import __version__ as PACKAGE_VERSION
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
            "google-adk-agents.googleAdkAgents",
            lambda: google_adk_agents("k", "q", CONTEXT),
        ),
        (
            "google-adk-agents.createGoogleAdkAgentsHandler",
            create_google_adk_agents_handler,
        ),
        ("google-adk-agents.googleAdkGraph", lambda: google_adk_graph("k")),
        (
            "google-adk-agents.googleAdkGraph",
            lambda: google_adk_graph("k").invoke("q", CONTEXT),
        ),
        ("google-adk-agents.toAdkAgents", lambda: to_adk_agents(None)),
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
