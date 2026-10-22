"""Tests for TESTING.md §3.27 — opencode-agents."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_opencode_agents import create_opencode_handler, opencode_agents
from launchdarkly_ai_opencode_agents._version import PACKAGE_NAME
from launchdarkly_ai_opencode_agents._version import __version__ as PACKAGE_VERSION
from launchdarkly_ai_server import SDK_INFO_CONTEXT, __version__, init_client, shutdown

CONTEXT = {"kind": "user", "key": "user-1"}


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
async def installed() -> Any:
    await shutdown()
    client = _fake()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=client)
    client.track.reset_mock()
    yield client
    await shutdown()


def _usage(client: MagicMock) -> list[Any]:
    return [
        entry
        for entry in client.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]


async def test_create_handler_sends_one_usage_event(installed: MagicMock) -> None:
    create_opencode_handler(cwd="/tmp", config="/tmp/opencode.json", env={})
    calls = _usage(installed)
    assert [entry.args[2].get("helper") for entry in calls] == [
        "opencode-agents.createOpenCodeHandler"
    ]
    assert calls[0].args[1] == SDK_INFO_CONTEXT
    assert calls[0].args[2] == {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": "opencode-agents.createOpenCodeHandler",
        "helperPackageName": PACKAGE_NAME,
        "helperPackageVersion": PACKAGE_VERSION,
    }
    assert calls[0].args[3] == 1


async def test_wrapper_sends_one_usage_event(installed: MagicMock) -> None:
    await _settle(
        opencode_agents(
            "k",
            "q",
            CONTEXT,
            cwd="/tmp",
            config="/no/such/opencode.json",
            env={},
        )
    )
    calls = _usage(installed)
    assert [entry.args[2].get("helper") for entry in calls] == [
        "opencode-agents.opencodeAgents"
    ]
    assert calls[0].args[2]["helperPackageName"] == PACKAGE_NAME
    assert calls[0].args[2]["helperPackageVersion"] == PACKAGE_VERSION
