"""Tests for SDK helper usage events — vercel-agents."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import SDK_INFO_CONTEXT, __version__, init_client, shutdown
from launchdarkly_ai_vercel_agents import (
    create_vercel_agents_handler,
    to_vercel_agents,
    vercel_agents,
    vercel_graph,
)
from launchdarkly_ai_vercel_agents._version import PACKAGE_NAME
from launchdarkly_ai_vercel_agents._version import __version__ as PACKAGE_VERSION

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
        ("vercel-agents.vercelAgents", lambda: vercel_agents("k", "q", CONTEXT)),
        ("vercel-agents.createVercelAgentsHandler", create_vercel_agents_handler),
        ("vercel-agents.vercelGraph", lambda: vercel_graph("k")),
        (
            "vercel-agents.vercelGraph",
            lambda: vercel_graph("k").invoke("q", CONTEXT),
        ),
        ("vercel-agents.toVercelAgents", lambda: to_vercel_agents(None)),
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
