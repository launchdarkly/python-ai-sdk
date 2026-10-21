"""Tests for TESTING.md §3.27 — langchain-agents."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_langchain_agents import (
    create_langchain_agents_handler,
    langchain_agents,
    langchain_graph,
    to_lang_graph,
)
from launchdarkly_ai_langchain_agents._version import PACKAGE_NAME
from launchdarkly_ai_langchain_agents._version import __version__ as PACKAGE_VERSION
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
        (
            "langchain-agents.langchainAgents",
            lambda: langchain_agents("k", "q", CONTEXT),
        ),
        (
            "langchain-agents.createLangChainAgentsHandler",
            create_langchain_agents_handler,
        ),
        ("langchain-agents.langchainGraph", lambda: langchain_graph("k")),
        ("langchain-agents.toLangGraph", lambda: to_lang_graph(None)),
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
    assert calls[0].args[1] == SDK_INFO_CONTEXT
    assert calls[0].args[2] == {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": helper,
        "helperPackageName": PACKAGE_NAME,
        "helperPackageVersion": PACKAGE_VERSION,
    }
    assert calls[0].args[3] == 1


def _helpers(client: MagicMock) -> list[str]:
    return [
        entry.args[2].get("helper")
        for entry in client.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]


async def _drain(stream: Any) -> None:
    try:
        async for _ in stream:
            pass
    except Exception:
        return


async def test_graph_wrapper_invoke_and_stream_report_only_the_wrapper(
    client: MagicMock,
) -> None:
    """The returned graph's methods are not separate ``client.graph.*`` calls."""
    instance = langchain_graph("k")
    await _settle(instance.invoke("q", CONTEXT))
    await _drain(instance.stream("q", CONTEXT))
    assert _helpers(client) == ["langchain-agents.langchainGraph"]


async def test_native_adapter_invoke_reports_only_the_adapter(
    client: MagicMock,
) -> None:
    async def _definition() -> Any:
        return SimpleNamespace(enabled=False, key="k")

    runner = to_lang_graph(_definition(), {"context": CONTEXT})
    with pytest.raises(ValueError, match="disabled"):
        await runner.invoke("q")
    assert _helpers(client) == ["langchain-agents.toLangGraph"]


async def test_held_helper_flushes_with_its_package() -> None:
    """A helper called before init keeps its package fields until the flush."""
    await shutdown()
    lifecycle_module._reset_for_testing()
    try:
        create_langchain_agents_handler()
    except Exception:
        pass  # The factory may need credentials; it reports first.
    installed = _fake()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=installed)
    calls = [
        entry.args[2]
        for entry in installed.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]
    assert calls == [
        {
            "aiSdkName": "launchdarkly-ai-server",
            "aiSdkVersion": __version__,
            "aiSdkLanguage": "python",
            "helper": "langchain-agents.createLangChainAgentsHandler",
            "helperPackageName": PACKAGE_NAME,
            "helperPackageVersion": PACKAGE_VERSION,
        }
    ]
    await shutdown()
