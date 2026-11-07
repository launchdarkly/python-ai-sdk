"""A handler receives only the tools the active config attaches.

``tool_handlers`` (and a ``Registry``) hold the tools of every variation, so core
narrows the map to ``config["tools"]`` before any handler runs, on the blocking
and the streaming path and on every graph node.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import NativeTool, ProviderHandler, config, graph
from launchdarkly_ai_server.tracking import execute_and_stream, execute_and_track

CONTEXT = {"kind": "user", "key": "u1"}
FN_TOOL = {"type": "function", "parameters": {}}


def _tool(name: str) -> dict[str, Any]:
    return {"name": name, "type": "function", "parameters": {}}


def _cfg(tools: dict[str, Any] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model": {"name": "m"},
        "provider": {"name": "TestProvider"},
        "instructions": "x",
    }
    if tools is not None:
        cfg["tools"] = tools
    return cfg


def _variation(tools: dict[str, Any] | None) -> dict[str, Any]:
    return {
        **_cfg(tools),
        "_ldMeta": {
            "enabled": True,
            "variationKey": "v1",
            "version": 1,
            "mode": "messages",
        },
    }


@pytest.fixture
def mock_ld_client() -> Any:
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    client.variation = AsyncMock(return_value=None)
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


def _capturing_handler(seen: list[dict[str, Any]]) -> ProviderHandler:
    async def fn(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        seen.append(dict(tool_handlers or {}))
        return {"output": "ok", "usage": {}}

    async def stream_fn(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        seen.append(dict(tool_handlers or {}))
        yield {"type": "done", "output": "ok", "usage": {}}

    return ProviderHandler(
        fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
    )


async def _run(
    path: str, cfg: dict[str, Any], tool_handlers: dict[str, Any] | None
) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {
        "config_key": "c",
        "config": cfg,
        "meta": {"variationKey": "v", "version": 1},
        "user_context": CONTEXT,
        "handler": _capturing_handler(seen),
        "user_input": "q",
        "tool_handlers": tool_handlers,
    }
    if path == "invoke":
        await execute_and_track(**kwargs)
    else:
        async for _ in execute_and_stream(**kwargs):
            pass
    assert len(seen) == 1
    return seen[0]


PATHS = ["invoke", "stream"]


@pytest.mark.parametrize("path", PATHS)
async def test_unattached_tools_never_reach_the_handler(
    mock_ld_client: Any, path: str
) -> None:
    privileged = MagicMock()
    received = await _run(
        path, _cfg({"safe": FN_TOOL}), {"safe": MagicMock(), "privileged": privileged}
    )
    assert set(received) == {"safe"}
    privileged.assert_not_called()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("tools", [None, {}])
async def test_no_config_tools_means_no_tools(
    mock_ld_client: Any, path: str, tools: dict[str, Any] | None
) -> None:
    assert await _run(path, _cfg(tools), {"privileged": MagicMock()}) == {}


@pytest.mark.parametrize("path", PATHS)
async def test_inherited_names_are_not_attached_tools(
    mock_ld_client: Any, path: str
) -> None:
    """A dict has no prototype, but attribute names must not resolve to a tool either."""
    tools = {"__class__": FN_TOOL, "keys": FN_TOOL, "get": FN_TOOL}
    assert await _run(path, _cfg(tools), {}) == {}


@pytest.mark.parametrize("path", PATHS)
async def test_native_tools_are_scoped_the_same_way(
    mock_ld_client: Any, path: str
) -> None:
    bash = NativeTool("Bash")
    received = await _run(
        path, _cfg({"safe": FN_TOOL}), {"safe": MagicMock(), "bash": bash}
    )
    assert set(received) == {"safe"}
    assert set(await _run(path, _cfg({"bash": FN_TOOL}), {"bash": bash})) == {"bash"}


@pytest.mark.parametrize("path", PATHS)
async def test_config_scopes_tools_and_registry_tools(
    mock_ld_client: Any, path: str
) -> None:
    mock_ld_client.variation = AsyncMock(
        return_value=_variation({"safe": _tool("safe")})
    )
    seen: list[dict[str, Any]] = []
    instance = config(
        key="flag",
        handler=_capturing_handler(seen),
        tool_handlers={"safe": MagicMock(), "privileged": MagicMock()},
    )
    if path == "invoke":
        await instance.invoke("q", CONTEXT)
    else:
        async for _ in instance.stream("q", CONTEXT):
            pass
    assert [set(m) for m in seen] == [{"safe"}]


async def test_graph_nodes_are_scoped_and_keep_their_handoff_tools(
    mock_ld_client: Any,
) -> None:
    """A routed node gets its own attached tools plus the synthetic handoff
    tools (they are on the routed config's tools), and nothing registered for
    another node."""

    async def variation(key: str, ctx: dict[str, Any], default: Any) -> Any:
        if key == "graph-key":
            return {"root": "root", "edges": {"root": [{"key": "a"}, {"key": "b"}]}}
        if key == "root":
            return _variation({"safe": _tool("safe")})
        return _variation({"leaf_tool": _tool("leaf_tool")})

    mock_ld_client.variation = AsyncMock(side_effect=variation)
    seen: list[dict[str, Any]] = []
    await graph(
        "graph-key",
        handlers=[_capturing_handler(seen)],
        tool_handlers={
            "safe": MagicMock(),
            "leaf_tool": MagicMock(),
            "privileged": MagicMock(),
        },
    ).invoke("hi", CONTEXT)
    root = seen[0]
    assert "safe" in root
    assert {k for k in root if k.startswith("__handoff_")}, root
    assert "leaf_tool" not in root and "privileged" not in root
