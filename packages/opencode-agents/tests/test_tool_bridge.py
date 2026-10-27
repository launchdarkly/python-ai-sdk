"""The loopback MCP bridge: JSON-RPC behavior, independent of ``opencode serve``.

OpenCode's own MCP client is not available in this test environment, so these
tests exercise the bridge's HTTP surface directly with ``httpx``, the same way
a conforming MCP streamable-HTTP client would: ``initialize``, then
``notifications/initialized``, then ``tools/list`` and ``tools/call``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx

from launchdarkly_ai_opencode_agents.tool_bridge import (
    ToolBridge,
    bridge_tool_name,
    sanitize,
)

_BRIDGE_RUNTIME_SECRET: str | None = None


def _bridge_runtime_probe(_args: dict[str, object]) -> str:
    return _BRIDGE_RUNTIME_SECRET or "missing"


def _weather_sync(args: dict[str, Any]) -> str:
    return f"sunny in {args['city']}"


async def _echo_async(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "echo": args}


def _weather_boom(_args: dict[str, Any]) -> str:
    raise ValueError("no such city")


def test_sanitize_matches_opencodes_own_regex() -> None:
    # mcp/catalog.ts: `value.replace(/[^a-zA-Z0-9_-]/g, "_")`.
    assert sanitize("get weather!") == "get_weather_"
    assert sanitize("already-ok_Name9") == "already-ok_Name9"


def test_bridge_tool_name_composes_server_and_key() -> None:
    assert bridge_tool_name("ld_abc123", "get weather") == "ld_abc123_get_weather"


class TestToolBridgeProtocol:
    async def _post(
        self, base_url: str, token: str, message: dict[str, Any] | None
    ) -> httpx.Response:
        async with httpx.AsyncClient() as http:
            return await http.post(
                f"{base_url}/mcp/run-1",
                json=message,
                headers={"Authorization": f"Bearer {token}"},
            )

    async def test_rejects_a_missing_or_wrong_token(self) -> None:
        bridge = ToolBridge()
        await bridge.start()
        try:
            async with httpx.AsyncClient() as http:
                response = await http.post(f"{bridge.base_url}/mcp/run-1", json={})
                assert response.status_code == 401
            response = await self._post(bridge.base_url, "wrong-token", {})
            assert response.status_code == 401
        finally:
            await bridge.stop()

    async def test_initialize_reports_tool_capability(self) -> None:
        bridge = ToolBridge()
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
            body = response.json()
            assert body["id"] == 1
            assert body["result"]["capabilities"] == {"tools": {}}
        finally:
            await bridge.stop()

    async def test_a_notification_gets_202_and_no_body(self) -> None:
        bridge = ToolBridge()
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            assert response.status_code == 202
            assert response.content == b""
        finally:
            await bridge.stop()

    async def test_tools_list_reports_the_registered_schema(self) -> None:
        bridge = ToolBridge()
        bridge.register(
            "get_weather",
            description="Look up the weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            fn=lambda args: f"sunny in {args['city']}",
        )
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            tools = response.json()["result"]["tools"]
            assert len(tools) == 1
            assert tools[0]["name"] == "get_weather"
            assert tools[0]["description"] == "Look up the weather"
            assert tools[0]["inputSchema"]["properties"]["city"]["type"] == "string"
        finally:
            await bridge.stop()

    async def test_tools_call_runs_callables_in_a_sandbox(self) -> None:
        global _BRIDGE_RUNTIME_SECRET
        _BRIDGE_RUNTIME_SECRET = "parent-only"
        bridge = ToolBridge()
        bridge.register(
            "probe",
            description="",
            parameters={},
            fn=_bridge_runtime_probe,
        )
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {
                    "jsonrpc": "2.0",
                    "id": 31,
                    "method": "tools/call",
                    "params": {"name": "probe", "arguments": {}},
                },
            )
            result = response.json()["result"]
            assert result["isError"] is False
            assert result["content"][0]["text"] == "missing"
        finally:
            await bridge.stop()

    async def test_tools_call_runs_a_sync_handler_and_wraps_the_result(self) -> None:
        bridge = ToolBridge()
        bridge.register(
            "get_weather",
            description="",
            parameters={},
            fn=_weather_sync,
        )
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "get_weather", "arguments": {"city": "Oslo"}},
                },
            )
            result = response.json()["result"]
            assert result["isError"] is False
            assert result["content"] == [{"type": "text", "text": "sunny in Oslo"}]
        finally:
            await bridge.stop()

    async def test_tools_call_awaits_an_async_handler(self) -> None:
        bridge = ToolBridge()
        bridge.register("echo", description="", parameters={}, fn=_echo_async)
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"x": 1}},
                },
            )
            result = response.json()["result"]
            assert result["isError"] is False
            # Non-string results are JSON-encoded (tool_result_string).
            assert json.loads(result["content"][0]["text"]) == {
                "ok": True,
                "echo": {"x": 1},
            }
        finally:
            await bridge.stop()

    async def test_a_raising_handler_comes_back_as_an_mcp_tool_error(self) -> None:
        bridge = ToolBridge()
        bridge.register("get_weather", description="", parameters={}, fn=_weather_boom)
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "get_weather", "arguments": {}},
                },
            )
            result = response.json()["result"]
            # A tool failure is an MCP-level `isError`, not a JSON-RPC protocol
            # error: OpenCode is expected to surface it as a normal completed
            # call whose part status is `error`, not reject the call outright.
            assert result["isError"] is True
            assert "no such city" in result["content"][0]["text"]
        finally:
            await bridge.stop()

    async def test_calling_an_unregistered_tool_is_a_protocol_error(self) -> None:
        bridge = ToolBridge()
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "nope", "arguments": {}},
                },
            )
            body = response.json()
            assert "error" in body
            assert body["error"]["code"] == -32602
        finally:
            await bridge.stop()

    async def test_an_unknown_method_is_a_protocol_error(self) -> None:
        bridge = ToolBridge()
        await bridge.start()
        try:
            response = await self._post(
                bridge.base_url,
                bridge.token,
                {"jsonrpc": "2.0", "id": 7, "method": "resources/list"},
            )
            body = response.json()
            assert body["error"]["code"] == -32601
        finally:
            await bridge.stop()

    async def test_get_opens_a_live_sse_stream_not_405(self) -> None:
        # OpenCode's MCP client (1.18.4) opens this GET right after
        # initialize and treats a non-2xx answer as a dead server, never
        # sending tools/list — confirmed live against the real binary. A
        # bare 405 here, which earlier code sent, reproduces that failure.
        # This test reads one live chunk rather than only the status line,
        # so a handler that writes headers then raises before any body
        # still fails it.
        bridge = ToolBridge()
        await bridge.start()
        try:
            async with (
                httpx.AsyncClient() as http,
                http.stream(
                    "GET",
                    f"{bridge.base_url}/mcp/run-1",
                    headers={"Authorization": f"Bearer {bridge.token}"},
                ) as response,
            ):
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"
                chunks = response.aiter_bytes()
                first = await asyncio.wait_for(chunks.__anext__(), timeout=5)
                assert b"connected" in first
        finally:
            await bridge.stop()

    async def test_stop_ends_the_sse_handler_while_the_client_is_connected(
        self,
    ) -> None:
        # The GET thread must exit when stop() runs, even if the client
        # has not closed. Otherwise HTTP/1.1 keep-alive leaves it blocked
        # in readline() and a shared process leaks one thread per call.
        bridge = ToolBridge()
        await bridge.start()
        before = {thread.ident for thread in threading.enumerate()}
        try:
            async with httpx.AsyncClient() as http:
                async with http.stream(
                    "GET",
                    f"{bridge.base_url}/mcp/run-1",
                    headers={"Authorization": f"Bearer {bridge.token}"},
                ) as response:
                    assert response.status_code == 200
                    await asyncio.wait_for(
                        response.aiter_bytes().__anext__(), timeout=5
                    )
                    await bridge.stop()
                    leftover: list[threading.Thread] = []
                    for _ in range(40):
                        leftover = [
                            thread
                            for thread in threading.enumerate()
                            if thread.ident not in before
                            and thread.is_alive()
                            and thread.name.startswith("Thread-")
                        ]
                        if not leftover:
                            break
                        await asyncio.sleep(0.05)
                    assert leftover == []
        finally:
            await bridge.stop()

    async def test_get_without_a_token_is_still_rejected(self) -> None:
        bridge = ToolBridge()
        await bridge.start()
        try:
            async with httpx.AsyncClient() as http:
                response = await http.get(f"{bridge.base_url}/mcp/run-1")
                assert response.status_code == 401
        finally:
            await bridge.stop()

    async def test_two_bridges_get_different_tokens_and_ports(self) -> None:
        a = ToolBridge()
        b = ToolBridge()
        await a.start()
        await b.start()
        try:
            assert a.token != b.token
            assert a.base_url != b.base_url
        finally:
            await a.stop()
            await b.stop()


async def test_stop_before_start_does_not_raise() -> None:
    bridge = ToolBridge()
    await bridge.stop()
