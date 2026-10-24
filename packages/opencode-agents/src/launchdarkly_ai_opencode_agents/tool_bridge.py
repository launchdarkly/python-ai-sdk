"""A loopback MCP server for AI Config tools backed by plain callables.

OpenCode has no route that accepts a tool definition alongside a message: the
message route's ``tools`` field is a deprecated boolean permission map, not a
place to describe a tool (see ``session/prompt.ts`` at the pinned OpenCode
tag). A plain callable in ``tool_handlers`` is therefore served from a
minimal MCP endpoint this module starts per call, which the handler attaches
to the running ``opencode serve`` instance with ``POST /mcp``.

This endpoint answers in MCP streamable-HTTP "JSON response" mode: every
``tools/call`` gets exactly one JSON-RPC response body, with no
server-initiated SSE stream. That is a conforming subset of the transport
(a server with nothing to push to the client is not required to stream), and
it is all this bridge needs: OpenCode is the only client, and it never needs
an unsolicited message from this side.

The stdlib has no asyncio HTTP server, and this package otherwise depends on
nothing beyond ``httpx`` and ``opentelemetry-api``. Bringing in an ASGI stack
for one loopback endpoint would be a much larger footprint than the endpoint
itself, so this runs on a background thread via ``http.server``, and crosses
back onto the event loop that is driving the call with
``asyncio.run_coroutine_threadsafe`` to invoke the tracked tool handler.

Each call gets its own :class:`ToolBridge`, its own ephemeral port, and its
own bearer token, so two calls sharing one ``opencode serve`` process
(``async with handler``) can never reach each other's tools.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import threading
from collections.abc import Awaitable, Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: OpenCode's own MCP tool-name sanitizer (``mcp/catalog.ts``'s ``sanitize``).
#: Matched exactly so the composed name this module predicts is the name
#: OpenCode actually shows the model.
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")

#: Generous on purpose. A served tool may call out (the broker, a database),
#: and the default MCP request timeout (5s, ``core/src/v1/config/mcp.ts``) is
#: far too short for that. ``register_mcp_bridge`` sends this as the
#: connection's own ``timeout``, in milliseconds.
CALL_TIMEOUT_SECONDS = 600.0

_PROTOCOL_VERSION = "2025-06-18"
_SERVER_NAME = "launchdarkly-ai-opencode-agents"


def sanitize(value: str) -> str:
    """OpenCode's own MCP tool-name sanitizer, reproduced so names agree."""
    return _SANITIZE_RE.sub("_", value)


def bridge_tool_name(server_name: str, key: str) -> str:
    """The name OpenCode shows the model for one bridged tool.

    Matches ``MCP.toolName`` (``sanitize(clientName) + "_" + sanitize(name)``)
    so a permission rule built from this name actually matches what OpenCode
    reports back on the tool part.
    """
    return f"{sanitize(server_name)}_{sanitize(key)}"


ToolCallable = Callable[[Any], "Awaitable[Any] | Any"]


class _ToolEntry:
    __slots__ = ("description", "fn", "parameters")

    def __init__(
        self, description: str, parameters: Mapping[str, Any], fn: ToolCallable
    ) -> None:
        self.description = description
        self.parameters = parameters
        self.fn = fn


class ToolBridge:
    """One loopback MCP endpoint, scoped to a single handler call.

    Register every tool before :meth:`start`, or afterward — ``tools/list``
    reads the live map either way. ``start`` binds an ephemeral port and
    returns its base URL; ``stop`` closes the socket and joins the serving
    thread.
    """

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}
        self._token = secrets.token_urlsafe(32)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._port = 0

    @property
    def token(self) -> str:
        """The bearer token a caller must present. Generated once per bridge."""
        return self._token

    @property
    def base_url(self) -> str:
        if not self._port:
            raise RuntimeError("ToolBridge has not started.")
        return f"http://127.0.0.1:{self._port}"

    def register(
        self,
        key: str,
        *,
        description: str,
        parameters: Mapping[str, Any],
        fn: ToolCallable,
    ) -> None:
        """Serves *fn* under *key* as an MCP tool. *key* is the raw (unsanitized) name."""
        self._tools[key] = _ToolEntry(description, parameters, fn)

    async def start(self) -> str:
        """Binds a loopback port and starts serving. Returns the base URL."""
        self._loop = asyncio.get_running_loop()
        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                # No server-initiated messages: this bridge never needs the
                # SSE half of the streamable-HTTP transport.
                self.send_error(405)

            def do_POST(self) -> None:
                if not bridge._authorized(self.headers):
                    self.send_error(401)
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length else b""
                try:
                    message = json.loads(raw.decode("utf-8")) if raw else {}
                except (ValueError, UnicodeDecodeError):
                    message = {}
                response = bridge._dispatch(
                    message if isinstance(message, dict) else {}
                )
                if response is None:
                    # A notification (no `id`): MCP streamable HTTP wants 202, no body.
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        self._server = server
        self._port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread = thread
        thread.start()
        return self.base_url

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=5.0)

    def _authorized(self, headers: Any) -> bool:
        header = headers.get("Authorization")
        if not isinstance(header, str):
            return False
        return secrets.compare_digest(header, f"Bearer {self._token}")

    def _dispatch(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": _SERVER_NAME, "version": "1"},
                },
            }
        if method is None or str(method).startswith("notifications/"):
            # A notification carries no `id` and gets no JSON-RPC response.
            return None
        if msg_id is None:
            return None
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": name,
                            "description": entry.description,
                            "inputSchema": entry.parameters
                            or {"type": "object", "properties": {}},
                        }
                        for name, entry in self._tools.items()
                    ]
                },
            }
        if method == "tools/call":
            params = message.get("params")
            params = params if isinstance(params, Mapping) else {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            entry = self._tools.get(name) if isinstance(name, str) else None
            if entry is None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": f"Unknown tool: {name!r}"},
                }
            return self._call_tool(msg_id, entry, arguments)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unknown method: {method!r}"},
        }

    def _call_tool(
        self, msg_id: Any, entry: _ToolEntry, arguments: Any
    ) -> dict[str, Any]:
        loop = self._loop
        if loop is None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": "bridge is not running"}],
                    "isError": True,
                },
            }
        future = asyncio.run_coroutine_threadsafe(_invoke(entry.fn, arguments), loop)
        try:
            text, failed = future.result(timeout=CALL_TIMEOUT_SECONDS)
        except Exception as exc:  # the handler raised, or the call timed out
            text, failed = str(exc), True
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": failed},
        }


async def _invoke(fn: ToolCallable, arguments: Any) -> tuple[str, bool]:
    from launchdarkly_ai_server import tool_result_string

    try:
        result = fn(arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return tool_result_string(result), False
    except Exception as exc:
        return str(exc), True
