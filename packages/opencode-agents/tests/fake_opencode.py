"""Fake ``opencode serve`` for handler tests.

The handler starts this executable. It binds the ``--port`` it was given,
speaks the stable HTTP routes, and exits when the handler stops the process.
"""

from __future__ import annotations

import base64
import json
import os
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Loopback(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Request(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        script: _Server = self.server.script  # type: ignore[attr-defined]
        if not script.authorized(self.headers):
            self.send_error(401)
            return
        path = self.path.split("?", 1)[0]
        script.note("GET", path)
        if path == "/global/health":
            body = b'{"healthy":true,"version":"fake"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path != "/event":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            script.serve_events(self)
            self._chunk(b"")
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        script: _Server = self.server.script  # type: ignore[attr-defined]
        if not script.authorized(self.headers):
            self.send_error(401)
            return
        path = self.path.split("?", 1)[0]
        script.note("POST", path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        status, payload = script.serve_post(path, body)
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


class _Server:
    def __init__(self, spec: dict[str, Any], state_dir: str) -> None:
        self.spec = spec
        self.state_dir = state_dir
        self.sessions: list[str] = []
        self.session_bodies: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.aborts: list[str] = []
        self.requests: list[dict[str, str]] = []
        self.mcp_calls: list[dict[str, Any]] = []
        self.mcp_disconnects: list[str] = []
        self.second_read = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def authorized(self, headers: Any) -> bool:
        expected = os.environ.get("OPENCODE_SERVER_PASSWORD", "")
        if not expected:
            return False
        header = headers.get("Authorization")
        if not isinstance(header, str) or not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:].strip(), validate=True).decode()
        except (ValueError, UnicodeDecodeError):
            return False
        return decoded == f"opencode:{expected}"

    def note(self, method: str, path: str) -> None:
        with self._lock:
            self.requests.append({"method": method, "path": path})
            self._write()

    def _write(self) -> None:
        if not self.state_dir:
            return
        payload = {
            "pid": os.getpid(),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "config": os.environ.get("OPENCODE_CONFIG", ""),
            "username": os.environ.get("OPENCODE_SERVER_USERNAME", ""),
            "password": os.environ.get("OPENCODE_SERVER_PASSWORD", ""),
            "sessions": self.sessions,
            "session_bodies": self.session_bodies,
            "messages": self.messages,
            "aborts": self.aborts,
            "requests": self.requests,
            "mcp_calls": self.mcp_calls,
            "mcp_disconnects": self.mcp_disconnects,
        }
        path = os.path.join(self.state_dir, f"{os.getpid()}.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def write(self) -> None:
        with self._lock:
            self._write()

    def serve_events(self, request: _Request) -> None:
        mode = self.spec.get("mode")
        if mode == "close_after_read":
            request._chunk(b":\n\n")
            self.second_read.set()
            sock = request.connection
            while True:
                readable, _, _ = select.select([sock], [], [], 0.05)
                if not readable:
                    continue
                try:
                    peeked = sock.recv(1, socket.MSG_PEEK)
                except OSError:
                    return
                if not peeked:
                    return
        events = list(self.spec.get("events") or [])
        if mode == "open":
            status = int(self.spec.get("message_status") or 200)
            delay = float(self.spec.get("delay_tail") or 0)
            if status < 400:
                events = [
                    {
                        "type": "message.updated",
                        "properties": {
                            "info": {
                                "id": "m1",
                                "role": "assistant",
                                "time": {"created": 1},
                            }
                        },
                    },
                    {
                        "type": "message.part.delta",
                        "properties": {
                            "sessionID": "ses",
                            "messageID": "m1",
                            "field": "text",
                            "delta": "Hi",
                            "partID": "p1",
                        },
                    },
                ]
            else:
                events = []
            for event in events:
                request._chunk(f"data: {json.dumps(event)}\n\n".encode())
            if delay:
                time.sleep(delay)
                tail = [
                    {
                        "type": "message.part.delta",
                        "properties": {
                            "sessionID": "ses",
                            "messageID": "m1",
                            "field": "text",
                            "delta": "!",
                            "partID": "p1",
                        },
                    },
                    {
                        "type": "message.updated",
                        "properties": {
                            "info": {
                                "id": "m1",
                                "role": "assistant",
                                "time": {"created": 1, "completed": 2},
                                "tokens": {"input": 1, "output": 4},
                            }
                        },
                    },
                    {
                        "type": "session.idle",
                        "properties": {"sessionID": "ses"},
                    },
                ]
                for event in tail:
                    request._chunk(f"data: {json.dumps(event)}\n\n".encode())
                return
            self.release.wait()
            return
        for event in events:
            request._chunk(f"data: {json.dumps(event)}\n\n".encode())

    def serve_post(self, path: str, body: bytes) -> tuple[int, dict[str, Any]]:
        action = ""
        with self._lock:
            if path == "/session":
                session_id = (
                    "ses" if not self.sessions else f"ses-{len(self.sessions) + 1}"
                )
                self.sessions.append(session_id)
                self.session_bodies.append(json.loads(body.decode()) if body else {})
                self._write()
                return 200, {"id": session_id}
            if path == "/mcp":
                parsed = json.loads(body.decode()) if body else {}
                self.mcp_calls.append(parsed)
                self._write()
                name = parsed.get("name", "")
                # The real route returns MCP.add's status map itself, keyed
                # by server name, not a wrapper around that map.
                status = self.spec.get("mcp_status", "connected")
                return 200, {name: {"status": status}}
            if path.endswith("/disconnect") and path.startswith("/mcp/"):
                name = path[len("/mcp/") : -len("/disconnect")]
                self.mcp_disconnects.append(name)
                self._write()
                return 200, True
            if path.endswith("/message"):
                self.messages.append(json.loads(body.decode()) if body else {})
                self._write()
                mode = self.spec.get("mode")
                if mode == "close_after_read":
                    action = "second_read"
                elif mode == "open":
                    status = int(self.spec.get("message_status") or 200)
                    if status >= 400:
                        return status, {}
                    if self.spec.get("delay_tail"):
                        return 200, {}
                    action = "release"
                else:
                    return 200, {}
            elif path.endswith("/abort"):
                self.aborts.append(path)
                self.release.set()
                self._write()
                return 200, {}
            else:
                return 404, {}
        if action == "second_read":
            self.second_read.wait(timeout=2)
            return 400, {}
        if action == "release":
            self.release.wait()
            return 200, {}
        return 404, {}


def _port(argv: list[str]) -> int:
    if "--port" not in argv:
        raise SystemExit(2)
    return int(argv[argv.index("--port") + 1])


def main() -> None:
    spec_path = os.environ["LD_OPENCODE_FAKE_SPEC"]
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise SystemExit(2)
    server = _Server(spec, os.environ.get("LD_OPENCODE_FAKE_DIR", ""))
    server.write()
    exit_stderr = spec.get("exit_stderr")
    if isinstance(exit_stderr, str) and exit_stderr:
        sys.stderr.write(exit_stderr)
        sys.stderr.flush()
        raise SystemExit(1)
    delay_ready = spec.get("delay_ready")
    if isinstance(delay_ready, int | float) and delay_ready > 0:
        time.sleep(delay_ready)
    httpd = _Loopback(("127.0.0.1", _port(sys.argv)), _Request)
    httpd.script = server  # type: ignore[attr-defined]
    httpd.serve_forever()


if __name__ == "__main__":
    main()
