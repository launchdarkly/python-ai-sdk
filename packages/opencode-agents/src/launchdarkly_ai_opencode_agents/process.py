"""Start and stop one ``opencode serve`` for a single handler call.

The process is loopback-only. Each start sets a generated basic-auth password
on the child, and the health probe sends it. Callers pass a working directory,
a config file, and an environment. They do not pass a URL, an HTTP client, or
a transport.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import BinaryIO, cast

import httpx

_STARTUP_TIMEOUT_SECONDS = 45.0
_SHUTDOWN_GRACE_SECONDS = 5.0
_PROBE_INTERVAL_SECONDS = 0.05
_STDERR_TAIL_BYTES = 2048

_CommandPart = str | os.PathLike[str]
Command = _CommandPart | Sequence[_CommandPart] | None


class OpenCodeServe:
    """One ``opencode serve`` child. ``start`` returns its loopback origin."""

    def __init__(
        self,
        *,
        command: list[str],
        cwd: str,
        config: str,
        env: Mapping[str, str],
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._config = config
        self._env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._password: str | None = None
        self._stderr: BinaryIO | None = None

    @property
    def password(self) -> str:
        """The basic-auth secret ``start`` gave this child."""
        if self._password is None:
            raise RuntimeError("opencode serve failed to start.")
        return self._password

    def spawn(self) -> OpenCodeServe:
        """A new child. Calls do not share a process."""
        return OpenCodeServe(
            command=list(self._command),
            cwd=self._cwd,
            config=self._config,
            env=dict(self._env),
        )

    async def start(self) -> str:
        cwd = os.path.abspath(self._cwd)
        config = os.path.abspath(self._config)
        if not os.path.isdir(cwd):
            raise FileNotFoundError("OpenCode working directory does not exist.")
        if not os.path.isfile(config):
            raise FileNotFoundError("OpenCode config file does not exist.")
        password = secrets.token_urlsafe(32)
        self._password = password
        port = _free_port()
        argv = [
            *self._command,
            "serve",
            "--port",
            str(port),
            "--hostname",
            "127.0.0.1",
        ]
        # A pipe fills and blocks when the server writes and nothing reads it.
        stderr = tempfile.TemporaryFile()
        self._stderr = stderr
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=_child_env(self._env, config, password),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr,
            )
        except OSError as exc:
            self._close_stderr()
            raise RuntimeError("opencode serve failed to start.") from exc
        base = f"http://127.0.0.1:{port}"
        try:
            await self._wait_until_listening(base, password)
        except BaseException:
            await self.stop()
            raise
        return base

    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        try:
            if proc is None:
                return
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), _SHUTDOWN_GRACE_SECONDS)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
            else:
                await proc.wait()
        except (ProcessLookupError, OSError):
            return
        finally:
            self._close_stderr()

    async def _wait_until_listening(self, base: str, password: str) -> None:
        proc = self._proc
        if proc is None:
            raise RuntimeError("opencode serve failed to start.")
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        async with httpx.AsyncClient(auth=("opencode", password)) as http:
            while time.monotonic() < deadline:
                if proc.returncode is not None:
                    raise self._startup_error(
                        "opencode serve exited during startup "
                        f"(code {proc.returncode})."
                    )
                try:
                    response = await http.get(f"{base}/global/health", timeout=2.0)
                except httpx.HTTPError:
                    response = None
                if response is not None and _is_healthy(response):
                    return
                await asyncio.sleep(_PROBE_INTERVAL_SECONDS)
        raise self._startup_error("opencode serve did not become healthy in time.")

    def _startup_error(self, message: str) -> RuntimeError:
        tail = self._stderr_tail()
        if not tail:
            return RuntimeError(message)
        return RuntimeError(f"{message}\n{tail}")

    def _stderr_tail(self) -> str:
        handle = self._stderr
        if handle is None:
            return ""
        try:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _STDERR_TAIL_BYTES), os.SEEK_SET)
            raw = handle.read()
        except OSError:
            return ""
        return raw.decode("utf-8", errors="replace").strip()

    def _close_stderr(self) -> None:
        handle = self._stderr
        self._stderr = None
        if handle is None:
            return
        try:
            handle.close()
        except OSError:
            return


def serve_from(
    *,
    cwd: str | os.PathLike[str],
    config: str | os.PathLike[str],
    env: Mapping[str, str],
    command: Command,
) -> OpenCodeServe:
    """Validate the public inputs and return a server that has not started."""
    return OpenCodeServe(
        command=_command_argv(command),
        cwd=_require_text(cwd, "working directory"),
        config=_require_text(config, "config file"),
        env=_require_env(env),
    )


def _require_text(value: str | os.PathLike[str], name: str) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"The OpenCode handler requires a {name}.")
    return value


def _require_env(env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(env, Mapping):
        raise TypeError("The OpenCode handler requires an env mapping.")
    copied: dict[str, str] = {}
    for key, item in env.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError("The OpenCode handler env must be strings.")
        copied[key] = item
    return copied


def _command_argv(command: Command) -> list[str]:
    if command is None:
        return ["opencode"]
    if isinstance(command, (str, os.PathLike)):
        return [_one_command(os.fspath(command))]
    if isinstance(command, Sequence):
        argv = [_one_command(os.fspath(part)) for part in command]
        if not argv:
            raise TypeError("The OpenCode handler takes a command, not a URL.")
        return argv
    raise TypeError("The OpenCode handler takes a command, not a URL.")


def _one_command(part: str) -> str:
    if not part or "://" in part:
        raise TypeError("The OpenCode handler takes a command, not a URL.")
    return part


def _child_env(env: Mapping[str, str], config: str, password: str) -> dict[str, str]:
    child = dict(env)
    child["OPENCODE_CONFIG"] = config
    child["OPENCODE_SERVER_PASSWORD"] = password
    return child


def _is_healthy(response: httpx.Response) -> bool:
    """``/global/health`` is ready only with 200 and ``healthy: true``."""
    if response.status_code != 200:
        return False
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(body, dict) and body.get("healthy") is True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])
