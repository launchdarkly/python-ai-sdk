"""Process-isolated execution for AI Config tool callables.

The OpenCode handler serves plain Python tools through a loopback MCP bridge.
Those callables run in a spawned child process so a tool cannot read the
handler's in-process state (module globals, open connections, and similar).

``NativeTool`` entries never pass through this module; OpenCode runs them
inside ``opencode serve``.
"""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import pickle
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

ToolCallable = Callable[[Any], Awaitable[Any] | Any]

_DEFAULT_TIMEOUT_SECONDS = 600.0


class SandboxError(RuntimeError):
    """The sandbox child failed or could not start."""


class Sandbox:
    """Runs *fn* in a spawned process when invoked.

    Authors may wrap a tool explicitly. The OpenCode loopback bridge also
    wraps bridged callables before registration so every served Python tool
    is isolated even when the author passes a plain function.
    """

    __slots__ = ("_fn",)

    def __init__(self, fn: ToolCallable) -> None:
        self._fn = fn

    @property
    def wrapped(self) -> ToolCallable:
        """The inner callable (before tracking wrappers)."""
        return self._fn

    def __call__(self, arguments: Any) -> Awaitable[Any] | Any:
        return invoke_tool_callable(self._fn, arguments)

    def __repr__(self) -> str:
        return f"Sandbox({self._fn!r})"


def _execute_pickled(payload: bytes) -> Any:
    fn, arguments = pickle.loads(payload)
    result = fn(arguments)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def _spawn_sync(fn: ToolCallable, arguments: Any, *, timeout: float) -> Any:
    try:
        payload = pickle.dumps((fn, arguments))
    except (pickle.PicklingError, AttributeError) as exc:
        raise SandboxError(
            "Tool callable is not picklable; define it at module scope "
            "instead of as a closure or lambda."
        ) from exc
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        future = pool.submit(_execute_pickled, payload)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            raise SandboxError("Tool callable timed out in the sandbox") from exc
        except Exception as exc:
            if isinstance(exc, SandboxError):
                raise
            raise SandboxError(str(exc)) from exc


async def invoke_tool_callable(
    fn: ToolCallable,
    arguments: Any,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """Run *fn* with *arguments* in a spawned child process."""
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, lambda: _spawn_sync(fn, arguments, timeout=timeout)),
        timeout=timeout + 1.0,
    )
