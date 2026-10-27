"""Process sandbox for AI Config tool callables."""

from __future__ import annotations

import pytest

from launchdarkly_ai_server import Sandbox, SandboxError, invoke_tool_callable

_RUNTIME_SECRET: str | None = None


def _runtime_secret_probe(_args: dict[str, object]) -> str:
    return _RUNTIME_SECRET or "missing"


async def test_invoke_tool_callable_does_not_see_parent_globals() -> None:
    global _RUNTIME_SECRET
    _RUNTIME_SECRET = "parent-only"
    result = await invoke_tool_callable(_runtime_secret_probe, {})
    assert result == "missing"


def _sync_echo(args: dict[str, object]) -> str:
    return str(args.get("city"))


async def _async_echo(args: dict[str, object]) -> dict[str, object]:
    return args


async def test_invoke_tool_callable_runs_sync_callables() -> None:
    result = await invoke_tool_callable(_sync_echo, {"city": "Oslo"})
    assert result == "Oslo"


async def test_sandbox_delegates_to_inner() -> None:
    boxed = Sandbox(_async_echo)
    assert await boxed({"x": 1}) == {"x": 1}


async def test_non_picklable_callable_raises_sandbox_error() -> None:
    with pytest.raises(SandboxError, match="picklable"):
        await invoke_tool_callable(lambda _args: "nope", {})
