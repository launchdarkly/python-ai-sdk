"""The forwarding probe itself: a key that crashes the probed call fails the probe."""

from __future__ import annotations

from typing import Any

import pytest

from tests.forwarding_spec import ProbeError, probe_forwarded_keys


async def test_a_key_that_changes_the_call_is_forwarded() -> None:
    async def call(parameters: dict[str, Any]) -> object:
        return {k: v for k, v in parameters.items() if k == "temperature"}

    assert await probe_forwarded_keys({"temperature", "user"}, call) == {"temperature"}


async def test_a_key_that_makes_the_call_raise_fails_the_probe() -> None:
    async def call(parameters: dict[str, Any]) -> object:
        if "tool_choice" in parameters:
            raise ValueError("tool_choice must be a string")
        return dict(parameters)

    with pytest.raises(ProbeError) as excinfo:
        await probe_forwarded_keys({"temperature", "tool_choice"}, call)

    assert set(excinfo.value.failures) == {"tool_choice"}
    assert "tool_choice must be a string" in str(excinfo.value)
