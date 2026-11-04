"""A raising tool in an evaluation answers the model with an error result."""

from __future__ import annotations

from typing import Any

import pytest

from launchdarkly_ai_server.evaluations.tools import return_tool_errors
from launchdarkly_ai_server.trajectory import TrajectoryRecorder
from launchdarkly_ai_server.types import NativeTool


def _no_station(args: dict[str, Any]) -> dict[str, Any]:
    raise ValueError(f"No weather station for {args['city']!r}")


async def _no_station_async(args: dict[str, Any]) -> dict[str, Any]:
    raise ValueError(f"No weather station for {args['city']!r}")


def _weather(args: dict[str, Any]) -> dict[str, Any]:
    return {"city": args["city"], "temp_c": 14}


def test_a_raising_sync_tool_returns_the_error_and_is_recorded_as_one() -> None:
    recorder = TrajectoryRecorder()
    tools = return_tool_errors(recorder.wrap({"get_weather": _no_station}))

    result = tools["get_weather"]({"city": "Reykjavik"})

    assert result == "Error: ValueError: No weather station for 'Reykjavik'"
    (call,) = recorder.invocations
    assert call.error == "No weather station for 'Reykjavik'"
    assert call.result is None


@pytest.mark.asyncio
async def test_a_raising_async_tool_returns_the_error() -> None:
    tools = return_tool_errors({"get_weather": _no_station_async})

    result = await tools["get_weather"]({"city": "Reykjavik"})

    assert result == "Error: ValueError: No weather station for 'Reykjavik'"


def test_a_working_tool_and_a_native_tool_are_unchanged() -> None:
    native = NativeTool("WebSearch")
    tools = return_tool_errors({"get_weather": _weather, "search": native})

    assert tools["get_weather"]({"city": "Tokyo"}) == {"city": "Tokyo", "temp_c": 14}
    assert tools["search"] is native
