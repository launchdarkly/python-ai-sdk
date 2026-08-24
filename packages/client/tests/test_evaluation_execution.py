from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from launchdarkly_ai_server.evaluations.runner import EvaluationsRunner
from launchdarkly_ai_server.evaluations.types import DatasetRow, ResolvedTool
from launchdarkly_ai_server.types import NativeTool


class UnusedApi:
    pass


def row(index: int, text: str = "Order {{id}}") -> DatasetRow:
    return DatasetRow(
        row_index=index,
        input=text,
        expected_output="Found {{id}}",
        variables={"id": str(index)},
        metadata=None,
    )


def test_handler_config_maps_prompts_tools_and_output_format() -> None:
    runner = EvaluationsRunner(UnusedApi())  # type: ignore[arg-type]
    config = runner._build_handler_config(
        {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "parameters": {"temperature": 0.2},
            "instructions": "Use {{snippet.policy}}",
            "prompt_snippets": {"policy": "care"},
            "output_format": {"type": "object"},
        },
        {
            "lookup": ResolvedTool(
                key="lookup",
                version=1,
                description="Look up an order",
                schema={"type": "object"},
            )
        },
    )

    assert config["instructions"] == "Use care"
    assert config["model"]["parameters"] == {"temperature": 0.2}
    assert config["tools"]["lookup"]["parameters"] == {"type": "object"}
    assert config["outputFormat"] == {"type": "object"}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 10])
async def test_execution_is_bounded_and_exactly_once(limit: int) -> None:
    runner = EvaluationsRunner(UnusedApi())  # type: ignore[arg-type]
    active = 0
    maximum = 0
    calls: list[int] = []

    async def handler(
        config: dict[str, Any],
        user_input: str | None,
        tools: dict[str, Callable[..., Any] | NativeTool],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal active, maximum
        del config, user_input, tools
        calls.append(int(variables["id"]))
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1
        return {"output": "ok", "usage": {"input_tokens": 1}}

    outcomes = await runner._run_rows(
        [row(index) for index in range(12)],
        handler,
        {"provider": {"name": "OpenAI"}},
        {"native": NativeTool("WebSearch")},
        limit,
    )

    assert maximum <= limit
    assert sorted(calls) == list(range(12))
    assert len(calls) == len(set(calls)) == 12
    assert all(outcome.error is None for outcome in outcomes)
    assert all(outcome.usage == {"input_tokens": 1} for outcome in outcomes)


@pytest.mark.asyncio
async def test_rendering_missing_usage_and_error_isolation_without_retry() -> None:
    runner = EvaluationsRunner(UnusedApi())  # type: ignore[arg-type]
    calls: list[str | None] = []

    async def handler(*args: Any) -> dict[str, Any]:
        user_input = args[1]
        calls.append(user_input)
        if user_input == "Order 1":
            raise RuntimeError("provider failed")
        return {"output": "ok"}

    outcomes = await runner._run_rows(
        [row(1), row(2)],
        handler,
        {"provider": {"name": "custom"}},
        {},
        2,
    )

    assert calls == ["Order 1", "Order 2"]
    assert outcomes[0].row.expected_output == "Found 1"
    assert outcomes[0].error == "handler raised: provider failed"
    assert outcomes[1].usage is None
    assert outcomes[1].output == "ok"
