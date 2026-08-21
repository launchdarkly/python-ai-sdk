from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from launchdarkly_ai_server.evaluations.flags import (
    ENABLE_TOOL_CALLS_IN_OFFLINE_EVALUATIONS_FLAG_KEY,
    should_skip_generation_result_ingestion,
)


@pytest.mark.asyncio
async def test_enabled_flag_selects_generation_result_ingestion_skip() -> None:
    client = MagicMock()
    client.variation = AsyncMock(return_value=True)

    assert await should_skip_generation_result_ingestion(client, "project-key") is True
    client.variation.assert_awaited_once_with(
        ENABLE_TOOL_CALLS_IN_OFFLINE_EVALUATIONS_FLAG_KEY,
        {"kind": "project", "key": "project-key"},
        False,
    )


@pytest.mark.asyncio
async def test_disabled_flag_preserves_generation_result_ingestion() -> None:
    client = MagicMock()
    client.variation = AsyncMock(return_value=False)

    assert await should_skip_generation_result_ingestion(client, "project-key") is False


@pytest.mark.asyncio
async def test_flag_evaluation_error_preserves_generation_result_ingestion() -> None:
    client = MagicMock()
    client.variation = AsyncMock(side_effect=RuntimeError("delivery unavailable"))

    assert await should_skip_generation_result_ingestion(client, "project-key") is False
