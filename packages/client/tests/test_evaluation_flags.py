from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from launchdarkly_ai_server.evaluations.flags import (
    ENABLE_BATCH_INGEST_IN_EVALS_FROM_CODE_FLAG_KEY,
    is_generation_result_batch_ingest_enabled,
)


@pytest.mark.asyncio
async def test_enabled_flag_enables_generation_result_batch_ingest() -> None:
    client = MagicMock()
    client.variation = AsyncMock(return_value=True)

    assert (
        await is_generation_result_batch_ingest_enabled(client, "project-key") is True
    )
    client.variation.assert_awaited_once_with(
        ENABLE_BATCH_INGEST_IN_EVALS_FROM_CODE_FLAG_KEY,
        {"kind": "project", "key": "project-key"},
        False,
    )


@pytest.mark.asyncio
async def test_disabled_flag_disables_generation_result_batch_ingest() -> None:
    client = MagicMock()
    client.variation = AsyncMock(return_value=False)

    assert (
        await is_generation_result_batch_ingest_enabled(client, "project-key") is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_value", [None, 1, "true", {}])
async def test_malformed_flag_disables_generation_result_batch_ingest(
    malformed_value: object,
) -> None:
    client = MagicMock()
    client.variation = AsyncMock(return_value=malformed_value)

    assert (
        await is_generation_result_batch_ingest_enabled(client, "project-key") is False
    )


@pytest.mark.asyncio
async def test_flag_evaluation_error_disables_generation_result_batch_ingest() -> None:
    client = MagicMock()
    client.variation = AsyncMock(side_effect=RuntimeError("delivery unavailable"))

    assert (
        await is_generation_result_batch_ingest_enabled(client, "project-key") is False
    )
