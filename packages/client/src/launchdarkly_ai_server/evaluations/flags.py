from __future__ import annotations

import inspect
import logging
from typing import Any, Final

from ..utils import to_ld_context

logger = logging.getLogger(__name__)

ENABLE_TOOL_CALLS_IN_OFFLINE_EVALUATIONS_FLAG_KEY: Final[str] = (
    "enable-tool-calls-in-offline-evaluations"
)
"""Canonical rollout flag for tool calls in offline evaluations."""


async def should_skip_generation_result_ingestion(
    client: Any,
    project_key: str,
) -> bool:
    """Return whether the rollout flag selects the no-ingest path.

    Flag evaluation is fail-safe: false, malformed, or failed evaluations retain
    the existing generation-result ingestion behavior.
    """
    try:
        context = to_ld_context(
            client,
            {"kind": "project", "key": project_key},
        )
        result = client.variation(
            ENABLE_TOOL_CALLS_IN_OFFLINE_EVALUATIONS_FLAG_KEY,
            context,
            False,
        )
        value = await result if inspect.isawaitable(result) else result
        return value is True
    except Exception:
        logger.warning(
            "Unable to evaluate %s; generation results will be ingested",
            ENABLE_TOOL_CALLS_IN_OFFLINE_EVALUATIONS_FLAG_KEY,
            exc_info=True,
        )
        return False
