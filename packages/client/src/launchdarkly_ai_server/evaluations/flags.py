from __future__ import annotations

import inspect
import logging
from typing import Any, Final

from ..utils import to_ld_context

logger = logging.getLogger(__name__)

ENABLE_BATCH_INGEST_IN_EVALS_FROM_CODE_FLAG_KEY: Final[str] = (
    "enable-batch-ingest-in-evals-from-code"
)
"""Canonical rollout flag for generation-result batch ingestion."""


async def is_generation_result_batch_ingest_enabled(
    client: Any,
    project_key: str,
) -> bool:
    """Return whether the rollout flag enables generation-result batch ingest.

    Flag evaluation is fail-safe: false, malformed, or failed evaluations disable
    the gated batch-ingest path.
    """
    try:
        context = to_ld_context(
            client,
            {"kind": "project", "key": project_key},
        )
        result = client.variation(
            ENABLE_BATCH_INGEST_IN_EVALS_FROM_CODE_FLAG_KEY,
            context,
            False,
        )
        value = await result if inspect.isawaitable(result) else result
        return value is True
    except Exception:
        logger.warning(
            "Unable to evaluate %s; generation results will not be batch ingested",
            ENABLE_BATCH_INGEST_IN_EVALS_FROM_CODE_FLAG_KEY,
            exc_info=True,
        )
        return False
