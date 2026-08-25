"""
Agent Skills — reference discovery.

Projects the skill references a resolved AI Config carries into typed values. A
pure projection: no network, no client, no store, no telemetry. Retrieving the
content those references point at is a separate layer.
"""

from __future__ import annotations

import logging

from .types import AiConfigRep, SkillReference
from .types_validation import (
    is_valid_skill_key,
    is_valid_skill_version,
    skill_key_rejection_reason,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference discovery
# ---------------------------------------------------------------------------
def skill_refs(config: AiConfigRep | None) -> list[SkillReference]:
    """
    Projects a resolved AI Config's ``skills`` array into typed references.

    A pure projection — no network, no client, no store, no telemetry. Returns
    ``[]`` when the config carries no skills.

    A config that came through ``parse_ai_config`` never contains an invalid
    entry — parsing fails closed on one. A hand-built dict can, and a silently
    shortened projection would leave a caller materializing a skill set it
    believes is complete, so every dropped entry is logged.
    """
    if not isinstance(config, dict):
        return []

    raw = config.get("skills")
    if not isinstance(raw, list):
        return []

    refs: list[SkillReference] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            logger.warning(
                "skills[%d] is not a {key, version} object; it was dropped "
                "from the projection",
                index,
            )
            continue
        key = entry.get("key")
        version = entry.get("version")
        # Branch on the TypeGuard predicate (not the reason string) so the type
        # checker narrows ``key`` to ``str`` for the reference below.
        if not is_valid_skill_key(key):
            logger.warning(
                "skills[%d].key %s; it was dropped from the projection",
                index,
                skill_key_rejection_reason(key),
            )
        elif not is_valid_skill_version(version):
            logger.warning(
                "skills[%d].version must be an integer >= 1; it was dropped "
                "from the projection",
                index,
            )
        else:
            refs.append(SkillReference(key=key, version=version))
    return refs
