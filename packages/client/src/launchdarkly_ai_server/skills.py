"""
Agent Skills — reference discovery and content accessors.

The public retrieval surface: projecting the skill references a resolved AI
Config carries, and retrieving skill content through an injectable store seam.

The three layers of the feature sit in three modules, and the dependencies run
one way only:

- ``skills_core.py`` — the store and telemetry seams, module state, integrity
  verification, and store resolution. Shared, and imports neither of the others.
- ``skills.py`` (this file) — ``skill_refs``, the accessors, and
  ``InMemorySkillStore``.
- ``skills_fs.py`` — the highest-blast-radius layer, the one that writes to a
  customer's disk. It owns the manifest format and the on-disk filenames;
  nothing here knows about the filesystem.

``_set_store``, ``_set_emitter_for_testing`` and ``_clear_state`` live here
because this module is the documented injection path; the
state they mutate lives in ``skills_core``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from . import skills_core
from .skills_core import (
    SKILL_OBJECT_KIND,
    list_raw_objects,
    reference_target,
    require_store,
    resolve_from_store,
    verify_raw_skill,
)
from .types import AiConfigRep, Skill, SkillReference
from .types_validation import (
    is_valid_skill_key,
    is_valid_skill_version,
    skill_key_rejection_reason,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injection points
# ---------------------------------------------------------------------------
#
# These three names are the documented seam: ``init_client`` and ``shutdown``
# call them, and tests inject through them. They delegate to ``skills_core``,
# which owns the state, so that there is exactly one store and one emitter no
# matter which layer reaches for them.


# Bound directly to the implementations rather than wrapped: a one-line
# delegation per name would give every state mutation two definitions and two
# docstrings to keep in agreement, which is the drift these names exist to
# avoid. ``_set_emitter_for_testing`` keeps its distinct name because it has no
# production caller.
_set_store = skills_core.set_store
_set_emitter_for_testing = skills_core.set_emitter
_clear_state = skills_core.clear_state


class InMemorySkillStore:
    """
    A skill store backed by a plain dict.

    Ships for local development, tests, and bring-your-own-content injection.
    Holds raw wire objects verbatim and performs no validation of its own —
    verification belongs at the accessor boundary, where it applies to every
    store equally.
    """

    def __init__(self, objects: dict[str, dict[str, Any]] | None = None) -> None:
        self._objects: dict[str, dict[str, Any]] = dict(objects or {})
        self._listeners: dict[str, list[Callable[[dict[str, Any]], Any]]] = {}

    def put(self, raw: dict[str, Any]) -> None:
        """
        Adds or replaces a raw skill object, keyed by its own ``key`` field.

        Notifies every skill-kind listener with the raw object as a single
        positional argument. No validation happens here — verification belongs at
        the accessor boundary, where it applies to every store equally — so a
        listener sees exactly what was put, unverified.
        """
        key = raw.get("key")
        if not isinstance(key, str):
            raise ValueError("a raw skill object must carry a string 'key'")
        self._objects[key] = raw
        for listener in self._listeners.get(SKILL_OBJECT_KIND, []):
            listener(raw)

    def get_object(self, kind: str, key: str) -> dict[str, Any] | None:
        if kind != SKILL_OBJECT_KIND:
            return None
        return self._objects.get(key)

    def all_objects(self, kind: str) -> dict[str, dict[str, Any]]:
        if kind != SKILL_OBJECT_KIND:
            return {}
        return dict(self._objects)

    def add_listener(self, kind: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        """
        Registers *fn* to be called with each raw object ``put`` under *kind*.

        Only ``kind == SKILL_OBJECT_KIND`` is ever notified, because ``put`` only
        accepts skill objects; a listener registered under any other kind is
        recorded and never fires.
        """
        self._listeners.setdefault(kind, []).append(fn)


# ---------------------------------------------------------------------------
# Reference discovery
# ---------------------------------------------------------------------------


def skill_refs(config: AiConfigRep | None) -> list[SkillReference]:
    """
    Projects a resolved AI Config's ``skills`` array into typed references.

    A pure projection — no network, no client, no store, no telemetry. Returns
    ``[]`` when the config carries no skills. Compose it with the accessors for
    per-context resolution: ``await get_skills(skill_refs(config))``.

    A config that came through ``parse_ai_config`` never contains an invalid
    entry — parsing fails closed on one. A hand-built dict can, and a silently
    shortened projection would let ``write_skills`` prune the dropped skill's
    on-disk copy, so every dropped entry is logged.
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


# ---------------------------------------------------------------------------
# Content accessors
# ---------------------------------------------------------------------------


async def get_skill(key: str, *, version: int | None = None) -> Skill | None:
    """
    Retrieves one verified skill by key.

    ``version=None`` means the newest version the store holds; a specific
    ``version`` returns the skill only when that exact version is available.
    Returns ``None`` — never raises — when the skill is missing, the requested
    version is not the one held, or verification fails. Raises ``RuntimeError``
    only when no skill store is configured.

    There is no context parameter: skills have no targeting, so the SDK
    credentials fully determine availability. Compose per-context resolution
    explicitly with ``get_skills(skill_refs(config))``.
    """
    return resolve_from_store(require_store(), key, version).skill


async def get_skills(refs: Sequence[SkillReference | str]) -> list[Skill]:
    """
    Retrieves a batch of verified skills.

    Accepts a mixed sequence of ``SkillReference`` values and bare key strings,
    where a string means "the latest version". Results follow input order for
    the skills that were found; entries that are missing, are the wrong version,
    or fail verification are omitted rather than returned as placeholders.
    """
    if isinstance(refs, str):
        # str satisfies Sequence[str], so this type-checks; iterating it would
        # silently look up one skill per character.
        raise TypeError(
            "get_skills takes a sequence of references; pass [key] rather than a "
            f"bare string. Got {refs!r}."
        )

    store = require_store()

    skills: list[Skill] = []
    for ref in refs:
        key, wanted = reference_target(ref)
        skill = resolve_from_store(store, key, wanted).skill
        if skill is not None:
            skills.append(skill)
    return skills


async def all_skills() -> list[Skill]:
    """
    Retrieves every verified skill the store currently holds.

    Skills that fail verification are omitted. Raises ``RuntimeError`` only when
    no skill store is configured.
    """
    objects, error = list_raw_objects(require_store())
    if error is not None:
        return []

    skills: list[Skill] = []
    for raw in objects.values():
        skill = verify_raw_skill(raw)
        if skill is not None:
            skills.append(skill)
    return skills
