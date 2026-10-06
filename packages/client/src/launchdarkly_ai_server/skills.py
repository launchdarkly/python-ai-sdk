"""
Agent Skills — reference discovery and content accessors.

- ``skill_refs`` reads the skill references a resolved AI Config carries.
- ``get_skill``, ``get_skill_result``, ``get_skills`` and ``all_skills`` return
  verified skill content from the configured store.
- ``InMemorySkillStore`` is a simple store for local development and tests.

Writing skills to disk lives in ``skills_fs`` (``write_skills``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from . import skills_core
from .skills_core import (
    SKILL_OBJECT_KIND,
    SkillStore,
    list_raw_objects,
    log_withholding_summary,
    newest_by_key,
    reference_target,
    require_store,
    resolve_from_store,
    verify_raw_skill,
)
from .types import AiConfigRep, Skill, SkillOutcome, SkillReference
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
# Used by ``set_skill_store`` and ``shutdown`` (and tests). The state itself lives
# in ``skills_core``, so there is exactly one store and one emitter.
_set_store = skills_core.set_store
_set_emitter_for_testing = skills_core.set_emitter
_clear_state = skills_core.clear_state


def set_skill_store(store: SkillStore | None) -> None:
    """
    Sets the store the skill accessors and ``write_skills`` read from.

    Applies on every call, including after ``init_client``, so a lazily
    initialized client can be given a store afterwards. ``None`` is ignored and
    never clears a configured store; ``shutdown()`` does that.

    Replacing a store does not close the previous one; close it yourself if it
    holds a connection. A running ``watch_skills`` keeps listening to the store
    it started with, so close the watcher and start a new one to follow the
    replacement.

    Args:
        store: ``FDv2SkillStore`` to receive skills from LaunchDarkly, or
            ``InMemorySkillStore`` for local development and tests.

    Raises:
        TypeError: If *store* is not ``None`` and has no callable
            ``get_object`` and ``all_objects``.
    """
    if store is None:
        return
    missing = [
        name
        for name in ("get_object", "all_objects")
        if not callable(getattr(store, name, None))
    ]
    if missing:
        raise TypeError(
            f"set_skill_store needs a SkillStore; {type(store).__name__} has no "
            f"callable {' or '.join(missing)}."
        )
    _set_store(store)


class InMemorySkillStore:
    """
    An in-memory skill store, for local development, tests, and
    bring-your-own-content.

    - Holds raw skill objects verbatim and does no validation; the accessors
      verify everything they return.
    - Can hold several versions of one key. ``get_object`` selects on
      ``(key, version)``; ``version=None`` means the newest held.
    - An object with an invalid ``version`` is still stored (under its key
      alone), so the accessors report it as an integrity failure rather than as
      absent.
    """

    def __init__(self, objects: dict[str, dict[str, Any]] | None = None) -> None:
        self._versions: dict[str, dict[int, dict[str, Any]]] = {}
        self._loose: dict[str, dict[str, Any]] = {}
        self._listeners: dict[str, list[Callable[[dict[str, Any]], Any]]] = {}
        for object_key, raw in (objects or {}).items():
            self._place(object_key, raw)

    def _place(self, fallback_key: str, raw: dict[str, Any]) -> None:
        """Files one raw object under its own identity, verbatim."""
        key = raw.get("key") if isinstance(raw, dict) else None
        if not isinstance(key, str):
            key = fallback_key
        version = raw.get("version") if isinstance(raw, dict) else None
        if is_valid_skill_version(version):
            self._versions.setdefault(key, {})[version] = raw
        else:
            self._loose[key] = raw

    def put(self, raw: dict[str, Any]) -> None:
        """
        Adds or replaces a raw skill object, keyed by its own ``key`` and
        ``version`` fields.

        A second version of a key is kept alongside the first; the same
        ``(key, version)`` replaces it. Then calls every ``"skill"`` listener with
        the raw, unverified object.

        Raises:
            ValueError: If *raw* has no string ``key``.
        """
        key = raw.get("key")
        if not isinstance(key, str):
            raise ValueError("a raw skill object must carry a string 'key'")
        self._place(key, raw)
        for listener in self._listeners.get(SKILL_OBJECT_KIND, []):
            listener(raw)

    def get_object(
        self, kind: str, key: str, version: int | None = None
    ) -> dict[str, Any] | None:
        if kind != SKILL_OBJECT_KIND:
            return None
        held = self._versions.get(key, {})
        if not held:
            # Only a malformed entry exists: serve it so verification reports it.
            # When well-formed versions exist, a missed pin is a plain miss.
            return self._loose.get(key)
        if version is not None:
            return held.get(version)
        return held[max(held)]

    def all_objects(self, kind: str) -> dict[str, dict[str, Any]]:
        """
        Every object held, one entry per ``(key, version)``.

        The dict keys are opaque identifiers: don't parse them or assume one
        entry per skill key.
        """
        if kind != SKILL_OBJECT_KIND:
            return {}
        out: dict[str, dict[str, Any]] = {
            f"{key}:{version}": raw
            for key, versions in self._versions.items()
            for version, raw in versions.items()
        }
        out.update(self._loose)
        return out

    def add_listener(self, kind: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        """
        Registers *fn* to be called with each raw object ``put`` under *kind*.

        Raises:
            ValueError: If *kind* is not ``"skill"``. This store notifies no other
                kind, and a listener that silently never fires would look like
                one whose skills never changed.
        """
        if kind != SKILL_OBJECT_KIND:
            raise ValueError(
                f"InMemorySkillStore notifies only {SKILL_OBJECT_KIND!r} "
                f"changes, so a listener on {kind!r} would never fire. Register "
                f"it on {SKILL_OBJECT_KIND!r}."
            )
        self._listeners.setdefault(kind, []).append(fn)

    def remove_listener(self, kind: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        """
        Unregisters *fn* from *kind*.

        Removes one registration per call. Removing a callable that is not
        registered is a no-op.
        """
        listeners = self._listeners.get(kind)
        if listeners is None:
            return
        try:
            listeners.remove(fn)
        except ValueError:
            return


# ---------------------------------------------------------------------------
# Reference discovery
# ---------------------------------------------------------------------------


def skill_refs(config: AiConfigRep | None) -> list[SkillReference]:
    """
    Returns the skill references attached to a resolved AI Config.

    Pure: no network, store, or telemetry. Returns ``[]`` when the config has no
    skills. Typical use: ``await get_skills(skill_refs(config))``.

    Invalid entries (possible only in a hand-built dict; ``parse_ai_config``
    rejects them) are dropped with a warning, because ``write_skills`` with
    ``prune=True`` would delete a dropped skill's files.
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
        # Branch on the TypeGuard so ``key`` narrows to ``str``.
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

    Skills have no targeting, so there is no context parameter. For the skills a
    given context's AI Config uses, call ``get_skills(skill_refs(config))``.

    Args:
        key: The skill key.
        version: The exact version to return. ``None`` (default) means the
            newest version the store holds.

    Returns:
        The ``Skill``, or ``None`` if it is missing, not at the requested
        version, or fails verification. Use ``get_skill_result`` to learn which.

    Raises:
        RuntimeError: If no skill store is configured.
    """
    return resolve_from_store(require_store(), key, version).skill


async def get_skill_result(key: str, *, version: int | None = None) -> SkillOutcome:
    """
    Retrieves one verified skill, reporting *why* when there is none.

    Behaves exactly like ``get_skill`` (same lookup, verification, and
    telemetry), but returns a ``SkillOutcome`` whose ``reason`` says what
    happened instead of collapsing every failure to ``None``. Use it to fail
    closed on tampering while tolerating a missing skill:

    ```python
    outcome = await get_skill_result("pdf-extraction")
    if outcome.reason == "integrity_failure":
        raise SystemExit(f"refusing to run: {outcome.detail}")
    if outcome.skill is not None:
        print(outcome.skill.content)
    ```

    ``detail`` is a human-readable message, safe to log: it never contains skill
    content or filesystem paths. Branch on ``reason``, not ``detail``.

    Raises:
        RuntimeError: If no skill store is configured.
    """
    resolved = resolve_from_store(require_store(), key, version)
    return SkillOutcome(
        skill=resolved.skill, reason=resolved.reason, detail=resolved.error
    )


async def get_skills(refs: Sequence[SkillReference | str]) -> list[Skill]:
    """
    Retrieves a batch of verified skills.

    Args:
        refs: ``SkillReference`` values and/or bare key strings (a string means
            the newest version).

    Returns:
        The skills found, in input order. Entries that are missing, at the
        wrong version, or fail verification are omitted. A warning logs how
        many failed verification; misses are not counted.

    Raises:
        TypeError: If *refs* is a single string; pass ``[key]`` instead.
        RuntimeError: If no skill store is configured.
    """
    if isinstance(refs, str):
        # A str type-checks as Sequence[str] but would be iterated per character.
        raise TypeError(
            "get_skills takes a sequence of references; pass [key] rather than a "
            f"bare string. Got {refs!r}."
        )

    store = require_store()

    skills: list[Skill] = []
    # Only what the store served counts toward the summary: a miss or an outage
    # is not a verification failure, and the summary would report it as one.
    served = 0
    for ref in refs:
        key, wanted = reference_target(ref)
        resolution = resolve_from_store(store, key, wanted)
        if resolution.skill is not None:
            skills.append(resolution.skill)
        if resolution.reason in ("ok", "integrity_failure"):
            served += 1
    log_withholding_summary("requested skills the store served", served, len(skills))
    return skills


async def all_skills() -> list[Skill]:
    """
    Retrieves every verified skill the store currently holds.

    Returns the newest version of each key. Skills that fail verification are
    omitted, and a warning logs how many.

    Raises:
        RuntimeError: If no skill store is configured.
    """
    objects, error = list_raw_objects(require_store())
    if error is not None:
        return []

    # The store may hold several versions per key; keep only the newest.
    candidates = newest_by_key(objects)
    skills: list[Skill] = []
    for _object_key, raw in candidates:
        skill = verify_raw_skill(raw)
        if skill is not None:
            skills.append(skill)
    log_withholding_summary("skills held by the store", len(candidates), len(skills))
    return skills
