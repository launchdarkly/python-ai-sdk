"""
Agent Skills internals shared by ``skills`` and ``skills_fs``: the store
interface and configured store, telemetry, integrity verification, and store
resolution.

Package-internal except ``SkillStore``, which the package root re-exports. This
module imports neither ``skills`` nor ``skills_fs``.

**Everything a store returns is untrusted.** Key, version, size and content hash
are revalidated on every pass, and no store-supplied value reaches a signal or
log line without a shape check.

Security and telemetry contracts: ``agents.md``, *Security posture* and
*Telemetry seam*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, get_args

from .types import Skill, SkillOutcomeReason, SkillReference
from .types_validation import is_valid_skill_key, is_valid_skill_version

logger = logging.getLogger(__name__)

SKILL_OBJECT_KIND = "skill"
"""
The kind the SDK passes to ``SkillStore.get_object`` and
``SkillStore.all_objects``. A store adapter may map it onto whatever its
transport uses.
"""

MAX_SKILL_CONTENT_BYTES = 10 * 1024 * 1024
"""
Hard cap on skill content; anything larger is withheld even if its hash matches.

A backstop against absurd input, set well above LaunchDarkly's own limit so that
limit can grow without this changing. Not the real limit, so do not pre-flight
skill sizes against it.
"""

_LANGUAGE = "python"

_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")
"""What a legitimate content hash looks like. Anything else is redacted before
it reaches telemetry, so a store cannot leak the skill body through
``contentHash``."""

_SIGNAL_INTEGRITY_FAILURE = "AgentControl Skill Integrity Failure"
_SIGNAL_MATERIALIZED = "AgentControl Skill Materialized"
_SIGNAL_REVOKED = "AgentControl Skill Revoked Received"

INTEGRITY_FAILURE_EVENT = "ld.skills.integrity_failure"
"""
Stable event name for the local integrity-failure log record. SIEM rules match
on it, so it must never be renamed.

It appears in the message text as well as ``extra``, because the default
formatter drops ``extra`` and this module also logs other errors at ERROR.
"""

_ACTION_WITHHELD = "withheld"
"""The only action an integrity failure results in: content is never returned."""

IntegrityReasonCode = Literal[
    "not_an_object",
    "invalid_key",
    "invalid_version",
    "missing_content",
    "missing_content_hash",
    "not_utf8",
    "over_size_cap",
    "hash_mismatch",
    "key_mismatch",
    "version_mismatch",
]
"""
The closed, stable ``reason_code`` vocabulary for integrity failures; detection
rules can match on these tokens, and every SDK language emits the same ones.

Eight come from ``verify_raw_skill`` and fire both the log record and the
signal. ``key_mismatch`` and ``version_mismatch`` are detected after
verification and fire the log record only (see ``record_key_mismatch``).
"""

INTEGRITY_REASON_CODES: frozenset[str] = frozenset(get_args(IntegrityReasonCode))
"""``IntegrityReasonCode`` as a runtime set, derived rather than restated."""

NO_STORE_MESSAGE = (
    "No skill store is configured, so skill content cannot be retrieved. Configure "
    'one with init_client(options={"skillStore": store}) — FDv2SkillStore receives '
    "content from LaunchDarkly, and InMemorySkillStore is available for local "
    "development and testing."
)
"""What the accessors report when no store is configured. Callers match on
"skill store"; keep that phrase if the wording changes."""


# ---------------------------------------------------------------------------
# The store interface
# ---------------------------------------------------------------------------


class SkillStore(Protocol):
    """
    Structural interface for a source of skill content: pass any object with
    these methods.

    Raw objects are wire-shaped, with camelCase field names::

        {"key": "pdf-extraction", "version": 2, "content": "---\\n...",
         "contentHash": "9f3a...", "name": "PDF Extraction", "description": "..."}

    - ``get_object(kind, key, version)`` returns one object, or ``None``. A store
      may hold several versions of a key, so the version is part of the lookup;
      ``version=None`` means the newest held.
    - ``all_objects(kind)`` returns one entry per *(key, version)*. Its dict keys
      are opaque; read identity from each object's ``key`` and ``version``.

    Optional methods, probed for rather than declared (declaring them would make
    them required):

    - ``is_initialized()``: whether initial data has arrived. Absent means
      initialized. Until it is, ``write_skills("*")`` does not prune, since an
      empty store would otherwise look like every skill was revoked.
    - ``add_listener(kind, fn)`` / ``remove_listener(kind, fn)``: push updates to
      consumers such as ``watch_skills``. ``remove_listener`` removes one
      registration and is a no-op if *fn* is not registered. Without it,
      listeners stay attached for the store's lifetime.
    """

    def get_object(
        self, kind: str, key: str, version: int | None = None
    ) -> dict[str, Any] | None: ...

    def all_objects(self, kind: str) -> dict[str, dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class _TelemetryEmitter(Protocol):
    def record(self, signal: str, properties: dict[str, Any]) -> None: ...


class _NoOpEmitter:
    """
    The default emitter: no skills telemetry leaves the process. Signals are
    still built, so a transport can be installed without touching call sites.
    """

    def record(self, signal: str, properties: dict[str, Any]) -> None:
        return None


_NOOP_EMITTER: _TelemetryEmitter = _NoOpEmitter()

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_store: SkillStore | None = None
_emitter: _TelemetryEmitter = _NOOP_EMITTER
"""Never ``None``; "no emitter" is the no-op emitter."""


def set_store(store: Any) -> None:
    """Replaces the configured store. Reached through ``skills._set_store``."""
    global _store
    _store = store


def set_emitter(emitter: Any) -> None:
    """Replaces the telemetry emitter."""
    global _emitter
    _emitter = emitter


def clear_state() -> None:
    """Drops both the store and the emitter."""
    global _store, _emitter
    _store = None
    _emitter = _NOOP_EMITTER


def get_store() -> SkillStore | None:
    """The configured store, or ``None``. The only reader of the global."""
    return _store


def require_store() -> SkillStore:
    store = get_store()
    if store is None:
        raise RuntimeError(NO_STORE_MESSAGE)
    return store


def store_is_initialized(store: Any) -> bool:
    """
    Whether *store* has received its initial data.

    ``True`` if the store has no ``is_initialized()``. A probe that raises counts
    as not initialized. An uninitialized store is reported unavailable, which
    suppresses pruning so a slow start cannot delete managed files.
    """
    probe = getattr(store, "is_initialized", None)
    if not callable(probe):
        return True
    try:
        return bool(probe())
    except Exception:
        logger.warning(
            "The skill store's is_initialized() raised; treating the store as "
            "not yet initialized",
            exc_info=True,
        )
        return False


def emit(signal: str, properties: dict[str, Any]) -> None:
    """Records one signal. Never raises: a broken emitter must not fail a
    retrieval or reconcile."""
    try:
        _emitter.record(signal, properties)
    except Exception:
        logger.warning("Skills telemetry emitter raised; ignoring", exc_info=True)


def record_integrity_failure(
    skill_key: str,
    reason: str,
    *,
    reason_code: IntegrityReasonCode,
    version: Any = None,
    expected_hash: Any = None,
    observed_hash: str | None = None,
) -> None:
    """
    Records an integrity failure as one local log record and one signal.

    Carries hashes and byte counts only, never the skill body. The log record
    works even with telemetry off; it adds the event name, action, reason and
    ``reason_code``, in both the message text and ``extra["ld_skills"]``.
    """
    # Key and expected hash are store-supplied and could carry the skill body:
    # shape-check, then redact. Any new store-supplied field needs the same.
    safe_key = skill_key if is_valid_skill_key(skill_key) else "<invalid-key>"
    properties: dict[str, Any] = {"skill_key": safe_key, "language": _LANGUAGE}
    if is_valid_skill_version(version):
        properties["version"] = version
    if isinstance(expected_hash, str):
        properties["expected_hash"] = (
            expected_hash
            if _SHA256_HEX.match(expected_hash)
            else "<not-a-sha256-digest>"
        )
    if observed_hash is not None:
        properties["observed_hash"] = observed_hash

    # Reuse the signal's properties so both redact the same fields. Absent
    # fields are omitted, never null.
    record: dict[str, Any] = {
        "event": INTEGRITY_FAILURE_EVENT,
        "action": _ACTION_WITHHELD,
        "reason_code": reason_code,
        "reason": reason,
        **properties,
    }
    # sort_keys is part of the format: it makes the line stable (and identical
    # across SDK languages, apart from ``language``). Do not drop it.
    logger.error(
        "%s %s",
        INTEGRITY_FAILURE_EVENT,
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        extra={"ld_skills": record},
    )
    emit(_SIGNAL_INTEGRITY_FAILURE, properties)


def record_key_mismatch(requested: Any, served: Any) -> None:
    """
    Records a store answering under a key other than the one requested.

    **Log record only, no signal.** A substituted skill may indicate tampering,
    so it is logged under ``INTEGRITY_FAILURE_EVENT`` for existing SIEM rules to
    catch. The usual cause, though, is a broken store adapter (stale cache,
    colliding key), so it is kept out of product telemetry.
    ``record_version_mismatch`` follows the same rule.

    Both keys are shape-checked and redacted, regardless of call order.
    """
    record: dict[str, Any] = {
        "event": INTEGRITY_FAILURE_EVENT,
        "action": _ACTION_WITHHELD,
        "reason_code": "key_mismatch",
        "reason": (
            "the skill store answered under a different key than the one requested"
        ),
        "language": _LANGUAGE,
        # The key the caller asked for, as on every other record.
        "skill_key": requested if is_valid_skill_key(requested) else "<invalid-key>",
        # The key the store answered under, for diagnosing the adapter.
        "served_key": served if is_valid_skill_key(served) else "<invalid-key>",
    }
    # No hashes or version: verification passed, so neither disqualified it.
    logger.error(
        "%s %s",
        INTEGRITY_FAILURE_EVENT,
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        extra={"ld_skills": record},
    )


def record_version_mismatch(key: Any, requested: Any, served: Any) -> None:
    """
    Records a store answering a version pin with a different version.

    **Log record only, no signal**, for the reason ``record_key_mismatch``
    gives. The built-in stores answer a pin with that version or ``None``, so
    only a custom adapter reaches this. ``get_skill`` returns ``None`` for
    ``wrong_version``, so this record is what makes it visible.

    *served* is shape-checked, which also keeps it an integer in the JSON.
    *requested* is the caller's own pin and is logged as given.
    """
    record: dict[str, Any] = {
        "action": _ACTION_WITHHELD,
        "event": INTEGRITY_FAILURE_EVENT,
        "language": _LANGUAGE,
        "reason": (
            "the skill store answered with a different version than the one requested"
        ),
        "reason_code": "version_mismatch",
        # The version the store answered with, for diagnosing the adapter.
        "served_version": (
            served if is_valid_skill_version(served) else "<invalid-version>"
        ),
        "skill_key": key if is_valid_skill_key(key) else "<invalid-key>",
        # The version requested, as ``version`` means on every other record.
        "version": requested,
    }
    # No hashes: verification passed, so they did not disqualify it.
    logger.error(
        "%s %s",
        INTEGRITY_FAILURE_EVENT,
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        extra={"ld_skills": record},
    )


def record_materialized(
    skill_key: str, content_bytes: int, content_hash: str, reconcile_action: str
) -> None:
    """
    Records a materialization. Carries no filesystem path; paths are reported in
    the returned ``ReconcileReport`` instead.
    """
    emit(
        _SIGNAL_MATERIALIZED,
        {
            "skill_key": skill_key,
            "content_bytes": content_bytes,
            "content_hash": content_hash,
            "reconcile_action": reconcile_action,
            "language": _LANGUAGE,
        },
    )


def record_revoked(skill_key: str, version: Any) -> None:
    """Records a revocation: a prune that removed a formerly managed skill."""
    # Both fields come from the manifest, which is untrusted: shape-check, then
    # redact.
    safe_key = skill_key if is_valid_skill_key(skill_key) else "<invalid-key>"
    properties: dict[str, Any] = {
        "skill_key": safe_key,
        "removed_from_disk": True,
        "language": _LANGUAGE,
    }
    if is_valid_skill_version(version):
        properties["version"] = version
    emit(_SIGNAL_REVOKED, properties)


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedContent:
    """Content that passed integrity verification."""

    encoded: bytes
    """The verbatim bytes, exactly as hashed."""
    content_hash: str
    """The locally computed sha256 — never the caller's expected value."""


@dataclass(frozen=True)
class VerificationFailure:
    """Why content did not pass. The reason is safe to show a caller."""

    reason: str


def verified_bytes(
    key: str, content: str | bytes, expected_hash: str, version: int
) -> VerifiedContent | VerificationFailure:
    """
    Verifies content: size cap, then UTF-8 encoding, then sha256 hash.

    The fixed order means content failing several checks always reports the same
    ``reason_code``. A ``str`` (from the wire) is UTF-8 encoded here; ``bytes``
    (a ``Skill.content``) is hashed directly.

    Returns the verbatim bytes and the locally computed hash, or a
    ``VerificationFailure`` after recording the integrity failure.

    Runs at the accessor boundary and again before a write, because callers can
    construct a ``Skill`` directly. Do not skip the second pass.
    """
    encodable = True
    if isinstance(content, bytes):
        encoded = content
    else:
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate (e.g. from a "\ud800" JSON escape) has no UTF-8
            # encoding. The replacement bytes are only used to check the size
            # cap; ``not_utf8`` returns before they could be hashed.
            encodable = False
            encoded = content.encode("utf-8", errors="replace")

    if len(encoded) > MAX_SKILL_CONTENT_BYTES:
        reason = (
            f"content is {len(encoded)} bytes, over the "
            f"{MAX_SKILL_CONTENT_BYTES} byte cap"
        )
        record_integrity_failure(
            key,
            reason,
            reason_code="over_size_cap",
            version=version,
            expected_hash=expected_hash,
        )
        return VerificationFailure(reason)

    if not encodable:
        reason = "content is not encodable as UTF-8"
        record_integrity_failure(
            key,
            reason,
            reason_code="not_utf8",
            version=version,
            expected_hash=expected_hash,
        )
        return VerificationFailure(reason)

    # sha256, lowercase hex, over the verbatim bytes; no canonicalization.
    observed_hash = hashlib.sha256(encoded).hexdigest()
    if observed_hash != expected_hash:
        record_integrity_failure(
            key,
            "content hash mismatch",
            reason_code="hash_mismatch",
            version=version,
            expected_hash=expected_hash,
            observed_hash=observed_hash,
        )
        return VerificationFailure("content hash mismatch")

    return VerifiedContent(encoded=encoded, content_hash=observed_hash)


def verify_raw_skill(raw: Any) -> Skill | None:
    """
    Turns one untrusted raw store object into a ``Skill``, or returns ``None``.

    On failure, records the integrity failure and logs an error. Unverified
    content is never returned.
    """
    if not isinstance(raw, dict):
        record_integrity_failure(
            "<unknown>",
            "raw skill object is not an object",
            reason_code="not_an_object",
        )
        return None

    key = raw.get("key")
    if not is_valid_skill_key(key):
        record_integrity_failure(
            key if isinstance(key, str) else "<unknown>",
            "key is not a valid skill key",
            reason_code="invalid_key",
        )
        return None

    version = raw.get("version")
    if not is_valid_skill_version(version):
        record_integrity_failure(
            key, "version is not an integer >= 1", reason_code="invalid_version"
        )
        return None

    content = raw.get("content")
    if not isinstance(content, str):
        record_integrity_failure(
            key,
            "content is missing or not a string",
            reason_code="missing_content",
            version=version,
        )
        return None

    expected_hash = raw.get("contentHash")
    if not isinstance(expected_hash, str):
        record_integrity_failure(
            key,
            "contentHash is missing or not a string",
            reason_code="missing_content_hash",
            version=version,
        )
        return None

    verified = verified_bytes(key, content, expected_hash, version)
    if isinstance(verified, VerificationFailure):
        return None

    name = raw.get("name")
    description = raw.get("description")
    return Skill(
        key=key,
        version=version,
        content=verified.encoded,
        content_hash=verified.content_hash,
        name=name if isinstance(name, str) else None,
        description=description if isinstance(description, str) else None,
    )


def log_withholding_summary(subject: str, requested: int, resolved: int) -> None:
    """
    Logs one WARN per batch when content was withheld, with the counts.

    Makes withholding visible at WARN level, especially when nothing verified
    and the empty result would look like "no skills".
    """
    withheld = requested - resolved
    if withheld <= 0:
        return
    if resolved == 0:
        logger.warning(
            "All %d %s were withheld and no skill content is available. Every "
            "object failed verification — check that the delivered objects carry "
            "a contentHash matching the sha256 of their content.",
            requested,
            subject,
        )
        return
    logger.warning(
        "%d of %d %s were withheld and are unavailable; see the preceding errors "
        "for the per-skill reason.",
        withheld,
        requested,
        subject,
    )


def store_raised(exc: Exception) -> str:
    """The one wording for "the store could not answer", used by every path."""
    return f"the skill store raised {type(exc).__name__}: {exc}"


def list_raw_objects(
    store: SkillStore,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """
    Every raw object the store holds, as ``(objects, None)``, or
    ``({}, reason)`` if the store raised.

    One entry per *(key, version)*; use ``newest_by_key`` for one per key. A
    non-dict answer is treated as a broken store, not an empty one, so it cannot
    read downstream as "every skill was revoked".
    """
    try:
        objects = store.all_objects(SKILL_OBJECT_KIND)
    except Exception as exc:
        logger.error("Skill store raised while listing skills", exc_info=True)
        return {}, store_raised(exc)
    if not isinstance(objects, dict):
        logger.error(
            "Skill store listed skills as %s rather than an object",
            type(objects).__name__,
        )
        return {}, (
            f"the skill store listed skills as {type(objects).__name__} "
            "rather than an object"
        )
    return objects, None


def newest_by_key(objects: dict[str, dict[str, Any]]) -> list[tuple[str, Any]]:
    """
    The highest version of each skill key, paired with its store key (used to
    attribute failures when the object's own key is unusable).

    Objects without a usable key and version are **kept** so verification
    reports them; dropping them would let prune delete the last good copy on
    disk. They are dropped only when their key resolved from another version.
    """
    best: dict[str, tuple[str, Any]] = {}
    unusable: list[tuple[str, Any]] = []
    for object_key, raw in objects.items():
        skill_key = raw.get("key") if isinstance(raw, dict) else None
        version = raw.get("version") if isinstance(raw, dict) else None
        if not is_valid_skill_key(skill_key) or not is_valid_skill_version(version):
            unusable.append((object_key, raw))
            continue
        held = best.get(skill_key)
        if held is None or version > held[1]["version"]:
            best[skill_key] = (object_key, raw)
    withheld = [
        (object_key, raw)
        for object_key, raw in unusable
        # is_valid_skill_key first: an unhashable key cannot be looked up.
        if not (
            is_valid_skill_key(raw.get("key") if isinstance(raw, dict) else None)
            and raw["key"] in best
        )
    ]
    return list(best.values()) + withheld


# ---------------------------------------------------------------------------
# Resolution internals — shared with the materialization path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """One key resolved against a store: the skill, or why there is none."""

    reason: SkillOutcomeReason
    """
    Which of the five public outcomes this is; ``get_skill_result`` publishes it.
    No default, so every construction site must choose one.
    """
    skill: Skill | None = None
    error: str | None = None
    unavailable: bool = False
    """
    ``True`` when the store could not answer, rather than answered "no"; set
    exactly when ``reason`` is ``store_unavailable``. Suppresses pruning, so an
    outage cannot delete managed files.
    """


def resolve_from_store(
    store: SkillStore, key: str, wanted_version: int | None
) -> Resolution:
    """
    Fetches one key from *store* and verifies it.

    ``wanted_version`` is passed to the store (``None`` for the newest). Because
    the store is untrusted, an answer with a different key or version is still
    withheld.
    """
    try:
        raw = store.get_object(SKILL_OBJECT_KIND, key, wanted_version)
    except Exception as exc:
        logger.error("Skill store raised while retrieving '%s'", key, exc_info=True)
        return Resolution(
            reason="store_unavailable",
            error=store_raised(exc),
            unavailable=True,
        )

    if not isinstance(raw, dict):
        return Resolution(
            reason="absent",
            error=f"skill '{key}' is not available from the configured skill store",
        )

    skill = verify_raw_skill(raw)
    if skill is None:
        return Resolution(
            reason="integrity_failure",
            error=f"skill '{key}' failed integrity verification and was withheld",
        )
    if skill.key != key:
        # integrity_failure, not absent: a substituted skill is something
        # callers should fail closed on. Log record only; see
        # record_key_mismatch.
        record_key_mismatch(key, skill.key)
        return Resolution(
            reason="integrity_failure",
            error=(
                f"skill '{key}' is not available: the store answered under "
                f"key '{skill.key}'"
            ),
        )
    if wanted_version is not None and skill.version != wanted_version:
        # Log record only; see record_version_mismatch.
        record_version_mismatch(key, wanted_version, skill.version)
        return Resolution(
            reason="wrong_version",
            error=(
                f"skill '{key}' version {wanted_version} is not available "
                f"(the store holds version {skill.version})"
            ),
        )
    return Resolution(reason="ok", skill=skill)


def reference_target(item: SkillReference | str) -> tuple[str, int | None]:
    """Normalizes a reference or bare key into ``(key, wanted version)``; a
    bare key wants the newest version."""
    if isinstance(item, str):
        return item, None
    return item.key, item.version
