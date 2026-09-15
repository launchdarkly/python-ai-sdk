"""
Agent Skills — the internals ``skills`` and ``skills_fs`` both need.

Package-internal: nothing here is exported from ``launchdarkly_ai_server``
except the two constants that are public API, and the dependency runs one way —
this module imports neither ``skills`` nor ``skills_fs``.

It holds the store interface and the configured store, the telemetry emitter,
integrity verification, and store resolution. Each lives here in one copy so
that the accessors and the materialization path cannot disagree: about whether a
store is configured, about which signals exist, about what verification accepts,
or about how a raising store is handled.

Everything the store hands back is untrusted input; the transport is not part of
the trust boundary. Key, version, size, and content hash are revalidated here on
every pass.
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
The kind this SDK asks a store for.

Internal, and deliberately not exported from the package root: it is what the
accessors pass to ``SkillStore.get_object`` and ``SkillStore.all_objects``, and
a store adapter is free to map it onto whatever its transport uses underneath.
A store that needs to agree on a kind agrees with whatever the SDK hands it,
reached through ``launchdarkly_ai_server.skills_core``.
"""

MAX_SKILL_CONTENT_BYTES = 10 * 1024 * 1024
"""
Hard cap on skill content. Legitimately delivered skills are well under this
bound, so anything larger is withheld regardless of whether its hash checks out.

Set well above LaunchDarkly's own limit on purpose. This is a backstop against
absurd input, not a second enforcement of the real bound, so the headroom lets
that bound grow without this constant moving.

Not exported from the package root, unlike the on-disk and on-the-wire constants
beside it: it is a local enforcement bound rather than a value a caller needs to
agree with, and a caller pre-flighting "will my skill fit?" against it would be
reading the client's guess rather than the real limit. ``verified_bytes``
reports the bound in its reason string when it is what withheld content.
"""

_LANGUAGE = "python"

_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")
"""What a legitimate content hash looks like. Anything else is redacted before
it reaches telemetry: ``contentHash`` is untrusted, and a store that put the
skill body there would otherwise leak it into a signal."""

_SIGNAL_INTEGRITY_FAILURE = "AgentControl Skill Integrity Failure"
_SIGNAL_MATERIALIZED = "AgentControl Skill Materialized"
_SIGNAL_REVOKED = "AgentControl Skill Revoked Received"

INTEGRITY_FAILURE_EVENT = "ld.skills.integrity_failure"
"""
Stable event identity for the local integrity-failure log record.

A compatibility surface, not an implementation detail: this is the string a SIEM
matches on, so it must never be renamed.

It appears verbatim **in the message text**, not only in ``extra``. Severity
alone cannot discriminate — ``list_raw_objects`` and ``resolve_from_store`` in
this module also log ERROR when a store raises — and the stdlib's default
formatter drops ``extra`` entirely, so under a plain ``logging.basicConfig()``
an ``extra``-only record would be invisible.
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
]
"""
The closed ``reason_code`` vocabulary — one token per
``record_integrity_failure`` call site. Stable: a detection rule written against
these tokens keeps working, so adding one is a deliberate edit here rather than
a new string invented at the call site that needed it.
"""

INTEGRITY_REASON_CODES: frozenset[str] = frozenset(get_args(IntegrityReasonCode))
"""``IntegrityReasonCode`` as a runtime set, derived rather than restated."""

NO_STORE_MESSAGE = (
    "No skill store is configured, so skill content cannot be retrieved. Configure "
    'one with init_client(options={"skillStore": store}) — FDv2SkillStore receives '
    "content from LaunchDarkly, and InMemorySkillStore is available for local "
    "development and testing."
)
"""
The first thing a user sees when no store is configured, so it names both stores,
``FDv2SkillStore`` first because it is the answer in production. Callers match on
"skill store"; keep that phrase if the wording changes.
"""


# ---------------------------------------------------------------------------
# The store interface
# ---------------------------------------------------------------------------


class SkillStore(Protocol):
    """
    Structural interface every source of skill content satisfies.

    Duck-typed on purpose, mirroring how the LaunchDarkly client interface works
    in this package: pass any object carrying these methods.

    Three members are **optional**, and are deliberately not declared here: a
    Protocol member is required for structural compatibility, so declaring them
    would reject every store that does not implement them. Each is probed for
    instead, and each has a defined behaviour when absent.

    ``is_initialized()`` reports whether the store has received its initial data
    — for a delivery transport, whether a payload has arrived yet. Absent means
    initialized, which is right for a store populated by hand. It matters
    because "the store holds nothing" and "the store has not heard yet" are the
    same answer through ``all_objects``, and ``write_skills("*")`` would read the
    first as "every skill was revoked": see ``store_is_initialized``.

    ``add_listener(kind, fn)`` lets a delivery transport push updates, and
    ``remove_listener(kind, fn)`` lets a consumer such as ``watch_skills`` stop
    receiving them; it removes one occurrence of *fn* under *kind* and is a
    no-op when *fn* is not registered. A store offering the first should offer
    the second: consumers skip detaching when it is absent, so such a store
    works at the cost of a listener that lives as long as it does.

    The raw objects a store serves are wire-shaped, with camelCase field names::

        {"key": "pdf-extraction", "version": 2, "content": "---\\n...",
         "contentHash": "9f3a...", "name": "PDF Extraction", "description": "..."}

    **Version is part of the lookup identity, not a filter applied afterwards.**
    A delivery payload holds the newest version of every skill *and* every
    version any variation currently pins, so two versions of one key coexist
    routinely. An interface keyed by key alone cannot express "the one this
    variation pinned": it would answer with the newest, and the caller rejecting
    it turns a pinned reference into a missing skill. So ``get_object`` takes the
    wanted version, and ``version=None`` means "the newest you hold".

    ``all_objects`` returns one entry per *(key, version)* the store holds. Its
    dict keys are **opaque store-internal identifiers** — do not parse them, and
    do not assume one entry per skill key. Identity is read off each object's own
    ``key`` and ``version`` fields, which are revalidated here anyway because
    everything a store serves is untrusted.
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
    The default emitter.

    No skills telemetry leaves the process: ``client.track()`` is the wrong
    channel for it — that needs an LD context, spends the application's event
    volume, and lands in its data export. Signals are still constructed and
    recorded, so a transport can be installed behind this interface without
    touching a call site.
    """

    def record(self, signal: str, properties: dict[str, Any]) -> None:
        return None


_NOOP_EMITTER: _TelemetryEmitter = _NoOpEmitter()

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_store: SkillStore | None = None
_emitter: _TelemetryEmitter = _NOOP_EMITTER
"""Never ``None``: "no emitter installed" is spelled as the no-op, so ``emit``
has one code path instead of re-deciding on every signal."""


def set_store(store: Any) -> None:
    """
    Replaces the configured store. Reached through ``skills._set_store``, the
    documented injection point.
    """
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

    ``True`` for a store that does not implement the optional
    ``is_initialized()``, since a hand-populated store is never waiting for
    anything. Probed rather than declared on the Protocol for the reason
    ``SkillStore`` gives: a declared member would reject every store without it.

    This is what keeps ``write_skills("*")`` from reading a store that has not
    yet received a payload as an environment whose every skill was revoked.
    Retrieval through such a store is reported unavailable, which suppresses
    pruning — the same treatment a raising store gets, and for the same reason:
    deleting the application's files because content could not be retrieved would
    turn a slow boot into data loss.

    A probe that raises counts as not initialized. A store that cannot answer
    whether it is ready is not one to authorize deletions on.
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
    """
    Records one signal. Never raises into the calling operation — a broken
    emitter must not be able to fail a retrieval or a reconcile.
    """
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
    Records an integrity failure on both surfaces: one local log record, one
    product signal.

    Carries hashes and byte counts only — the skill body never appears in a
    signal, a log line, or an error message.

    The two are deliberately different sizes. The signal is product telemetry:
    no-op by default, with a fixed property set. The log record is the
    application's own detection path — the only one that works when telemetry is
    off — so it additionally carries the stable event name, the action taken, the
    human-readable reason, and the machine-parseable ``reason_code``.

    The record is emitted in two forms because neither alone is enough: the
    message text carries ``INTEGRITY_FAILURE_EVENT`` followed by compact JSON, so
    it survives ``logging.basicConfig()`` and is greppable and ``jq``-able under
    any handler configuration, and ``extra["ld_skills"]`` carries the same
    mapping unflattened for a structured handler that would rather not reparse.
    """
    # Both of these come off the wire, so neither may be echoed verbatim: a store
    # that set contentHash (or key) to the skill body would otherwise publish the
    # body itself. Shape-check, then redact. Every field the log record adds on
    # top is either a literal or SDK-authored, so the record introduces no new
    # untrusted value — anything added later needs this same treatment.
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

    # Spread the signal's properties rather than rebuilding them, so the record
    # cannot drift from the signal on the fields they share — in particular on
    # which of them are redacted and which are omitted. Absent optional fields
    # stay absent; the record never carries a null.
    record: dict[str, Any] = {
        "event": INTEGRITY_FAILURE_EVENT,
        "action": _ACTION_WITHHELD,
        "reason_code": reason_code,
        "reason": reason,
        **properties,
    }
    # ``sort_keys`` is part of the record's format rather than cosmetic: it is
    # what makes the serialized line stable for a given input, so a detection
    # rule can match on it. Do not drop it, and do not reorder the keys above
    # expecting the output to follow.
    logger.error(
        "%s %s",
        INTEGRITY_FAILURE_EVENT,
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        extra={"ld_skills": record},
    )
    emit(_SIGNAL_INTEGRITY_FAILURE, properties)


def record_materialized(
    skill_key: str, content_bytes: int, content_hash: str, reconcile_action: str
) -> None:
    """
    Records a materialization. Deliberately carries no ``target_path`` and no
    filesystem path of any kind — the same reasoning that keeps the skill body
    out of telemetry keeps the application's directory layout out. Paths live in the
    returned ``ReconcileReport``, which is user-facing API rather than telemetry.
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
    """
    Records a revocation — a prune that removed a formerly managed skill.

    Lives here with the other two recorders rather than at the prune site, so
    every signal this SDK can emit is visible in one place and nothing outside
    this module touches ``emit``.
    """
    # Both fields come off the manifest, which is untrusted — same rule as
    # ``record_integrity_failure``: shape-check, then redact, so a hand-edited
    # manifest cannot plant an arbitrary string in a signal.
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
    The whole content half of integrity verification: encode, size, hash.

    Accepts either shape content legitimately arrives in. Wire-shaped ``str``
    input — a raw store object's JSON string — is UTF-8 encoded here, once, and
    this is the only place that encode happens. ``bytes`` input is an already
    verified ``Skill.content`` being re-verified, and is hashed directly: those
    bytes are the verbatim value, so re-encoding does not apply.

    Returns the verbatim bytes and their locally computed sha256, or a
    human-readable reason — having already recorded the integrity signal, so the
    signal does not depend on which caller noticed. The hash handed back is
    always the one computed here, never the caller's expected value, which keeps
    an untrusted string out of ``Skill``.

    This runs twice per skill by design: once at the accessor boundary, and again
    immediately before a write, because a ``Skill`` can also be constructed
    directly by a caller. The second pass re-hashes bytes the first pass already
    hashed, which is negligible next to the write it guards — and carrying the
    first pass's verdict forward would put a "trust the value computed upstream"
    branch inside the one function whose job is not to.
    """
    if isinstance(content, bytes):
        encoded = content
    else:
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError:
            # json.loads turns a "\ud800" escape into an unpaired surrogate, which
            # has no UTF-8 encoding. There are no bytes the server could have
            # hashed, so this is not authentic content. Never use
            # errors="surrogatepass" here: that would fabricate bytes and could
            # satisfy the hash comparison.
            reason = "content is not encodable as UTF-8"
            record_integrity_failure(
                key,
                reason,
                reason_code="not_utf8",
                version=version,
                expected_hash=expected_hash,
            )
            return VerificationFailure(reason)

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

    # sha256, lowercase hex, over the verbatim bytes — no canonicalization and
    # no content parsing of any kind anywhere in the integrity path.
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
    Turns one untrusted raw store object into a ``Skill``, or withholds it.

    On any failure the skill is treated as missing, the integrity signal is
    recorded, and an error is logged. No unverified content is ever returned to
    user code.
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
    One WARN per run when content was withheld, naming the counts.

    Every individual withholding already records an integrity signal and an error
    line, but a caller reading logs at WARN sees neither — and the case that
    matters most is a run where *nothing* verified, because the result is then an
    empty list indistinguishable from "this project has no skills".

    Called once per batch, not once per skill, so a large withholding run does
    not itself become the noise.
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
    Every raw object the store holds, or the reason it could not answer.

    One entry per *(key, version)*, under keys that are opaque to this SDK — see
    ``SkillStore``. Callers that need one skill per key have to collapse the
    result themselves; ``newest_by_key`` does it.

    Returns the reason rather than raising, because both callers need the
    distinction between "no skills" and "the store is broken", worded
    identically.

    An answer that is not a mapping is a broken store, on the same footing as
    one that raised — **not** an empty one. Collapsing it to ``{}`` would make a
    store that served nothing usable indistinguishable from a store that holds
    no skills, which reads downstream as "every skill was revoked".
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
    One raw object per skill key — the highest version of each, paired with the
    store key it was served under.

    ``all_objects`` may hold several versions of one key, and both whole-store
    callers want one skill per key: ``all_skills``, because a list holding two
    versions of one key is not a set of skills, and the ``"*"`` reconcile,
    because ``<root>/<key>/SKILL.md`` is a single path. The store key is carried
    through because the reconcile attributes a failure to it when the object's
    own key is unusable.

    An object too malformed to carry a usable key and version is **kept**, so
    verification is what withholds it: dropped silently it would fall out of the
    requested set, and prune would then delete the last known-good copy on disk.
    The exception is an object whose key resolved from another version anyway —
    that key is already in the requested set, so keeping the malformed sibling
    would only report a withholding for a key that resolved.
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
        # ``is_valid_skill_key`` first: an unhashable key cannot be looked up.
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
    Which of the five public outcomes this resolution is.

    Declared first and **without a default**, so every construction site has to
    state which public token it maps to rather than inheriting one.

    ``get_skill_result`` publishes this value directly. It is carried as a token
    rather than recovered from ``error``, because pattern-matching prose for a
    decision a caller fails closed on is the fragility the typed outcome removes.

    Distinct from ``unavailable``, which answers a different question (may prune
    run?), but the two can only disagree by a bug: ``unavailable`` is ``True`` in
    exactly the ``store_unavailable`` case.
    """
    skill: Skill | None = None
    error: str | None = None
    unavailable: bool = False
    """
    ``True`` when the *store* could not answer — it raised — rather than when it
    answered "no". Only the former suppresses pruning: deleting managed files
    because a lookup failed would turn an outage into data loss.
    """


def resolve_from_store(
    store: SkillStore, key: str, wanted_version: int | None
) -> Resolution:
    """
    Fetches one key and verifies it — the sequence the accessors and the
    materialization path share, written once so the two cannot drift apart on
    the policy for a raising store.

    ``wanted_version`` goes *into* the lookup, because a store may hold several
    versions of one key and only it can pick between them; ``None`` asks for the
    newest. The equality check afterwards is kept as a **defense**, not as the
    selection mechanism: the store is untrusted, so an answer that is not the
    version that was asked for is withheld rather than returned. The key is
    checked the same way and for the same reason: identity is read off the
    object itself, so an answer served under a different key would otherwise be
    returned under the caller's key while carrying its own.
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
        # ``integrity_failure`` rather than ``absent``: content was delivered
        # and its identity did not verify, which is the one token a caller is
        # expected to fail closed on. Reporting ``absent`` would file a store
        # that substitutes one skill for another in the bucket the same caller
        # is invited to tolerate. It is not ``wrong_version`` either — that
        # token names a version mismatch specifically, and there is deliberately
        # no ``wrong_key`` to parallel it.
        return Resolution(
            reason="integrity_failure",
            error=(
                f"skill '{key}' is not available: the store answered under "
                f"key '{skill.key}'"
            ),
        )
    if wanted_version is not None and skill.version != wanted_version:
        return Resolution(
            reason="wrong_version",
            error=(
                f"skill '{key}' version {wanted_version} is not available "
                f"(the store holds version {skill.version})"
            ),
        )
    return Resolution(reason="ok", skill=skill)


def reference_target(item: SkillReference | str) -> tuple[str, int | None]:
    """Normalises a reference-or-key into ``(key, wanted version)``.

    A bare string means "the latest version the store holds".
    """
    if isinstance(item, str):
        return item, None
    return item.key, item.version
