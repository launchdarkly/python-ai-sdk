"""
Agent Skills — the internals ``skills`` and ``skills_fs`` both need.

Extracted so the two layers above it share one implementation through an
explicit surface instead of reaching into each other's privates. Everything
here is package-internal — nothing in this module is exported from
``launchdarkly_ai_server`` except the two constants that are public API — and
the dependency runs one way: this module imports neither ``skills`` nor
``skills_fs``.

What lives here, and why it has to be one copy:

- **The store seam and the configured store.** One place holds the store, so
  the accessors and the materialization path cannot disagree about whether one
  is configured.
- **The telemetry seam.** Every signal the feature can emit is constructed by a
  ``record_*`` function in this file and nowhere else, which is what makes the
  three-signal allowlist enforceable by reading one section. ``emit`` is never
  called from outside this module.
- **Integrity verification.** ``verified_bytes`` runs twice per skill by design
  — once at the accessor boundary, and again immediately before a write, since a
  ``Skill`` can also be constructed directly by a caller. Sharing the
  implementation is what keeps the two passes from drifting — the signal's
  property keys must match whichever layer caught the defect.
- **Store resolution.** ``resolve_from_store`` is the fetch-and-verify sequence
  the accessors and the reconcile share, so its call sites cannot drift apart —
  in particular on how a raising store is handled.

Everything the store hands back is untrusted input; the transport is not part of
the trust boundary. Key, version, size, and content hash are revalidated here on
every pass.

The store and emitter are injected through ``skills._set_store`` and
``skills._set_emitter_for_testing`` — those names are the documented seam,
and they delegate here.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .types import Skill, SkillReference
from .types_validation import is_valid_skill_key, is_valid_skill_version

logger = logging.getLogger(__name__)

SKILL_OBJECT_KIND = "skill"
"""
The kind this SDK asks a store for.

An **internal seam value**, deliberately not exported from the package root. It
is the string ``skills.py`` and ``skills_fs.py`` pass to ``SkillStore.get_object``
and ``SkillStore.all_objects``, and a store adapter is free to map it onto
whatever the transport underneath actually uses — a delivery payload may well
carry skills under a broader kind with a narrower category, in which case
translating that pair to this one value is the adapter's job.

Exporting it would publish an SDK-side seam string as though it were the wire
contract, which is a claim this side cannot make and would be hard to walk back
once a caller depends on it. A store that needs to agree on a kind agrees with
whatever the SDK hands it, which is this constant reached through
``launchdarkly_ai_server.skills_core``.
"""

MAX_SKILL_CONTENT_BYTES = 64 * 1024
"""
Hard cap on skill content. Legitimately delivered skills are well under this
bound, so anything larger is withheld regardless of whether its hash checks out.

Deliberately **not** exported from the package root, unlike the on-disk and
on-the-wire constants beside it. Those are values this SDK defines and a caller
may need to agree with; this one is a local enforcement bound on content the
platform produces, so publishing it would semver-lock a number this side does
not own — and a caller pre-flighting "will my skill fit?" against it would be
reading the client's guess rather than the real limit. The reason string from
``verified_bytes`` already reports the bound when it is what withheld content.
"""

_LANGUAGE = "python"

_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")
"""What a legitimate content hash looks like. Anything else is redacted before
it reaches telemetry — ``contentHash`` is attacker-controlled, and a store that
put the skill body there would otherwise leak it into a signal."""

_SIGNAL_INTEGRITY_FAILURE = "AgentControl Skill Integrity Failure"
_SIGNAL_MATERIALIZED = "AgentControl Skill Materialized"
_SIGNAL_REVOKED = "AgentControl Skill Revoked Received"

NO_STORE_MESSAGE = (
    "No skill store is configured, so skill content cannot be retrieved. Configure "
    'one with init_client(options={"skillStore": store}) — InMemorySkillStore is '
    "available for local development and testing."
)


# ---------------------------------------------------------------------------
# The store seam
# ---------------------------------------------------------------------------


class SkillStore(Protocol):
    """
    Structural interface every source of skill content satisfies.

    Duck-typed on purpose, mirroring how the LaunchDarkly client interface works
    in this package: pass any object carrying these methods.

    ``add_listener(kind, fn)`` is part of the seam but
    **optional**, which is why it is deliberately not declared here: a Protocol
    member is required for structural compatibility, so declaring it would reject
    every store that does not implement it. Nothing in this module calls it — it
    exists for the delivery transport to push updates through.

    The raw objects a store serves are wire-shaped, with camelCase field names
    identical across language implementations::

        {"key": "pdf-extraction", "version": 2, "content": "---\\n...",
         "contentHash": "9f3a...", "name": "PDF Extraction", "description": "..."}

    **Version is part of the lookup identity, not a filter applied afterwards.**
    A delivery payload holds the newest version of every skill *and* every
    version any variation currently pins, so two versions of one key coexist
    routinely. A seam keyed by key alone cannot express "the one this variation
    pinned": it would answer with the newest and the caller would then have to
    reject it, which turns a pinned reference into a missing skill. So
    ``get_object`` takes the wanted version, and ``version=None`` means "the
    newest you hold".

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
# Telemetry seam
# ---------------------------------------------------------------------------


class _TelemetryEmitter(Protocol):
    def record(self, signal: str, properties: dict[str, Any]) -> None: ...


class _NoOpEmitter:
    """
    The default emitter.

    No skills telemetry leaves the process in this release: ``client.track()`` is
    the wrong channel (it needs an LD context, spends the customer's event
    volume, and lands in their data export), and the diagnostic-event channel
    that would be right has no wrapper-SDK extension point yet. Signals are
    recorded through this seam so the eventual transport drops in behind it
    without touching a single call site.
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
    Replaces the configured store.

    Reached through ``skills._set_store``, which is the documented seam; see that
    function for who calls it and why it has no test-only twin.
    """
    global _store
    _store = store


def set_emitter(emitter: Any) -> None:
    """Replaces the telemetry emitter. Reached through
    ``skills._set_emitter_for_testing``."""
    global _emitter
    _emitter = emitter


def clear_state() -> None:
    """Drops both the store and the emitter. Reached through ``skills._clear_state``."""
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
    version: Any = None,
    expected_hash: Any = None,
    observed_hash: str | None = None,
) -> None:
    """
    Records an integrity failure. Carries hashes and byte counts only — the skill
    body never appears in a signal, a log line, or an error message.
    """
    # Both of these come off the wire, so neither may be echoed verbatim: a store
    # that set contentHash (or key) to the skill body would otherwise publish the
    # body itself. Shape-check, then redact.
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

    logger.error("Skill '%s' failed integrity verification: %s", safe_key, reason)
    emit(_SIGNAL_INTEGRITY_FAILURE, properties)


def record_materialized(
    skill_key: str, content_bytes: int, content_hash: str, reconcile_action: str
) -> None:
    """
    Records a materialization. Deliberately carries no ``target_path`` and no
    filesystem path of any kind — the same reasoning that keeps the skill body
    out of telemetry keeps the customer's directory layout out. Paths live in the
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

    Lives here with the other two recorders rather than at the prune site so the
    signal allowlist is maintained in one place: every signal this SDK can emit
    is visible in this section of this module, and nothing outside it touches
    ``emit``.
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
    signal's property set cannot depend on which caller noticed. The hash handed
    back is the one computed here, never the caller's expected value: the two are
    equal on this path by construction, and returning the locally derived one
    keeps an attacker-supplied string out of ``Skill``.

    The two outcomes are distinct types rather than a ``tuple | str`` union so a
    call site reads as "verification failed" instead of "the result is a string",
    and so a future success payload carrying a ``str`` cannot silently invert the
    discrimination.

    This runs twice per skill by design: once at the accessor boundary, and again
    immediately before a write, because a ``Skill`` can also be constructed
    directly by a caller. Sharing the implementation is what keeps those two
    passes from drifting — the property keys must match.

    The second pass re-hashes bytes the first pass already hashed. That
    redundancy is deliberate: it is negligible next to the write it guards, and
    carrying the first pass's verdict forward would put a "trust the value
    computed upstream" branch inside the one function whose entire job is not to.
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
                key, reason, version=version, expected_hash=expected_hash
            )
            return VerificationFailure(reason)

    if len(encoded) > MAX_SKILL_CONTENT_BYTES:
        reason = (
            f"content is {len(encoded)} bytes, over the "
            f"{MAX_SKILL_CONTENT_BYTES} byte cap"
        )
        record_integrity_failure(
            key, reason, version=version, expected_hash=expected_hash
        )
        return VerificationFailure(reason)

    # sha256, lowercase hex, over the verbatim bytes — no canonicalization and
    # no content parsing of any kind anywhere in the integrity path.
    observed_hash = hashlib.sha256(encoded).hexdigest()
    if observed_hash != expected_hash:
        record_integrity_failure(
            key,
            "content hash mismatch",
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
        record_integrity_failure("<unknown>", "raw skill object is not an object")
        return None

    key = raw.get("key")
    if not is_valid_skill_key(key):
        record_integrity_failure(
            key if isinstance(key, str) else "<unknown>",
            "key is not a valid skill key",
        )
        return None

    version = raw.get("version")
    if not is_valid_skill_version(version):
        record_integrity_failure(key, "version is not an integer >= 1")
        return None

    content = raw.get("content")
    if not isinstance(content, str):
        record_integrity_failure(
            key, "content is missing or not a string", version=version
        )
        return None

    expected_hash = raw.get("contentHash")
    if not isinstance(expected_hash, str):
        record_integrity_failure(
            key, "contentHash is missing or not a string", version=version
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
    log line, but a caller reading logs at WARN sees neither. That matters most in
    the case where *nothing* verified — a payload built before ``contentHash`` is
    populated, say — because the feature then returns an empty result that is
    indistinguishable from "this project has no skills". A run-level summary is
    the difference between a silent no-op and a visible one.

    Called once per batch retrieval, not once per skill, so a large withholding
    run does not itself become the noise.
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
    distinction between "no skills" and "the store is broken" — and they need it
    worded identically. Letting the exception out instead would make each of
    them re-derive the log line and the message, which is the drift this module
    exists to prevent.
    """
    try:
        objects = store.all_objects(SKILL_OBJECT_KIND)
    except Exception as exc:
        logger.error("Skill store raised while listing skills", exc_info=True)
        return {}, store_raised(exc)
    return (objects if isinstance(objects, dict) else {}), None


def newest_by_key(objects: dict[str, dict[str, Any]]) -> list[tuple[str, Any]]:
    """
    One raw object per skill key — the highest version of each, paired with the
    store key it was served under.

    ``all_objects`` may hold several versions of one key, and both callers that
    consume the whole store want one skill per key: ``all_skills`` because a list
    holding two versions of one key is not a set of skills, and the ``"*"``
    reconcile because ``<root>/<key>/SKILL.md`` is a single path and writing it
    twice in one run is a bug rather than a policy.

    The store key is carried through rather than discarded because the reconcile
    attributes a failure to it when the object's own key is unusable.

    Objects too malformed to carry a usable key and version are **kept**, not
    dropped, so verification is what withholds them: a silently dropped object
    falls out of the requested set, and prune would then delete the last
    known-good copy already on disk.
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
    return list(best.values()) + unusable


# ---------------------------------------------------------------------------
# Resolution internals — shared with the materialization path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """One key resolved against a store: the skill, or why there is none."""

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
    materialization path share.

    Written once on purpose, so the call sites cannot drift apart — in
    particular on the policy for a raising store.

    ``wanted_version`` goes *into* the lookup, because a store may hold several
    versions of one key and only it can pick between them; ``None`` asks for the
    newest. The equality check afterwards is kept as a **defense**, not as the
    selection mechanism: the store is untrusted, so an answer that is not the
    version that was asked for is withheld rather than returned.
    """
    try:
        raw = store.get_object(SKILL_OBJECT_KIND, key, wanted_version)
    except Exception as exc:
        logger.error("Skill store raised while retrieving '%s'", key, exc_info=True)
        return Resolution(error=store_raised(exc), unavailable=True)

    if not isinstance(raw, dict):
        return Resolution(
            error=f"skill '{key}' is not available from the configured skill store"
        )

    skill = verify_raw_skill(raw)
    if skill is None:
        return Resolution(
            error=f"skill '{key}' failed integrity verification and was withheld"
        )
    if wanted_version is not None and skill.version != wanted_version:
        return Resolution(
            error=(
                f"skill '{key}' version {wanted_version} is not available "
                f"(the store holds version {skill.version})"
            )
        )
    return Resolution(skill=skill)


def reference_target(item: SkillReference | str) -> tuple[str, int | None]:
    """Normalises a reference-or-key into ``(key, wanted version)``.

    A bare string means "the latest version the store holds".
    """
    if isinstance(item, str):
        return item, None
    return item.key, item.version
