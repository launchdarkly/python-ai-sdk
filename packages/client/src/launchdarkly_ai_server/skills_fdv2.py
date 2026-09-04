"""
Agent Skills — the FDv2 delivery transport.

The store implementation that actually talks to LaunchDarkly. It sits *below*
the ``SkillStore`` seam, not above it: it produces raw wire objects in the shape
``skills_core`` documents, and everything above — the accessors, integrity
verification, the ``Skill`` dataclass, materialization — is unchanged and
unaware of it. That is the whole point of the seam, and the fact that replacing
the transport design wholesale cost nothing above this line is the evidence it
was drawn in the right place.

Layering::

    launchdarkly_ai_server
      └─ SkillStore protocol (skills_core)     ── duck-typed accessor surface
            └─ FDv2SkillStore (this module)    ── deserialize, hold, serve
                  └─ the SDK-facing FDv2 channel on FDCore
                     GET /sdk/poll, GET /sdk/stream, authenticated with the
                     environment's server-side SDK key

Dependencies run one way. This module imports ``skills_core`` for the seam's
kind constant and nothing else from the feature; ``skills.py`` and
``skills_fs.py`` do not import it. It uses only the standard library, so the
content path adds no dependency to a package whose sole runtime dependency is
``opentelemetry-api`` and whose LaunchDarkly base-SDK dependency is optional.

**There is no bespoke private route here, deliberately.** An earlier design had
this adapter poll ``/private/flagdlv/payloads/{id}/latest/obj/skill/{key}``.
Those are gonfalon private endpoints authenticated by Cognito machine-token
OAuth scopes with no per-tenant authorization; the security review ruled out
both relaxing that auth and shipping a machine credential to a customer host.
This transport uses the genuinely SDK-facing channel instead, which is also the
channel payload signing will eventually cover. Do not reintroduce the private
route.

What this module does *not* do, on purpose:

- **It does not verify content.** Verification lives at the accessor boundary in
  ``skills_core`` so that it applies to every store equally, including
  ``InMemorySkillStore`` and a customer's own. A transport that verified would
  make integrity depend on which store you configured.
- **It does not skip verification when the wire envelope has no
  ``contentHash``.** See ``_SkillObjectSet.put`` and ``StoreDiagnostics``: a
  hashless object is stored verbatim and *withheld* by verification with
  ``missing_content_hash``, and this module's job is to make that outcome loud
  rather than to paper over it.
- **It does not evaluate anything.** No flags, no segments, no targeting. Skills
  have no targeting; the SDK key fully determines the payload.
"""

from __future__ import annotations

import json
import logging
import random
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .skills_core import SKILL_OBJECT_KIND
from .types_validation import is_valid_skill_version

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The wire contract
# ---------------------------------------------------------------------------

FDV2_OBJECT_KIND = "inline-resource"
"""
The FDv2 ``kind`` skills are delivered under.

Distinct from ``skills_core.SKILL_OBJECT_KIND`` (``"skill"``), which is the
*seam* value the SDK asks a store for. Translating this pair — kind
``inline-resource`` plus category ``skill`` — onto that single value is exactly
the adapter's job, and the reason ``SKILL_OBJECT_KIND`` is documented as a seam
string rather than as the wire contract.
"""

FDV2_OBJECT_CATEGORY = "skill"
"""The ``category`` that narrows ``inline-resource`` to an agent skill."""

DEFAULT_BASE_URI = "https://sdk.launchdarkly.com"
"""Where the SDK-facing FDv2 endpoints live. Overridable for Federal instances,
private instances, and the fake endpoint the tests run against."""

POLL_PATH = "/sdk/poll"
STREAM_PATH = "/sdk/stream"

SDK_DATA_MODEL_VERSION = 1
"""
The ``mv`` request parameter — the SDK data model version this adapter speaks.

Overridable through ``FDv2SkillStore(data_model_version=...)`` because it is the
one request parameter this side cannot verify: the LaunchDarkly base SDK's own
FDv2 data source does not send ``mv`` at all today, and the streamer branch that
carries skills is unmerged, so the value the server expects has not been
observed. Confirm it with FDN before Beta rather than trusting this default.
"""

_EVENT_SERVER_INTENT = "server-intent"
_EVENT_PUT_OBJECT = "put-object"
_EVENT_DELETE_OBJECT = "delete-object"
_EVENT_PAYLOAD_TRANSFERRED = "payload-transferred"
_EVENT_HEARTBEAT = "heart-beat"
_EVENT_GOODBYE = "goodbye"
_EVENT_ERROR = "error"

_INTENT_TRANSFER_FULL = "xfer-full"
_INTENT_TRANSFER_CHANGES = "xfer-changes"
_INTENT_TRANSFER_NONE = "none"

_ENVELOPE_FIELDS = ("contentType", "content", "contentHash", "name", "description")
"""
The skill object envelope's fields, copied through verbatim.

``contentHash`` is listed here and is the field the whole content path waits on;
see ``StoreDiagnostics``. Nothing here is coerced, defaulted, or normalised —
everything a store serves is untrusted input and is revalidated above the seam,
so a transport that "helpfully" filled in a field would be forging the very
thing verification exists to check.
"""

Mode = Literal["stream", "poll"]

_MOBILE_KEY_PREFIX = "mob-"
_SERVER_KEY_PREFIX = "sdk-"
_CLIENT_SIDE_ID = re.compile(r"\A[0-9a-f]{20,}\Z")
"""
A client-side environment ID: bare lowercase hex, no prefix. Server-side keys
and mobile keys both carry a prefix, so "hex with no prefix" is an
unambiguous client-side credential rather than a heuristic.
"""


# ---------------------------------------------------------------------------
# Server-side only
# ---------------------------------------------------------------------------


def _require_server_side_credential(sdk_key: str) -> None:
    """
    Refuses a mobile key or a client-side environment ID.

    Skills are for server-side agent runtimes. The payload assignment that
    carries them is shared by every auth type, so the skill payload ID is
    appended for mobile and environment-ID auth too — which means a client-side
    credential may well *succeed* against these endpoints and deliver
    customer-confidential skill content to a client-side process. Failing here
    is the SDK-side half of that boundary; excluding skills at assignment time
    is the platform-side half, and is an open ask on FDN (design §3.1c).

    Raises ``ValueError`` rather than logging, because there is no degraded mode
    that is correct: a store built on the wrong credential should not exist.
    """
    if not isinstance(sdk_key, str) or not sdk_key.strip():
        raise ValueError(
            "FDv2SkillStore requires a LaunchDarkly server-side SDK key "
            "(sdk-...); none was given."
        )
    key = sdk_key.strip()
    if key.startswith(_MOBILE_KEY_PREFIX):
        raise ValueError(
            "FDv2SkillStore was given a mobile key (mob-...). Agent Skills are a "
            "server-side feature: skill content is customer-confidential and is "
            "never delivered to a mobile or client-side process. Use the "
            "environment's server-side SDK key (sdk-...)."
        )
    if _CLIENT_SIDE_ID.match(key):
        raise ValueError(
            "FDv2SkillStore was given what looks like a client-side environment "
            "ID. Agent Skills are a server-side feature: skill content is "
            "customer-confidential and is never delivered to a client-side "
            "process. Use the environment's server-side SDK key (sdk-...)."
        )
    if not key.startswith(_SERVER_KEY_PREFIX):
        # Not rejected: private instances and test doubles issue credentials that
        # do not carry the public prefix, and refusing them would break a
        # deployment that is perfectly correct. The two shapes above are refused
        # because they are unambiguously *not* server-side.
        logger.warning(
            "The credential given to FDv2SkillStore does not look like a "
            "LaunchDarkly server-side SDK key (sdk-...). Skills are delivered "
            "only to server-side credentials; if this is a client-side or mobile "
            "credential the connection will be rejected or will deliver nothing."
        )


# ---------------------------------------------------------------------------
# Diagnostics — and the contentHash gap in particular
# ---------------------------------------------------------------------------


@dataclass
class StoreDiagnostics:
    """
    What the transport has seen. Read-only from a caller's perspective.

    Not part of the ``SkillStore`` seam — nothing above the seam reads this — but
    the difference between "this environment has no skills" and "every skill was
    withheld" is the single most confusing failure this feature can produce, and
    a counter a caller can assert on beats reading logs.
    """

    payloads_transferred: int = 0
    """Completed ``payload-transferred`` commits since the store started."""
    skill_objects_received: int = 0
    """``put-object`` events identified as skills, across all payloads."""
    objects_ignored: int = 0
    """Objects skipped because they were not skills — flags, segments, and any
    future kind. Skipping is the contract, not a failure; the count exists so a
    mixed payload is visibly mixed."""
    objects_revoked: int = 0
    """``delete-object`` events applied to skills."""
    hashless_objects: int = 0
    """
    Skill objects whose envelope carried no ``contentHash``.

    **Nonzero means skills are being withheld.** Verification withholds a
    hashless object with ``missing_content_hash``, so every one of these is a
    skill that will never resolve. The field exists so that outcome is a number
    a caller can read rather than an empty store they have to explain.
    """
    connection_failures: int = 0
    """Recoverable transport failures since the last successful transfer."""
    last_error: str | None = None
    """The most recent transport error, if any. Human-readable; do not parse."""


# ---------------------------------------------------------------------------
# Deserialization — where objectVersion is not version
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tombstone:
    """A ``delete-object`` narrowed to the identity it revokes."""

    key: str
    object_version: int | None


def is_skill_event(data: Any) -> bool:
    """
    Whether one ``put-object`` / ``delete-object`` payload is a skill.

    ``kind == "inline-resource" and category == "skill"``, and nothing else. Both
    halves are required: ``inline-resource`` is a broad kind that may carry other
    categories, and flags and segments omit ``category`` entirely.

    Every other kind is **ignored, not rejected**. An environment's payload
    assignment carries the flagging payload alongside the agent-skill payload, so
    a connection delivers flag and segment objects as a matter of course. Erroring
    on them would turn a normal payload into a permanent failure — which is
    exactly the unknown-kind reconnect loop this feature must not reproduce.
    """
    if not isinstance(data, dict):
        return False
    return (
        data.get("kind") == FDV2_OBJECT_KIND
        and data.get("category") == FDV2_OBJECT_CATEGORY
    )


def seam_object_from_put(data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Translates one FDv2 skill ``put-object`` into a seam-shaped raw object.

    ``None`` when the event cannot be filed at all — only when ``key`` is not a
    string, since a keyless object has no identity to store it under and no key
    to attribute a failure to. Every other defect is carried through verbatim so
    that *verification* withholds it, with a reason code and an integrity signal,
    rather than the transport dropping it silently. A silent drop is
    indistinguishable from "no such skill" and would additionally let a prune
    delete the last known-good copy on disk.

    **The translation this whole module exists to get right:**

        wire ``objectVersion``  →  seam ``version``      (the skill's own version)
        wire ``version``        →  dropped                (the *payload* version)

    ``objectVersion`` is what a ``{key, version}`` reference pins. ``version`` is
    the version of the payload the object arrived in — it changes when anything
    in the environment changes, including a flag that has nothing to do with
    skills. Reading it as the skill's version resolves the wrong content with no
    error anywhere: the object verifies, the hash matches, and the caller is
    handed a skill under a version number that means nothing. Flags and segments
    carry only ``version``, which is why the two fields look interchangeable and
    are not.
    """
    key = data.get("key")
    if not isinstance(key, str) or not key:
        logger.warning(
            "An FDv2 skill put-object carried no string 'key' and could not be "
            "stored under any identity; it was dropped."
        )
        return None

    raw: dict[str, Any] = {"key": key}

    # The single translation. Written as a membership test rather than a `.get`
    # default so an explicitly-null objectVersion stays null and reaches
    # verification as `invalid_version`, instead of being invented here.
    if "objectVersion" in data:
        raw["version"] = data["objectVersion"]

    envelope = data.get("object")
    if isinstance(envelope, dict):
        for wire_field in _ENVELOPE_FIELDS:
            if wire_field in envelope:
                raw[wire_field] = envelope[wire_field]
    return raw


def tombstone_from_delete(data: dict[str, Any]) -> _Tombstone | None:
    """
    Narrows one FDv2 skill ``delete-object`` to the identity it revokes.

    A delete for an inline resource **is revocation** — the object leaves the
    payload, this store drops it, the accessors stop resolving it, and the next
    reconcile prunes its files. Same ``objectVersion`` translation as a put.

    ``object_version`` of ``None`` means the delete named no usable version, and
    is read as "revoke every version of this key". That is the safe direction:
    the alternative is ignoring an unparseable revocation and continuing to serve
    content LaunchDarkly has withdrawn.
    """
    key = data.get("key")
    if not isinstance(key, str) or not key:
        logger.warning(
            "An FDv2 skill delete-object carried no string 'key'; it was ignored."
        )
        return None
    object_version = data.get("objectVersion")
    return _Tombstone(
        key=key,
        object_version=object_version
        if is_valid_skill_version(object_version)
        else None,
    )


# ---------------------------------------------------------------------------
# The held object set
# ---------------------------------------------------------------------------


class _SkillObjectSet:
    """
    Raw skill objects held in memory, keyed by ``(key, objectVersion)``.

    Lookup semantics are deliberately identical to ``InMemorySkillStore``'s, down
    to the fall-through to a version-less entry, so that the store a caller
    configures cannot change how a pinned reference resolves. They are
    reimplemented here rather than inherited because the transport needs two
    operations a hand-populated store does not have — ``delete`` and the atomic
    ``replace`` a full transfer requires — and reaching into another store's
    privates to get them would couple the two far harder than a test that asserts
    they agree. ``test_skills_fdv2.py`` carries that parity test.

    Several versions of one key coexist, because they coexist in a real payload:
    the newest version of every skill plus every version a variation currently
    pins. An object too malformed to carry a usable version is still held, under
    its key alone, so verification withholds it with a signal rather than the
    transport dropping it into indistinguishable absence.
    """

    def __init__(self) -> None:
        self._versions: dict[str, dict[int, dict[str, Any]]] = {}
        self._loose: dict[str, dict[str, Any]] = {}

    def put(self, raw: dict[str, Any]) -> None:
        key = raw["key"]
        version = raw.get("version")
        if is_valid_skill_version(version):
            self._versions.setdefault(key, {})[version] = raw
        else:
            self._loose[key] = raw

    def delete(self, tombstone: _Tombstone) -> list[dict[str, Any]]:
        """
        Removes what *tombstone* revokes; returns the raw objects that went away.

        A tombstone with no usable version removes every version of the key — see
        ``tombstone_from_delete`` for why that is the safe reading.
        """
        removed: list[dict[str, Any]] = []
        if tombstone.object_version is None:
            held = self._versions.pop(tombstone.key, {})
            removed.extend(held.values())
            loose = self._loose.pop(tombstone.key, None)
            if loose is not None:
                removed.append(loose)
            return removed

        held = self._versions.get(tombstone.key, {})
        gone = held.pop(tombstone.object_version, None)
        if gone is not None:
            removed.append(gone)
        if not held:
            self._versions.pop(tombstone.key, None)
        return removed

    def get(self, key: str, version: int | None) -> dict[str, Any] | None:
        held = self._versions.get(key, {})
        if version is not None:
            # Fall through to the version-less entry when the pin matches nothing
            # well-formed, so a malformed object reaches verification and is
            # withheld with a signal rather than reading as simply absent.
            return held.get(version) or self._loose.get(key)
        if held:
            return held[max(held)]
        return self._loose.get(key)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """One entry per ``(key, version)``, under keys opaque to the SDK."""
        out: dict[str, dict[str, Any]] = {
            f"{key}:{version}": raw
            for key, versions in self._versions.items()
            for version, raw in versions.items()
        }
        out.update(self._loose)
        return out

    def all_raw(self) -> list[dict[str, Any]]:
        return list(self.snapshot().values())

    def replace_with(self, other: _SkillObjectSet) -> None:
        """Adopts *other*'s contents wholesale — how a full transfer commits."""
        self._versions = other._versions
        self._loose = other._loose

    def copy(self) -> _SkillObjectSet:
        clone = _SkillObjectSet()
        clone._versions = {key: dict(v) for key, v in self._versions.items()}
        clone._loose = dict(self._loose)
        return clone

    def __len__(self) -> int:
        return sum(len(v) for v in self._versions.values()) + len(self._loose)


# ---------------------------------------------------------------------------
# The protocol state machine — pure, no I/O
# ---------------------------------------------------------------------------


@dataclass
class _Change:
    """One committed change, as handed to a listener."""

    raw: dict[str, Any]


@dataclass
class _TransferOutcome:
    """What one event did. Aggregated by the caller; nothing here does I/O."""

    committed: bool = False
    changes: list[dict[str, Any]] = field(default_factory=list)
    basis: str | None = None
    fatal: str | None = None
    disconnect: str | None = None


class _ProtocolReader:
    """
    Applies FDv2 events to an object set. Pure — no sockets, no threads, no clock.

    Split out so the protocol is testable without a server: every wire case in
    ``test_skills_fdv2.py`` drives this directly, and the HTTP layer above it only
    has to turn bytes into ``(event name, data)`` pairs.

    **Changes are buffered and committed at ``payload-transferred``**, matching
    how the base SDK's FDv2 data source applies a change set. A payload version
    is the unit of consistency: applying half of one would publish a state the
    server never described, and on a full transfer it would briefly empty the
    store — which, with pruning on, is the difference between a reconcile and
    deleting a customer's skill files. Listeners therefore fire once per commit,
    not once per object, which is also exactly the granularity the re-reconcile
    wants.
    """

    def __init__(self, committed: _SkillObjectSet) -> None:
        self._committed = committed
        self._intent: str | None = None
        self._pending: _SkillObjectSet | None = None
        self._changes: list[dict[str, Any]] = []
        self.diagnostics = StoreDiagnostics()

    # -- events ------------------------------------------------------------

    def handle(self, name: str, data: Any) -> _TransferOutcome:
        """Routes one event. Unknown event names are ignored, by contract."""
        if name == _EVENT_SERVER_INTENT:
            return self._server_intent(data)
        if name == _EVENT_PUT_OBJECT:
            return self._put_object(data)
        if name == _EVENT_DELETE_OBJECT:
            return self._delete_object(data)
        if name == _EVENT_PAYLOAD_TRANSFERRED:
            return self._payload_transferred(data)
        if name == _EVENT_ERROR:
            return self._error(data)
        if name == _EVENT_GOODBYE:
            return self._goodbye(data)
        if name == _EVENT_HEARTBEAT:
            return _TransferOutcome()
        logger.debug("Ignoring unknown FDv2 event '%s'", name)
        return _TransferOutcome()

    def _server_intent(self, data: Any) -> _TransferOutcome:
        payloads = data.get("payloads") if isinstance(data, dict) else None
        if not isinstance(payloads, list) or not payloads:
            return _TransferOutcome(
                disconnect="server-intent carried no payload description"
            )
        first = payloads[0]
        intent = first.get("intentCode") if isinstance(first, dict) else None
        self._intent = intent
        self._changes = []
        if intent == _INTENT_TRANSFER_FULL:
            # A fresh set: the payload about to arrive replaces everything held.
            # Built alongside the live set rather than in place, so an interrupted
            # transfer leaves last-known-good intact.
            self._pending = _SkillObjectSet()
        elif intent == _INTENT_TRANSFER_CHANGES:
            self._pending = self._committed.copy()
        elif intent == _INTENT_TRANSFER_NONE:
            self._pending = None
        else:
            logger.debug("Ignoring FDv2 server-intent with intentCode %r", intent)
            self._pending = None
        return _TransferOutcome()

    def _target(self) -> _SkillObjectSet | None:
        if self._pending is None and self._intent in (
            _INTENT_TRANSFER_FULL,
            _INTENT_TRANSFER_CHANGES,
        ):
            # An object arrived before any server-intent. Treat it as a delta
            # against what we hold rather than dropping it.
            self._pending = self._committed.copy()
        return self._pending

    def _put_object(self, data: Any) -> _TransferOutcome:
        if not is_skill_event(data):
            self.diagnostics.objects_ignored += 1
            return _TransferOutcome()
        if self._pending is None and self._intent is None:
            self._intent = _INTENT_TRANSFER_CHANGES
        target = self._target()
        if target is None:
            return _TransferOutcome()

        raw = seam_object_from_put(data)
        if raw is None:
            return _TransferOutcome()
        target.put(raw)
        self._changes.append(raw)
        self.diagnostics.skill_objects_received += 1
        if not isinstance(raw.get("contentHash"), str):
            self.diagnostics.hashless_objects += 1
            _warn_hashless(raw)
        return _TransferOutcome()

    def _delete_object(self, data: Any) -> _TransferOutcome:
        if not is_skill_event(data):
            self.diagnostics.objects_ignored += 1
            return _TransferOutcome()
        if self._pending is None and self._intent is None:
            self._intent = _INTENT_TRANSFER_CHANGES
        target = self._target()
        if target is None:
            return _TransferOutcome()

        tombstone = tombstone_from_delete(data)
        if tombstone is None:
            return _TransferOutcome()
        target.delete(tombstone)
        self.diagnostics.objects_revoked += 1
        # A tombstone, not a skill object: it carries identity and no content, so
        # a listener that only needs "something changed" works unchanged while one
        # that reads content sees no `content` key. Documented on
        # ``FDv2SkillStore.add_listener``.
        self._changes.append(
            {"key": tombstone.key, "version": tombstone.object_version}
        )
        return _TransferOutcome()

    def _payload_transferred(self, data: Any) -> _TransferOutcome:
        state = data.get("state") if isinstance(data, dict) else None
        version = data.get("version") if isinstance(data, dict) else None
        if self._pending is not None:
            hashless_before = self.diagnostics.hashless_objects
            self._committed.replace_with(self._pending)
            _warn_if_nothing_can_verify(self._committed, hashless_before)
        self._pending = None
        self._intent = None
        changes = self._changes
        self._changes = []
        self.diagnostics.payloads_transferred += 1
        logger.debug(
            "FDv2 payload transferred: payload version %s, %d skill object(s) held",
            version,
            len(self._committed),
        )
        return _TransferOutcome(
            committed=True,
            changes=changes,
            basis=state if isinstance(state, str) and state else None,
        )

    def _error(self, data: Any) -> _TransferOutcome:
        reason = data.get("reason") if isinstance(data, dict) else None
        # An error abandons the in-flight payload and keeps what is committed.
        self._pending = None
        self._intent = None
        self._changes = []
        return _TransferOutcome(disconnect=f"server sent error: {reason}")

    def _goodbye(self, data: Any) -> _TransferOutcome:
        reason = data.get("reason") if isinstance(data, dict) else None
        catastrophe = bool(data.get("catastrophe")) if isinstance(data, dict) else False
        silent = bool(data.get("silent")) if isinstance(data, dict) else False
        self._pending = None
        self._intent = None
        self._changes = []
        if not silent:
            logger.info("FDv2 connection closing: %s", reason)
        if catastrophe:
            return _TransferOutcome(
                fatal=f"server sent a catastrophic goodbye: {reason}"
            )
        return _TransferOutcome(disconnect=f"server said goodbye: {reason}")


_HASHLESS_ADVICE = (
    "The delivered skill object carries no 'contentHash', so integrity "
    "verification withholds it with reason_code 'missing_content_hash' and its "
    "content will never resolve. This is not a fault in this store and not "
    "something the SDK can work around: verification hashes the verbatim bytes "
    "and compares, and there is nothing to compare against. The field is "
    "specified as an additive sha256-over-verbatim-UTF-8 value on the skill "
    "envelope (LaunchDarkly AIC-2905) and has not shipped yet. Until it does, "
    "expect an empty result from every skill accessor."
)

_warned_hashless: set[tuple[str, Any]] = set()
_warned_lock = threading.Lock()


def _warn_hashless(raw: dict[str, Any]) -> None:
    """
    One ERROR per ``(key, version)`` whose envelope had no ``contentHash``.

    At ERROR rather than WARN, and per object rather than once per process,
    because this is the difference between a broken deployment and an
    empty-by-design one — the exact confusion the blocking gap produces. Deduped
    so a re-delivered payload does not multiply it; a store that is restarted
    reports again.
    """
    identity = (raw["key"], raw.get("version"))
    with _warned_lock:
        if identity in _warned_hashless:
            return
        _warned_hashless.add(identity)
    logger.error(
        "Skill '%s' version %s arrived without a contentHash and will be withheld. %s",
        raw["key"],
        raw.get("version"),
        _HASHLESS_ADVICE,
        extra={"ld_skill_key": raw["key"], "ld_skill_version": raw.get("version")},
    )


def _warn_if_nothing_can_verify(
    committed: _SkillObjectSet, hashless_before_this_payload: int
) -> None:
    """
    One ERROR per committed payload in which *nothing* the store now holds can
    possibly verify.

    ``log_withholding_summary`` already reports a wholly-withheld batch at the
    accessor boundary, but only once a caller asks. This fires at delivery time,
    so the condition is visible in a process that boots, materializes nothing,
    and exits — which is the shape a skills deployment fails in.
    """
    del hashless_before_this_payload  # counted for the store, not for this check
    held = committed.all_raw()
    if not held:
        return
    hashless = [raw for raw in held if not isinstance(raw.get("contentHash"), str)]
    if len(hashless) != len(held):
        return
    logger.error(
        "All %d skill object(s) in the delivered payload arrived without a "
        "contentHash. No skill content will resolve from this store. %s",
        len(held),
        _HASHLESS_ADVICE,
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class _FatalTransportError(Exception):
    """A failure retrying cannot fix: bad credential, forbidden, wrong URI."""


class _RecoverableTransportError(Exception):
    """A failure worth retrying. Carries a server-requested delay when given one."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_FORBIDDEN_ADVICE = (
    "FDv2 is opt-in per account: the 'fdv2-protocol-control' setting defaults to "
    "'forbid', which is served as HTTP 403. Skill delivery over this channel "
    "needs that flag flipped for the account, and needs the FDCore/streamer "
    "inline-resource support merged and deployed."
)


def _retry_after_seconds(headers: Any) -> float | None:
    """``Retry-After`` in seconds, when the server sent a usable one."""
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        seconds: float = float(str(raw).strip())
    except ValueError:
        # The HTTP-date form is legal and rare; falling back to our own backoff
        # is better than parsing a date to honour it approximately.
        return None
    return max(0.0, seconds)


def _classify_status(status: int, headers: Any) -> Exception:
    """Turns an HTTP error status into the right exception type."""
    if status == 401:
        return _FatalTransportError(
            "LaunchDarkly rejected the SDK key (HTTP 401). Skill delivery cannot "
            "start. Check that the key is the environment's server-side SDK key."
        )
    if status == 403:
        return _FatalTransportError(
            f"LaunchDarkly returned HTTP 403. {_FORBIDDEN_ADVICE}"
        )
    if status == 404:
        return _FatalTransportError(
            "LaunchDarkly returned HTTP 404 for the FDv2 endpoint. Check the base "
            "URI, and that this instance serves /sdk/poll and /sdk/stream."
        )
    if status in (400, 405, 406, 414, 501):
        return _FatalTransportError(
            f"LaunchDarkly returned HTTP {status}, which retrying will not fix. "
            "The request this adapter sent was not understood; the 'mv' data "
            f"model version ({SDK_DATA_MODEL_VERSION}) is the parameter most "
            "likely to be wrong."
        )
    return _RecoverableTransportError(
        f"LaunchDarkly returned HTTP {status}", _retry_after_seconds(headers)
    )


def _interrupt_read(response: Any) -> None:
    """
    Best-effort interruption of a read blocked on *response*, from another thread.

    Closing the response is not enough: CPython's buffered reader stays parked in
    ``readline`` until bytes arrive, so a ``close`` from another thread does not
    unblock it. Shutting the *socket* down underneath it does, immediately.

    Reaching the socket means walking urllib's private attribute chain, so every
    step is guarded and a failure here is silent by design. It is an
    optimisation, not a correctness requirement: the delivery thread is a daemon
    and ``close``'s join timeout is the backstop, so the worst case of this not
    finding a socket is a shutdown that takes as long as the join allows.
    """
    for path in (("fp", "raw", "_sock"), ("fp", "_sock"), ("_sock",)):
        found: Any = response
        for name in path:
            found = getattr(found, name, None)
            if found is None:
                break
        if found is not None and hasattr(found, "shutdown"):
            try:
                found.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return


class _StreamConnection:
    """
    One open streaming connection: an event iterator plus a way to interrupt it.

    Exists because ``close`` runs on a *different* thread from the read. The
    delivery thread spends nearly all its life blocked in a socket read on a
    long-lived stream, where a stop flag it cannot check is no use. Without an
    interruption a store's ``close`` would block for its whole join timeout on
    every shutdown of a *healthy* stream — a hang in the caller's shutdown path,
    paid every time.
    """

    def __init__(self, response: Any) -> None:
        self._response = response
        self.events = _iter_sse(response)

    def close(self) -> None:
        """Interrupts the read. Safe to call from any thread, and twice."""
        _interrupt_read(self._response)
        try:
            self._response.close()
        except Exception:
            pass


@dataclass(frozen=True)
class _PollResult:
    not_modified: bool
    events: list[tuple[str, Any]]
    etag: str | None


class _Requester:
    """
    The only place this module opens a socket.

    Standard library only, on purpose: this package's sole runtime dependency is
    ``opentelemetry-api`` and its LaunchDarkly base-SDK dependency is optional, so
    the content path must not smuggle in an HTTP client.
    """

    def __init__(
        self,
        sdk_key: str,
        base_uri: str,
        *,
        connect_timeout: float,
        read_timeout: float,
        data_model_version: int,
        opener: Any = None,
    ) -> None:
        self._sdk_key = sdk_key
        self._base_uri = base_uri.rstrip("/")
        self._read_timeout = read_timeout
        self._connect_timeout = connect_timeout
        self._data_model_version = data_model_version
        # Injectable so the tests drive a fake endpoint without a socket; the
        # default is urllib's global opener.
        self._opener = opener or urllib.request.build_opener()

    def _url(self, path: str, basis: str | None) -> str:
        params: dict[str, str] = {"mv": str(self._data_model_version)}
        if basis:
            params["basis"] = basis
        return f"{self._base_uri}{path}?{urllib.parse.urlencode(params)}"

    def _request(
        self, path: str, basis: str | None, headers: dict[str, str]
    ) -> urllib.request.Request:
        all_headers = {"Authorization": self._sdk_key, **headers}
        return urllib.request.Request(
            self._url(path, basis), headers=all_headers, method="GET"
        )

    def poll(self, basis: str | None, etag: str | None) -> _PollResult:
        """
        One ``GET /sdk/poll``. Honours ``If-None-Match`` and returns 304 as a
        first-class outcome rather than as an error.
        """
        headers = {"Accept": "application/json"}
        if etag:
            headers["If-None-Match"] = etag
        request = self._request(POLL_PATH, basis, headers)
        try:
            with self._opener.open(request, timeout=self._read_timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status == 304:
                    return _PollResult(not_modified=True, events=[], etag=etag)
                body = response.read()
                new_etag = response.headers.get("ETag") or etag
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                # urllib raises on 304 when no redirect handler swallows it.
                return _PollResult(not_modified=True, events=[], etag=etag)
            raise _classify_status(exc.code, exc.headers) from exc
        except _FatalTransportError:
            raise
        except Exception as exc:
            raise _RecoverableTransportError(
                f"polling request failed: {type(exc).__name__}: {exc}"
            ) from exc

        return _PollResult(
            not_modified=False, events=_decode_poll_body(body), etag=new_etag
        )

    def stream(self, basis: str | None) -> _StreamConnection:
        """
        Opens ``GET /sdk/stream``.

        Returns a ``_StreamConnection`` rather than a bare generator so the
        caller can interrupt a blocked read from another thread; see that class.
        """
        request = self._request(
            STREAM_PATH,
            basis,
            {"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        )
        try:
            response = self._opener.open(request, timeout=self._read_timeout)
        except urllib.error.HTTPError as exc:
            raise _classify_status(exc.code, exc.headers) from exc
        except _FatalTransportError:
            raise
        except Exception as exc:
            raise _RecoverableTransportError(
                f"streaming request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _StreamConnection(response)


def _decode_poll_body(body: bytes) -> list[tuple[str, Any]]:
    """
    Unwraps ``{"events": [...]}``.

    Polling and streaming carry the *identical* event objects — polling just wraps
    them in an envelope — which is why the protocol state machine above is shared
    and neither mode has its own copy of the semantics.
    """
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _RecoverableTransportError(
            f"polling response was not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
        raise _RecoverableTransportError("polling response had no 'events' array")
    events: list[tuple[str, Any]] = []
    for entry in parsed["events"]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("event")
        if isinstance(name, str):
            events.append((name, entry.get("data")))
    return events


def _iter_sse(response: Any) -> Any:
    """
    Decodes an SSE body into ``(event name, data)`` pairs.

    Minimal on purpose — this consumes one LaunchDarkly endpoint, not the whole
    spec: ``event:``/``data:`` fields, multi-line ``data`` joined with newlines,
    a blank line dispatching, and ``:`` comments skipped.
    """
    try:
        name: str | None = None
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if name is not None:
                    payload = "\n".join(data_lines)
                    try:
                        parsed = json.loads(payload) if payload else None
                    except json.JSONDecodeError:
                        logger.warning(
                            "Discarding FDv2 '%s' event whose data was not JSON", name
                        )
                        parsed = None
                    else:
                        yield name, parsed
                name = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            field_name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field_name == "event":
                name = value
            elif field_name == "data":
                data_lines.append(value)
    finally:
        try:
            response.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def backoff_delay(
    attempt: int, *, base: float, maximum: float, jitter: float = 0.5
) -> float:
    """
    Exponential backoff with decorrelating jitter, capped at *maximum*.

    Jitter is subtractive over the *whole* range rather than added on top, so the
    cap is a real ceiling: a fleet of agent processes restarted together must not
    reconnect in lockstep, and must not exceed the interval the cap promises.
    """
    # float(2 ** n) rather than 2 ** n: the integer power is untyped to mypy,
    # and the whole expression is a duration, not a count.
    ceiling: float = min(maximum, base * float(2 ** max(0, attempt - 1)))
    return ceiling * (1.0 - jitter * random.random())


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class FDv2SkillStore:
    """
    A ``SkillStore`` fed by LaunchDarkly's SDK-facing FDv2 delivery channel.

    The transport half of Agent Skills. Constructed with the environment's
    server-side SDK key, started explicitly, and passed to ``init_client``::

        store = FDv2SkillStore(sdk_key=os.environ["LD_SDK_KEY"])
        store.start()
        store.wait_for_skills(timeout=10)
        await init_client(options={"skillStore": store})

        skill = await get_skill("pdf-extraction")
        ...
        store.close()

    It also works as a context manager, which is the shape to prefer when the
    process's lifetime is a block.

    **Server-side only.** Skills are for server-side agent runtimes and skill
    content is customer-confidential. A mobile key or a client-side environment
    ID is refused in the constructor — see ``_require_server_side_credential``.

    **Delivery is in the background; retrieval is not.** ``SkillStore`` is a
    synchronous seam, so a daemon thread owns the connection and fills memory,
    and ``get_object`` only ever reads what has already arrived. Nothing here
    blocks a retrieval on the network. The corollary is that a process which
    calls ``get_skill`` immediately after ``start()`` may see an empty store;
    ``wait_for_skills`` is how you order boot against the first payload.

    **Last known good survives an outage.** A transport failure never empties the
    store and never makes ``get_object`` raise: it keeps serving what it last
    received, which is what makes ``write_skills(on_unavailable="keep")``
    correct. ``diagnostics`` and ``failed`` report the degradation.

    **What arrives is untrusted.** This store holds raw wire objects verbatim and
    verifies nothing — integrity verification lives at the accessor boundary so
    it applies to every store equally. In particular an object with no
    ``contentHash`` is held and then *withheld* by verification; see
    ``StoreDiagnostics.hashless_objects``.
    """

    def __init__(
        self,
        sdk_key: str,
        *,
        base_uri: str = DEFAULT_BASE_URI,
        mode: Mode = "stream",
        poll_interval: float = 30.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 300.0,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        max_consecutive_failures: int = 10,
        data_model_version: int = SDK_DATA_MODEL_VERSION,
        _requester: Any = None,
    ) -> None:
        """
        *mode* is ``"stream"`` by default. Prefer it: a ``delete-object`` reaches a
        live stream in seconds, which is what makes revocation seconds-latent
        instead of restart-latent, and is why the change-listener re-reconcile is
        worth wiring at all. ``"poll"`` exists for environments that cannot hold a
        long-lived connection, and revocation there is one ``poll_interval`` late.

        *max_consecutive_failures* bounds the retry loop. On exceeding it the
        transport stops, logs an error, and the store keeps serving last known
        good rather than pretending to be live — ``failed`` reports it.
        """
        _require_server_side_credential(sdk_key)
        if mode not in ("stream", "poll"):
            raise ValueError(f'mode must be "stream" or "poll", got {mode!r}')
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive, got {poll_interval!r}")

        self._mode: Mode = mode
        self._poll_interval = poll_interval
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._max_consecutive_failures = max_consecutive_failures

        self._objects = _SkillObjectSet()
        self._reader = _ProtocolReader(self._objects)
        self._lock = threading.RLock()
        self._listeners: dict[str, list[Callable[[dict[str, Any]], Any]]] = {}

        self._basis: str | None = None
        self._etag: str | None = None

        self._requester = _requester or _Requester(
            sdk_key.strip(),
            base_uri,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            data_model_version=data_model_version,
        )

        self._stop = threading.Event()
        self._first_payload = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed_reason: str | None = None
        self._connection: Any = None
        """The open streaming connection, so ``close`` can interrupt its read."""

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> FDv2SkillStore:
        """
        Starts the delivery thread. Idempotent; returns ``self`` so it chains.

        Does not block: use ``wait_for_skills`` when boot ordering matters.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="ld-ai-skills-fdv2", daemon=True
            )
            self._thread.start()
        return self

    def close(self, timeout: float = 5.0) -> None:
        """
        Stops delivery. Idempotent, and safe to call from any thread.

        Held content is *not* dropped: a closed store still answers from what it
        received, so shutting the transport down does not turn into an integrity
        failure or an empty reconcile mid-flight. ``shutdown()`` is what detaches
        the store from the accessors.
        """
        self._stop.set()
        # Interrupt the read before joining. The delivery thread is normally
        # blocked in a socket read that no flag can reach, so without this the
        # join below waits out its full timeout on every shutdown.
        with self._lock:
            connection = self._connection
        if connection is not None:
            connection.close()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

    def __enter__(self) -> FDv2SkillStore:
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def wait_for_skills(self, timeout: float = 10.0) -> bool:
        """
        Blocks until the first payload has been committed, or *timeout* elapses.

        ``True`` means a payload arrived — not that any skill in it verified, and
        not that the environment has any skills. Boot ordering is all this
        answers; ``diagnostics`` answers the rest.
        """
        return self._first_payload.wait(timeout=timeout)

    @property
    def failed(self) -> str | None:
        """Why delivery stopped for good, or ``None`` while it is running."""
        with self._lock:
            return self._failed_reason

    @property
    def diagnostics(self) -> StoreDiagnostics:
        """A snapshot of what the transport has seen. See ``StoreDiagnostics``."""
        with self._lock:
            return StoreDiagnostics(**vars(self._reader.diagnostics))

    # -- the SkillStore seam ----------------------------------------------

    def get_object(
        self, kind: str, key: str, version: int | None = None
    ) -> dict[str, Any] | None:
        if kind != SKILL_OBJECT_KIND:
            return None
        with self._lock:
            return self._objects.get(key, version)

    def all_objects(self, kind: str) -> dict[str, dict[str, Any]]:
        if kind != SKILL_OBJECT_KIND:
            return {}
        with self._lock:
            return self._objects.snapshot()

    def add_listener(self, kind: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        """
        Registers *fn* to be called once per committed change.

        Fires **once per changed object at payload-transferred**, not as objects
        stream in: a payload version is the unit of consistency, and a listener
        that reacted to a half-applied full transfer would see the store briefly
        empty. ``skills_watch.watch_skills`` is the intended consumer.

        A put notifies with the raw skill object. A revocation notifies with a
        ``{"key", "version"}`` tombstone — it names what went away and carries no
        content, since there is none. A listener that only needs "something
        changed" works with both; one that reads content must check for
        ``content`` rather than assume it.

        *fn* runs on the delivery thread. Keep it cheap and non-blocking: work
        done there delays the next event. An exception it raises is logged and
        swallowed, because a broken listener must not be able to kill delivery.
        """
        with self._lock:
            self._listeners.setdefault(kind, []).append(fn)

    def _notify(self, changes: list[dict[str, Any]]) -> None:
        with self._lock:
            listeners = list(self._listeners.get(SKILL_OBJECT_KIND, []))
        for raw in changes:
            for listener in listeners:
                try:
                    listener(raw)
                except Exception:
                    logger.error(
                        "A skill store change listener raised; delivery continues",
                        exc_info=True,
                    )

    # -- the delivery loop -------------------------------------------------

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                if self._mode == "stream":
                    self._stream_once()
                else:
                    self._poll_once()
                failures = 0
                with self._lock:
                    self._reader.diagnostics.connection_failures = 0
            except _FatalTransportError as exc:
                self._give_up(str(exc))
                return
            except _RecoverableTransportError as exc:
                failures += 1
                with self._lock:
                    self._reader.diagnostics.connection_failures = failures
                    self._reader.diagnostics.last_error = str(exc)
                if failures > self._max_consecutive_failures:
                    self._give_up(
                        f"gave up after {failures} consecutive failures; "
                        f"last error: {exc}"
                    )
                    return
                delay = exc.retry_after
                if delay is None:
                    delay = backoff_delay(
                        failures, base=self._initial_backoff, maximum=self._max_backoff
                    )
                logger.warning(
                    "Skill delivery failed (%s); retrying in %.1fs", exc, delay
                )
                if self._stop.wait(delay):
                    return
                continue
            except Exception as exc:  # pragma: no cover - belt and braces
                self._give_up(f"unexpected error in skill delivery: {exc!r}")
                logger.error("Unexpected error in skill delivery", exc_info=True)
                return

            if self._mode == "poll" and self._stop.wait(self._poll_interval):
                return

    def _give_up(self, reason: str) -> None:
        with self._lock:
            self._failed_reason = reason
            self._reader.diagnostics.last_error = reason
        logger.error(
            "Skill delivery has stopped and will not retry: %s. The store keeps "
            "serving the last content it received; skills will not update until "
            "the process restarts with a working connection.",
            reason,
        )
        # Unblock anyone waiting on a first payload that is never coming, rather
        # than making them eat the full timeout.
        self._first_payload.set()

    def _apply(self, name: str, data: Any) -> _TransferOutcome:
        with self._lock:
            outcome = self._reader.handle(name, data)
            if outcome.committed and outcome.basis is not None:
                self._basis = outcome.basis
        if outcome.committed:
            self._first_payload.set()
            if outcome.changes:
                self._notify(outcome.changes)
        return outcome

    def _poll_once(self) -> None:
        with self._lock:
            basis, etag = self._basis, self._etag
        result = self._requester.poll(basis, etag)
        with self._lock:
            self._etag = result.etag
        if result.not_modified:
            logger.debug("Skill payload unchanged (HTTP 304)")
            # A 304 is a successful, current answer: the payload we hold is the
            # payload the server has. It counts as a first payload so a boot that
            # reconnects with a cached basis is not blocked on a transfer the
            # server has no reason to send.
            self._first_payload.set()
            return
        for name, data in result.events:
            outcome = self._apply(name, data)
            if outcome.fatal:
                raise _FatalTransportError(outcome.fatal)
            if outcome.disconnect:
                raise _RecoverableTransportError(outcome.disconnect)

    def _stream_once(self) -> None:
        with self._lock:
            basis = self._basis
        connection = self._requester.stream(basis)
        with self._lock:
            self._connection = connection
        try:
            for name, data in connection.events:
                if self._stop.is_set():
                    return
                outcome = self._apply(name, data)
                if outcome.fatal:
                    raise _FatalTransportError(outcome.fatal)
                if outcome.disconnect:
                    raise _RecoverableTransportError(outcome.disconnect)
        except Exception:
            if self._stop.is_set():
                # `close` interrupted the read on purpose; unwinding quietly is
                # the point, not a failure to report or retry.
                return
            raise
        finally:
            connection.close()
            with self._lock:
                self._connection = None
        # A stream that ends without a goodbye is a dropped connection, not a
        # completed operation: reconnect through the backoff path.
        raise _RecoverableTransportError("the FDv2 stream closed unexpectedly")
