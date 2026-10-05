"""
Agent Skills — the FDv2 delivery transport.

``FDv2SkillStore`` is the ``SkillStore`` that receives skills from LaunchDarkly
over ``GET /sdk/poll`` or ``GET /sdk/stream``, authenticated with the
environment's server-side SDK key. It holds raw wire objects and serves them to
the accessors; it uses only the standard library.

What it does not do:

- **Verify content.** Integrity verification happens at the accessor boundary,
  so it applies to every store, including one you supply yourself.
- **Work around a missing ``contentHash``.** Such an object is held as-is and
  then withheld by verification with ``missing_content_hash``; the store logs an
  error and counts it in ``StoreDiagnostics.hashless_objects``.
- **Evaluate flags.** Non-skill objects are skipped and counted.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import socket
import threading
import time
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

FDV2_OBJECT_KIND = "skill"
"""
The FDv2 object ``kind`` of a skill (exact, lower-case match).

Distinct from ``FDV2_PAYLOAD_KIND``, the kind of the payload skills arrive in.
"""

FDV2_PAYLOAD_KIND = "agent-skill"
"""
The FDv2 payload kind declared on every request as ``?kinds=``.

Required: a request without it is served the flag payload and no skills.
"""

FDV2_KEY_DELIMITER = ":"
"""
Separates key from version in a skill object's wire ``key``
(``<key>:<version>``). Each version of a skill is a separate object.
"""

DEFAULT_BASE_URI = "https://sdk.launchdarkly.com"
"""Where ``GET /sdk/poll`` is served. Override for Federal, private, or relay
deployments."""

DEFAULT_STREAM_URI = "https://stream.launchdarkly.com"
"""
Where ``GET /sdk/stream`` is served. LaunchDarkly streams from a different host
than it polls from; a *base_uri* given without a *stream_uri* is used for both.
"""

POLL_PATH = "/sdk/poll"
STREAM_PATH = "/sdk/stream"

DEFAULT_POLL_TIMEOUT = 10.0
"""Default ``read_timeout`` in ``"poll"`` mode: the bound on one whole request."""

DEFAULT_STREAM_READ_TIMEOUT = 300.0
"""Default ``read_timeout`` in ``"stream"`` mode: the longest gap tolerated
between two reads. LaunchDarkly's heartbeats arrive well inside this."""

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
"""The most the transport holds in memory from one poll body or one streamed event.

A memory backstop set far above any real payload, separate from the per-skill
content limit verification enforces. Crossing it is fatal: nothing is applied,
the store keeps its current content, and ``failed`` is set. The payload's size
belongs to the environment, so a retry would be refused the same way, after
downloading up to this much again."""

_READ_CHUNK_BYTES = 64 * 1024

_BACKOFF_RESET_INTERVAL = 60.0
"""
Seconds a stream must stay open before its end resets the backoff delay, as in
the base server-side SDKs. Overridden in tests.
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
Envelope fields copied verbatim. Never coerced or defaulted: filling in a
missing field would forge what verification checks.
"""

_PAYLOAD_SELECTOR = re.compile(r"\(p:([^:()]+):\d+\)")
"""
The payload id inside a transfer's selector, ``(p:<id>:<version>)``. Used when
the intent named no ``id``.
"""

Mode = Literal["stream", "poll"]

_MOBILE_KEY_PREFIX = "mob-"
_SERVER_KEY_PREFIX = "sdk-"
_CLIENT_SIDE_ID = re.compile(r"\A[0-9a-f]{20,}\Z")
"""A client-side environment ID: unprefixed lowercase hex."""


# ---------------------------------------------------------------------------
# Server-side only
# ---------------------------------------------------------------------------


def _require_server_side_credential(sdk_key: str) -> None:
    """
    Raises ``ValueError`` for a missing key, a mobile key, or a client-side
    environment ID.

    Skill content is confidential, and the server may not refuse a client-side
    credential itself, so the SDK does.
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
        # Warn only: private instances and test doubles may use unprefixed keys.
        logger.warning(
            "The credential given to FDv2SkillStore does not look like a "
            "LaunchDarkly server-side SDK key (sdk-...). Skills are delivered "
            "only to server-side credentials; if this is a client-side or mobile "
            "credential the connection will be rejected or will deliver nothing."
        )


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
"""The only hosts a plain ``http://`` base URI may name: a local test double."""


def _require_https_uri(base_uri: str, option: str = "base_uri") -> None:
    """
    Raises ``ValueError`` for a URI that would send the SDK key in cleartext.

    Requires ``https://`` with a host; plain ``http://`` is allowed only to a
    loopback host (``localhost``, ``127.0.0.1``, ``::1``) for local test doubles.
    """
    if not isinstance(base_uri, str) or not base_uri.strip():
        raise ValueError(
            f"FDv2SkillStore requires an https:// URI for {option}; none was given."
        )
    parts = urllib.parse.urlsplit(base_uri.strip())
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS:
        return
    if parts.scheme == "http":
        raise ValueError(
            f"FDv2SkillStore refuses {option} {base_uri!r}: a plain http:// URI "
            "would send the server-side SDK key in cleartext. Use https:// "
            "(the defaults are https://sdk.launchdarkly.com for polling and "
            "https://stream.launchdarkly.com for streaming). Plain http:// is "
            "allowed only for a loopback host (localhost, 127.0.0.1, ::1) "
            "serving a local test double."
        )
    raise ValueError(
        f"FDv2SkillStore refuses {option} {base_uri!r}: expected an https:// URI "
        "with a host, such as https://sdk.launchdarkly.com."
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class StoreDiagnostics:
    """
    Counters describing what the transport has seen. Read-only.

    Useful for telling "this environment has no skills" apart from "every skill
    was withheld".
    """

    payloads_transferred: int = 0
    """Completed ``payload-transferred`` commits since the store started."""
    skill_objects_received: int = 0
    """``put-object`` events identified as skills, across all payloads."""
    objects_ignored: int = 0
    """Objects skipped because they were not skills."""
    objects_revoked: int = 0
    """Skill revocations: each ``delete-object`` that removed something, plus each
    key a full transfer dropped entirely (not a key whose version moved)."""
    payloads_ignored: int = 0
    """Transfers not applied because they completed a payload other than the
    skill payload. Normally zero."""
    hashless_objects: int = 0
    """
    Skill objects whose envelope carried no ``contentHash``.

    **Nonzero means skills are being withheld** with ``missing_content_hash``.
    Cumulative: read it as "this has happened", not as the current count.
    """
    connection_failures: int = 0
    """Recoverable transport failures in a row; reset by a completed exchange."""
    last_error: str | None = None
    """The most recent transport error, if any. Human-readable; do not parse."""


# ---------------------------------------------------------------------------
# Deserialisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tombstone:
    """A ``delete-object`` narrowed to the identity it revokes."""

    key: str
    object_version: int | None


def _is_skill_event(data: Any) -> bool:
    """
    Whether one ``put-object`` / ``delete-object`` payload is a skill.

    Other kinds are ignored rather than rejected, so an unrecognised kind cannot
    cause a reconnect loop.
    """
    if not isinstance(data, dict):
        return False
    return data.get("kind") == FDV2_OBJECT_KIND


@dataclass(frozen=True)
class _WireIdentity:
    """A skill object's wire ``key``, split into the skill's key and version."""

    key: str
    version: Any
    """``int`` when the wire carried one; the offending text when it did not;
    absent (``_NO_VERSION``) when the wire key had no delimiter at all."""


_NO_VERSION = object()


def _split_wire_key(wire_key: Any) -> _WireIdentity | None:
    """
    Splits ``<key>:<version>`` from one object's wire ``key``.

    Malformed versions are kept so verification can report them under a
    recognisable key:

    - No delimiter: no version; verification reports ``invalid_version``.
    - Non-digit version (``"pdf:latest"``, ``"pdf:"``, ``"a:1:2"``): the text
      is kept as the version.
    - Empty key (``":3"``): ``None``; the object is dropped.

    Leading zeros are accepted (``"pdf:03"`` is version 3).
    """
    if not isinstance(wire_key, str) or not wire_key:
        return None
    key, delimiter, version_text = wire_key.partition(FDV2_KEY_DELIMITER)
    if not key:
        return None
    if not delimiter:
        return _WireIdentity(key=key, version=_NO_VERSION)
    if version_text.isascii() and version_text.isdigit():
        return _WireIdentity(key=key, version=int(version_text))
    return _WireIdentity(key=key, version=version_text)


def _store_object_from_put(data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Translates one FDv2 skill ``put-object`` into a raw ``SkillStore`` object.

        wire ``key``      →  stored ``key`` and ``version``  (split on ``:``)
        wire ``version``  →  dropped                          (the *payload* version)

    The event's ``version`` is the payload's version, not the skill's; using it
    would serve content under a meaningless version number.

    Returns ``None`` only when the wire ``key`` has no skill key. Other defects
    are carried through so verification withholds them with a reason code.
    """
    identity = _split_wire_key(data.get("key"))
    if identity is None:
        logger.warning(
            "An FDv2 skill put-object carried no usable 'key' (%r) and could not "
            "be stored under any identity; it was dropped.",
            data.get("key"),
        )
        return None

    raw: dict[str, Any] = {"key": identity.key}

    # Absent or malformed versions pass through for verification to report.
    if identity.version is not _NO_VERSION:
        raw["version"] = identity.version

    envelope = data.get("object")
    if isinstance(envelope, dict):
        for wire_field in _ENVELOPE_FIELDS:
            if wire_field in envelope:
                raw[wire_field] = envelope[wire_field]
    return raw


def _payload_id_of(intent: Any) -> str | None:
    """The payload id one payload intent names, when it names a usable one."""
    if not isinstance(intent, dict):
        return None
    value = intent.get("id")
    return value if isinstance(value, str) and value else None


def _payload_id_from_selector(state: Any) -> str | None:
    """The payload id inside a transfer's selector, when it carries one."""
    if not isinstance(state, str):
        return None
    match = _PAYLOAD_SELECTOR.search(state)
    return match.group(1) if match else None


def _tombstone_from_delete(data: dict[str, Any]) -> _Tombstone | None:
    """
    Narrows one FDv2 skill ``delete-object`` to the identity it revokes.

    A delete with no usable version (``object_version=None``) revokes every
    version of the key, including a version-less malformed entry. Erring this way
    avoids serving withdrawn content.
    """
    identity = _split_wire_key(data.get("key"))
    if identity is None:
        logger.warning(
            "An FDv2 skill delete-object carried no usable 'key' (%r); it was ignored.",
            data.get("key"),
        )
        return None
    return _Tombstone(
        key=identity.key,
        object_version=identity.version
        if is_valid_skill_version(identity.version)
        else None,
    )


# ---------------------------------------------------------------------------
# The held object set
# ---------------------------------------------------------------------------


class _SkillObjectSet:
    """
    Raw skill objects held in memory, keyed by ``(key, version)``.

    Lookups match ``InMemorySkillStore``. An object with no usable version is
    held under its key alone so verification can withhold it with a reason.
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
        """Removes what *tombstone* revokes; returns the raw objects that went away."""
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
        if not held:
            # Only a version-less entry exists: serve it so verification reports
            # it. A pin that misses while valid versions exist is a plain miss.
            return self._loose.get(key)
        if version is not None:
            return held.get(version)
        return held[max(held)]

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
class _TransferOutcome:
    """What one event did. Aggregated by the caller; nothing here does I/O."""

    committed: bool = False
    changes: list[dict[str, Any]] = field(default_factory=list)
    basis: str | None = None
    fatal: str | None = None
    disconnect: str | None = None
    up_to_date: bool = False
    """A ``none`` intent: the content held is current. Counts as a healthy
    exchange though it commits nothing, like a 304 to a poll."""
    recycled: bool = False
    """The disconnect is a ``goodbye``: routine if the connection has already
    completed an exchange."""


def _identity_of(raw: dict[str, Any]) -> tuple[str, Any]:
    """One object's comparable ``(key, version)``; an unusable version is ``None``."""
    version = raw.get("version")
    return (raw["key"], version if is_valid_skill_version(version) else None)


def _revocations_between(
    current: _SkillObjectSet, pending: _SkillObjectSet
) -> list[dict[str, Any]]:
    """
    Tombstones for every ``(key, version)`` *pending* no longer holds.

    A full transfer revokes by omission; this recovers those revocations. A key
    whose version moved yields a put for the new version and a tombstone for the
    old one.
    """
    surviving = {_identity_of(raw) for raw in pending.all_raw()}
    return [
        {"key": key, "version": version}
        for key, version in (_identity_of(raw) for raw in current.all_raw())
        if (key, version) not in surviving
    ]


def _keys_fully_revoked(revoked: list[dict[str, Any]], pending: _SkillObjectSet) -> int:
    """
    How many keys in *revoked* left the payload entirely (not version moves).

    This is what ``objects_revoked`` counts; ``changes`` carries every tombstone.
    """
    return len(
        {
            tombstone["key"]
            for tombstone in revoked
            if pending.get(tombstone["key"], None) is None
        }
    )


class _ProtocolReader:
    """
    Applies FDv2 events to an object set. Pure: no sockets, threads, or clock.

    - **Changes commit at ``payload-transferred``**, never half-applied, so a
      full transfer never briefly empties the store and listeners fire only at
      commit.
    - **Only the first payload intent is read**, as the protocol requires, and
      it is taken to be the skill payload. The reader learns which payload
      carries skills and declines transfers of any other, which would otherwise
      empty the skill set.
    """

    def __init__(self, committed: _SkillObjectSet) -> None:
        self._committed = committed
        self._intent: str | None = None
        self._pending: _SkillObjectSet | None = None
        self._changes: list[dict[str, Any]] = []
        self.diagnostics = StoreDiagnostics()
        # Per reader, so each store reports hashless objects independently.
        self._warned_hashless: set[tuple[str, Any]] = set()
        # The payload the current intent describes, and the one skills arrive
        # on; kept apart so a transfer of another payload can be declined.
        self._intent_payload_id: str | None = None
        self._skill_payload_id: str | None = None
        self._skills_in_payload = 0
        self._warned_multiple_payloads = False
        self._warned_foreign_payload = False

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
        if len(payloads) > 1:
            self._warn_multiple_payloads(payloads)
        # The first payload only, as the protocol requires.
        first = payloads[0]
        intent = first.get("intentCode") if isinstance(first, dict) else None
        self._intent = intent
        self._intent_payload_id = _payload_id_of(first)
        self._changes = []
        self._skills_in_payload = 0
        if intent == _INTENT_TRANSFER_FULL:
            # Built beside the live set so an interrupted transfer keeps it.
            self._pending = _SkillObjectSet()
        elif intent == _INTENT_TRANSFER_CHANGES:
            self._pending = self._committed.copy()
        elif intent == _INTENT_TRANSFER_NONE:
            # Current, but not finished: later edits arrive on this connection
            # with no second intent, so expect changes, as the base SDK's
            # ``ChangeSetBuilder.expect_changes()`` does. The pending set is
            # copied when the first object arrives.
            self._intent = _INTENT_TRANSFER_CHANGES
            self._pending = None
            return _TransferOutcome(up_to_date=True)
        else:
            logger.debug("Ignoring FDv2 server-intent with intentCode %r", intent)
            self._pending = None
        return _TransferOutcome()

    def _target_for(self, data: Any) -> _SkillObjectSet | None:
        """
        The pending set a skill event applies to, or ``None`` when the event is
        not a skill or the current intent carries no objects. An object with no
        preceding ``server-intent`` is treated as a delta.
        """
        if not _is_skill_event(data):
            self.diagnostics.objects_ignored += 1
            return None
        if self._pending is None:
            if self._intent is None:
                self._intent = _INTENT_TRANSFER_CHANGES
            if self._intent not in (_INTENT_TRANSFER_FULL, _INTENT_TRANSFER_CHANGES):
                return None
            self._pending = self._committed.copy()
        return self._pending

    def _put_object(self, data: Any) -> _TransferOutcome:
        target = self._target_for(data)
        if target is None:
            return _TransferOutcome()
        raw = _store_object_from_put(data)
        if raw is None:
            return _TransferOutcome()
        target.put(raw)
        self._changes.append(raw)
        self.diagnostics.skill_objects_received += 1
        self._skills_in_payload += 1
        if not isinstance(raw.get("contentHash"), str):
            self.diagnostics.hashless_objects += 1
            self._warn_hashless(raw)
        return _TransferOutcome()

    def _delete_object(self, data: Any) -> _TransferOutcome:
        target = self._target_for(data)
        if target is None:
            return _TransferOutcome()
        tombstone = _tombstone_from_delete(data)
        if tombstone is None:
            return _TransferOutcome()
        removed = target.delete(tombstone)
        if removed:
            # Counted only if it removed something; listeners see it either way.
            self.diagnostics.objects_revoked += 1
        # A revocation identifies the skill payload just as a put does.
        self._skills_in_payload += 1
        self._changes.append(
            {"key": tombstone.key, "version": tombstone.object_version}
        )
        return _TransferOutcome()

    def _payload_transferred(self, data: Any) -> _TransferOutcome:
        state = data.get("state") if isinstance(data, dict) else None
        version = data.get("version") if isinstance(data, dict) else None
        payload_id = self._intent_payload_id or _payload_id_from_selector(state)
        # Checked even with no pending set (a ``none`` intent), so a foreign
        # payload's selector never becomes the resume point.
        foreign = self._is_foreign_payload(payload_id)
        # A ``none`` intent builds no pending set, nor does an intent code this
        # SDK does not recognise, and a foreign payload's contents are declined
        # below.
        applied = not foreign and self._pending is not None
        if foreign:
            self._warn_foreign_payload(payload_id)
            self.diagnostics.payloads_ignored += 1
            self._changes = []
        elif self._pending is not None:
            if self._intent == _INTENT_TRANSFER_FULL:
                # A full transfer revokes by omission. Diff before the swap so
                # those departures reach listeners as tombstones.
                revoked = _revocations_between(self._committed, self._pending)
                self._changes.extend(revoked)
                self.diagnostics.objects_revoked += _keys_fully_revoked(
                    revoked, self._pending
                )
            self._committed.replace_with(self._pending)
            _warn_if_nothing_can_verify(self._committed)
            if self._skills_in_payload and payload_id is not None:
                # The payload that carried a skill put or delete is the skill payload.
                self._skill_payload_id = payload_id
        self._pending = None
        self._intent = None
        self._intent_payload_id = None
        self._skills_in_payload = 0
        changes = self._changes
        self._changes = []
        self.diagnostics.payloads_transferred += 1
        logger.debug(
            "FDv2 payload transferred: payload version %s, %d skill object(s) held",
            version,
            len(self._committed),
        )
        if not applied:
            # Nothing applied, so nothing to report — and in particular not a
            # commit, because a commit publishes the first payload, and
            # ``is_initialized`` (what ``write_skills("*")`` authorises a prune
            # on) must not go true over a store that received nothing. Not up to
            # date either: only the ``none`` intent says that, on its own event.
            # The selector is withheld too — it names a payload never applied.
            return _TransferOutcome()
        return _TransferOutcome(
            committed=True,
            changes=changes,
            basis=state if isinstance(state, str) and state else None,
        )

    def _abandon_in_flight(self) -> None:
        """Drops the in-flight payload and keeps what is committed."""
        self._pending = None
        self._intent = None
        self._intent_payload_id = None
        self._skills_in_payload = 0
        self._changes = []

    def _error(self, data: Any) -> _TransferOutcome:
        reason = data.get("reason") if isinstance(data, dict) else None
        self._abandon_in_flight()
        return _TransferOutcome(disconnect=f"server sent error: {reason}")

    def _goodbye(self, data: Any) -> _TransferOutcome:
        reason = data.get("reason") if isinstance(data, dict) else None
        catastrophe = bool(data.get("catastrophe")) if isinstance(data, dict) else False
        silent = bool(data.get("silent")) if isinstance(data, dict) else False
        self._abandon_in_flight()
        if not silent:
            # Debug only: a goodbye after a completed exchange is a routine
            # recycle, and the delivery loop warns when one is not.
            logger.debug("FDv2 connection closing: %s", reason)
        if catastrophe:
            return _TransferOutcome(
                fatal=f"server sent a catastrophic goodbye: {reason}"
            )
        return _TransferOutcome(
            disconnect=f"server said goodbye: {reason}", recycled=True
        )

    # -- payload identity ----------------------------------------------------

    def _is_foreign_payload(self, payload_id: str | None) -> bool:
        """
        Whether a transfer completes a payload other than the skill payload.
        ``False`` unless both payload ids are known.
        """
        return (
            self._skill_payload_id is not None
            and payload_id is not None
            and payload_id != self._skill_payload_id
        )

    # -- diagnostics ---------------------------------------------------------

    def _warn_multiple_payloads(self, payloads: list[Any]) -> None:
        """
        One WARNING per reader for an intent describing more than one payload:
        the first is then not guaranteed to be the skill payload.
        """
        if self._warned_multiple_payloads:
            return
        self._warned_multiple_payloads = True
        logger.warning(
            "An FDv2 server-intent described %d payloads (%s). Only the first is "
            "read, as the protocol requires, and it is taken to be the payload "
            "skills arrive on. If skills stop resolving from this point, that is "
            "the assumption that broke; contact LaunchDarkly support.",
            len(payloads),
            ", ".join(str(_payload_id_of(p)) for p in payloads),
        )

    def _warn_foreign_payload(self, payload_id: str | None) -> None:
        """One WARNING per reader for a transfer this layer declined to apply."""
        if self._warned_foreign_payload:
            return
        self._warned_foreign_payload = True
        logger.warning(
            "An FDv2 transfer of payload %s was not applied to the skills held, "
            "which arrive on payload %s. Applying it would have replaced them "
            "with whatever that payload carried — nothing, in the case of a flag "
            "payload. The skills held are unchanged.",
            payload_id,
            self._skill_payload_id,
        )

    def _warn_hashless(self, raw: dict[str, Any]) -> None:
        """
        One ERROR per ``(key, version)`` whose envelope had no ``contentHash``,
        so withheld skills are not mistaken for an environment with none.
        """
        identity = (raw["key"], raw.get("version"))
        if identity in self._warned_hashless:
            return
        self._warned_hashless.add(identity)
        logger.error(
            "Skill '%s' version %s arrived without a contentHash and will be withheld. %s",
            raw["key"],
            raw.get("version"),
            _HASHLESS_ADVICE,
            extra={"ld_skill_key": raw["key"], "ld_skill_version": raw.get("version")},
        )


_HASHLESS_ADVICE = (
    "The delivered skill object carries no 'contentHash', so integrity "
    "verification withholds it with reason_code 'missing_content_hash' and its "
    "content will not resolve. The SDK cannot work around this: verification "
    "hashes the delivered bytes and compares them against the envelope's "
    "'contentHash', and there is nothing to compare against. 'contentHash' is a "
    "sha256 over the verbatim UTF-8 content. Contact LaunchDarkly support if "
    "skills in your environment arrive without one."
)


def _warn_if_nothing_can_verify(committed: _SkillObjectSet) -> None:
    """
    One ERROR per committed payload in which *nothing* held can verify.

    Fires at delivery time, so it shows even in a process that never reads a
    skill.
    """
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


class _ResponseTooLargeError(_FatalTransportError):
    """A poll body, stream line or stream event over ``MAX_RESPONSE_BYTES``."""


class _RecoverableTransportError(Exception):
    """A failure worth retrying. Carries a server-requested delay when given one."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        *,
        recycled: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.recycled = recycled
        """A ``goodbye`` after a completed exchange: the server recycling the
        stream. Reconnected, but not counted or reported as a failure."""


class _StaleRequestStateError(_RecoverableTransportError):
    """
    An HTTP 400. If the request carried client state (``basis`` or an
    ``If-None-Match`` etag), that state is dropped and a full transfer requested
    once; a 400 for a request with no state is fatal.
    """


_REQUEST_ADVICE = (
    "The request this adapter sent was not understood. It carries only the SDK "
    "key, a 'kinds' parameter declaring the skill payload, and, after the first "
    "payload, a 'basis' selector, so check the base URI and that the endpoint "
    "speaks FDv2."
)

_FORBIDDEN_ADVICE = (
    "The FDv2 protocol is opt-in per LaunchDarkly account and is served as HTTP "
    "403 while it is off. Skill delivery needs it enabled; contact LaunchDarkly "
    "support to enable it for your account."
)


def _retry_after_seconds(headers: Any) -> float | None:
    """
    ``Retry-After`` in seconds, or ``None`` when absent or unusable (the
    HTTP-date form, or a non-finite number), in which case normal backoff applies.
    """
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
        return None
    if not math.isfinite(seconds):
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
    if 300 <= status < 400 and status != 304:
        return _FatalTransportError(
            f"LaunchDarkly returned HTTP {status}, a redirect. Redirects are not "
            "followed, so the SDK key is never forwarded to a host other than the "
            "base URI. The SDK-facing FDv2 endpoints do not redirect; check the "
            "base URI, and any proxy in between, for the address being redirected "
            "to."
        )
    if status == 404:
        # Typically a mistyped base URI; retrying will not help.
        return _FatalTransportError(
            "LaunchDarkly returned HTTP 404 for the FDv2 endpoint. Check the "
            "base URI, and that this instance serves /sdk/poll and /sdk/stream."
        )
    if status == 400:
        # The selector sent may be stale: recoverable once (see
        # ``_StaleRequestStateError``).
        return _StaleRequestStateError(
            f"LaunchDarkly returned HTTP 400. {_REQUEST_ADVICE}"
        )
    if status == 422:
        # No payload matching the declared kinds is available on this
        # connection, most often because of a view-scoped SDK key. Fatal.
        return _FatalTransportError(
            "LaunchDarkly will not deliver Agent Skills on this connection "
            "(HTTP 422). The usual cause is a view-scoped SDK key. Check your "
            "SDK key or contact LaunchDarkly support."
        )
    if status in (405, 406, 414, 501):
        return _FatalTransportError(
            f"LaunchDarkly returned HTTP {status}, which retrying will not fix. "
            f"{_REQUEST_ADVICE}"
        )
    return _RecoverableTransportError(
        f"LaunchDarkly returned HTTP {status}", _retry_after_seconds(headers)
    )


def _interrupt_read(response: Any) -> None:
    """
    Best-effort interruption of a read blocked on *response*, from another thread.

    Closing the response does not unblock a parked ``readline``; shutting down
    the socket does. That socket is reached through urllib's private attributes,
    so every step is guarded and failure is silent.
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
    """One open streaming connection: an event iterator that ``close`` can interrupt."""

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


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """
    A redirect handler that follows nothing.

    Following a redirect would forward the ``Authorization`` header (the SDK key)
    to whatever host ``Location`` names. Refused 3xx responses surface as
    ``HTTPError`` and are fatal. The FDv2 endpoints never redirect.
    """

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: Any,
        msg: Any,
        headers: Any,
        newurl: Any,
    ) -> None:
        return None


def _build_opener() -> urllib.request.OpenerDirector:
    """The default opener with its redirect handler replaced by a refusing one."""
    return urllib.request.build_opener(_RefuseRedirects)


@dataclass(frozen=True)
class _PollResult:
    not_modified: bool
    events: list[tuple[str, Any]]
    etag: str | None


class _Requester:
    """
    The only place this module opens a socket.

    *read_timeout* bounds every socket operation of a request (connect, headers,
    each body read); ``urllib`` has no separate connect timeout.
    """

    def __init__(
        self,
        sdk_key: str,
        base_uri: str,
        *,
        read_timeout: float,
        stream_uri: str | None = None,
        opener: Any = None,
    ) -> None:
        self._sdk_key = sdk_key
        self._base_uri = base_uri.rstrip("/")
        self._stream_uri = (stream_uri or base_uri).rstrip("/")
        self._read_timeout = read_timeout
        # The default opener never follows a redirect; see ``_RefuseRedirects``.
        self._opener = opener or _build_opener()
        self._lock = threading.Lock()
        # The in-flight poll response, so ``interrupt`` can reach its socket.
        # Streams are interrupted through their ``_StreamConnection``.
        self._in_flight: Any = None

    def interrupt(self) -> None:
        """
        Unblocks a poll parked in its body read, from another thread.

        Best effort; a no-op when nothing is in flight. A request still
        connecting is bounded only by ``read_timeout``.
        """
        with self._lock:
            response = self._in_flight
        if response is not None:
            _interrupt_read(response)

    def _url(self, origin: str, path: str, basis: str | None) -> str:
        """
        The request URL: ``?kinds=`` on every request (see
        ``FDV2_PAYLOAD_KIND``), plus ``basis`` once a payload has committed.

        No ``mv`` parameter: it selects the flag data model and does not apply
        to skills.
        """
        query: dict[str, str] = {"kinds": FDV2_PAYLOAD_KIND}
        if basis:
            query["basis"] = basis
        return f"{origin}{path}?{urllib.parse.urlencode(query)}"

    def _request(
        self, origin: str, path: str, basis: str | None, headers: dict[str, str]
    ) -> urllib.request.Request:
        all_headers = {"Authorization": self._sdk_key, **headers}
        return urllib.request.Request(
            self._url(origin, path, basis), headers=all_headers, method="GET"
        )

    def poll(self, basis: str | None, etag: str | None) -> _PollResult:
        """One ``GET /sdk/poll``. A 304 is a first-class outcome, not an error."""
        headers = {"Accept": "application/json"}
        if etag:
            headers["If-None-Match"] = etag
        request = self._request(self._base_uri, POLL_PATH, basis, headers)
        try:
            with self._opener.open(request, timeout=self._read_timeout) as response:
                with self._lock:
                    self._in_flight = response
                try:
                    status = getattr(response, "status", None) or response.getcode()
                    if status == 304:
                        return _PollResult(not_modified=True, events=[], etag=etag)
                    body = _read_bounded(response, MAX_RESPONSE_BYTES)
                    new_etag = response.headers.get("ETag") or etag
                finally:
                    with self._lock:
                        self._in_flight = None
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                # urllib raises on 304; here it means "unchanged".
                return _PollResult(not_modified=True, events=[], etag=etag)
            raise _classify_status(exc.code, exc.headers) from exc
        except (_RecoverableTransportError, _FatalTransportError):
            raise
        except Exception as exc:
            raise _RecoverableTransportError(
                f"polling request failed: {type(exc).__name__}: {exc}"
            ) from exc

        return _PollResult(
            not_modified=False, events=_decode_poll_body(body), etag=new_etag
        )

    def stream(self, basis: str | None) -> _StreamConnection:
        """Opens ``GET /sdk/stream``."""
        request = self._request(
            self._stream_uri,
            STREAM_PATH,
            basis,
            {"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        )
        try:
            response = self._opener.open(request, timeout=self._read_timeout)
        except urllib.error.HTTPError as exc:
            raise _classify_status(exc.code, exc.headers) from exc
        except Exception as exc:
            raise _RecoverableTransportError(
                f"streaming request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _StreamConnection(response)


def _read_bounded(response: Any, limit: int) -> bytes:
    """
    Reads a whole poll body in chunks, abandoning it as soon as it exceeds
    *limit* bytes.
    """
    chunks: list[bytes] = []
    seen = 0
    while True:
        chunk = response.read(min(_READ_CHUNK_BYTES, limit + 1 - seen))
        if not chunk:
            return b"".join(chunks)
        seen += len(chunk)
        if seen > limit:
            raise _ResponseTooLargeError(
                f"polling response exceeded the {limit}-byte transport bound "
                f"(at least {seen} bytes received); nothing from it was applied"
            )
        chunks.append(chunk)


def _decode_poll_body(body: bytes) -> list[tuple[str, Any]]:
    """
    Unwraps ``{"events": [...]}``. Poll and stream events are identical, so both
    modes share one protocol reader.
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


def _iter_stream_lines(response: Any, limit: int) -> Any:
    """
    Yields a streaming body's raw lines, refusing any line over *limit* bytes.

    Any read failure (timeout, reset, truncation) is re-raised as
    ``_RecoverableTransportError``; the delivery loop treats other exceptions as
    bugs and stops. ``readline`` is given a size so a line that never ends is not
    buffered whole.
    """
    try:
        while True:
            line = response.readline(limit + 1)
            if not line:
                return
            if len(line) > limit:
                raise _ResponseTooLargeError(
                    f"an FDv2 stream line exceeded the {limit}-byte transport "
                    "bound; the connection was dropped and nothing from the "
                    "in-flight payload was applied"
                )
            yield line
    except (_RecoverableTransportError, _FatalTransportError):
        raise
    except Exception as exc:
        raise _RecoverableTransportError(
            f"reading the FDv2 stream failed: {type(exc).__name__}: {exc}"
        ) from exc


def _iter_sse(response: Any) -> Any:
    """
    Decodes an SSE body into ``(event name, data)`` pairs.

    Minimal: ``event:``/``data:`` fields, multi-line ``data`` joined with
    newlines, blank line dispatches, ``:`` comments skipped. An event over
    ``MAX_RESPONSE_BYTES`` raises a fatal error and the in-flight payload is
    abandoned.
    """
    limit = MAX_RESPONSE_BYTES
    try:
        name: str | None = None
        data_lines: list[str] = []
        event_bytes = 0
        for raw_line in _iter_stream_lines(response, limit):
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
                event_bytes = 0
                continue
            if line.startswith(":"):
                continue
            event_bytes += len(raw_line)
            if event_bytes > limit:
                raise _ResponseTooLargeError(
                    f"an FDv2 stream event exceeded the {limit}-byte transport "
                    f"bound (at least {event_bytes} bytes received); the "
                    "connection was dropped and nothing from the in-flight "
                    "payload was applied"
                )
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


def _backoff_delay(
    attempt: int, *, base: float, maximum: float, jitter: float = 0.5
) -> float:
    """
    Exponential backoff with jitter, capped at *maximum*.

    Jitter is subtracted, never added, so *maximum* is a true ceiling.
    """
    # Clamped: retries are unbounded, and ``float(2 ** n)`` overflows past
    # about 1024. ``2 ** 62`` exceeds any real cap. ``float(...)``: the integer
    # power is untyped to mypy.
    exponent = min(max(0, attempt - 1), 62)
    ceiling: float = min(maximum, base * float(2**exponent))
    return ceiling * (1.0 - jitter * random.random())


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class FDv2SkillStore:
    """
    A ``SkillStore`` fed by LaunchDarkly's SDK-facing FDv2 delivery channel.

    Constructed with the environment's server-side SDK key, started explicitly,
    and passed to ``init_client``::

        store = FDv2SkillStore(sdk_key=os.environ["LD_SDK_KEY"])
        store.start()
        if not store.wait_for_skills(timeout=10):
            ...  # no payload yet: see is_initialized
        await init_client(options={"skillStore": store})

        skill = await get_skill("pdf-extraction")
        ...
        store.close()

    It also works as a context manager.

    - **Server-side only.** A mobile key or client-side environment ID raises.
    - **The SDK key stays where you point it.** URIs must be ``https://``
      (``http://`` only to loopback) and redirects are never followed.
    - **Delivery runs in the background.** A daemon thread fills memory;
      ``get_object`` reads only what has arrived. Use ``wait_for_skills`` or
      ``is_initialized`` before relying on content.
    - **Last known good survives an outage.** A transport failure never empties
      the store or makes ``get_object`` raise; ``diagnostics`` and ``failed``
      report it.
    - **``close`` is final.** A closed store still serves what it received, but
      ``start`` raises; construct a new store instead.
    - **Content is verified by the accessors, not here.** An object with no
      ``contentHash`` is held and then withheld; see
      ``StoreDiagnostics.hashless_objects``.
    """

    def __init__(
        self,
        sdk_key: str,
        *,
        base_uri: str | None = None,
        stream_uri: str | None = None,
        mode: Mode = "stream",
        poll_interval: float = 30.0,
        read_timeout: float | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        _requester: Any = None,
    ) -> None:
        """
        Raises ``ValueError`` for an invalid credential, URI, mode, or interval.

        - *base_uri*: origin for ``GET /sdk/poll`` (default ``DEFAULT_BASE_URI``).
          Given without *stream_uri*, it is used for streaming too (relays and
          private instances).
        - *stream_uri*: origin for ``GET /sdk/stream`` (default
          ``DEFAULT_STREAM_URI``). Both must be ``https://``; ``http://`` is
          accepted only for ``localhost``, ``127.0.0.1`` or ``::1``.
        - *mode*: ``"stream"`` (default, recommended: revocations arrive in
          seconds) or ``"poll"`` (revocations arrive within one *poll_interval*).
        - *poll_interval*: seconds between polls; positive and finite.
        - *read_timeout*: the only network timeout, positive and finite. In
          ``"poll"`` mode it bounds the whole request (``DEFAULT_POLL_TIMEOUT``);
          in ``"stream"`` mode, each wait for more bytes
          (``DEFAULT_STREAM_READ_TIMEOUT``).
        - *initial_backoff*: the first retry delay; positive and finite.
        - *max_backoff*: caps every retry delay, including ``Retry-After``;
          positive, finite, and at least *initial_backoff*.

        Recoverable failures are retried for as long as the store runs; only a
        fatal status stops delivery and sets ``failed``.
        """
        _require_server_side_credential(sdk_key)
        # A lone ``base_uri`` serves both endpoints.
        if stream_uri is None:
            stream_uri = DEFAULT_STREAM_URI if base_uri is None else base_uri
        if base_uri is None:
            base_uri = DEFAULT_BASE_URI
        _require_https_uri(base_uri)
        _require_https_uri(stream_uri, "stream_uri")
        if mode not in ("stream", "poll"):
            raise ValueError(f'mode must be "stream" or "poll", got {mode!r}')
        # ``nan`` and ``inf`` pass a ``<= 0`` check, but would make the poll
        # loop spin or never poll again.
        if not (math.isfinite(poll_interval) and poll_interval > 0):
            raise ValueError(f"poll_interval must be positive, got {poll_interval!r}")
        if read_timeout is None:
            read_timeout = (
                DEFAULT_STREAM_READ_TIMEOUT
                if mode == "stream"
                else DEFAULT_POLL_TIMEOUT
            )
        elif not (math.isfinite(read_timeout) and read_timeout > 0):
            raise ValueError(f"read_timeout must be positive, got {read_timeout!r}")
        # With no failure bound these are the only limit on the retry loop: a
        # zero or negative value reconnects as fast as the network allows.
        for option, value in (
            ("initial_backoff", initial_backoff),
            ("max_backoff", max_backoff),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{option} must be positive and finite, got {value!r}")
        if initial_backoff > max_backoff:
            raise ValueError(
                f"initial_backoff ({initial_backoff!r}) must not exceed "
                f"max_backoff ({max_backoff!r})"
            )

        self._mode: Mode = mode
        self._poll_interval = poll_interval
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

        self._objects = _SkillObjectSet()
        self._reader = _ProtocolReader(self._objects)
        self._lock = threading.RLock()
        self._listeners: dict[str, list[Callable[[dict[str, Any]], Any]]] = {}

        self._basis: str | None = None
        self._etag: str | None = None
        # The basis ``_etag`` was issued for; the etag is only sent with it.
        self._etag_basis: str | None = None

        self._requester = _requester or _Requester(
            sdk_key.strip(),
            base_uri,
            read_timeout=read_timeout,
            stream_uri=stream_uri,
        )

        self._closed = False
        """``close`` has been called. Final: ``start`` raises afterwards."""
        self._stop = threading.Event()
        self._first_payload = threading.Event()
        """A payload has committed. The fact ``wait_for_skills`` reports."""
        self._delivery_ended = threading.Event()
        """Delivery has stopped, by ``close`` or ``_give_up``."""
        self._released = threading.Event()
        """Set by either of the above; what ``wait_for_skills`` waits on."""
        self._thread: threading.Thread | None = None
        self._failed_reason: str | None = None
        # The open streaming connection, so ``close`` can interrupt its read.
        self._connection: Any = None
        # Recoverable failures in a row; reset by a completed exchange, not by a
        # connection ending (a stream only ends by being dropped).
        self._failures = 0
        # The backoff's attempt number, kept apart from ``_failures``: reset
        # only by a stream that stayed open ``_BACKOFF_RESET_INTERVAL``, or by
        # a completed poll, so a server that answers and drops is backed off.
        self._backoff_attempts = 0
        # When the current stream connected, for the reset above.
        self._connected_at: float | None = None
        # Whether the current attempt got a complete answer, which separates a
        # recycled healthy stream from a failed one.
        self._attempt_answered = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> FDv2SkillStore:
        """
        Starts the delivery thread. Idempotent; returns ``self`` so it chains.

        Does not block: use ``wait_for_skills`` when boot ordering matters.

        Raises ``RuntimeError`` if the store has been closed. A store whose
        delivery stopped on its own (``failed`` is set) can be started again,
        with its backoff reset.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "This FDv2SkillStore has been closed, and close() is final: "
                    "delivery cannot be resumed, so a restarted store would "
                    "report itself started and never deliver. Construct a new "
                    "FDv2SkillStore to resume delivery. Held content is still "
                    "readable from the closed store."
                )
            # Read before the rearm clears ``_failed_reason``: a thread inside
            # ``_give_up`` is still alive but no longer delivering.
            delivering = (
                self._failed_reason is None
                and self._thread is not None
                and self._thread.is_alive()
            )
            self._rearm_waiters()
            if delivering:
                return self
            self._thread = threading.Thread(
                target=self._run, name="ld-ai-skills-fdv2", daemon=True
            )
            self._thread.start()
        return self

    def _rearm_waiters(self) -> None:
        """
        Resets per-run state for a store being started again after it gave up:
        the ended-delivery flag, ``failed``, and the failure count, including
        the one ``diagnostics`` reports. A payload already held still answers
        ``wait_for_skills``. Call with the lock held.
        """
        self._delivery_ended.clear()
        self._failed_reason = None
        self._failures = 0
        self._reader.diagnostics.connection_failures = 0
        self._backoff_attempts = 0
        if not self._first_payload.is_set():
            self._released.clear()

    def close(self, timeout: float = 5.0) -> None:
        """
        Stops delivery and waits up to *timeout* seconds for the thread to exit.
        Idempotent and safe from any thread.

        **Final:** ``start`` raises afterwards; construct a new store to resume.
        Held content is kept, so a closed store still serves what it received.
        ``launchdarkly_ai_server.shutdown()`` detaches the store from the
        accessors.
        """
        with self._lock:
            self._closed = True
        self._stop.set()
        # Release any ``wait_for_skills`` caller now.
        self._end_delivery()
        # Unblock the socket read the delivery thread is parked in (stream or poll).
        with self._lock:
            connection = self._connection
        if connection is not None:
            connection.close()
        self._requester.interrupt()
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
        Blocks until the first payload arrives, or *timeout* seconds elapse.

        Returns ``True`` once a payload has committed; a 304 alone does not
        count. That does not mean any skill verified, or that the environment
        has skills; see ``diagnostics``.

        Returns ``False`` on timeout, or early if delivery ends first (``close``,
        or a failure that will not be retried).
        """
        self._released.wait(timeout=timeout)
        return self._first_payload.is_set()

    def _publish_first_payload(self) -> None:
        """Records the first committed payload and lets any waiter go."""
        self._first_payload.set()
        self._released.set()

    def _end_delivery(self) -> None:
        """Records that delivery has stopped and lets any waiter go."""
        self._delivery_ended.set()
        self._released.set()

    def is_initialized(self) -> bool:
        """
        Whether a payload has arrived: ``wait_for_skills`` without the wait.

        Optional ``SkillStore`` method. ``write_skills("*")`` checks it so a
        reconcile before first delivery reports "unavailable" instead of pruning
        every skill. Stays ``True`` after ``close``.
        """
        return self._first_payload.is_set()

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

    # -- the SkillStore interface -----------------------------------------

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
        Registers *fn* to be called once per changed object, when a payload
        commits (never mid-transfer).

        - A put passes the raw skill object.
        - A revocation passes a ``{"key", "version"}`` tombstone with no
          ``content``; check for ``content`` before reading it.

        *fn* runs on the delivery thread: keep it cheap and non-blocking.
        Exceptions it raises are logged and swallowed.

        Raises ``ValueError`` for any *kind* other than ``SKILL_OBJECT_KIND``,
        since such a listener would never fire.
        """
        if kind != SKILL_OBJECT_KIND:
            raise ValueError(
                f"FDv2SkillStore notifies only {SKILL_OBJECT_KIND!r} changes, so "
                f"a listener on {kind!r} would never fire. Register it on "
                f"{SKILL_OBJECT_KIND!r}."
            )
        with self._lock:
            self._listeners.setdefault(kind, []).append(fn)

    def remove_listener(self, kind: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        """
        Unregisters one occurrence of *fn* from *kind*; a no-op if it is not
        registered. Safe from any thread, including inside a listener (takes
        effect from the next commit).
        """
        with self._lock:
            listeners = self._listeners.get(kind)
            if listeners is None:
                return
            try:
                listeners.remove(fn)
            except ValueError:
                return

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
        while not self._stop.is_set():
            with self._lock:
                self._attempt_answered = False
            try:
                if self._mode == "stream":
                    self._stream_once()
                else:
                    self._poll_once()
                # A returned poll is a current answer, even a 304. Stream
                # successes are recorded in ``_apply``.
                self._record_success()
                with self._lock:
                    self._backoff_attempts = 0
            except _FatalTransportError as exc:
                self._give_up(str(exc))
                return
            except _RecoverableTransportError as exc:
                if self._stop.is_set():
                    # ``close`` interrupted the request; not a failure.
                    return
                if isinstance(exc, _StaleRequestStateError):
                    # With no basis or etag to drop, the 400 is fatal. Otherwise
                    # drop them and request a full transfer once.
                    with self._lock:
                        exhausted = self._basis is None and self._etag is None
                        if not exhausted:
                            self._basis = None
                            self._etag = None
                            self._etag_basis = None
                    if exhausted:
                        self._give_up(str(exc))
                        return
                with self._lock:
                    # Discard any partial payload from the dropped connection.
                    self._reader._abandon_in_flight()
                    answered = self._attempt_answered
                    # A goodbye after a completed exchange is a routine recycle.
                    recycled = exc.recycled and answered
                    if not recycled:
                        self._failures += 1
                        self._reader.diagnostics.connection_failures = self._failures
                        self._reader.diagnostics.last_error = str(exc)
                    connected_at = self._connected_at
                    self._connected_at = None
                    if (
                        connected_at is not None
                        and time.monotonic() - connected_at >= _BACKOFF_RESET_INTERVAL
                    ):
                        self._backoff_attempts = 0
                    self._backoff_attempts += 1
                    attempt = self._backoff_attempts
                delay = exc.retry_after
                if delay is None or not math.isfinite(delay):
                    delay = _backoff_delay(
                        attempt, base=self._initial_backoff, maximum=self._max_backoff
                    )
                else:
                    # Floor at ``initial_backoff`` so ``Retry-After: 0`` cannot
                    # cause a tight reconnect loop.
                    delay = max(delay, self._initial_backoff)
                # Cap at ``max_backoff`` even if ``Retry-After`` asks for more.
                delay = min(delay, self._max_backoff)
                if answered:
                    # A stream recycled after a complete answer is routine.
                    logger.debug(
                        "The FDv2 stream ended after a complete answer (%s); "
                        "reconnecting in %.1fs",
                        exc,
                        delay,
                    )
                else:
                    logger.warning(
                        "Skill delivery failed (%s); retrying in %.1fs", exc, delay
                    )
                if self._stop.wait(delay):
                    return
                continue
            except Exception as exc:  # pragma: no cover - defensive
                self._give_up(f"unexpected error in skill delivery: {exc!r}")
                logger.error("Unexpected error in skill delivery", exc_info=True)
                return

            if self._mode == "poll" and self._stop.wait(self._poll_interval):
                return

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._attempt_answered = True
            self._reader.diagnostics.connection_failures = 0

    def _give_up(self, reason: str) -> None:
        with self._lock:
            self._failed_reason = reason
            self._reader.diagnostics.last_error = reason
            # Under the lock with the reason, so a concurrent ``start`` cannot
            # have its fresh waiters released by this dying run.
            self._end_delivery()
        # Tests match on this message's prefix.
        logger.error(
            "Skill delivery has stopped and will not retry: %s. The store keeps "
            "serving the last content it received, and skills will not update "
            "until delivery runs again: call start() on this store once the "
            "cause is fixed, or restart the process.",
            reason,
        )

    def _apply(self, name: str, data: Any) -> bool:
        """
        Feeds one event to the reader, publishes a commit, and raises the
        transport error the event calls for, if any.

        Returns whether the event completed an exchange — a commit, or the
        server confirming the content held is current — which is what
        ``_poll_once`` adopts an etag on.
        """
        with self._lock:
            outcome = self._reader.handle(name, data)
            if outcome.committed and outcome.basis is not None:
                self._basis = outcome.basis
        if outcome.committed or outcome.up_to_date:
            # Both are completed exchanges. Counting ``up_to_date`` keeps a
            # stream for an unchanging environment from reading as failing.
            self._record_success()
        if outcome.committed:
            self._publish_first_payload()
            if outcome.changes:
                self._notify(outcome.changes)
        if outcome.fatal:
            raise _FatalTransportError(outcome.fatal)
        if outcome.disconnect:
            raise _RecoverableTransportError(
                outcome.disconnect, recycled=outcome.recycled
            )
        return outcome.committed or outcome.up_to_date

    def _poll_once(self) -> None:
        with self._lock:
            basis = self._basis
            # Send the etag only with the basis it was issued for; a 304 to a
            # stale pair would describe the previous request.
            etag = self._etag if self._etag_basis == basis else None
        result = self._requester.poll(basis, etag)
        if result.not_modified:
            logger.debug("Skill payload unchanged (HTTP 304)")
            # A 304 confirms the payload this store holds; it cannot establish
            # one. ``is_initialized`` stays false until something commits, so a
            # store that has received nothing never authorises a prune of the
            # files on disk. ``_run`` counts the poll as a healthy answer.
            return
        completed = False
        for name, data in result.events:
            completed = self._apply(name, data) or completed
        if not completed:
            # An etag describes the body it came with, so it is adopted only when
            # that body is also what the store now holds: a commit, or a ``none``
            # intent. A transfer this SDK could not apply is neither, and keeping
            # its etag would let the next 304 confirm content never applied. Any
            # etag already held stays valid, so it is left alone, not cleared.
            return
        with self._lock:
            self._etag = result.etag
            self._etag_basis = basis

    def _stream_once(self) -> None:
        with self._lock:
            basis = self._basis
        connection = self._requester.stream(basis)
        with self._lock:
            self._connection = connection
            self._connected_at = time.monotonic()
        try:
            # ``close`` may have run during the connect, before there was a
            # connection to interrupt.
            if self._stop.is_set():
                return
            for name, data in connection.events:
                if self._stop.is_set():
                    return
                self._apply(name, data)
        except Exception:
            if self._stop.is_set():
                # ``close`` interrupted the read on purpose.
                return
            raise
        finally:
            connection.close()
            with self._lock:
                self._connection = None
        # A stream that ends without a goodbye is a dropped connection.
        raise _RecoverableTransportError("the FDv2 stream closed unexpectedly")
