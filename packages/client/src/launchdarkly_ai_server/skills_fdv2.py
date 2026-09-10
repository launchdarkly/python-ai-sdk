"""
Agent Skills — the FDv2 delivery transport.

The store implementation that talks to LaunchDarkly. It sits *below* the
``SkillStore`` interface: it produces raw wire objects in the shape
``skills_core`` documents, and everything above — the accessors, integrity
verification, the ``Skill`` dataclass, materialization — is unaware of it.

Layering::

    launchdarkly_ai_server
      └─ SkillStore protocol (skills_core)     ── the interface accessors call
            └─ FDv2SkillStore (this module)    ── deserialise, hold, serve
                  └─ LaunchDarkly's SDK-facing FDv2 channel
                     GET /sdk/poll, GET /sdk/stream, authenticated with the
                     environment's server-side SDK key

Dependencies run one way: this module imports ``skills_core`` for the
interface's kind constant and nothing else from the feature, and nothing in the
feature imports it. It uses only the standard library, so it adds no dependency
to a package whose sole runtime dependency is ``opentelemetry-api``.

Three things this module does *not* do, on purpose:

- **It does not verify content.** Verification lives at the accessor boundary in
  ``skills_core`` so that it applies to every store equally, including a
  customer's own.
- **It does not work around a missing ``contentHash``.** A hashless object is
  held verbatim and *withheld* by verification with ``missing_content_hash``.
  This module's job is to make that outcome loud — see ``StoreDiagnostics``.
- **It does not evaluate anything.** Flag and segment objects that share the
  connection are skipped and counted, nothing more.

The design rationale — why the skill's version is read from the object's
``key`` and never from ``version``, why changes commit at
``payload-transferred``, why there is one network timeout — is in
``agents.md`` under *The delivery transport*.
"""

from __future__ import annotations

import json
import logging
import math
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

FDV2_OBJECT_KIND = "skill"
"""
The FDv2 ``kind`` skills are delivered under.

Object kinds on the SDK-facing channel are open strings: the agent-skill payload
is classified ``generic`` and every object in it carries the kind its producer
registered, which for skills is the bare category name. Delivery lower-cases the
kind, so an exact comparison is the whole test. The kind happens to equal
``skills_core.SKILL_OBJECT_KIND`` today; they are still separate constants,
because one is a wire value LaunchDarkly owns and the other is an SDK seam.
"""

FDV2_KEY_DELIMITER = ":"
"""
What separates a skill's key from its version inside the object's wire ``key``.

A generic object is identified on the wire as ``<key>:<version>`` — the skill's
own key, one delimiter, the skill's own version — because each version of a
skill is a distinct object in the payload. Delivery forbids the delimiter inside
a registered category and skill keys cannot contain it, so a well-formed wire key
has exactly one.
"""

DEFAULT_BASE_URI = "https://sdk.launchdarkly.com"
"""Where the SDK-facing FDv2 endpoints live. Overridable for Federal and private
instances."""

POLL_PATH = "/sdk/poll"
STREAM_PATH = "/sdk/stream"

DEFAULT_POLL_TIMEOUT = 10.0
"""Default ``read_timeout`` in ``"poll"`` mode: the bound on one whole request."""

DEFAULT_STREAM_READ_TIMEOUT = 300.0
"""Default ``read_timeout`` in ``"stream"`` mode: the longest gap tolerated
between two reads. LaunchDarkly's heartbeats arrive well inside this."""

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
The skill object envelope's fields, copied through verbatim. Nothing is coerced
or defaulted: a transport that filled in a missing field would be forging the
very thing verification exists to check.
"""

_PAYLOAD_SELECTOR = re.compile(r"\(p:([^:()]+):\d+\)")
"""
The payload identity inside a transfer's selector, ``(p:<id>:<version>)``.

The selector is the only place a completed transfer names its own payload:
``put-object``, ``delete-object`` and ``payload-transferred`` carry no payload id
of their own. ``_ProtocolReader`` reads it as a fallback for an intent that named
no ``id``.
"""

Mode = Literal["stream", "poll"]

_MOBILE_KEY_PREFIX = "mob-"
_SERVER_KEY_PREFIX = "sdk-"
_CLIENT_SIDE_ID = re.compile(r"\A[0-9a-f]{20,}\Z")
"""A client-side environment ID: bare lowercase hex. Server-side and mobile keys
both carry a prefix, so this shape is unambiguous rather than heuristic."""


# ---------------------------------------------------------------------------
# Server-side only
# ---------------------------------------------------------------------------


def _require_server_side_credential(sdk_key: str) -> None:
    """
    Refuses a mobile key or a client-side environment ID.

    Skill content is customer-confidential, and payload assignment is shared
    across credential types, so a client-side credential may well *succeed*
    against these endpoints. Raises rather than logs: a store built on the wrong
    credential should not exist.
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
        # Not rejected: private instances and test doubles issue credentials
        # without the public prefix. Only the two unambiguous shapes above are.
        logger.warning(
            "The credential given to FDv2SkillStore does not look like a "
            "LaunchDarkly server-side SDK key (sdk-...). Skills are delivered "
            "only to server-side credentials; if this is a client-side or mobile "
            "credential the connection will be rejected or will deliver nothing."
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class StoreDiagnostics:
    """
    What the transport has seen. Read-only from a caller's perspective.

    Not part of the ``SkillStore`` interface. It exists because "this environment
    has no skills" and "every skill was withheld" are easy to mistake for each
    other, and a counter is easier to assert on than a log line.
    """

    payloads_transferred: int = 0
    """Completed ``payload-transferred`` commits since the store started."""
    skill_objects_received: int = 0
    """``put-object`` events identified as skills, across all payloads."""
    objects_ignored: int = 0
    """Objects skipped because they were not skills: flags, segments, and any
    future kind. Skipping is the contract, not a failure."""
    objects_revoked: int = 0
    """``delete-object`` events applied to skills."""
    payloads_ignored: int = 0
    """
    Transfers not applied because they completed a payload other than the one
    skills arrive on. Zero while delivery sends one payload per connection.
    """
    hashless_objects: int = 0
    """
    Skill objects whose envelope carried no ``contentHash``.

    **Nonzero means skills are being withheld**: verification withholds every one
    of these with ``missing_content_hash``.
    """
    connection_failures: int = 0
    """Recoverable transport failures since the last successful transfer."""
    last_error: str | None = None
    """The most recent transport error, if any. Human-readable; do not parse."""


# ---------------------------------------------------------------------------
# Deserialisation — where the skill's version lives in the key, not in version
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tombstone:
    """A ``delete-object`` narrowed to the identity it revokes."""

    key: str
    object_version: int | None


def _is_skill_event(data: Any) -> bool:
    """
    Whether one ``put-object`` / ``delete-object`` payload is a skill.

    The kind alone decides it. Every other kind is ignored, not rejected,
    because flag and segment objects share the connection and erroring on them
    would turn a normal payload into a reconnect loop.
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
    Reads ``<key>:<version>`` off one object's wire ``key``.

    Lenient where leniency keeps the object diagnosable and strict only where
    there is nothing to diagnose:

    - No delimiter: the whole wire key is the skill key and there is no version,
      so the object is held version-less and verification reports
      ``invalid_version`` under a key the caller can recognise.
    - A version that is not a run of digits (``"pdf:latest"``, ``"pdf:"``,
      ``"a:1:2"``): the text is carried through *as the version*, for the same
      reason — the caller learns that ``pdf`` arrived broken, not that it is
      absent.
    - An empty key before the delimiter (``":3"``): there is no identity to hold
      it under, so ``None``, and the caller drops it.

    Leading zeros are accepted (``"pdf:03"`` is version 3) since ``int`` is the
    identity a reference pins, not the spelling.
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
    Translates one FDv2 skill ``put-object`` into the raw object shape the
    ``SkillStore`` interface defines.

    **The one translation this adapter must get right:**

        wire ``key``      →  stored ``key`` and ``version``  (split on ``:``)
        wire ``version``  →  dropped                          (the *payload* version)

    Each version of a skill is its own object on the wire, identified as
    ``<key>:<version>``; that version is what a ``{key, version}`` reference
    pins. The event's ``version`` field is the version of the payload the object
    arrived in and moves whenever anything in the environment moves. Confusing
    them fails silently: the object verifies and the caller gets content under a
    version number that means nothing.

    Returns ``None`` only when the wire ``key`` carries no skill key at all,
    since such an object has no identity to store it under. Every other defect
    is carried through so that verification withholds it with a reason code
    rather than the transport dropping it into indistinguishable absence.
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

    # Absent stays absent and malformed stays malformed, so verification sees
    # what arrived (as `invalid_version`) rather than something invented here.
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
    Narrows one FDv2 skill ``delete-object`` to the identity it revokes, reading
    the wire ``key`` the same way a put does.

    An ``object_version`` of ``None`` means the delete named no usable version
    and is read as "revoke every version of this key". That is the safe
    direction: the alternative is continuing to serve content LaunchDarkly has
    withdrawn. It also removes whatever a malformed put of the same wire key
    left held, since that was stored version-less under the same skill key.
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

    Lookup semantics are identical to ``InMemorySkillStore``'s, down to the
    fall-through to a version-less entry, so that the store a caller configures
    cannot change how a pinned reference resolves; ``TestInterfaceParity``
    asserts it. Reimplemented rather than inherited because the transport needs
    ``delete`` and the atomic ``replace_with`` a full transfer requires.

    An object too malformed to carry a usable version is still held, under its
    key alone, so verification withholds it with a signal rather than the
    transport dropping it.
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
        if version is not None:
            # Fall through to the version-less entry so a malformed object
            # reaches verification rather than reading as simply absent.
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
class _TransferOutcome:
    """What one event did. Aggregated by the caller; nothing here does I/O."""

    committed: bool = False
    changes: list[dict[str, Any]] = field(default_factory=list)
    basis: str | None = None
    fatal: str | None = None
    disconnect: str | None = None


class _ProtocolReader:
    """
    Applies FDv2 events to an object set. Pure — no sockets, no threads, no clock —
    so every wire case is testable without a server.

    **Changes are buffered and committed at ``payload-transferred``.** A payload
    version is the unit of consistency: applying half of one would publish a
    state the server never described, and on a full transfer would briefly empty
    the store. Listeners therefore fire once per commit, not once per object.

    **The first payload intent is read, and is assumed to be the skill payload.**
    Delivery provides one payload per credential and the protocol requires a
    client to ignore all but the first payload intent, so ``payloads[0]`` is both
    what arrives and what the protocol says to read. If that ever widens, an
    ``xfer-full`` for somebody else's payload would empty the skill set and the
    next ``payload-transferred`` would publish it empty — with pruning on, the
    difference between a reconcile and deleting a customer's files. This layer
    therefore learns which payload skills arrive on and declines to apply a
    transfer of any other, once at WARNING and counted. The residual is the first
    transfer of a connection: before a skill has arrived there is nothing to
    compare a payload against.
    """

    def __init__(self, committed: _SkillObjectSet) -> None:
        self._committed = committed
        self._intent: str | None = None
        self._pending: _SkillObjectSet | None = None
        self._changes: list[dict[str, Any]] = []
        self.diagnostics = StoreDiagnostics()
        # Identities already reported by ``_warn_hashless``. Per reader, so a
        # recreated store reports again and two stores never quieten each other.
        self._warned_hashless: set[tuple[str, Any]] = set()
        # The payload the current intent describes, and the payload skills have
        # actually arrived on. One payload per connection makes these the same
        # payload; the class docstring says why they are kept apart regardless.
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
            # Built alongside the live set rather than in place, so an
            # interrupted transfer leaves last known good intact.
            self._pending = _SkillObjectSet()
        elif intent == _INTENT_TRANSFER_CHANGES:
            self._pending = self._committed.copy()
        else:
            if intent != _INTENT_TRANSFER_NONE:
                logger.debug("Ignoring FDv2 server-intent with intentCode %r", intent)
            self._pending = None
        return _TransferOutcome()

    def _target_for(self, data: Any) -> _SkillObjectSet | None:
        """
        The pending set a skill object event applies to, or ``None`` when the
        event is not a skill or the current intent carries no objects.

        An object arriving with no ``server-intent`` at all is treated as a
        delta against what is held rather than dropped.
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
        target.delete(tombstone)
        self.diagnostics.objects_revoked += 1
        # A revocation identifies the payload as ours just as a put does.
        self._skills_in_payload += 1
        # A tombstone carries identity and no content; see
        # ``FDv2SkillStore.add_listener`` for what listeners should expect.
        self._changes.append(
            {"key": tombstone.key, "version": tombstone.object_version}
        )
        return _TransferOutcome()

    def _payload_transferred(self, data: Any) -> _TransferOutcome:
        state = data.get("state") if isinstance(data, dict) else None
        version = data.get("version") if isinstance(data, dict) else None
        payload_id = self._intent_payload_id or _payload_id_from_selector(state)
        if self._pending is not None and self._is_foreign_payload(payload_id):
            self._warn_foreign_payload(payload_id)
            self.diagnostics.payloads_ignored += 1
            self._changes = []
        elif self._pending is not None:
            self._committed.replace_with(self._pending)
            _warn_if_nothing_can_verify(self._committed)
            if self._skills_in_payload and payload_id is not None:
                # Learnt, not configured: nothing below the interface is told
                # which payload is which, so the payload that carried a skill
                # put or revocation is the payload skills arrive on.
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
            logger.info("FDv2 connection closing: %s", reason)
        if catastrophe:
            return _TransferOutcome(
                fatal=f"server sent a catastrophic goodbye: {reason}"
            )
        return _TransferOutcome(disconnect=f"server said goodbye: {reason}")

    # -- payload identity ----------------------------------------------------

    def _is_foreign_payload(self, payload_id: str | None) -> bool:
        """
        Whether a transfer completes a payload other than the one skills arrive on.

        ``False`` unless both payloads are known, so one-payload delivery and the
        first transfer of a connection behave exactly as they did before this
        check existed.
        """
        return (
            self._skill_payload_id is not None
            and payload_id is not None
            and payload_id != self._skill_payload_id
        )

    # -- diagnostics ---------------------------------------------------------

    def _warn_multiple_payloads(self, payloads: list[Any]) -> None:
        """
        One WARNING per reader for an intent describing more than one payload.

        Not an error: reading only the first is what the protocol asks for. But it
        means the first payload is no longer *guaranteed* to be the skill payload,
        and an intent for another payload arriving before any skill has been seen
        is the one case ``_is_foreign_payload`` cannot catch.
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
        One ERROR per ``(key, version)`` whose envelope had no ``contentHash``.

        ERROR rather than WARN because an empty accessor result is otherwise
        indistinguishable from an environment that has no skills.
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
    One ERROR per committed payload in which *nothing* held can possibly verify.

    ``log_withholding_summary`` reports the same condition at the accessor
    boundary, but only once a caller asks. This fires at delivery time, so it is
    visible in a process that boots, materializes nothing, and exits.
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


class _RecoverableTransportError(Exception):
    """A failure worth retrying. Carries a server-requested delay when given one."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_FORBIDDEN_ADVICE = (
    "The FDv2 protocol is opt-in per LaunchDarkly account and is served as HTTP "
    "403 while it is off. Skill delivery needs it enabled; contact LaunchDarkly "
    "support to enable it for your account."
)


def _retry_after_seconds(headers: Any) -> float | None:
    """
    ``Retry-After`` in seconds, when the server sent a usable one.

    The HTTP-date form, and non-finite values such as ``inf`` or ``1e309`` that
    ``float`` accepts, fall back to our own backoff: none of them is a delay,
    and an infinite one would overflow the wait that honours it.
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
    if status in (400, 405, 406, 414, 501):
        return _FatalTransportError(
            f"LaunchDarkly returned HTTP {status}, which retrying will not fix. "
            "The request this adapter sent was not understood. It carries only "
            "the SDK key and, after the first payload, a 'basis' selector, so "
            "check the base URI and that the endpoint speaks FDv2."
        )
    return _RecoverableTransportError(
        f"LaunchDarkly returned HTTP {status}", _retry_after_seconds(headers)
    )


def _interrupt_read(response: Any) -> None:
    """
    Best-effort interruption of a read blocked on *response*, from another thread.

    Closing the response is not enough: CPython's buffered reader stays parked in
    ``readline`` until bytes arrive. Shutting the *socket* down underneath it
    unblocks it immediately. Reaching the socket means walking urllib's private
    attribute chain, so every step is guarded and failure is silent: the
    delivery thread is a daemon and ``close``'s join timeout is the backstop.
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
    One open streaming connection: an event iterator plus a way to interrupt it
    from another thread, which is what ``FDv2SkillStore.close`` needs.
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
    The only place this module opens a socket. Standard library only, on purpose.

    *read_timeout* is applied to every socket operation of a request. ``urllib``
    has no separate connect timeout: its ``timeout`` becomes the socket timeout
    for the whole operation, so connecting, waiting for headers and each body
    read are all bounded by the same value.
    """

    def __init__(
        self,
        sdk_key: str,
        base_uri: str,
        *,
        read_timeout: float,
        opener: Any = None,
    ) -> None:
        self._sdk_key = sdk_key
        self._base_uri = base_uri.rstrip("/")
        self._read_timeout = read_timeout
        # Injectable so tests can drive a fake endpoint without a socket.
        self._opener = opener or urllib.request.build_opener()

    def _url(self, path: str, basis: str | None) -> str:
        """
        The request URL: the path, plus ``basis`` once a payload has committed.

        Deliberately no ``mv`` (data model version). That parameter selects the
        *flag* data model and the connection rejects any value but the flag
        default; the agent-skill payload is generic, is served regardless of it,
        and has no model version of its own to ask for.
        """
        if not basis:
            return f"{self._base_uri}{path}"
        return f"{self._base_uri}{path}?{urllib.parse.urlencode({'basis': basis})}"

    def _request(
        self, path: str, basis: str | None, headers: dict[str, str]
    ) -> urllib.request.Request:
        all_headers = {"Authorization": self._sdk_key, **headers}
        return urllib.request.Request(
            self._url(path, basis), headers=all_headers, method="GET"
        )

    def poll(self, basis: str | None, etag: str | None) -> _PollResult:
        """One ``GET /sdk/poll``. A 304 is a first-class outcome, not an error."""
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


def _decode_poll_body(body: bytes) -> list[tuple[str, Any]]:
    """
    Unwraps ``{"events": [...]}``. Polling and streaming carry identical event
    objects, which is why the protocol reader is shared between the two modes.
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

    Minimal on purpose: ``event:``/``data:`` fields, multi-line ``data`` joined
    with newlines, a blank line dispatching, and ``:`` comments skipped.
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


def _backoff_delay(
    attempt: int, *, base: float, maximum: float, jitter: float = 0.5
) -> float:
    """
    Exponential backoff with jitter, capped at *maximum*.

    Jitter is subtractive over the whole range rather than added on top, so the
    cap is a real ceiling: a fleet restarted together must not reconnect in
    lockstep, and must not exceed the interval the cap promises.
    """
    # float(2 ** n): the integer power is untyped to mypy.
    ceiling: float = min(maximum, base * float(2 ** max(0, attempt - 1)))
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
        store.wait_for_skills(timeout=10)
        await init_client(options={"skillStore": store})

        skill = await get_skill("pdf-extraction")
        ...
        store.close()

    It also works as a context manager.

    **Server-side only.** A mobile key or a client-side environment ID is
    refused in the constructor.

    **Delivery is in the background; retrieval is not.** A daemon thread owns
    the connection and fills memory, and ``get_object`` only ever reads what has
    already arrived. A process that calls ``get_skill`` immediately after
    ``start()`` may see an empty store; ``wait_for_skills`` orders boot against
    the first payload.

    **Last known good survives an outage.** A transport failure never empties
    the store and never makes ``get_object`` raise, which is what makes
    ``write_skills(on_unavailable="keep")`` correct. ``diagnostics`` and
    ``failed`` report the degradation.

    **What arrives is untrusted.** Raw wire objects are held verbatim and
    verified at the accessor boundary, not here. In particular an object with no
    ``contentHash`` is held and then *withheld*; see
    ``StoreDiagnostics.hashless_objects``.
    """

    def __init__(
        self,
        sdk_key: str,
        *,
        base_uri: str = DEFAULT_BASE_URI,
        mode: Mode = "stream",
        poll_interval: float = 30.0,
        read_timeout: float | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        max_consecutive_failures: int = 10,
        _requester: Any = None,
    ) -> None:
        """
        *mode* is ``"stream"`` by default. Prefer it: a ``delete-object`` reaches
        a live stream in seconds. ``"poll"`` exists for environments that cannot
        hold a long-lived connection, and revocation there is one
        ``poll_interval`` late.

        *read_timeout* is the only network timeout and bounds every socket
        operation of a request, so its meaning and default follow the mode: in
        ``"poll"`` it bounds the whole request (``DEFAULT_POLL_TIMEOUT``); in
        ``"stream"`` it bounds each wait for the next bytes
        (``DEFAULT_STREAM_READ_TIMEOUT``). Must be positive when given.

        *max_backoff* caps every delay between retries, including one the server
        asks for with ``Retry-After``.

        *max_consecutive_failures* bounds the retry loop. On exceeding it the
        transport stops, logs an error, and the store keeps serving last known
        good; ``failed`` reports it. Only failures in a row count: a committed
        payload resets the count.
        """
        _require_server_side_credential(sdk_key)
        if mode not in ("stream", "poll"):
            raise ValueError(f'mode must be "stream" or "poll", got {mode!r}')
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive, got {poll_interval!r}")
        if read_timeout is None:
            read_timeout = (
                DEFAULT_STREAM_READ_TIMEOUT
                if mode == "stream"
                else DEFAULT_POLL_TIMEOUT
            )
        elif not (math.isfinite(read_timeout) and read_timeout > 0):
            raise ValueError(f"read_timeout must be positive, got {read_timeout!r}")

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
            read_timeout=read_timeout,
        )

        self._stop = threading.Event()
        self._first_payload = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed_reason: str | None = None
        # The open streaming connection, so ``close`` can interrupt its read.
        self._connection: Any = None
        # Recoverable failures since the last committed payload. Reset at the
        # commit rather than when a connection returns: a stream only ever ends
        # by being dropped, so resetting on return would count every healthy,
        # server-recycled connection as a failure.
        self._failures = 0

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
        received. Detaching the store from the accessors is the job of the
        package-level ``launchdarkly_ai_server.shutdown()`` coroutine.
        """
        self._stop.set()
        # The delivery thread is normally blocked in a socket read that no flag
        # can reach; without this the join waits out its full timeout.
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
        not that the environment has any skills. ``diagnostics`` answers the rest.
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
        Registers *fn* to be called once per changed object, at
        ``payload-transferred`` rather than as objects stream in.

        A put notifies with the raw skill object. A revocation notifies with a
        ``{"key", "version"}`` tombstone carrying no content, so a listener that
        reads content must check for ``content`` rather than assume it.

        *fn* runs on the delivery thread. Keep it cheap and non-blocking. An
        exception it raises is logged and swallowed, because a broken listener
        must not be able to kill delivery.
        """
        with self._lock:
            self._listeners.setdefault(kind, []).append(fn)

    def remove_listener(self, kind: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        """
        Unregisters *fn* from *kind*. Safe to call from any thread, including
        from inside a listener: a removal during one commit takes effect from
        the next.

        Removes one occurrence; removing a callable that is not registered is a
        no-op, so ``SkillWatcher.close`` can detach unconditionally.
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
            try:
                if self._mode == "stream":
                    self._stream_once()
                else:
                    self._poll_once()
                # A poll that returned is a current answer even when it committed
                # nothing (HTTP 304). A stream never returns normally; its
                # successes are counted at each commit in ``_apply``.
                self._record_success()
            except _FatalTransportError as exc:
                self._give_up(str(exc))
                return
            except _RecoverableTransportError as exc:
                with self._lock:
                    self._failures += 1
                    failures = self._failures
                    self._reader.diagnostics.connection_failures = failures
                    self._reader.diagnostics.last_error = str(exc)
                if failures > self._max_consecutive_failures:
                    self._give_up(
                        f"gave up after {failures} consecutive failures; "
                        f"last error: {exc}"
                    )
                    return
                delay = exc.retry_after
                if delay is None or not math.isfinite(delay):
                    delay = _backoff_delay(
                        failures, base=self._initial_backoff, maximum=self._max_backoff
                    )
                # ``Retry-After`` is a request and ``max_backoff`` is a promise.
                # The header may come from a proxy rather than LaunchDarkly, and
                # a value in the hours would park revocation for that long.
                delay = min(delay, self._max_backoff)
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
            self._reader.diagnostics.connection_failures = 0

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
        # Unblock anyone waiting on a first payload that is never coming.
        self._first_payload.set()

    def _apply(self, name: str, data: Any) -> None:
        """
        Feeds one event to the reader, publishes a commit, and raises the
        transport error the event calls for, if any.
        """
        with self._lock:
            outcome = self._reader.handle(name, data)
            if outcome.committed and outcome.basis is not None:
                self._basis = outcome.basis
        if outcome.committed:
            # A commit breaks the row of consecutive failures.
            self._record_success()
            self._first_payload.set()
            if outcome.changes:
                self._notify(outcome.changes)
        if outcome.fatal:
            raise _FatalTransportError(outcome.fatal)
        if outcome.disconnect:
            raise _RecoverableTransportError(outcome.disconnect)

    def _poll_once(self) -> None:
        with self._lock:
            basis, etag = self._basis, self._etag
        result = self._requester.poll(basis, etag)
        with self._lock:
            self._etag = result.etag
        if result.not_modified:
            logger.debug("Skill payload unchanged (HTTP 304)")
            # A 304 counts as a first payload, so a boot that reconnects with a
            # cached basis is not blocked on a transfer the server will not send.
            self._first_payload.set()
            return
        for name, data in result.events:
            self._apply(name, data)

    def _stream_once(self) -> None:
        with self._lock:
            basis = self._basis
        connection = self._requester.stream(basis)
        with self._lock:
            self._connection = connection
        try:
            # ``close`` may have run while the connect was in flight and found
            # no connection to interrupt; this is the last chance to notice
            # before the read below blocks.
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
