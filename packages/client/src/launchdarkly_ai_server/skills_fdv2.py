"""
Agent Skills — the FDv2 delivery protocol.

The half of the delivery transport that has no I/O: identifying skill objects
on the wire, translating them into the raw object shape the ``SkillStore``
interface defines, holding them by ``(key, version)``, and applying a
payload's events as one consistent commit. ``FDv2SkillStore``, the store that
puts a network connection underneath this, follows in a separate change.

It sits *below* the ``SkillStore`` interface, and everything above — the
accessors, integrity verification, the ``Skill`` dataclass, materialization —
is unaware of it.

Layering::

    launchdarkly_ai_server
      └─ SkillStore protocol (skills_core)     ── the interface accessors call
            └─ FDv2SkillStore (this module)    ── deserialise, hold, serve
                  └─ LaunchDarkly's SDK-facing FDv2 channel
                     GET /sdk/poll, GET /sdk/stream, authenticated with the
                     environment's server-side SDK key

Dependencies run one way: this module imports nothing from the feature beyond
the version validator in ``types_validation``, and nothing in the feature
imports it. It uses only the standard library, so it adds no dependency
to a package whose sole runtime dependency is ``opentelemetry-api``.

Three things this layer does *not* do, on purpose:

- **It does not verify content.** Verification lives at the accessor boundary in
  ``skills_core`` so that it applies to every store equally, including a
  customer's own.
- **It does not work around a missing ``contentHash``.** A hashless object is
  held verbatim and *withheld* by verification with ``missing_content_hash``.
  This module's job is to make that outcome loud — see ``StoreDiagnostics``.
- **It does not evaluate anything.** Flag and segment objects that share the
  connection are skipped and counted, nothing more.

One assumption it *does* make, and states: **the payload intent it reads is the
payload skills arrive on.** Delivery sends one payload per credential and the
protocol tells a client to read only the first payload intent, so today those are
the same payload. ``_ProtocolReader`` keeps the pair apart anyway, because the
cost of conflating them is an emptied skill set.

The design rationale — why the skill's version is read from the object's
``key`` and never from ``version``, why changes commit at
``payload-transferred`` — is in ``agents.md`` under *The delivery transport*.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

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
        # No lock: ``handle`` runs only on its owner's single delivery thread.
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
        # A tombstone carries identity and no content, so a listener that reads
        # content must check for ``content`` rather than assume it.
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
