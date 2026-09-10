"""
Tests for the FDv2 skill delivery protocol.

Wire semantics — which objects are skills, ``objectVersion`` versus ``version``,
revocation, mixed payloads, the commit at ``payload-transferred`` — are asserted
against ``_ProtocolReader``, which has no I/O, so each case reads as the contract
it is rather than as a server script.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

import pytest

from launchdarkly_ai_server import InMemorySkillStore
from launchdarkly_ai_server.skills_core import SKILL_OBJECT_KIND
from launchdarkly_ai_server.skills_fdv2 import (
    FDV2_OBJECT_CATEGORY,
    FDV2_OBJECT_KIND,
    _is_skill_event,
    _ProtocolReader,
    _SkillObjectSet,
    _store_object_from_put,
    _tombstone_from_delete,
)

pytestmark = pytest.mark.usefixtures("reset_skill_state")

SKILL_BODY = "---\nname: PDF Extraction\n---\nExtract text from PDFs.\n"


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Wire builders — one place that knows the shape, so a contract change is one edit
# ---------------------------------------------------------------------------


def put_skill(
    key: str = "pdf-extraction",
    *,
    object_version: Any = 3,
    payload_version: int = 42,
    content: str = SKILL_BODY,
    content_hash: Any = None,
    omit_hash: bool = False,
    name: str = "PDF Extraction",
) -> dict[str, Any]:
    """One skill ``put-object`` event's data, in the shape the wire delivers it."""
    envelope: dict[str, Any] = {
        "contentType": "text/markdown",
        "content": content,
        "name": name,
        "description": "Extracts text",
    }
    if not omit_hash:
        envelope["contentHash"] = (
            content_hash if content_hash is not None else _hash(content)
        )
    return {
        "key": key,
        "kind": FDV2_OBJECT_KIND,
        "category": FDV2_OBJECT_CATEGORY,
        "objectVersion": object_version,
        "version": payload_version,
        "object": envelope,
    }


def delete_skill(
    key: str = "pdf-extraction", *, object_version: Any = 3, payload_version: int = 43
) -> dict[str, Any]:
    return {
        "key": key,
        "kind": FDV2_OBJECT_KIND,
        "category": FDV2_OBJECT_CATEGORY,
        "objectVersion": object_version,
        "version": payload_version,
    }


def put_flag(key: str = "my-flag", version: int = 17) -> dict[str, Any]:
    """A flag ``put-object``: no ``category``, no ``objectVersion``."""
    return {
        "key": key,
        "kind": "flag",
        "version": version,
        "object": {
            "key": key,
            "version": version,
            "on": True,
            "variations": [True, False],
        },
    }


def put_segment(key: str = "beta-users", version: int = 4) -> dict[str, Any]:
    return {
        "key": key,
        "kind": "segment",
        "version": version,
        "object": {"key": key, "version": version, "included": []},
    }


def server_intent(
    code: str = "xfer-full", payload_id: str = "agent-skill"
) -> dict[str, Any]:
    return {
        "payloads": [
            {"id": payload_id, "target": 1, "intentCode": code, "reason": "test"}
        ]
    }


def transferred(state: str = "basis-1", version: int = 42) -> dict[str, Any]:
    return {"state": state, "version": version}


def events(*pairs: tuple[str, Any]) -> list[dict[str, Any]]:
    return [{"event": name, "data": data} for name, data in pairs]


def full_payload(
    *object_events: tuple[str, Any], state: str = "basis-1"
) -> list[dict[str, Any]]:
    return events(
        ("server-intent", server_intent("xfer-full")),
        *object_events,
        ("payload-transferred", transferred(state)),
    )


# ---------------------------------------------------------------------------
# Identifying skill objects, and ignoring everything else
# ---------------------------------------------------------------------------


class TestObjectIdentification:
    def test_kind_and_category_together_identify_a_skill(self) -> None:
        assert _is_skill_event(put_skill()) is True

    def test_a_flag_is_not_a_skill(self) -> None:
        assert _is_skill_event(put_flag()) is False

    def test_a_segment_is_not_a_skill(self) -> None:
        assert _is_skill_event(put_segment()) is False

    def test_inline_resource_of_another_category_is_not_a_skill(self) -> None:
        """``inline-resource`` is a broad kind, so the category is required too."""
        other = put_skill()
        other["category"] = "prompt-template"
        assert _is_skill_event(other) is False

    def test_skill_category_under_another_kind_is_not_a_skill(self) -> None:
        other = put_skill()
        other["kind"] = "some-future-kind"
        assert _is_skill_event(other) is False

    def test_a_flag_shaped_object_with_no_category_is_not_a_skill(self) -> None:
        """Flags and segments omit ``category`` entirely — the documented shape."""
        assert "category" not in put_flag()
        assert "objectVersion" not in put_flag()

    @pytest.mark.parametrize("value", [None, "skill", 3, [], ()])
    def test_non_dict_events_are_not_skills(self, value: Any) -> None:
        assert _is_skill_event(value) is False


# ---------------------------------------------------------------------------
# objectVersion is not version
# ---------------------------------------------------------------------------


class TestVersionTranslation:
    def test_object_version_becomes_the_seam_version(self) -> None:
        raw = _store_object_from_put(put_skill(object_version=3, payload_version=42))
        assert raw is not None
        assert raw["version"] == 3

    def test_the_payload_version_never_reaches_the_seam(self) -> None:
        """
        The failure this asserts against is silent: a store that read ``version``
        would serve verifiable content under a version number that means nothing,
        and every pinned reference would resolve to the wrong thing with no error.
        """
        raw = _store_object_from_put(put_skill(object_version=3, payload_version=42))
        assert raw is not None
        assert raw["version"] != 42
        assert 42 not in raw.values()

    def test_the_two_are_distinguished_even_when_the_payload_version_is_lower(
        self,
    ) -> None:
        raw = _store_object_from_put(put_skill(object_version=99, payload_version=1))
        assert raw is not None
        assert raw["version"] == 99

    def test_a_missing_object_version_is_not_defaulted_from_the_payload(self) -> None:
        wire = put_skill()
        del wire["objectVersion"]
        raw = _store_object_from_put(wire)
        assert raw is not None
        assert "version" not in raw

    def test_an_explicitly_null_object_version_is_carried_through_as_null(self) -> None:
        """Carried, not invented: verification reports ``invalid_version``."""
        raw = _store_object_from_put(put_skill(object_version=None))
        assert raw is not None
        assert raw["version"] is None

    def test_a_delete_translates_object_version_too(self) -> None:
        tombstone = _tombstone_from_delete(
            delete_skill(object_version=3, payload_version=43)
        )
        assert tombstone is not None
        assert tombstone.object_version == 3

    def test_a_delete_with_no_usable_object_version_revokes_every_version(self) -> None:
        tombstone = _tombstone_from_delete(delete_skill(object_version=None))
        assert tombstone is not None
        assert tombstone.object_version is None

    def test_a_keyless_put_is_dropped_because_it_has_no_identity(self) -> None:
        wire = put_skill()
        del wire["key"]
        assert _store_object_from_put(wire) is None

    def test_the_envelope_is_copied_verbatim(self) -> None:
        raw = _store_object_from_put(put_skill())
        assert raw is not None
        assert raw["content"] == SKILL_BODY
        assert raw["contentHash"] == _hash(SKILL_BODY)
        assert raw["name"] == "PDF Extraction"
        assert raw["contentType"] == "text/markdown"

    def test_an_absent_envelope_field_is_absent_rather_than_defaulted(self) -> None:
        wire = put_skill()
        del wire["object"]["name"]
        raw = _store_object_from_put(wire)
        assert raw is not None
        assert "name" not in raw


# ---------------------------------------------------------------------------
# The protocol reader
# ---------------------------------------------------------------------------


def drive(reader: _ProtocolReader, payload_events: list[dict[str, Any]]) -> list[Any]:
    return [reader.handle(e["event"], e.get("data")) for e in payload_events]


class TestProtocolReader:
    def test_a_full_transfer_commits_at_payload_transferred(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        outcomes = drive(reader, full_payload(("put-object", put_skill())))
        assert len(held) == 1
        assert outcomes[-1].committed is True
        assert outcomes[-1].basis == "basis-1"

    def test_nothing_is_visible_before_payload_transferred(self) -> None:
        """A payload version is the unit of consistency; half of one is not a state."""
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-full")),
                ("put-object", put_skill()),
            ),
        )
        assert len(held) == 0

    def test_an_interrupted_full_transfer_leaves_last_known_good_intact(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill(object_version=1))))
        assert held.get("pdf-extraction", None) is not None

        # A second full transfer starts and never completes.
        drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-full")),
                ("put-object", put_skill(object_version=2)),
            ),
        )
        still_held = held.get("pdf-extraction", None)
        assert still_held is not None
        assert still_held["version"] == 1

    def test_a_full_transfer_replaces_rather_than_merges(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill("first"))))
        drive(
            reader, full_payload(("put-object", put_skill("second")), state="basis-2")
        )
        assert held.get("first", None) is None
        assert held.get("second", None) is not None

    def test_a_change_transfer_applies_deltas_over_what_is_held(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill("first"))))
        drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("put-object", put_skill("second")),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert held.get("first", None) is not None
        assert held.get("second", None) is not None

    def test_a_delete_object_revokes_the_skill(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill(object_version=3))))
        drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("delete-object", delete_skill(object_version=3)),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert held.get("pdf-extraction", None) is None
        assert reader.diagnostics.objects_revoked == 1

    def test_a_delete_notifies_with_a_tombstone_carrying_no_content(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill())))
        outcomes = drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("delete-object", delete_skill()),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        (change,) = outcomes[-1].changes
        assert change == {"key": "pdf-extraction", "version": 3}
        assert "content" not in change

    def test_a_delete_for_one_version_leaves_the_other_held(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(
            reader,
            full_payload(
                ("put-object", put_skill(object_version=2)),
                ("put-object", put_skill(object_version=3)),
            ),
        )
        drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("delete-object", delete_skill(object_version=3)),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert held.get("pdf-extraction", 2) is not None
        assert held.get("pdf-extraction", None)["version"] == 2

    def test_flag_and_segment_objects_are_skipped_cleanly(self) -> None:
        """
        The mixed payload is the normal case, not an edge one: an environment's
        assignment carries its flag payload alongside its agent-skill payload.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        outcomes = drive(
            reader,
            full_payload(
                ("put-object", put_flag("flag-a")),
                ("put-object", put_skill("pdf-extraction")),
                ("put-object", put_segment("beta-users")),
                ("put-object", put_flag("flag-b")),
                ("delete-object", put_flag("flag-c")),
            ),
        )
        assert len(held) == 1
        assert held.get("pdf-extraction", None) is not None
        assert reader.diagnostics.objects_ignored == 4
        assert reader.diagnostics.skill_objects_received == 1
        assert all(o.fatal is None and o.disconnect is None for o in outcomes)

    def test_an_unknown_kind_is_ignored_rather_than_fatal(self) -> None:
        """
        Erroring on an unrecognised kind would turn a normal payload into a
        permanent reconnect loop — a flag-delivery outage caused by a skills
        rollout.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        exotic = {
            "key": "x",
            "kind": "quantum-widget",
            "version": 1,
            "object": {"a": 1},
        }
        outcomes = drive(reader, full_payload(("put-object", exotic)))
        assert len(held) == 0
        assert all(o.fatal is None and o.disconnect is None for o in outcomes)

    def test_an_unknown_event_name_is_ignored(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        outcome = reader.handle("some-future-event", {"anything": True})
        assert outcome.fatal is None
        assert outcome.disconnect is None

    def test_a_heartbeat_does_nothing(self) -> None:
        reader = _ProtocolReader(_SkillObjectSet())
        outcome = reader.handle("heart-beat", None)
        assert outcome == type(outcome)()

    def test_an_error_event_abandons_the_in_flight_payload(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill(object_version=1))))
        outcomes = drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-full")),
                ("put-object", put_skill(object_version=2)),
                (
                    "error",
                    {"payloadId": "agent-skill", "reason": "backend unavailable"},
                ),
            ),
        )
        assert outcomes[-1].disconnect is not None
        assert held.get("pdf-extraction", None)["version"] == 1

    def test_a_goodbye_asks_for_a_reconnect(self) -> None:
        reader = _ProtocolReader(_SkillObjectSet())
        outcome = reader.handle("goodbye", {"reason": "rebalancing", "silent": False})
        assert outcome.disconnect is not None
        assert outcome.fatal is None

    def test_a_catastrophic_goodbye_is_fatal(self) -> None:
        reader = _ProtocolReader(_SkillObjectSet())
        outcome = reader.handle(
            "goodbye", {"reason": "no", "silent": False, "catastrophe": True}
        )
        assert outcome.fatal is not None

    def test_transfer_none_holds_everything_and_commits(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill())))
        drive(
            reader,
            events(
                ("server-intent", server_intent("none")),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert len(held) == 1

    def test_an_object_arriving_with_no_intent_is_treated_as_a_delta(self) -> None:
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(
            reader,
            events(
                ("put-object", put_skill()),
                ("payload-transferred", transferred("basis-1")),
            ),
        )
        assert len(held) == 1


# ---------------------------------------------------------------------------
# Interface parity with InMemorySkillStore
# ---------------------------------------------------------------------------


class TestInterfaceParity:
    """
    The two stores must resolve identically. ``_SkillObjectSet`` reimplements the
    lookup rather than inheriting it — see its docstring for why — so this is the
    test that stops the two from drifting.
    """

    RAWS: ClassVar[list[dict[str, Any]]] = [
        {"key": "a", "version": 1, "content": "x", "contentHash": _hash("x")},
        {"key": "a", "version": 4, "content": "y", "contentHash": _hash("y")},
        {"key": "b", "version": 2, "content": "z", "contentHash": _hash("z")},
        {"key": "malformed", "version": "not-a-version", "content": "q"},
    ]

    def _both(self) -> tuple[InMemorySkillStore, _SkillObjectSet]:
        memory = InMemorySkillStore()
        objects = _SkillObjectSet()
        for raw in self.RAWS:
            memory.put(dict(raw))
            objects.put(dict(raw))
        return memory, objects

    @pytest.mark.parametrize(
        "key,version",
        [
            ("a", None),
            ("a", 1),
            ("a", 4),
            ("a", 9),
            ("b", 2),
            ("b", None),
            ("missing", None),
            ("missing", 1),
            ("malformed", None),
            ("malformed", 7),
        ],
    )
    def test_get_agrees(self, key: str, version: int | None) -> None:
        memory, objects = self._both()
        assert memory.get_object(SKILL_OBJECT_KIND, key, version) == objects.get(
            key, version
        )

    def test_snapshot_agrees(self) -> None:
        memory, objects = self._both()
        assert memory.all_objects(SKILL_OBJECT_KIND) == objects.snapshot()


# ---------------------------------------------------------------------------
# The contentHash gap
# ---------------------------------------------------------------------------


def _per_object_hashless_errors(caplog: Any) -> list[Any]:
    """The per-object ERROR, as distinct from the whole-payload summary."""
    return [
        r
        for r in caplog.records
        if r.levelname == "ERROR"
        and "arrived without a contentHash" in r.getMessage()
        and "No skill content will resolve" not in r.getMessage()
    ]


class TestMissingContentHash:
    """
    A skill delivered without a ``contentHash``, asserted as behaviour.

    An envelope with no ``contentHash`` must produce a *withheld* skill with the
    ``missing_content_hash`` reason — loudly, diagnosably, and without a crash.
    There is deliberately no fallback that skips verification: a hash the SDK
    computed from the content it was handed would certify the content against
    itself and verify nothing.
    """

    def test_a_redelivered_hashless_object_logs_once_per_store(
        self, caplog: Any
    ) -> None:
        """Re-delivering the same ``(key, version)`` to one store must not
        multiply the ERROR: a polling store sees every object on every poll."""
        reader = _ProtocolReader(_SkillObjectSet())
        payload = full_payload(("put-object", put_skill(omit_hash=True)))
        with caplog.at_level("ERROR"):
            drive(reader, payload)
            drive(reader, payload)
        assert len(_per_object_hashless_errors(caplog)) == 1

    def test_a_recreated_store_reports_the_same_hashless_object_again(
        self, caplog: Any
    ) -> None:
        """
        The dedupe belongs to the store, not the process. A host that rebuilds
        its store (reconnect wrapper, config reload, credential rotation) must
        get the ERROR again, since it is the loudest signal that a deployment is
        broken rather than empty by design.
        """
        payload = full_payload(("put-object", put_skill(omit_hash=True)))
        with caplog.at_level("ERROR"):
            drive(_ProtocolReader(_SkillObjectSet()), payload)
            first = len(_per_object_hashless_errors(caplog))
            drive(_ProtocolReader(_SkillObjectSet()), payload)
        assert first == 1
        assert len(_per_object_hashless_errors(caplog)) == 2

    def test_two_live_stores_do_not_suppress_each_other(self, caplog: Any) -> None:
        """Two stores in one process (say, two environments) each report."""
        one = _ProtocolReader(_SkillObjectSet())
        two = _ProtocolReader(_SkillObjectSet())
        payload = full_payload(("put-object", put_skill(omit_hash=True)))
        with caplog.at_level("ERROR"):
            drive(one, payload)
            drive(two, payload)
            # And each still dedupes its own re-deliveries.
            drive(one, payload)
            drive(two, payload)
        assert len(_per_object_hashless_errors(caplog)) == 2
