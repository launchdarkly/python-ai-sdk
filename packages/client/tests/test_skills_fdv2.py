"""
Tests for the FDv2 skill delivery transport.

Two layers, deliberately:

- **A real fake endpoint.** ``_FakeFDv2Endpoint`` is an in-process
  ``ThreadingHTTPServer`` that implements the wire contract — ``basis`` and
  ``mv`` query parameters, ``Authorization``, ``If-None-Match``/304, the
  ``{"events": [...]}`` polling envelope, and SSE for streaming. The store under
  test opens real sockets against it, so request construction and header
  handling are exercised rather than mocked. This is what stands in for a live
  server while the backend work is unmerged.
- **The protocol reader driven directly.** Wire semantics — which objects are
  skills, ``objectVersion`` versus ``version``, revocation, mixed payloads — are
  asserted against ``_ProtocolReader``, which has no I/O, so those cases read as
  the contract they are instead of as a server script.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

import pytest

from launchdarkly_ai_server import (
    FDv2SkillStore,
    InMemorySkillStore,
    all_skills,
    get_skill,
    get_skill_result,
    init_client,
    watch_skills,
)
from launchdarkly_ai_server.skills_core import SKILL_OBJECT_KIND
from launchdarkly_ai_server.skills_fdv2 import (
    FDV2_OBJECT_CATEGORY,
    FDV2_OBJECT_KIND,
    _ProtocolReader,
    _RecoverableTransportError,
    _SkillObjectSet,
    backoff_delay,
    is_skill_event,
    seam_object_from_put,
    tombstone_from_delete,
)
from launchdarkly_ai_server.skills_fdv2 import _warned_hashless as _hashless_dedupe

pytestmark = pytest.mark.usefixtures("reset_skill_state")

SDK_KEY = "sdk-00000000-0000-4000-8000-000000000000"
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
    """One skill ``put-object`` event's data, exactly as §2.3 specifies it."""
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
# The fake endpoint
# ---------------------------------------------------------------------------


class _FakeFDv2Endpoint:
    """
    An in-process server implementing the SDK-facing FDv2 contract.

    Scripted per request: ``queue_poll`` appends a response for the next
    ``/sdk/poll``, ``queue_stream`` appends a sequence of SSE events for the next
    ``/sdk/stream``. Every request's method, path, query and headers are recorded
    in ``requests`` so the tests can assert on what the store actually sent —
    which is the only way ``basis`` round-tripping and ``If-None-Match`` can be
    checked at all.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._polls: list[dict[str, Any]] = []
        self._streams: list[list[dict[str, Any]]] = []
        self._lock = threading.Lock()
        self.hold_stream_open = False
        self._release = threading.Event()

        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                with endpoint._lock:
                    endpoint.requests.append(
                        {
                            "path": parsed.path,
                            "query": query,
                            "authorization": self.headers.get("Authorization"),
                            "if_none_match": self.headers.get("If-None-Match"),
                            "accept": self.headers.get("Accept"),
                        }
                    )
                if parsed.path == "/sdk/poll":
                    endpoint._serve_poll(self)
                elif parsed.path == "/sdk/stream":
                    endpoint._serve_stream(self)
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        class Server(ThreadingHTTPServer):
            # Handler threads are not joined on shutdown: a test that ends while
            # a stream is deliberately held open should not pay for the hold.
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        # A short poll interval so `shutdown` is prompt: the default 0.5s is
        # paid at the teardown of every test that touches the endpoint.
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.01), daemon=True
        )
        self._thread.start()

    @property
    def base_uri(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    # -- scripting ---------------------------------------------------------

    def queue_poll(
        self,
        payload_events: list[dict[str, Any]] | None = None,
        *,
        status: int = 200,
        etag: str | None = None,
        retry_after: str | None = None,
    ) -> None:
        with self._lock:
            self._polls.append(
                {
                    "status": status,
                    "events": payload_events or [],
                    "etag": etag,
                    "retry_after": retry_after,
                }
            )

    def queue_stream(self, payload_events: list[dict[str, Any]]) -> None:
        with self._lock:
            self._streams.append(payload_events)

    # -- serving -----------------------------------------------------------

    def _serve_poll(self, handler: BaseHTTPRequestHandler) -> None:
        with self._lock:
            response = (
                self._polls.pop(0) if self._polls else {"status": 304, "events": []}
            )
        status = response["status"]
        handler.send_response(status)
        if response.get("etag"):
            handler.send_header("ETag", response["etag"])
        if response.get("retry_after"):
            handler.send_header("Retry-After", response["retry_after"])
        if status in (200,):
            body = json.dumps({"events": response["events"]}).encode("utf-8")
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def _serve_stream(self, handler: BaseHTTPRequestHandler) -> None:
        with self._lock:
            payload_events = self._streams.pop(0) if self._streams else []
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for event in payload_events:
            chunk = (
                f"event: {event['event']}\ndata: {json.dumps(event.get('data'))}\n\n"
            ).encode()
            handler.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            handler.wfile.flush()
        if self.hold_stream_open:
            # Keeps the connection up so a test can assert on the store's state
            # without racing the reconnect path. Released on ``close`` so the
            # hold costs the suite nothing once the test is done with it.
            self._release.wait(timeout=10)
        handler.wfile.write(b"0\r\n\r\n")

    def close(self) -> None:
        self._release.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def endpoint() -> Any:
    server = _FakeFDv2Endpoint()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _clear_hashless_dedupe() -> Any:
    """The hashless-object ERROR is deduped per process; per test here."""
    _hashless_dedupe.clear()
    yield
    _hashless_dedupe.clear()


def poll_store(endpoint: Any, **kwargs: Any) -> FDv2SkillStore:
    return FDv2SkillStore(
        SDK_KEY,
        base_uri=endpoint.base_uri,
        mode="poll",
        poll_interval=kwargs.pop("poll_interval", 0.05),
        initial_backoff=kwargs.pop("initial_backoff", 0.01),
        max_backoff=kwargs.pop("max_backoff", 0.05),
        read_timeout=kwargs.pop("read_timeout", 5.0),
        **kwargs,
    )


def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# Identifying skill objects, and ignoring everything else
# ---------------------------------------------------------------------------


class TestObjectIdentification:
    def test_kind_and_category_together_identify_a_skill(self) -> None:
        assert is_skill_event(put_skill()) is True

    def test_a_flag_is_not_a_skill(self) -> None:
        assert is_skill_event(put_flag()) is False

    def test_a_segment_is_not_a_skill(self) -> None:
        assert is_skill_event(put_segment()) is False

    def test_inline_resource_of_another_category_is_not_a_skill(self) -> None:
        """``inline-resource`` is a broad kind; the category is load-bearing."""
        other = put_skill()
        other["category"] = "prompt-template"
        assert is_skill_event(other) is False

    def test_skill_category_under_another_kind_is_not_a_skill(self) -> None:
        other = put_skill()
        other["kind"] = "some-future-kind"
        assert is_skill_event(other) is False

    def test_a_flag_shaped_object_with_no_category_is_not_a_skill(self) -> None:
        """Flags and segments omit ``category`` entirely — the documented shape."""
        assert "category" not in put_flag()
        assert "objectVersion" not in put_flag()

    @pytest.mark.parametrize("value", [None, "skill", 3, [], ()])
    def test_non_dict_events_are_not_skills(self, value: Any) -> None:
        assert is_skill_event(value) is False


# ---------------------------------------------------------------------------
# objectVersion is not version. This is the whole ballgame.
# ---------------------------------------------------------------------------


class TestVersionTranslation:
    def test_object_version_becomes_the_seam_version(self) -> None:
        raw = seam_object_from_put(put_skill(object_version=3, payload_version=42))
        assert raw is not None
        assert raw["version"] == 3

    def test_the_payload_version_never_reaches_the_seam(self) -> None:
        """
        The failure this asserts against is silent: a store that read ``version``
        would serve verifiable content under a version number that means nothing,
        and every pinned reference would resolve to the wrong thing with no error.
        """
        raw = seam_object_from_put(put_skill(object_version=3, payload_version=42))
        assert raw is not None
        assert raw["version"] != 42
        assert 42 not in raw.values()

    def test_the_two_are_distinguished_even_when_the_payload_version_is_lower(
        self,
    ) -> None:
        raw = seam_object_from_put(put_skill(object_version=99, payload_version=1))
        assert raw is not None
        assert raw["version"] == 99

    def test_a_missing_object_version_is_not_defaulted_from_the_payload(self) -> None:
        wire = put_skill()
        del wire["objectVersion"]
        raw = seam_object_from_put(wire)
        assert raw is not None
        assert "version" not in raw

    def test_an_explicitly_null_object_version_is_carried_through_as_null(self) -> None:
        """Carried, not invented: verification reports ``invalid_version``."""
        raw = seam_object_from_put(put_skill(object_version=None))
        assert raw is not None
        assert raw["version"] is None

    def test_a_delete_translates_object_version_too(self) -> None:
        tombstone = tombstone_from_delete(
            delete_skill(object_version=3, payload_version=43)
        )
        assert tombstone is not None
        assert tombstone.object_version == 3

    def test_a_delete_with_no_usable_object_version_revokes_every_version(self) -> None:
        tombstone = tombstone_from_delete(delete_skill(object_version=None))
        assert tombstone is not None
        assert tombstone.object_version is None

    def test_a_keyless_put_is_dropped_because_it_has_no_identity(self) -> None:
        wire = put_skill()
        del wire["key"]
        assert seam_object_from_put(wire) is None

    def test_the_envelope_is_copied_verbatim(self) -> None:
        raw = seam_object_from_put(put_skill())
        assert raw is not None
        assert raw["content"] == SKILL_BODY
        assert raw["contentHash"] == _hash(SKILL_BODY)
        assert raw["name"] == "PDF Extraction"
        assert raw["contentType"] == "text/markdown"

    def test_an_absent_envelope_field_is_absent_rather_than_defaulted(self) -> None:
        wire = put_skill()
        del wire["object"]["name"]
        raw = seam_object_from_put(wire)
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
        assignment carries the flagging payload alongside the agent-skill payload.
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
        Erroring here is the unknown-kind reconnect loop this feature must not
        reproduce — a flag-delivery outage caused by a skills rollout.
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
# Seam parity with InMemorySkillStore
# ---------------------------------------------------------------------------


class TestSeamParity:
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
# The store against the fake endpoint
# ---------------------------------------------------------------------------


class TestPollingAgainstTheEndpoint:
    def test_a_polled_skill_becomes_retrievable_through_the_accessors(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            raw = store.get_object(SKILL_OBJECT_KIND, "pdf-extraction")
            assert raw is not None
            assert raw["version"] == 3

    def test_the_request_carries_the_sdk_key_and_the_data_model_version(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
        first = endpoint.requests[0]
        assert first["path"] == "/sdk/poll"
        assert first["authorization"] == SDK_KEY
        assert first["query"]["mv"] == "1"

    def test_the_first_request_sends_no_basis(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
        assert "basis" not in endpoint.requests[0]["query"]

    def test_the_basis_from_payload_transferred_is_echoed_on_the_next_request(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(
            full_payload(("put-object", put_skill()), state="selector-abc")
        )
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            assert wait_until(lambda: len(endpoint.requests) >= 2)
        assert endpoint.requests[1]["query"]["basis"] == "selector-abc"

    def test_the_basis_advances_across_successive_payloads(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill()), state="basis-1"))
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("put-object", put_skill("second")),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        endpoint.queue_poll(status=304)
        with poll_store(endpoint):
            assert wait_until(lambda: len(endpoint.requests) >= 3)
        bases = [r["query"].get("basis") for r in endpoint.requests[:3]]
        assert bases == [None, "basis-1", "basis-2"]

    def test_an_etag_is_returned_as_if_none_match(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())), etag='W/"v1"')
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            assert wait_until(lambda: len(endpoint.requests) >= 2)
        assert endpoint.requests[1]["if_none_match"] == 'W/"v1"'

    def test_a_304_keeps_the_held_content(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())), etag='W/"v1"')
        endpoint.queue_poll(status=304)
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            assert wait_until(lambda: len(endpoint.requests) >= 3)
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
            assert store.diagnostics.payloads_transferred == 1
            assert store.failed is None

    def test_a_304_before_any_payload_still_releases_wait_for_skills(
        self, endpoint: Any
    ) -> None:
        """A reconnect with a cached basis has nothing to transfer; boot must not
        block on a payload the server has no reason to send."""
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True

    def test_a_mixed_payload_over_the_wire_yields_only_the_skill(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_flag("flag-a")),
                ("put-object", put_segment("beta")),
                ("put-object", put_skill("pdf-extraction")),
                ("put-object", put_flag("flag-b")),
            )
        )
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            held = store.all_objects(SKILL_OBJECT_KIND)
            assert len(held) == 1
            assert next(iter(held.values()))["key"] == "pdf-extraction"
            assert store.diagnostics.objects_ignored == 3

    def test_a_revocation_over_the_wire_removes_the_skill(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("delete-object", delete_skill()),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            assert wait_until(
                lambda: (
                    store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is None
                    and store.diagnostics.objects_revoked == 1
                )
            )

    def test_the_store_asks_for_only_the_kind_it_serves(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            assert store.get_object("flag", "pdf-extraction") is None
            assert store.all_objects("flag") == {}


class TestStreamingAgainstTheEndpoint:
    def test_a_streamed_payload_lands(self, endpoint: Any) -> None:
        endpoint.hold_stream_open = True
        endpoint.queue_stream(full_payload(("put-object", put_skill())))
        store = FDv2SkillStore(
            SDK_KEY, base_uri=endpoint.base_uri, mode="stream", initial_backoff=0.01
        )
        try:
            store.start()
            assert store.wait_for_skills(timeout=5) is True
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
        finally:
            store.close()

    def test_the_stream_request_advertises_event_stream(self, endpoint: Any) -> None:
        endpoint.hold_stream_open = True
        endpoint.queue_stream(full_payload(("put-object", put_skill())))
        store = FDv2SkillStore(SDK_KEY, base_uri=endpoint.base_uri, mode="stream")
        try:
            store.start()
            store.wait_for_skills(timeout=5)
        finally:
            store.close()
        assert endpoint.requests[0]["path"] == "/sdk/stream"
        assert endpoint.requests[0]["accept"] == "text/event-stream"

    def test_a_streamed_revocation_arrives_without_a_restart(
        self, endpoint: Any
    ) -> None:
        endpoint.hold_stream_open = True
        endpoint.queue_stream(
            full_payload(("put-object", put_skill()))
            + events(
                ("server-intent", server_intent("xfer-changes")),
                ("delete-object", delete_skill()),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        store = FDv2SkillStore(SDK_KEY, base_uri=endpoint.base_uri, mode="stream")
        try:
            store.start()
            assert wait_until(
                lambda: (
                    store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is None
                    and store.diagnostics.objects_revoked == 1
                )
            )
        finally:
            store.close()

    def test_a_dropped_stream_reconnects_with_the_basis_it_reached(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_stream(
            full_payload(("put-object", put_skill()), state="basis-1")
        )
        endpoint.queue_stream(events(("heart-beat", None)))
        store = FDv2SkillStore(
            SDK_KEY,
            base_uri=endpoint.base_uri,
            mode="stream",
            initial_backoff=0.01,
            max_backoff=0.05,
        )
        try:
            store.start()
            assert wait_until(lambda: len(endpoint.requests) >= 2)
        finally:
            store.close()
        assert endpoint.requests[1]["query"]["basis"] == "basis-1"

    def test_close_returns_promptly_while_a_stream_is_open(self, endpoint: Any) -> None:
        """
        The delivery thread is blocked in a socket read that no stop flag can
        reach, so ``close`` closes the connection under it. Without that, every
        shutdown of a healthy stream waits out the join timeout.
        """
        endpoint.hold_stream_open = True
        endpoint.queue_stream(full_payload(("put-object", put_skill())))
        store = FDv2SkillStore(SDK_KEY, base_uri=endpoint.base_uri, mode="stream")
        store.start()
        assert store.wait_for_skills(timeout=5) is True
        started = time.monotonic()
        store.close(timeout=5.0)
        assert time.monotonic() - started < 1.0

    def test_an_interrupted_stream_is_not_reported_as_a_failure(
        self, endpoint: Any
    ) -> None:
        endpoint.hold_stream_open = True
        endpoint.queue_stream(full_payload(("put-object", put_skill())))
        store = FDv2SkillStore(SDK_KEY, base_uri=endpoint.base_uri, mode="stream")
        store.start()
        store.wait_for_skills(timeout=5)
        store.close()
        assert store.failed is None

    def test_content_survives_a_reconnect(self, endpoint: Any) -> None:
        endpoint.queue_stream(full_payload(("put-object", put_skill())))
        endpoint.queue_stream(events(("heart-beat", None)))
        store = FDv2SkillStore(
            SDK_KEY, base_uri=endpoint.base_uri, mode="stream", initial_backoff=0.01
        )
        try:
            store.start()
            assert wait_until(lambda: len(endpoint.requests) >= 2)
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class _ScriptedConnection:
    """Stands in for ``_StreamConnection``: an event iterator plus a close."""

    def __init__(self, payload_events: Any) -> None:
        self.events = iter(payload_events)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ScriptedRequester:
    """Raises a scripted sequence, so backoff is asserted without real sockets."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str | None, str | None]] = []

    def poll(self, basis: str | None, etag: str | None) -> Any:
        self.calls.append((basis, etag))
        outcome = (
            self.outcomes.pop(0) if self.outcomes else _RecoverableTransportError("x")
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def stream(self, basis: str | None) -> Any:
        self.calls.append((basis, None))
        outcome = (
            self.outcomes.pop(0) if self.outcomes else _RecoverableTransportError("x")
        )
        if isinstance(outcome, Exception):
            raise outcome
        return _ScriptedConnection(outcome)


class TestFailureHandling:
    def test_a_403_stops_delivery_and_names_the_protocol_control_flag(
        self, endpoint: Any, caplog: Any
    ) -> None:
        endpoint.queue_poll(status=403)
        with caplog.at_level("ERROR"):
            with poll_store(endpoint) as store:
                assert wait_until(lambda: store.failed is not None)
        assert "403" in store.failed
        assert "fdv2-protocol-control" in store.failed
        assert any("fdv2-protocol-control" in r.getMessage() for r in caplog.records)

    def test_a_401_stops_delivery(self, endpoint: Any) -> None:
        endpoint.queue_poll(status=401)
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
        assert "401" in store.failed

    def test_a_fatal_failure_releases_wait_for_skills_rather_than_hanging(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(status=401)
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert store.failed is not None

    def test_a_fatal_failure_keeps_last_known_good_servable(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=403)
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None

    def test_a_500_is_retried(self, endpoint: Any) -> None:
        endpoint.queue_poll(status=500)
        endpoint.queue_poll(status=503)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert store.failed is None
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None

    def test_a_retry_resets_the_failure_count_on_success(self, endpoint: Any) -> None:
        endpoint.queue_poll(status=500)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5)
            assert wait_until(lambda: store.diagnostics.connection_failures == 0)

    def test_retries_are_bounded(self) -> None:
        store = FDv2SkillStore(
            SDK_KEY,
            mode="poll",
            poll_interval=0.01,
            initial_backoff=0.001,
            max_backoff=0.002,
            max_consecutive_failures=3,
            _requester=_ScriptedRequester(),
        )
        try:
            store.start()
            assert wait_until(lambda: store.failed is not None)
            assert "3 consecutive failures" in store.failed or "gave up" in store.failed
        finally:
            store.close()

    def test_a_retry_after_header_is_honoured(self) -> None:
        requester = _ScriptedRequester(
            _RecoverableTransportError("slow down", retry_after=0.25),
        )
        store = FDv2SkillStore(
            SDK_KEY,
            mode="poll",
            poll_interval=10.0,
            initial_backoff=5.0,
            _requester=requester,
        )
        try:
            started = time.monotonic()
            store.start()
            assert wait_until(lambda: len(requester.calls) >= 2, timeout=3)
            elapsed = time.monotonic() - started
            # The server asked for 0.25s; our own backoff would have been 5s.
            assert 0.2 <= elapsed < 3.0
        finally:
            store.close()

    def test_a_retry_after_header_is_parsed_off_the_wire(self, endpoint: Any) -> None:
        endpoint.queue_poll(status=429, retry_after="0")
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint, initial_backoff=5.0) as store:
            # If Retry-After were ignored the 5s backoff would blow the timeout.
            assert store.wait_for_skills(timeout=3) is True

    def test_backoff_is_exponential_and_capped(self) -> None:
        assert backoff_delay(1, base=1.0, maximum=30.0, jitter=0.0) == 1.0
        assert backoff_delay(2, base=1.0, maximum=30.0, jitter=0.0) == 2.0
        assert backoff_delay(3, base=1.0, maximum=30.0, jitter=0.0) == 4.0
        assert backoff_delay(20, base=1.0, maximum=30.0, jitter=0.0) == 30.0

    def test_jitter_never_exceeds_the_cap(self) -> None:
        for attempt in range(1, 12):
            for _ in range(50):
                assert 0.0 <= backoff_delay(attempt, base=1.0, maximum=5.0) <= 5.0

    def test_a_malformed_polling_envelope_is_recoverable_not_fatal(
        self, endpoint: Any
    ) -> None:
        store = FDv2SkillStore(
            SDK_KEY,
            mode="poll",
            poll_interval=0.01,
            initial_backoff=0.001,
            _requester=_ScriptedRequester(
                _RecoverableTransportError("polling response had no 'events' array")
            ),
        )
        try:
            store.start()
            assert wait_until(lambda: store.diagnostics.connection_failures >= 1)
            assert store.failed is None
        finally:
            store.close()

    def test_a_listener_that_raises_does_not_kill_delivery(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill("first"))))
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("put-object", put_skill("second")),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            store.add_listener(SKILL_OBJECT_KIND, lambda _raw: 1 / 0)
            assert wait_until(
                lambda: store.get_object(SKILL_OBJECT_KIND, "second") is not None
            )
            assert store.failed is None


# ---------------------------------------------------------------------------
# The contentHash gap
# ---------------------------------------------------------------------------


class TestMissingContentHash:
    """
    The blocking backend gap, asserted as behaviour rather than assumed.

    An envelope with no ``contentHash`` must produce a *withheld* skill with the
    ``missing_content_hash`` reason — loudly, diagnosably, and without a crash.
    There is deliberately no fallback that skips verification: a hash the SDK
    computed from the content it was handed would certify the content against
    itself and verify nothing.
    """

    async def test_a_hashless_skill_is_withheld_with_the_right_reason(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill(omit_hash=True))))
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())

            outcome = await get_skill_result("pdf-extraction")
            assert outcome.skill is None
            assert outcome.reason == "integrity_failure"
            assert await get_skill("pdf-extraction") is None
            assert await all_skills() == []

    async def test_the_object_is_still_held_so_the_outcome_is_not_absent(
        self, endpoint: Any
    ) -> None:
        """
        Holding it is what makes the failure diagnosable. Dropping it at the
        transport would report ``absent`` — indistinguishable from "no such
        skill" — and would additionally let a prune delete the last known-good
        copy already on disk.
        """
        endpoint.queue_poll(full_payload(("put-object", put_skill(omit_hash=True))))
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            raw = store.get_object(SKILL_OBJECT_KIND, "pdf-extraction")
            assert raw is not None
            assert "contentHash" not in raw
            await init_client(options={"skillStore": store}, client=object())
            assert (await get_skill_result("pdf-extraction")).reason != "absent"

    def test_the_store_counts_hashless_objects(self, endpoint: Any) -> None:
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill("a", omit_hash=True)),
                ("put-object", put_skill("b", omit_hash=True)),
                ("put-object", put_skill("c")),
            )
        )
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            assert store.diagnostics.hashless_objects == 2
            assert store.diagnostics.skill_objects_received == 3

    def test_a_hashless_object_logs_an_error_naming_the_reason_code(
        self, endpoint: Any, caplog: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill(omit_hash=True))))
        with caplog.at_level("ERROR"):
            with poll_store(endpoint) as store:
                store.wait_for_skills(timeout=5)
        rendered = "\n".join(r.getMessage() for r in caplog.records)
        assert "missing_content_hash" in rendered
        assert "pdf-extraction" in rendered
        assert "contentHash" in rendered

    def test_a_wholly_hashless_payload_says_so_once(
        self, endpoint: Any, caplog: Any
    ) -> None:
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill("a", omit_hash=True)),
                ("put-object", put_skill("b", omit_hash=True)),
            )
        )
        with caplog.at_level("ERROR"):
            with poll_store(endpoint) as store:
                store.wait_for_skills(timeout=5)
        summaries = [
            r
            for r in caplog.records
            if "No skill content will resolve" in r.getMessage()
        ]
        assert len(summaries) == 1
        assert "All 2 skill object(s)" in summaries[0].getMessage()

    def test_a_partly_hashed_payload_does_not_claim_total_failure(
        self, endpoint: Any, caplog: Any
    ) -> None:
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill("a", omit_hash=True)),
                ("put-object", put_skill("b")),
            )
        )
        with caplog.at_level("ERROR"):
            with poll_store(endpoint) as store:
                store.wait_for_skills(timeout=5)
        rendered = "\n".join(r.getMessage() for r in caplog.records)
        assert "No skill content will resolve" not in rendered

    async def test_a_hash_that_does_not_match_is_a_different_failure(
        self, endpoint: Any
    ) -> None:
        """``missing_content_hash`` and ``hash_mismatch`` must not collapse: one is
        a backend gap and the other is possible tampering."""
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill(content_hash=_hash("something else")))
            )
        )
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            assert store.diagnostics.hashless_objects == 0
            await init_client(options={"skillStore": store}, client=object())
            assert (
                await get_skill_result("pdf-extraction")
            ).reason == "integrity_failure"

    async def test_a_hashed_skill_resolves_end_to_end(self, endpoint: Any) -> None:
        """The positive control: everything above is a gap, not a broken adapter."""
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())

            skill = await get_skill("pdf-extraction")
            assert skill is not None
            assert skill.key == "pdf-extraction"
            assert skill.version == 3
            assert skill.content == SKILL_BODY.encode("utf-8")
            assert skill.content_hash == _hash(SKILL_BODY)
            assert skill.name == "PDF Extraction"

    async def test_a_pinned_reference_resolves_to_the_pinned_object_version(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill(object_version=2, content="v2 body")),
                ("put-object", put_skill(object_version=5, content="v5 body")),
            )
        )
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())

            pinned = await get_skill("pdf-extraction", version=2)
            assert pinned is not None
            assert pinned.content == b"v2 body"
            newest = await get_skill("pdf-extraction")
            assert newest is not None
            assert newest.version == 5

    async def test_the_payload_version_is_not_resolvable_as_a_skill_version(
        self, endpoint: Any
    ) -> None:
        """
        The end-to-end form of the ``objectVersion``/``version`` assertion.

        Asking for the payload version resolves nothing — reported ``absent``,
        because the store answers "I hold no such version" rather than answering
        with the wrong one. The version that *does* resolve is ``objectVersion``.
        """
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill(object_version=3, payload_version=42))
            )
        )
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())
            by_payload_version = await get_skill_result("pdf-extraction", version=42)
            assert by_payload_version.skill is None
            assert by_payload_version.reason == "absent"
            assert await get_skill("pdf-extraction", version=3) is not None


# ---------------------------------------------------------------------------
# Server-side only
# ---------------------------------------------------------------------------


class TestServerSideOnly:
    def test_a_mobile_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mobile key"):
            FDv2SkillStore("mob-00000000-0000-4000-8000-000000000000")

    def test_a_client_side_environment_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="client-side"):
            FDv2SkillStore("0123456789abcdef01234567")

    def test_an_empty_credential_is_refused(self) -> None:
        with pytest.raises(ValueError, match="server-side SDK key"):
            FDv2SkillStore("   ")

    def test_a_server_side_key_is_accepted(self) -> None:
        assert FDv2SkillStore(SDK_KEY) is not None

    def test_an_unrecognised_credential_shape_warns_but_is_allowed(
        self, caplog: Any
    ) -> None:
        """Private instances and test doubles issue keys without the public prefix."""
        with caplog.at_level("WARNING"):
            FDv2SkillStore("my-private-instance-credential")
        assert any("server-side SDK key" in r.message for r in caplog.records)

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="stream"):
            FDv2SkillStore(SDK_KEY, mode="mobile")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The eager re-reconcile
# ---------------------------------------------------------------------------


class TestWatchSkills:
    async def test_a_revocation_prunes_without_a_restart(
        self, endpoint: Any, tmp_path: Any
    ) -> None:
        """
        AV-1, closed at this layer. The store's change listener drives the
        reconcile, so the file goes away seconds after the ``delete-object``
        rather than at the next process start.
        """
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("delete-object", delete_skill()),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        endpoint.queue_poll(status=304)

        with poll_store(endpoint, poll_interval=0.2) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())
            report, watcher = await watch_skills(
                "*", tmp_path / "skills", debounce=0.05
            )
            try:
                written = tmp_path / "skills" / "pdf-extraction" / "SKILL.md"
                assert written.exists()
                assert any(a.action == "written" for a in report.actions)
                assert wait_until(lambda: not written.exists(), timeout=10)
            finally:
                watcher.close()

    async def test_a_new_version_is_rewritten_without_a_restart(
        self, endpoint: Any, tmp_path: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill(content="first"))))
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-full")),
                ("put-object", put_skill(object_version=4, content="second")),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        endpoint.queue_poll(status=304)

        with poll_store(endpoint, poll_interval=0.2) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())
            _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
            try:
                written = tmp_path / "s" / "pdf-extraction" / "SKILL.md"
                assert written.read_text() == "first"
                assert wait_until(lambda: written.read_text() == "second", timeout=10)
            finally:
                watcher.close()

    async def test_a_burst_of_changes_coalesces_into_few_reconciles(
        self, endpoint: Any, tmp_path: Any
    ) -> None:
        endpoint.queue_poll(
            full_payload(*[("put-object", put_skill(f"skill-{i}")) for i in range(12)])
        )
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())
            _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.1)
            try:
                time.sleep(0.5)
                # Twelve objects committed in one payload fire twelve listener
                # calls; without coalescing that is twelve reconciles of one root.
                assert watcher.reconciles <= 2
            finally:
                watcher.close()

    async def test_the_default_keeps_last_known_good_during_an_outage(
        self, endpoint: Any, tmp_path: Any
    ) -> None:
        """``on_unavailable="keep"`` is the endorsed default: an outage must not
        read as "everything was revoked"."""
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=500)
        with poll_store(endpoint, poll_interval=0.05) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())
            _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
            try:
                written = tmp_path / "s" / "pdf-extraction" / "SKILL.md"
                assert written.exists()
                # ``last_error`` rather than ``connection_failures``: the counter
                # resets on the next successful poll, so asserting on it races
                # the retry that is supposed to happen.
                assert wait_until(
                    lambda: store.diagnostics.last_error is not None, timeout=10
                )
                time.sleep(0.3)
                assert written.exists()
            finally:
                watcher.close()

    async def test_a_store_with_no_listener_support_is_refused_loudly(
        self, tmp_path: Any
    ) -> None:
        class NoListeners:
            def get_object(self, *_a: Any, **_k: Any) -> None:
                return None

            def all_objects(self, _kind: str) -> dict[str, Any]:
                return {}

        await init_client(options={"skillStore": NoListeners()}, client=object())
        with pytest.raises(RuntimeError, match="add_listener"):
            await watch_skills("*", tmp_path / "s")

    async def test_no_store_configured_raises(self, tmp_path: Any) -> None:
        with pytest.raises(RuntimeError, match="configured skill store"):
            await watch_skills("*", tmp_path / "s")

    async def test_the_in_memory_store_can_also_drive_a_watch(
        self, tmp_path: Any
    ) -> None:
        """The watcher is wired to the seam, not to the FDv2 store."""
        store = InMemorySkillStore()
        store.put(
            {
                "key": "a",
                "version": 1,
                "content": "body",
                "contentHash": _hash("body"),
            }
        )
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
        try:
            written = tmp_path / "s" / "a" / "SKILL.md"
            assert written.read_text() == "body"
            store.put(
                {
                    "key": "a",
                    "version": 2,
                    "content": "new body",
                    "contentHash": _hash("new body"),
                }
            )
            assert wait_until(lambda: written.read_text() == "new body", timeout=10)
        finally:
            watcher.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_is_idempotent(self, endpoint: Any) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        store = poll_store(endpoint)
        try:
            assert store.start() is store
            assert store.start() is store
            assert store.wait_for_skills(timeout=5)
        finally:
            store.close()

    def test_close_is_idempotent(self, endpoint: Any) -> None:
        store = poll_store(endpoint)
        store.start()
        store.close()
        store.close()

    def test_a_closed_store_still_answers_from_what_it_received(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        store = poll_store(endpoint)
        store.start()
        store.wait_for_skills(timeout=5)
        store.close()
        assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None

    def test_wait_for_skills_times_out_rather_than_hanging(self) -> None:
        store = FDv2SkillStore(
            SDK_KEY, mode="poll", poll_interval=60, _requester=_ScriptedRequester()
        )
        try:
            assert store.wait_for_skills(timeout=0.05) is False
        finally:
            store.close()

    def test_the_store_satisfies_the_seam_before_it_starts(self) -> None:
        store = FDv2SkillStore(SDK_KEY)
        assert store.get_object(SKILL_OBJECT_KIND, "anything") is None
        assert store.all_objects(SKILL_OBJECT_KIND) == {}
