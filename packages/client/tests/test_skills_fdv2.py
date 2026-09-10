"""
Tests for the FDv2 skill delivery transport.

Two layers, deliberately:

- **A real fake endpoint.** ``_FakeFDv2Endpoint`` is an in-process
  ``ThreadingHTTPServer`` that implements the wire contract — ``basis`` and
  ``mv`` query parameters, ``Authorization``, ``If-None-Match``/304, the
  ``{"events": [...]}`` polling envelope, and SSE for streaming. The store under
  test opens real sockets against it, so request construction and header
  handling are exercised rather than mocked.
- **The protocol reader driven directly.** Wire semantics — which objects are
  skills, ``objectVersion`` versus ``version``, revocation, mixed payloads — are
  asserted against ``_ProtocolReader``, which has no I/O, so those cases read as
  the contract they are instead of as a server script.
"""

from __future__ import annotations

import hashlib
import json
import socket
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
    DEFAULT_POLL_TIMEOUT,
    DEFAULT_STREAM_READ_TIMEOUT,
    FDV2_OBJECT_CATEGORY,
    FDV2_OBJECT_KIND,
    _ProtocolReader,
    _RecoverableTransportError,
    _Requester,
    _retry_after_seconds,
    _SkillObjectSet,
    backoff_delay,
    is_skill_event,
    seam_object_from_put,
    tombstone_from_delete,
)

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
        """``inline-resource`` is a broad kind, so the category is required too."""
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
# objectVersion is not version
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


class _RecyclingRequester:
    """
    A healthy server that recycles connections: every ``stream`` call succeeds,
    transfers a full payload, and then ends the connection, as LaunchDarkly and
    any proxy in between do to a long-lived stream.
    """

    def __init__(self) -> None:
        self.connections = 0

    def stream(self, basis: str | None) -> Any:
        self.connections += 1
        return _ScriptedConnection(
            [
                (e["event"], e["data"])
                for e in full_payload(
                    ("put-object", put_skill()), state=f"basis-{self.connections}"
                )
            ]
        )


class _BlockingConnection:
    """A stream that never produces an event until it is closed."""

    def __init__(self) -> None:
        self._closed = threading.Event()

    @property
    def events(self) -> Any:
        self._closed.wait()
        return iter(())

    def close(self) -> None:
        self._closed.set()


class _SlowConnectRequester:
    """
    A ``stream`` whose connect does not return until the test releases it,
    standing in for a slow TLS handshake, followed by a read that never yields.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def stream(self, basis: str | None) -> Any:
        self.entered.set()
        self.release.wait(timeout=10)
        return _BlockingConnection()


def stream_store(**kwargs: Any) -> FDv2SkillStore:
    return FDv2SkillStore(
        SDK_KEY,
        mode="stream",
        initial_backoff=kwargs.pop("initial_backoff", 0.001),
        max_backoff=kwargs.pop("max_backoff", 0.002),
        **kwargs,
    )


class TestFailureHandling:
    def test_a_403_stops_delivery_and_explains_why(
        self, endpoint: Any, caplog: Any
    ) -> None:
        endpoint.queue_poll(status=403)
        with caplog.at_level("ERROR"):
            with poll_store(endpoint) as store:
                assert wait_until(lambda: store.failed is not None)
        assert "403" in store.failed
        assert "opt-in" in store.failed
        assert any("opt-in" in r.getMessage() for r in caplog.records)

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
            # Four, not three: the bound is the number of failures *tolerated*,
            # so the run that exceeds it is the one that gives up.
            assert "gave up after 4 consecutive failures" in store.failed
        finally:
            store.close()

    def test_recycled_stream_connections_are_not_failures(self) -> None:
        # A streaming connection only ever ends by being dropped, so a loop
        # that counted every drop as a failure would give up on a healthy
        # server after max_consecutive_failures + 1 recycles, and delivery
        # (including revocation) would silently stop for the process lifetime.
        requester = _RecyclingRequester()
        store = stream_store(max_consecutive_failures=3, _requester=requester)
        try:
            store.start()
            assert wait_until(lambda: requester.connections >= 8)
            assert store.failed is None
            assert store.diagnostics.payloads_transferred >= 8
            # A drop is a failure until the next commit clears it, so the count
            # may read 1 mid-reconnect. What it must never do is climb.
            assert store.diagnostics.connection_failures <= 1
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
        finally:
            store.close()

    def test_a_stream_commit_resets_the_failure_count(self) -> None:
        payload = [
            (e["event"], e["data"]) for e in full_payload(("put-object", put_skill()))
        ]
        requester = _ScriptedRequester(
            _RecoverableTransportError("x"),
            _RecoverableTransportError("x"),
            _RecoverableTransportError("x"),
            payload,
        )
        store = stream_store(max_consecutive_failures=3, _requester=requester)
        try:
            store.start()
            assert store.wait_for_skills(timeout=5)
            # Three failures reach the bound, then a commit, then the exhausted
            # requester fails on every reconnect. The count must start again at
            # the commit: the stream's own drop is failure one, and three more
            # connects are owed before giving up. Carrying the three over would
            # give up on the drop itself, with no further connect at all.
            assert wait_until(lambda: store.failed is not None)
            assert "gave up after 4 consecutive failures" in store.failed
            assert "last error: x" in store.failed
            assert len(requester.calls) == 7
        finally:
            store.close()

    def test_stream_retries_are_bounded(self) -> None:
        store = stream_store(
            max_consecutive_failures=3, _requester=_ScriptedRequester()
        )
        try:
            store.start()
            assert wait_until(lambda: store.failed is not None)
            assert "gave up after 4 consecutive failures" in store.failed
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

    @pytest.mark.parametrize("raw", ["inf", "Infinity", "-inf", "nan", "1e309"])
    def test_a_non_finite_retry_after_is_ignored(self, raw: str) -> None:
        assert _retry_after_seconds({"Retry-After": raw}) is None

    def test_retry_after_parsing_keeps_its_edges(self) -> None:
        assert _retry_after_seconds({"Retry-After": "0"}) == 0.0
        assert _retry_after_seconds({"Retry-After": "-5"}) == 0.0
        assert (
            _retry_after_seconds({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
            is None
        )
        assert _retry_after_seconds({"Retry-After": "2.5"}) == 2.5

    @pytest.mark.parametrize("retry_after", [float("inf"), float("nan"), 86400.0])
    def test_an_unreasonable_retry_after_neither_kills_delivery_nor_parks_it(
        self, retry_after: float
    ) -> None:
        # An infinite wait would overflow inside the retry handler and kill the
        # thread with `failed` still None; a day-long one would be honoured to
        # the second. Both must fall back to the max_backoff cap and carry on.
        requester = _ScriptedRequester(
            _RecoverableTransportError("slow down", retry_after=retry_after),
            [
                (e["event"], e["data"])
                for e in full_payload(("put-object", put_skill()))
            ],
        )
        store = stream_store(max_backoff=0.05, _requester=requester)
        try:
            store.start()
            assert store.wait_for_skills(timeout=3) is True
            assert store.failed is None
            assert store._thread is not None and store._thread.is_alive()
        finally:
            store.close()

    def test_a_non_finite_retry_after_off_the_wire_falls_back_to_backoff(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(status=429, retry_after="inf")
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=3) is True
            assert store.failed is None

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
        """``missing_content_hash`` and ``hash_mismatch`` must not collapse: one
        means the envelope carried no hash, the other means the content did not
        match the hash it carried."""
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
        """The positive control: a well-formed envelope resolves end to end."""
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
        The store's change listener drives the reconcile, so the file goes away
        seconds after the ``delete-object`` rather than at the next process start.
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
        """``on_unavailable="keep"`` is the default: an outage must not read as
        "everything was revoked"."""
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
        """The watcher is wired to the ``SkillStore`` interface, not to the FDv2
        store."""
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


class TestWatcherDetachesOnClose:
    """``SkillWatcher.close`` unregisters ``notify``, so a closed watcher is
    neither called nor kept alive by the store."""

    @staticmethod
    def _skill_listeners(store: Any) -> list[Any]:
        return list(store._listeners.get(SKILL_OBJECT_KIND, []))

    async def test_a_closed_watcher_is_no_longer_notified(
        self, endpoint: Any, tmp_path: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill(content="first"))))
        endpoint.queue_poll(status=304)
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-full")),
                ("put-object", put_skill(object_version=4, content="second")),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        endpoint.queue_poll(status=304)

        with poll_store(endpoint, poll_interval=0.1) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())
            _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
            assert watcher.notify in self._skill_listeners(store)

            watcher.close()

            assert watcher.notify not in self._skill_listeners(store)
            written = tmp_path / "s" / "pdf-extraction" / "SKILL.md"
            assert wait_until(
                lambda: (
                    store.get_object(SKILL_OBJECT_KIND, "pdf-extraction", 4) is not None
                ),
                timeout=10,
            )
            time.sleep(0.3)
            assert written.read_text() == "first"
            assert watcher.reconciles == 0

    async def test_close_twice_does_not_raise(self, tmp_path: Any) -> None:
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
        watcher.close()
        watcher.close()
        assert self._skill_listeners(store) == []

    async def test_repeated_watchers_leave_no_listeners_behind(
        self, tmp_path: Any
    ) -> None:
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        for _ in range(5):
            _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
            assert len(self._skill_listeners(store)) == 1
            watcher.close()
        assert self._skill_listeners(store) == []

    async def test_a_store_without_remove_listener_still_closes(
        self, tmp_path: Any
    ) -> None:
        """``remove_listener`` is optional: an older store keeps working, at the
        cost of the listener staying registered."""

        class AddOnly:
            def __init__(self) -> None:
                self.listeners: list[Any] = []

            def get_object(self, *_a: Any, **_k: Any) -> None:
                return None

            def all_objects(self, _kind: str) -> dict[str, Any]:
                return {}

            def add_listener(self, _kind: str, fn: Any) -> None:
                self.listeners.append(fn)

        store = AddOnly()
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
        assert store.listeners == [watcher.notify]

        watcher.close()
        watcher.close()

        assert store.listeners == [watcher.notify]

    def test_fdv2_remove_listener_of_an_unregistered_callable_is_a_no_op(
        self, endpoint: Any
    ) -> None:
        with poll_store(endpoint) as store:
            store.remove_listener(SKILL_OBJECT_KIND, print)
            store.add_listener(SKILL_OBJECT_KIND, print)
            store.remove_listener("flag", print)
            store.remove_listener(SKILL_OBJECT_KIND, print)
            store.remove_listener(SKILL_OBJECT_KIND, print)
            assert self._skill_listeners(store) == []


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

    def test_close_during_a_slow_connect_returns_promptly(self) -> None:
        # Before the connect returns there is no connection for close() to
        # interrupt. If the delivery thread then enters the read anyway, close()
        # sits out its whole join timeout on a stream that will never speak.
        requester = _SlowConnectRequester()
        store = stream_store(_requester=requester)
        store.start()
        assert requester.entered.wait(timeout=5)
        threading.Timer(0.1, requester.release.set).start()
        started = time.monotonic()
        store.close(timeout=5.0)
        elapsed = time.monotonic() - started
        assert elapsed < 2.0
        assert store._thread is not None
        assert not store._thread.is_alive()

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


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


class _BlackHole:
    """
    A listening socket that accepts connections and never sends a byte.

    This is the host ``read_timeout`` exists for: the TCP handshake completes, so
    nothing fails fast, and then no response ever comes. A request against it can
    only end by timing out, which makes the elapsed time a direct measurement of
    the timeout actually applied.
    """

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._accepted: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_forever, daemon=True)
        self._thread.start()
        host, port = self._listener.getsockname()
        self.base_uri = f"http://{host}:{port}"

    def _accept_forever(self) -> None:
        self._listener.settimeout(0.05)
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                continue
            self._accepted.append(conn)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        for conn in self._accepted:
            conn.close()
        self._listener.close()


@pytest.fixture
def black_hole() -> Any:
    server = _BlackHole()
    yield server
    server.close()


class TestTimeouts:
    """
    ``read_timeout`` is the only network timeout, and every request honours it.

    The bounds asserted here are loose on purpose: the point is that a request
    against an unresponsive host fails in roughly ``read_timeout`` rather than in
    minutes, and that a regression back to a much longer default fails this
    suite quickly instead of hanging it.
    """

    def test_a_poll_against_an_unresponsive_host_fails_within_read_timeout(
        self, black_hole: Any
    ) -> None:
        requester = _Requester(
            SDK_KEY, black_hole.base_uri, read_timeout=0.3, data_model_version=1
        )
        started = time.monotonic()
        with pytest.raises(_RecoverableTransportError) as excinfo:
            requester.poll(None, None)
        elapsed = time.monotonic() - started
        assert 0.2 <= elapsed < 2.0
        assert "timed out" in str(excinfo.value)

    def test_a_stream_against_an_unresponsive_host_fails_within_read_timeout(
        self, black_hole: Any
    ) -> None:
        requester = _Requester(
            SDK_KEY, black_hole.base_uri, read_timeout=0.3, data_model_version=1
        )
        started = time.monotonic()
        with pytest.raises(_RecoverableTransportError):
            requester.stream(None)
        assert time.monotonic() - started < 2.0

    def test_the_store_reports_the_timeout_and_keeps_going(
        self, black_hole: Any
    ) -> None:
        store = FDv2SkillStore(
            SDK_KEY,
            base_uri=black_hole.base_uri,
            mode="poll",
            poll_interval=0.05,
            initial_backoff=0.01,
            max_backoff=0.05,
            read_timeout=0.3,
        )
        try:
            store.start()
            assert wait_until(lambda: store.diagnostics.connection_failures >= 1)
            assert store.failed is None
            assert "timed out" in (store.diagnostics.last_error or "")
        finally:
            store.close()

    def test_the_default_bound_depends_on_the_mode(self) -> None:
        assert DEFAULT_POLL_TIMEOUT == 10.0
        assert DEFAULT_STREAM_READ_TIMEOUT == 300.0
        polling = FDv2SkillStore(SDK_KEY, mode="poll")
        streaming = FDv2SkillStore(SDK_KEY, mode="stream")
        assert polling._requester._read_timeout == DEFAULT_POLL_TIMEOUT
        assert streaming._requester._read_timeout == DEFAULT_STREAM_READ_TIMEOUT

    @pytest.mark.parametrize("mode", ["poll", "stream"])
    def test_an_explicit_read_timeout_overrides_the_default(self, mode: Any) -> None:
        store = FDv2SkillStore(SDK_KEY, mode=mode, read_timeout=42.0)
        assert store._requester._read_timeout == 42.0

    @pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_non_positive_read_timeout_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="read_timeout"):
            FDv2SkillStore(SDK_KEY, read_timeout=value)

    def test_there_is_no_separate_connect_timeout(self) -> None:
        # ``urllib`` cannot bound the connect separately from the reads, so the
        # constructor does not offer a parameter that would only pretend to.
        with pytest.raises(TypeError):
            FDv2SkillStore(SDK_KEY, connect_timeout=2.0)  # type: ignore[call-arg]
