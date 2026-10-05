"""
Tests for the FDv2 skill delivery transport.

Two layers, deliberately:

- **A real fake endpoint.** ``_FakeFDv2Endpoint`` is an in-process
  ``ThreadingHTTPServer`` that implements the wire contract — the ``basis``
  query parameter, ``Authorization``, ``If-None-Match``/304, the
  ``{"events": [...]}`` polling envelope, and SSE for streaming. The store under
  test opens real sockets against it, so request construction and header
  handling are exercised rather than mocked.
- **The protocol reader driven directly.** Wire semantics — which objects are
  skills, the skill's version in the wire ``key`` versus the payload's in
  ``version``, revocation, mixed payloads — are
  asserted against ``_ProtocolReader``, which has no I/O, so those cases read as
  the contract they are instead of as a server script.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import socket
import threading
import time
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    skills_core,
    skills_fdv2,
    watch_skills,
    write_skills,
)
from launchdarkly_ai_server import skills as skills_module
from launchdarkly_ai_server.skills_core import SKILL_OBJECT_KIND
from launchdarkly_ai_server.skills_fdv2 import (
    DEFAULT_BASE_URI,
    DEFAULT_POLL_TIMEOUT,
    DEFAULT_STREAM_READ_TIMEOUT,
    DEFAULT_STREAM_URI,
    FDV2_KEY_DELIMITER,
    FDV2_OBJECT_KIND,
    FDV2_PAYLOAD_KIND,
    MAX_RESPONSE_BYTES,
    StoreDiagnostics,
    _backoff_delay,
    _classify_status,
    _FatalTransportError,
    _is_skill_event,
    _iter_sse,
    _ProtocolReader,
    _RecoverableTransportError,
    _Requester,
    _ResponseTooLargeError,
    _retry_after_seconds,
    _SkillObjectSet,
    _StaleRequestStateError,
    _store_object_from_put,
    _StreamConnection,
    _tombstone_from_delete,
)

pytestmark = pytest.mark.usefixtures("reset_skill_state")

SDK_KEY = "sdk-00000000-0000-4000-8000-000000000000"
SKILL_BODY = "---\nname: PDF Extraction\n---\nExtract text from PDFs.\n"


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Wire builders — one place that knows the shape, so a contract change is one edit
# ---------------------------------------------------------------------------


def wire_key(key: str, object_version: Any) -> str:
    """
    The wire ``key`` of one skill object: ``<key>:<version>``.

    ``None`` builds a key with no version at all, which is how the tests spell a
    malformed object; anything else is spelled after the delimiter verbatim.
    """
    if object_version is None:
        return key
    return f"{key}{FDV2_KEY_DELIMITER}{object_version}"


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
        "key": wire_key(key, object_version),
        "kind": FDV2_OBJECT_KIND,
        "version": payload_version,
        "object": envelope,
    }


def delete_skill(
    key: str = "pdf-extraction", *, object_version: Any = 3, payload_version: int = 43
) -> dict[str, Any]:
    return {
        "key": wire_key(key, object_version),
        "kind": FDV2_OBJECT_KIND,
        "version": payload_version,
    }


def put_flag(key: str = "my-flag", version: int = 17) -> dict[str, Any]:
    """A flag ``put-object``: the same envelope fields, a different ``kind``."""
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
        # When set, every ``/sdk/stream`` answers 307 to this URL instead of
        # streaming, so a test can check that the store refuses to follow it.
        self.redirect_stream_to: str | None = None
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
        location: str | None = None,
    ) -> None:
        with self._lock:
            self._polls.append(
                {
                    "status": status,
                    "events": payload_events or [],
                    "etag": etag,
                    "retry_after": retry_after,
                    "location": location,
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
        if response.get("location"):
            handler.send_header("Location", response["location"])
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
        if self.redirect_stream_to:
            handler.send_response(307)
            handler.send_header("Location", self.redirect_stream_to)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
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


def _assigned_literal(module: Any, name: str) -> Any:
    """The literal *name* is assigned in *module*'s own source, or ``None``.

    Read from the syntax rather than from the attribute, because the attribute
    cannot answer the question being asked: a module that bound *name* to some
    other module's constant would expose an identical value.
    """
    tree = ast.parse(inspect.getsource(module))
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        named = any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        )
        if named and isinstance(node.value, ast.Constant):
            return node.value.value
    return None


# ---------------------------------------------------------------------------
# Identifying skill objects, and ignoring everything else
# ---------------------------------------------------------------------------


class TestObjectIdentification:
    def test_the_kind_alone_identifies_a_skill(self) -> None:
        assert _is_skill_event(put_skill()) is True

    def test_the_payload_kind_and_the_object_kind_are_separate_constants(
        self,
    ) -> None:
        """
        Neither is derived from the other: the store asks for a payload of kind
        ``agent-skill`` and reads objects of kind ``skill`` out of it. Held
        apart so a rename of either cannot silently move the other.
        """
        assert FDV2_PAYLOAD_KIND == "agent-skill"
        assert FDV2_OBJECT_KIND == "skill"
        assert FDV2_PAYLOAD_KIND != FDV2_OBJECT_KIND

    def test_the_wire_object_kind_is_not_the_interface_kind_renamed(self) -> None:
        """
        Three related constants, and no one of them is an alias of another: the
        wire payload kind and the wire object kind above, plus the kind the
        accessors ask a store for. Only the first two are LaunchDarkly's.

        ``FDV2_OBJECT_KIND`` and ``SKILL_OBJECT_KIND`` hold the same string
        today, which is exactly why this needs asserting rather than comparing:
        equal values cannot tell an intentional coincidence from one module
        re-exporting the other's constant, and identity cannot either, since
        CPython interns both. What fixes it is the definition — each module
        states its own value, so renaming LaunchDarkly's wire kind cannot drag
        this SDK's interface kind along with it, or the reverse.
        """
        # Pinned so that the day the coincidence ends, this test is what a
        # reader lands on rather than a puzzling failure somewhere else.
        assert FDV2_OBJECT_KIND == SKILL_OBJECT_KIND
        assert _assigned_literal(skills_fdv2, "FDV2_OBJECT_KIND") == "skill"
        assert _assigned_literal(skills_fdv2, "FDV2_PAYLOAD_KIND") == "agent-skill"
        assert _assigned_literal(skills_core, "SKILL_OBJECT_KIND") == "skill"

    def test_the_kind_is_the_bare_category_name(self) -> None:
        """
        Object kinds on the channel are open strings, so a skill arrives under
        the kind its producer registered — ``skill`` — not under a broader
        wrapper kind, and not under the kind of the payload carrying it.
        """
        assert FDV2_OBJECT_KIND == "skill"

    def test_a_flag_is_not_a_skill(self) -> None:
        assert _is_skill_event(put_flag()) is False

    def test_a_segment_is_not_a_skill(self) -> None:
        assert _is_skill_event(put_segment()) is False

    def test_another_generic_kind_is_not_a_skill(self) -> None:
        """A generic payload may carry other registered kinds one day."""
        other = put_skill()
        other["kind"] = "prompt-template"
        assert _is_skill_event(other) is False

    def test_a_skill_shaped_envelope_under_another_kind_is_not_a_skill(self) -> None:
        other = put_skill()
        other["kind"] = "some-future-kind"
        assert _is_skill_event(other) is False

    def test_nothing_but_the_kind_is_consulted(self) -> None:
        """No secondary field narrows the kind, and none may be required."""
        assert set(put_skill()) == {"key", "kind", "version", "object"}

    @pytest.mark.parametrize("value", [None, "skill", 3, [], ()])
    def test_non_dict_events_are_not_skills(self, value: Any) -> None:
        assert _is_skill_event(value) is False


# ---------------------------------------------------------------------------
# The skill's version is in the wire key; `version` is the payload's
# ---------------------------------------------------------------------------


class TestVersionTranslation:
    def test_the_wire_key_is_key_colon_version(self) -> None:
        assert (
            put_skill("pdf-extraction", object_version=3)["key"] == "pdf-extraction:3"
        )

    def test_the_version_after_the_delimiter_becomes_the_stored_version(
        self,
    ) -> None:
        raw = _store_object_from_put(put_skill(object_version=3, payload_version=42))
        assert raw is not None
        assert raw["version"] == 3
        assert isinstance(raw["version"], int)

    def test_the_key_before_the_delimiter_becomes_the_stored_key(self) -> None:
        """A caller asks for ``pdf-extraction``, never for ``pdf-extraction:3``."""
        raw = _store_object_from_put(put_skill("pdf-extraction", object_version=3))
        assert raw is not None
        assert raw["key"] == "pdf-extraction"

    def test_the_payload_version_is_never_stored(self) -> None:
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

    def test_a_key_with_no_delimiter_is_held_version_less(self) -> None:
        """Not defaulted from the payload version, and not dropped: verification
        reports ``invalid_version`` under a key the caller recognises."""
        raw = _store_object_from_put(put_skill(object_version=None))
        assert raw is not None
        assert raw["key"] == "pdf-extraction"
        assert "version" not in raw

    @pytest.mark.parametrize("spelling", ["latest", "", "3.0", "-1", "1:2", "３"])
    def test_a_version_that_is_not_digits_is_carried_through_as_invalid(
        self, spelling: str
    ) -> None:
        """Carried, not invented: verification reports ``invalid_version`` for
        the object rather than the transport reporting it absent."""
        raw = _store_object_from_put(put_skill(object_version=spelling))
        assert raw is not None
        assert raw["key"] == "pdf-extraction"
        assert raw["version"] == spelling

    def test_leading_zeros_spell_the_same_version(self) -> None:
        raw = _store_object_from_put(put_skill(object_version="03"))
        assert raw is not None
        assert raw["version"] == 3

    def test_a_delete_reads_the_wire_key_the_same_way(self) -> None:
        tombstone = _tombstone_from_delete(
            delete_skill(object_version=3, payload_version=43)
        )
        assert tombstone is not None
        assert tombstone.key == "pdf-extraction"
        assert tombstone.object_version == 3

    @pytest.mark.parametrize("spelling", [None, "latest", "0"])
    def test_a_delete_with_no_usable_version_revokes_every_version(
        self, spelling: Any
    ) -> None:
        tombstone = _tombstone_from_delete(delete_skill(object_version=spelling))
        assert tombstone is not None
        assert tombstone.key == "pdf-extraction"
        assert tombstone.object_version is None

    @pytest.mark.parametrize("bad_key", [":3", "", None, 3])
    def test_a_put_with_no_skill_key_is_dropped_because_it_has_no_identity(
        self, bad_key: Any
    ) -> None:
        wire = put_skill()
        wire["key"] = bad_key
        assert _store_object_from_put(wire) is None

    def test_a_keyless_put_is_dropped_because_it_has_no_identity(self) -> None:
        wire = put_skill()
        del wire["key"]
        assert _store_object_from_put(wire) is None

    def test_a_delete_with_no_skill_key_is_ignored(self) -> None:
        wire = delete_skill()
        wire["key"] = ":3"
        assert _tombstone_from_delete(wire) is None

    def test_the_stored_identity_round_trips_to_the_wire_key(self) -> None:
        """``_SkillObjectSet.snapshot`` spells its opaque keys the way the wire
        does, so a held object can be matched back to the event that carried it."""
        held = _SkillObjectSet()
        wire = put_skill("pdf-extraction", object_version=3)
        raw = _store_object_from_put(wire)
        assert raw is not None
        held.put(raw)
        assert set(held.snapshot()) == {wire["key"]}

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

    def test_an_up_to_date_intent_is_reported_as_such(self) -> None:
        """``intentCode: "none"`` is the stream's 304: current, nothing to send."""
        reader = _ProtocolReader(_SkillObjectSet())
        outcome = reader.handle("server-intent", server_intent("none"))
        assert outcome.up_to_date is True
        assert outcome.committed is False
        assert outcome.disconnect is None
        # A transfer intent is a promise of content, not an up-to-date answer.
        transfer = reader.handle("server-intent", server_intent("xfer-full"))
        assert transfer.up_to_date is False

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

    def test_a_full_transfer_that_omits_a_skill_publishes_a_tombstone(self) -> None:
        """
        A full transfer states the whole payload, so it revokes by omission and
        no ``delete-object`` ever says so. Without the diff the store empties
        while ``changes`` stays empty, and the case pruning exists for — the
        environment's last skill revoked — wakes no listener at all.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill(object_version=3))))
        outcomes = drive(reader, full_payload(state="basis-2"))
        assert len(held) == 0
        assert outcomes[-1].changes == [{"key": "pdf-extraction", "version": 3}]
        assert reader.diagnostics.objects_revoked == 1

    def test_an_omitted_version_less_object_is_reported_as_departed(self) -> None:
        """
        An object too malformed to carry a version is held under its key alone,
        and leaves the same way: as a tombstone with no version, which is what a
        ``delete-object`` naming no version spells too.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill(object_version=None))))
        outcomes = drive(reader, full_payload(state="basis-2"))
        assert outcomes[-1].changes == [{"key": "pdf-extraction", "version": None}]
        assert reader.diagnostics.objects_revoked == 1

    def test_a_version_move_reports_both_ends_and_counts_no_revocation(self) -> None:
        """
        The diff runs at ``(key, version)``, so a key whose version moved yields
        a put for the arrival and a tombstone for the departure — what a
        listener that reads versions needs. ``objects_revoked`` counts per key,
        though, and this key never left the payload: counting it would tell an
        operator a revocation landed when a publish did.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill(object_version=3))))
        outcomes = drive(
            reader,
            full_payload(("put-object", put_skill(object_version=4)), state="basis-2"),
        )
        arrived, departed = outcomes[-1].changes
        assert (arrived["key"], arrived["version"]) == ("pdf-extraction", 4)
        assert departed == {"key": "pdf-extraction", "version": 3}
        assert reader.diagnostics.objects_revoked == 0

    def test_a_change_transfer_revokes_nothing_by_omission(self) -> None:
        """Only a full transfer states the whole payload. A delta that carries
        no tombstone revoked nothing, and diffing one would drop every skill it
        simply had no reason to mention."""
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill())))
        outcomes = drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert held.get("pdf-extraction", None) is not None
        assert outcomes[-1].changes == []
        assert reader.diagnostics.objects_revoked == 0

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

    def test_a_delete_for_a_key_never_held_is_not_counted_as_a_revocation(
        self,
    ) -> None:
        """``objects_revoked`` counts what went away, not tombstones seen.

        The counter is operator-facing, and it is read precisely when somebody
        is working out whether a revocation landed. A delete for a key the store
        never held revoked nothing, so counting it inflates the one number that
        answers that question. The tombstone still reaches listeners through
        ``changes``, which is where "every revocation the server stated" lives.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill(key="kept"))))
        outcomes = drive(
            reader,
            events(
                ("server-intent", server_intent("xfer-changes")),
                ("delete-object", delete_skill(key="never-delivered")),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert reader.diagnostics.objects_revoked == 0
        assert held.get("kept", None) is not None
        # Reported, just not counted.
        assert outcomes[-1].changes == [{"key": "never-delivered", "version": 3}]

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

    def test_transfer_none_holds_everything_and_commits_nothing(self) -> None:
        """
        ``none`` is the server saying the payload held is current, so nothing is
        applied and nothing is committed. It is a *completed exchange* — which
        is what breaks a row of failures, and what lets the next request offer
        the body's etag — but it is not a payload. Publishing one would make
        ``is_initialized`` true over whatever the store happens to hold, and
        that is the fact ``write_skills("*")`` prunes on.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill())))
        intent, outcome = drive(
            reader,
            events(
                ("server-intent", server_intent("none")),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert len(held) == 1
        assert intent.up_to_date is True
        assert outcome.committed is False
        # The intent event above already carried the up-to-date answer; the
        # transfer completing it adds nothing to report.
        assert outcome.up_to_date is False
        # Nor does it move the resume point. ``basis-2`` names a payload this
        # store was never sent, and resuming from it would ask every later
        # connection for changes since a payload it never applied.
        assert outcome.basis is None

    def test_a_transfer_that_applied_nothing_is_not_a_commit(self) -> None:
        """
        Three shapes reach ``payload-transferred`` with no pending set to apply:
        an intent code this SDK does not recognise, a ``none`` intent, and a
        lone transfer under no intent at all. None of them applied anything, so
        none of them claims anything — neither a commit, which is what publishes
        the first payload ``write_skills("*")`` prunes on, nor an up-to-date
        answer, which only the server can give and only the ``none`` intent
        does, on its own event.

        The transfer is still a wire fact, counted either way.
        """
        for payload_events in (
            events(
                ("server-intent", server_intent("xfer-future")),
                ("put-object", put_skill()),
                ("payload-transferred", transferred("basis-1")),
            ),
            events(
                ("server-intent", server_intent("none")),
                ("payload-transferred", transferred("basis-1")),
            ),
            events(("payload-transferred", transferred("basis-1"))),
        ):
            held = _SkillObjectSet()
            reader = _ProtocolReader(held)
            outcome = drive(reader, payload_events)[-1]
            assert outcome.committed is False
            assert outcome.up_to_date is False
            assert outcome.basis is None
            assert len(held) == 0
            assert reader.diagnostics.payloads_transferred == 1

    def test_a_put_after_a_none_intent_is_applied(self) -> None:
        """``none`` means current, not finished: later edits follow it on the
        same connection with no second intent, and must land."""
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(
            reader,
            events(
                ("server-intent", server_intent("none")),
                ("put-object", put_skill(object_version=4)),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert held.get("pdf-extraction", 4) is not None
        assert reader.diagnostics.objects_ignored == 0

    def test_a_delete_after_a_none_intent_revokes(self) -> None:
        """The case that matters: a skill revoked after a routine reconnect.

        Dropping the delete while adopting the selector would keep serving the
        revoked skill, and a reconnect would not recover it, because the basis
        has already moved past the change.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill())))
        outcomes = drive(
            reader,
            events(
                ("server-intent", server_intent("none")),
                ("delete-object", delete_skill()),
                ("payload-transferred", transferred("basis-2")),
            ),
        )
        assert len(held) == 0
        assert reader.diagnostics.objects_revoked == 1
        assert reader.diagnostics.objects_ignored == 0
        assert outcomes[-1].changes == [{"key": "pdf-extraction", "version": 3}]
        assert outcomes[-1].basis == "basis-2"

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
# Which payload a transfer completed
# ---------------------------------------------------------------------------


def _payload_warnings(caplog: Any, fragment: str) -> list[Any]:
    return [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and fragment in r.getMessage()
    ]


def skill_payload(
    *object_events: tuple[str, Any],
    payload_id: str = "agent-skill",
    code: str = "xfer-full",
    state: str = "basis-1",
) -> list[dict[str, Any]]:
    """One payload's events, with the payload it belongs to named explicitly."""
    return events(
        ("server-intent", server_intent(code, payload_id)),
        *object_events,
        ("payload-transferred", transferred(state)),
    )


class TestPayloadIdentity:
    """
    Which payload a transfer completed, and why this layer tracks it at all.

    Delivery provides one payload per credential and the protocol requires a
    client to read only the first payload intent, so today the payload read is
    the payload skills arrive on. These assert the behaviour that survives if
    the first of those stops holding: another payload's ``xfer-full`` must not
    publish an empty skill set, because with pruning on that deletes a
    customer's materialized files.
    """

    def test_only_the_first_payload_intent_is_read(self) -> None:
        """Reading only the first is what the protocol asks for, however many
        arrive — the point of the rest of this class is to make that safe."""
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(
            reader,
            events(
                (
                    "server-intent",
                    {
                        "payloads": [
                            {
                                "id": "agent-skill",
                                "target": 1,
                                "intentCode": "xfer-full",
                            },
                            {"id": "env-flags", "target": 2, "intentCode": "none"},
                        ]
                    },
                ),
                ("put-object", put_skill()),
                ("payload-transferred", transferred()),
            ),
        )
        assert len(held) == 1

    def test_more_than_one_payload_intent_warns_once(self, caplog: Any) -> None:
        reader = _ProtocolReader(_SkillObjectSet())
        intent = {
            "payloads": [
                {"id": "env-flags", "target": 1, "intentCode": "xfer-changes"},
                {"id": "agent-skill", "target": 2, "intentCode": "xfer-changes"},
            ]
        }
        with caplog.at_level("WARNING"):
            reader.handle("server-intent", intent)
            reader.handle("server-intent", intent)
        assert len(_payload_warnings(caplog, "described 2 payloads")) == 1

    def test_one_payload_intent_warns_about_nothing(self, caplog: Any) -> None:
        with caplog.at_level("WARNING"):
            drive(
                _ProtocolReader(_SkillObjectSet()),
                skill_payload(("put-object", put_skill())),
            )
        assert _payload_warnings(caplog, "payload") == []

    def test_another_payloads_full_transfer_does_not_empty_the_skills_held(
        self, caplog: Any
    ) -> None:
        """
        The case this guard exists for. A flag payload's ``xfer-full`` starts an
        empty pending set; applying it at ``payload-transferred`` would publish
        every skill as revoked, which a reconcile with pruning on reads as
        "delete these files".
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, skill_payload(("put-object", put_skill())))
        with caplog.at_level("WARNING"):
            outcomes = drive(
                reader,
                skill_payload(
                    ("put-object", put_flag()), payload_id="env-flags", state="basis-2"
                ),
            )
        assert held.get("pdf-extraction", None) is not None
        assert reader.diagnostics.payloads_ignored == 1
        assert len(_payload_warnings(caplog, "was not applied")) == 1
        # Nothing changed, so no listener is woken to reconcile against it.
        assert outcomes[-1].changes == []

    def test_a_declined_transfer_warns_once_however_often_it_repeats(
        self, caplog: Any
    ) -> None:
        """A polling connection sees the other payload on every poll."""
        reader = _ProtocolReader(_SkillObjectSet())
        drive(reader, skill_payload(("put-object", put_skill())))
        foreign = skill_payload(("put-object", put_flag()), payload_id="env-flags")
        with caplog.at_level("WARNING"):
            drive(reader, foreign)
            drive(reader, foreign)
        assert len(_payload_warnings(caplog, "was not applied")) == 1
        assert reader.diagnostics.payloads_ignored == 2

    def test_a_declined_transfer_does_not_move_the_resume_point(self) -> None:
        """
        Ignoring a foreign payload's contents while adopting its resume point
        would ask the next poll or stream to resume from someone else's
        payload: skill updates could stop arriving while every diagnostic read
        healthy.
        """
        reader = _ProtocolReader(_SkillObjectSet())
        ours = drive(
            reader, skill_payload(("put-object", put_skill()), state="skills-basis")
        )
        assert ours[-1].basis == "skills-basis"
        outcomes = drive(
            reader,
            skill_payload(
                ("put-object", put_flag()), payload_id="env-flags", state="flag-basis"
            ),
        )
        assert outcomes[-1].basis is None
        assert reader.diagnostics.payloads_ignored == 1

    def test_a_none_intent_for_another_payload_moves_nothing(self) -> None:
        """
        A ``none`` intent builds no pending set, and the foreign check must not
        be gated on one: the transfer that follows still names a payload, and
        adopting its selector would resume the next connection from someone
        else's payload with every diagnostic reading healthy.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, full_payload(("put-object", put_skill()), state="basis-skills"))
        outcomes = drive(
            reader,
            events(
                ("server-intent", server_intent("none", "env-flags")),
                ("payload-transferred", transferred("basis-flags")),
            ),
        )
        assert outcomes[-1].basis is None
        assert reader.diagnostics.payloads_ignored == 1
        assert held.get("pdf-extraction", None) is not None

    def test_a_full_transfer_of_the_skill_payload_still_empties_it(self) -> None:
        """Every skill deleted is a real state, and the guard must not mask it."""
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, skill_payload(("put-object", put_skill())))
        drive(reader, skill_payload(state="basis-2"))
        assert len(held) == 0
        assert reader.diagnostics.payloads_ignored == 0

    def test_a_revocation_identifies_the_payload_as_the_skill_payload(self) -> None:
        """A payload that only revokes is still a payload skills arrive on."""
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(
            reader,
            skill_payload(("delete-object", delete_skill()), code="xfer-changes"),
        )
        drive(
            reader, skill_payload(("put-object", put_skill()), payload_id="env-flags")
        )
        assert reader.diagnostics.payloads_ignored == 1

    def test_the_payload_is_identified_from_the_selector_when_no_id_is_named(
        self,
    ) -> None:
        """``payload-transferred``'s selector is the only other place a completed
        transfer names its payload."""
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        unnamed = {"payloads": [{"target": 1, "intentCode": "xfer-full"}]}
        drive(
            reader,
            events(
                ("server-intent", unnamed),
                ("put-object", put_skill()),
                ("payload-transferred", transferred("(p:agent-skill:53)")),
            ),
        )
        drive(
            reader,
            events(
                ("server-intent", unnamed),
                ("put-object", put_flag()),
                ("payload-transferred", transferred("(p:env-flags:12)")),
            ),
        )
        assert held.get("pdf-extraction", None) is not None
        assert reader.diagnostics.payloads_ignored == 1

    def test_an_unidentifiable_payload_is_applied_rather_than_withheld(self) -> None:
        """
        A transfer naming no payload at all is the store's own, since delivery
        sends it one payload. Withholding it would break the common case to
        defend against a hypothetical one.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        drive(reader, skill_payload(("put-object", put_skill())))
        drive(
            reader,
            events(
                ("server-intent", {"payloads": [{"intentCode": "xfer-full"}]}),
                ("put-object", put_skill(object_version=4)),
                ("payload-transferred", {"version": 44}),
            ),
        )
        assert held.get("pdf-extraction", None)["version"] == 4
        assert reader.diagnostics.payloads_ignored == 0

    def test_the_first_transfer_of_a_connection_is_the_residual(
        self, caplog: Any
    ) -> None:
        """
        Before a skill has arrived there is nothing to compare a payload
        against, so another payload's ``xfer-full`` arriving first cannot be
        told apart. The multiple-payload WARNING is the only signal there is,
        which is why it exists.
        """
        held = _SkillObjectSet()
        reader = _ProtocolReader(held)
        with caplog.at_level("WARNING"):
            drive(
                reader,
                events(
                    (
                        "server-intent",
                        {
                            "payloads": [
                                {"id": "env-flags", "intentCode": "xfer-full"},
                                {"id": "agent-skill", "intentCode": "xfer-full"},
                            ]
                        },
                    ),
                    ("put-object", put_flag()),
                    ("payload-transferred", transferred()),
                ),
            )
        assert len(held) == 0
        assert len(_payload_warnings(caplog, "described 2 payloads")) == 1


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
        # A key holding a well-formed version *and* a version-less entry. The
        # quadrant the fixtures above miss: "malformed" has no usable version
        # at all, and "a"/"b" have no version-less entry, so neither exercises
        # what happens when a pin misses a key that has both.
        {"key": "mixed", "version": 2, "content": "m", "contentHash": _hash("m")},
        {"key": "mixed", "version": "not-a-version", "content": "n"},
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
            ("mixed", None),
            ("mixed", 2),
            ("mixed", 7),
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

    def test_the_request_carries_the_sdk_key_and_no_data_model_version(
        self, endpoint: Any
    ) -> None:
        """
        No ``mv``: that parameter selects the *flag* data model, and delivery
        overrides it with the payload's own default for any non-flagging
        payload, so sending it would state a preference that is ignored.
        """
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
        first = endpoint.requests[0]
        assert first["path"] == "/sdk/poll"
        assert first["authorization"] == SDK_KEY
        assert "mv" not in first["query"]

    def test_every_request_declares_the_agent_skill_payload_kind(
        self, endpoint: Any
    ) -> None:
        """
        Delivery narrows a connection to the kinds it declares and defaults to
        flags, so a request without this parameter is served the environment's
        flag payload and no skills at all. It is on the first request too: the
        declaration selects what the connection is served rather than describing
        what it holds, so there is no state for it to wait on.
        """
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=304)
        with poll_store(endpoint, poll_interval=0.02) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert wait_until(lambda: len(endpoint.requests) >= 2)
        first, second = endpoint.requests[0], endpoint.requests[1]
        assert first["query"]["kinds"] == "agent-skill"
        assert "basis" not in first["query"]
        assert second["query"]["kinds"] == "agent-skill"
        assert second["query"]["basis"] == "basis-1"

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

    def test_the_basis_stays_on_the_skill_payload_when_another_transfers(
        self, endpoint: Any
    ) -> None:
        """The wire half of the declined-payload case: the store must resume from
        the payload skills arrive on, not from the one it just threw away."""
        endpoint.queue_poll(
            full_payload(("put-object", put_skill()), state="skills-basis")
        )
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-full", "env-flags")),
                ("put-object", put_flag()),
                ("payload-transferred", transferred("flag-basis")),
            )
        )
        endpoint.queue_poll(status=304)
        with poll_store(endpoint):
            assert wait_until(lambda: len(endpoint.requests) >= 3)
        bases = [r["query"].get("basis") for r in endpoint.requests[:3]]
        assert bases == [None, "skills-basis", "skills-basis"]

    def test_an_etag_is_returned_for_the_basis_it_was_issued_against(
        self, endpoint: Any
    ) -> None:
        """
        An ETag validates one representation of one resource, and the basis is
        part of the request that names it. ``W/"v1"`` answers the request that
        carried no basis at all, so it is not offered once the payload it came
        with moved the basis on; ``W/"v2"`` answers a request from ``basis-1``,
        which is still the question being asked, so it is.
        """
        endpoint.queue_poll(
            full_payload(("put-object", put_skill()), state="basis-1"), etag='W/"v1"'
        )
        endpoint.queue_poll(
            events(("server-intent", server_intent("none"))), etag='W/"v2"'
        )
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            assert wait_until(lambda: len(endpoint.requests) >= 3)
        bases = [r["query"].get("basis") for r in endpoint.requests[:3]]
        assert bases == [None, "basis-1", "basis-1"]
        offered = [r["if_none_match"] for r in endpoint.requests[:3]]
        assert offered == [None, None, 'W/"v2"']

    def test_the_etag_of_a_body_never_applied_is_not_offered(
        self, endpoint: Any
    ) -> None:
        """
        The body announced a transfer and then broke off, so the payload it
        described was never committed. Offering its etag would invite a 304 that
        reports a store still missing that payload as current and healthy — and
        unlike the 200 it replaces, a 304 carries nothing to notice that on.
        """
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-full")),
                ("put-object", put_skill()),
                ("error", {"reason": "cut off mid-payload"}),
            ),
            etag='W/"v1"',
        )
        endpoint.queue_poll(
            full_payload(("put-object", put_skill()), state="basis-1"), etag='W/"v2"'
        )
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert endpoint.requests[1]["if_none_match"] is None
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None

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
            # A payload arrived, and a 304 does not take that back.
            assert store.is_initialized() is True

    def test_a_304_confirms_a_payload_but_cannot_establish_one(
        self, endpoint: Any
    ) -> None:
        """
        A 304 answers for content this store already holds, and the etag that
        asked for it is only ever adopted from a body that completed an
        exchange. Reaching one with nothing held therefore takes a server
        answering a request that carried no etag at all, and that 304 says
        nothing about a payload this store never received.

        Releasing ``wait_for_skills`` on it would make ``is_initialized`` true
        over an empty committed set — the fact ``write_skills("*")`` prunes on —
        so a reconcile that raced delivery would delete every managed skill on
        disk instead of reporting the retrieval unavailable (§3.21). Failing
        closed costs a boot that is genuinely waiting nothing it was not already
        waiting for.
        """
        endpoint.queue_poll(status=304)
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=0.5) is False
            assert store.is_initialized() is False
            # Not a failure either: the poll was answered, and the store is
            # still asking.
            assert store.failed is None
            assert endpoint.requests[0]["if_none_match"] is None

    def test_an_intent_it_cannot_apply_does_not_lend_its_etag_to_a_304(
        self, endpoint: Any
    ) -> None:
        """
        The chain this closes: a body under a future intent code announces and
        transfers a payload this SDK cannot apply, its etag is adopted as though
        the body had been applied in full, and the next 304 reports the empty
        store it left behind as current. That store is initialized, healthy, and
        authorises a prune of every managed skill on disk, with nothing in the
        304 to notice it on.

        An etag is adopted only from a body that completed an exchange, so the
        second request carries none and the endpoint's standing 304 cannot
        answer for content that never arrived.
        """
        endpoint.queue_poll(
            events(
                ("server-intent", server_intent("xfer-future")),
                ("put-object", put_skill()),
                ("payload-transferred", transferred("basis-1")),
            ),
            etag='W/"v1"',
        )
        with poll_store(endpoint) as store:
            assert wait_until(lambda: len(endpoint.requests) >= 2)
            assert store.is_initialized() is False
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is None
            assert endpoint.requests[1]["if_none_match"] is None
            # Nor is the selector of a payload it could not apply a resume point.
            assert [r["query"].get("basis") for r in endpoint.requests[:2]] == [
                None,
                None,
            ]

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

    def test_the_stream_request_declares_the_payload_kind(self, endpoint: Any) -> None:
        """Asserted separately from polling: the two paths build their own URLs
        against different origins."""
        endpoint.hold_stream_open = True
        endpoint.queue_stream(full_payload(("put-object", put_skill())))
        store = FDv2SkillStore(SDK_KEY, base_uri=endpoint.base_uri, mode="stream")
        try:
            store.start()
            store.wait_for_skills(timeout=5)
        finally:
            store.close()
        assert endpoint.requests[0]["query"]["kinds"] == "agent-skill"

    def test_a_revocation_after_an_up_to_date_answer_arrives(
        self, endpoint: Any
    ) -> None:
        """A reconnect answered ``none``, then a revocation on the same stream.

        The first connection ends after its payload, as a recycled stream does.
        """
        endpoint.queue_stream(full_payload(("put-object", put_skill())))
        endpoint.queue_stream(
            events(
                ("server-intent", server_intent("none")),
                ("delete-object", delete_skill()),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        store = FDv2SkillStore(
            SDK_KEY,
            base_uri=endpoint.base_uri,
            mode="stream",
            initial_backoff=0.01,
            max_backoff=0.02,
        )
        notified: list[dict[str, Any]] = []
        store.add_listener(SKILL_OBJECT_KIND, notified.append)
        try:
            store.start()
            assert store.wait_for_skills(timeout=5) is True
            assert wait_until(
                lambda: store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is None
            )
            assert store.diagnostics.objects_revoked == 1
            assert {"key": "pdf-extraction", "version": 3} in notified
        finally:
            store.close()
        # The premise: the revocation came on a reconnect carrying a basis.
        assert endpoint.requests[1]["query"].get("basis") == "basis-1"

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


class _DyingResponse:
    """
    A streaming body that transfers a payload and then fails mid-read.

    This is how a live stream actually ends: not with a clean end of body but
    with a read timeout on a stream that went quiet, or a reset from the server
    or a proxy in between.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self._lines: list[bytes] = []
        for event in full_payload(("put-object", put_skill())):
            self._lines.append(f"event: {event['event']}\n".encode())
            self._lines.append(f"data: {json.dumps(event['data'])}\n".encode())
            self._lines.append(b"\n")

    def readline(self, size: int = -1) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        raise self._exc

    def close(self) -> None:
        pass


class _FakeRequester:
    """
    Base for the requester fakes: supplies the ``interrupt`` the store calls on
    ``close``, so each fake only scripts the part it is about.
    """

    def interrupt(self) -> None:
        """No real socket to reach; these fakes end their own connections."""


class _DyingStreamRequester(_FakeRequester):
    """Every connection transfers a payload, then dies with *exc* mid-read."""

    def __init__(self, exc: BaseException) -> None:
        self.connections = 0
        self._exc = exc

    def stream(self, basis: str | None) -> Any:
        self.connections += 1
        return _StreamConnection(_DyingResponse(self._exc))


class _ScriptedConnection:
    """Stands in for ``_StreamConnection``: an event iterator plus a close."""

    def __init__(self, payload_events: Any) -> None:
        self.events = iter(payload_events)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ScriptedRequester(_FakeRequester):
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


class _RecyclingRequester(_FakeRequester):
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


class _UpToDateRecyclingRequester(_FakeRequester):
    """
    A healthy server with nothing new to say: every connection answers
    ``intentCode: "none"`` — the stream's equivalent of a 304 — transfers
    nothing, and is then recycled. This is the steady state of an environment
    whose skills are not changing, which is most environments most of the time.
    """

    def __init__(self, farewell: bool = False) -> None:
        self.connections = 0
        self._farewell = farewell

    def stream(self, basis: str | None) -> Any:
        self.connections += 1
        script: list[tuple[str, Any]] = [
            ("server-intent", server_intent("none")),
            ("heart-beat", {}),
        ]
        if self._farewell:
            # A recycle is often announced rather than abrupt.
            script.append(("goodbye", {"reason": "connection recycled"}))
        return _ScriptedConnection(script)


class _SlowPollRequester(_FakeRequester):
    """
    A poll whose request does not return until the test releases it, standing in
    for one blocked where no interrupt can reach: inside its connect.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def poll(self, basis: str | None, etag: str | None) -> Any:
        self.entered.set()
        self.release.wait(timeout=10)
        raise _RecoverableTransportError("released")


class _SilentStreamRequester(_FakeRequester):
    """A stream that connects and then delivers nothing until it is closed."""

    def stream(self, basis: str | None) -> Any:
        return _BlockingConnection()


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


class _SlowConnectRequester(_FakeRequester):
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


def _record_backoff_attempts(monkeypatch: Any) -> list[int]:
    """Records the attempt number of every backoff the store computes."""
    attempts: list[int] = []
    real = skills_fdv2._backoff_delay

    def recording(attempt: int, **kwargs: Any) -> float:
        attempts.append(attempt)
        return real(attempt, **kwargs)

    monkeypatch.setattr(skills_fdv2, "_backoff_delay", recording)
    return attempts


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

    def test_the_give_up_line_points_at_start_not_a_process_restart(
        self, endpoint: Any, caplog: Any
    ) -> None:
        """One line for every fatal, so it has to be true of every fatal.

        ``_give_up`` ends the run and not the store, and the test below is what
        proves a restarted store delivers. Telling an operator to restart their
        process is therefore an overstatement wherever it appears, and this
        line appears on all of them — a 401, a 403, a 404, and a 422 alike.
        """
        endpoint.queue_poll(status=401)
        with caplog.at_level("ERROR"):
            with poll_store(endpoint) as store:
                assert wait_until(lambda: store.failed is not None)
        lines = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("Skill delivery has stopped")
        ]
        assert len(lines) == 1
        assert "start()" in lines[0]
        assert "until the process restarts" not in lines[0]
        # It still says the held content survives, which is the other half of
        # what an operator reading this needs to know.
        assert "last content it received" in lines[0]

    def test_a_restarted_store_does_not_report_the_old_failure(
        self, endpoint: Any
    ) -> None:
        """``failed`` says why delivery stopped *for good*.

        A store started again is delivering, so the terminal reason from the
        previous run is no longer true of it. Leaving it would have a healthy
        store reporting a failure it has recovered from.
        """
        endpoint.queue_poll(status=401)
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
            assert "401" in store.failed

            endpoint.queue_poll(full_payload(("put-object", put_skill())))
            store.start()
            assert store.wait_for_skills(timeout=5) is True
            assert store.failed is None

    def test_a_restart_during_the_give_up_still_delivers(
        self, endpoint: Any, monkeypatch: Any
    ) -> None:
        """A restart that races the dying thread must not adopt it.

        ``failed`` becomes readable while the delivery thread is still alive and
        winding down. A ``start`` in that window has to spawn a replacement: a
        live thread is not by itself a delivering one, and treating it as one
        leaves a store reporting no failure with nothing left to deliver.

        The window is held open by blocking the give-up log line, which the
        dying thread emits after publishing the reason.
        """
        giving_up = threading.Event()
        may_finish = threading.Event()
        real_error = skills_fdv2.logger.error

        def blocking_error(msg: Any, *args: Any, **kwargs: Any) -> None:
            if isinstance(msg, str) and msg.startswith("Skill delivery has stopped"):
                giving_up.set()
                may_finish.wait(timeout=5)
            real_error(msg, *args, **kwargs)

        monkeypatch.setattr(skills_fdv2.logger, "error", blocking_error)

        endpoint.queue_poll(status=401)
        with poll_store(endpoint) as store:
            assert giving_up.wait(timeout=5)
            # The premise of the test: the reason is readable and the thread
            # that published it has not returned yet.
            assert store.failed is not None
            assert store._thread is not None
            assert store._thread.is_alive()

            endpoint.queue_poll(full_payload(("put-object", put_skill())))
            store.start()
            may_finish.set()

            assert store.wait_for_skills(timeout=5) is True
            assert store.failed is None

    def test_a_restart_resets_the_failure_count(self) -> None:
        """The failure count belongs to the run that accumulated it.

        Carried into a restarted run, it would start that run's backoff part
        way up and report failures the new run never had.
        """
        requester = _ScriptedRequester(
            _RecoverableTransportError("x"),
            _RecoverableTransportError("x"),
            _FatalTransportError("401"),
        )
        store = FDv2SkillStore(
            SDK_KEY,
            mode="poll",
            initial_backoff=0.001,
            max_backoff=0.002,
            _requester=requester,
        )
        with store:
            store.start()
            assert wait_until(lambda: store.failed is not None)
            assert store.diagnostics.connection_failures == 2

            requester.outcomes = [
                _RecoverableTransportError("x"),
                _FatalTransportError("401"),
            ]
            store.start()
            assert wait_until(lambda: store.failed is not None)
            # One failure in the new run, not three.
            assert store.diagnostics.connection_failures == 1

    def test_a_404_stops_delivery_immediately(self, endpoint: Any) -> None:
        """A 404 means the endpoint does not exist for this credential.

        A mistyped base URI, typically, or an instance that does not serve the
        FDv2 endpoints. No reconnect produces one, so it is fatal rather than
        retried — and it is the exception to the shape the recoverable-by-default
        rule would suggest.
        """
        endpoint.queue_poll(status=404)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
            assert store.wait_for_skills(timeout=1) is False
        assert "404" in store.failed
        assert "/sdk/poll" in store.failed
        # Fatal means one request, not a retry that happened to find the payload.
        assert len(endpoint.requests) == 1
        assert store.diagnostics.connection_failures == 0

    def test_a_400_reconnects_once_from_scratch_and_is_then_fatal(
        self, endpoint: Any
    ) -> None:
        """A 400 is what a stale ``basis`` selector looks like.

        The selector and the etag are the only client state the request carries,
        so a fresh connection built from nothing is the one repair available.
        It gets exactly one: the one-retry limit is what keeps this from being
        "400 is recoverable".
        """
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=400)
        endpoint.queue_poll(status=400)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert wait_until(lambda: store.failed is not None)
        assert "400" in store.failed
        # The first payload, then the retried request, then the fatal one. The
        # fourth queued payload is never asked for.
        assert len(endpoint.requests) == 3
        # The premise: the rejected request did carry client state to drop.
        assert "basis" in endpoint.requests[1]["query"]
        # The retry was from scratch: no selector and no etag on the way back.
        # The kind declaration stays; it is not client state.
        retried = endpoint.requests[2]
        assert retried["query"] == {"kinds": FDV2_PAYLOAD_KIND}
        assert retried["if_none_match"] is None
        # Last known good survives both.
        assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None

    def test_a_400_carrying_no_client_state_is_fatal_at_once(
        self, endpoint: Any
    ) -> None:
        """There is nothing to drop on a first connection, so nothing to repair.

        A request that carried neither a selector nor an etag and was still
        refused was refused on its own terms.
        """
        endpoint.queue_poll(status=400)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
        assert "400" in store.failed
        assert len(endpoint.requests) == 1

    def test_the_exceptional_statuses_are_classified_apart(self) -> None:
        # The classification is the contract; the end-to-end tests above are
        # what prove the loop honours it.
        assert isinstance(_classify_status(404, None), _FatalTransportError)
        assert isinstance(_classify_status(400, None), _StaleRequestStateError)
        # A stale-state error is still a recoverable one, so the retry path
        # reaches it at all.
        assert isinstance(_classify_status(400, None), _RecoverableTransportError)
        for status in (405, 406, 414, 501):
            assert isinstance(_classify_status(status, None), _FatalTransportError)
        assert isinstance(_classify_status(503, None), _RecoverableTransportError)
        assert not isinstance(_classify_status(503, None), _StaleRequestStateError)
        # 422 is fatal, and the platform chose the status to be exactly that:
        # a code SDKs stop on rather than retry.
        refused = _classify_status(422, None)
        assert isinstance(refused, _FatalTransportError)
        assert not isinstance(refused, _RecoverableTransportError)

    def test_every_status_is_either_recoverable_or_fatal(self) -> None:
        """
        There is no third class. A status classified as neither broken nor
        terminal is a retry loop with no bound and no budget — retrying for the
        life of the process, invisible to ``failed`` and ``connection_failures``
        alike — which is what held 422 before it was classified as fatal.
        """
        for status in range(300, 600):
            classified = _classify_status(status, None)
            fatal = isinstance(classified, _FatalTransportError)
            recoverable = isinstance(classified, _RecoverableTransportError)
            assert fatal ^ recoverable, (
                f"HTTP {status} classified as {type(classified).__name__}, "
                "which is neither exactly recoverable nor exactly fatal"
            )

    def test_there_is_no_expected_recoverable_error_class(self) -> None:
        """
        Asserted gone by name: the class existed only to hold 422, and a
        reintroduction is otherwise visible only in a log line no test reads.
        """
        assert not hasattr(skills_fdv2, "_NoSkillPayloadError")

    def test_diagnostics_does_not_count_an_unavailable_payload(self) -> None:
        """
        ``StoreDiagnostics`` is public API from the moment it ships, so its
        field list is the contract. A field counting "no payload of the kind you
        declared" would count the 422, which is fatal — no recurring event to
        accumulate, and no running store to accumulate it on.
        """
        assert "payload_unavailable" not in StoreDiagnostics.__dataclass_fields__
        assert not hasattr(StoreDiagnostics(), "payload_unavailable")
        # The whole list, so a reintroduction under any other name fails too.
        assert set(StoreDiagnostics.__dataclass_fields__) == {
            "payloads_transferred",
            "skill_objects_received",
            "objects_ignored",
            "objects_revoked",
            "payloads_ignored",
            "hashless_objects",
            "connection_failures",
            "last_error",
        }

    def test_a_422_on_the_first_response_stops_delivery(self, endpoint: Any) -> None:
        """
        A 422 means this connection will never be assigned a skill payload, not
        that the environment has no skills yet — an environment holding zero
        skills is assigned an empty payload that commits normally. Every cause
        is permanent, so the first 422 is enough to stop on.
        """
        endpoint.queue_poll(status=422)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
        assert "422" in store.failed
        # The queued payload is never asked for.
        assert len(endpoint.requests) == 1
        assert store.is_initialized() is False

    def test_the_422_message_names_its_one_actionable_cause(
        self, endpoint: Any
    ) -> None:
        """
        The message is what a customer pastes into a support ticket, so it names
        the one cause that is theirs to fix — a view-scoped SDK key — and sends
        every other case to support. Asserted on substance rather than prose, so
        the wording stays free to improve.

        Word-for-word the TypeScript SDK's, which is why it is this short.
        """
        endpoint.queue_poll(status=422)
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
        message = store.failed.lower()
        assert "view-scoped" in message
        assert "support" in message

        # What it must not say. The remaining causes are not a customer's to
        # fix, so they go to support unenumerated rather than being listed at
        # the reader. The other two were simply false — the condition is not
        # about whether any skill exists, and a *process* restart is not what
        # clears it.
        assert "account" not in message
        assert "enabled" not in message
        assert "first skill" not in message
        assert "restart" not in message

    def test_a_store_that_gave_up_on_a_422_resumes_on_start(
        self, endpoint: Any
    ) -> None:
        """The recovery the README documents, asserted rather than claimed.

        A fatal 422 stops the run, not the store: ``_give_up`` does not close
        it, and ``close`` is the only thing ``start`` refuses. So a 422 whose
        cause is fixed while the process runs is recovered by starting this
        store again, and neither the docs nor the message may send a customer
        to restart their service instead.
        ``test_a_restarted_store_does_not_report_the_old_failure`` covers the
        general case; this pins the one the 422's docs promise.
        """
        endpoint.queue_poll(status=422)
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)

            # The key is replaced: nothing about the store has changed.
            endpoint.queue_poll(full_payload(("put-object", put_skill())))
            store.start()
            assert store.wait_for_skills(timeout=5) is True
            assert store.failed is None
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None

    def test_a_fatal_422_is_not_counted_against_the_retry_bound(
        self, endpoint: Any
    ) -> None:
        """
        ``connection_failures`` measures consecutive *recoverable* failures
        against the retry bound. A fatal never retries, so counting one would
        make a store that gave up on its first response indistinguishable from
        one that exhausted its attempts. ``_give_up`` already accounts 401 and
        404 this way, so routing 422 through it makes this free — asserted
        rather than implemented.
        """
        endpoint.queue_poll(status=422)
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
        assert store.diagnostics.connection_failures == 0
        assert store.diagnostics.last_error is not None
        assert store.failed == store.diagnostics.last_error

    def test_a_fatal_422_releases_wait_for_skills_at_once(self, endpoint: Any) -> None:
        """
        As much the point of the classification as the stopped retries are: a
        boot gated on skills would otherwise pay its whole timeout on every
        start, against a store that knew the answer on its first response. The
        assertion is on the value *and* the elapsed time.
        """
        endpoint.queue_poll(status=422)
        with poll_store(endpoint) as store:
            started = time.monotonic()
            assert store.wait_for_skills(timeout=30.0) is False
            elapsed = time.monotonic() - started
        assert elapsed < 5.0, (
            f"waited {elapsed:.1f}s for an answer delivery already had"
        )

    async def test_a_store_that_gave_up_on_a_422_prunes_nothing(
        self, endpoint: Any, tmp_path: Any
    ) -> None:
        """
        "LaunchDarkly will not deliver skills to this connection" and "every
        skill here was revoked" are the two readings of an empty answer, and
        only the second may delete a customer's files. A 422 commits no payload,
        so the readiness probe stays false and the wildcard reconcile withholds
        the prune for the whole run.

        The composition is under test rather than either half: "delivery gave
        up, therefore the store is empty, therefore prune" is the inference an
        implementation makes when it assembles this from two sections. It is
        also the path a filesystem-agent deployment takes against a connection
        skills are never delivered on.
        """
        root = tmp_path / "skills"
        stale = root / "left-behind"
        stale.mkdir(parents=True)
        (stale / "SKILL.md").write_text("not ours to delete", encoding="utf-8")

        endpoint.queue_poll(status=422)
        store = poll_store(endpoint)
        with store:
            assert wait_until(lambda: store.failed is not None)
            assert store.is_initialized() is False
            assert store.wait_for_skills(timeout=0.1) is False
            await init_client(options={"skillStore": store}, client=object())
            report = await write_skills("*", root)
        assert report.ok is False
        assert (stale / "SKILL.md").read_text(encoding="utf-8") == "not ours to delete"
        assert not any(a.action == "removed" for a in report.actions)

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
            started = time.monotonic()
            # Released promptly, and ``False``: no payload arrived, and saying
            # otherwise would send a caller on to read a store holding nothing.
            assert store.wait_for_skills(timeout=5) is False
            assert time.monotonic() - started < 2.0
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

    def test_there_is_no_consecutive_failure_option(self) -> None:
        """A count bound would freeze a give-up contract into the public API.

        Asserted by name, so a reintroduction is caught here rather than in a
        process that stopped receiving revocations after a short outage.
        """
        assert (
            "max_consecutive_failures"
            not in inspect.signature(FDv2SkillStore.__init__).parameters
        )
        with pytest.raises(TypeError):
            FDv2SkillStore(SDK_KEY, max_consecutive_failures=3)  # type: ignore[call-arg]

    @pytest.mark.parametrize("option", ["initial_backoff", "max_backoff"])
    @pytest.mark.parametrize("value", [0, 0.0, -5, math.nan, math.inf])
    def test_a_backoff_option_must_be_positive_and_finite(
        self, option: str, value: float
    ) -> None:
        """With no failure bound, these two numbers are the only limit on the
        retry loop: zero or less reconnects as fast as the network allows."""
        kwargs = {"initial_backoff": 1.0, "max_backoff": 30.0, option: value}
        with pytest.raises(ValueError, match=option):
            FDv2SkillStore(SDK_KEY, **kwargs)

    def test_the_initial_backoff_may_not_exceed_the_cap(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            FDv2SkillStore(SDK_KEY, initial_backoff=5.0, max_backoff=1.0)
        # Equal is a fixed delay, and fine.
        FDv2SkillStore(SDK_KEY, initial_backoff=2.0, max_backoff=2.0)

    def test_wait_for_skills_runs_to_its_timeout_during_an_outage(self) -> None:
        """Recoverable failures no longer end delivery, so the store does not
        know the answer yet, and the wait is not cut short."""
        store = stream_store(_requester=_ScriptedRequester())
        try:
            store.start()
            started = time.monotonic()
            assert store.wait_for_skills(timeout=0.5) is False
            assert time.monotonic() - started >= 0.4
            assert store.failed is None
            assert store.diagnostics.connection_failures > 0
        finally:
            store.close()

    def test_a_400_after_a_recoverable_failure_still_repairs(
        self, endpoint: Any
    ) -> None:
        """The repair still goes out from scratch when an outage came first."""
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=500)
        endpoint.queue_poll(status=400)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert wait_until(lambda: len(endpoint.requests) >= 4)
            assert store.failed is None
        # The 500 left the basis in place; the 400 is what dropped it.
        assert "basis" in endpoint.requests[2]["query"]
        repair = endpoint.requests[3]
        assert repair["query"] == {"kinds": FDV2_PAYLOAD_KIND}
        assert repair["if_none_match"] is None

    def test_the_second_400_still_stops_with_a_recoverable_failure_between(
        self, endpoint: Any
    ) -> None:
        """A recoverable failure between them does not earn the 400 another
        repair: the request after it still carries no client state."""
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=400)
        endpoint.queue_poll(status=500)
        endpoint.queue_poll(status=400)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert wait_until(lambda: store.failed is not None)
        assert "400" in store.failed
        # The fifth queued payload is never asked for.
        assert len(endpoint.requests) == 4

    def test_recoverable_failures_are_retried_indefinitely(self) -> None:
        """Well past ten in a row, the bound this replaces, and still retrying."""
        requester = _ScriptedRequester()
        store = FDv2SkillStore(
            SDK_KEY,
            mode="poll",
            poll_interval=0.01,
            initial_backoff=0.001,
            max_backoff=0.002,
            _requester=requester,
        )
        try:
            store.start()
            assert wait_until(lambda: store.diagnostics.connection_failures >= 25)
            assert store.failed is None
            assert store.diagnostics.last_error is not None
        finally:
            store.close()
        assert len(requester.calls) >= 25

    def test_recycled_stream_connections_are_not_failures(self) -> None:
        # A streaming connection only ever ends by being dropped, so a loop
        # that counted every drop as a failure would back a healthy server off
        # to ``max_backoff`` and report it as failing.
        requester = _RecyclingRequester()
        store = stream_store(_requester=requester)
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

    @pytest.mark.parametrize("farewell", [False, True], ids=["dropped", "goodbye"])
    def test_an_up_to_date_recycled_stream_is_not_a_failure(
        self, farewell: bool
    ) -> None:
        # Resetting at a commit covers only a connection that carried new
        # content. An environment whose skills are not changing answers every
        # reconnect with ``intentCode: "none"`` and transfers nothing, so a loop
        # that counted those drops would back a *healthy* idle stream off to
        # ``max_backoff`` and report it as failing.
        requester = _UpToDateRecyclingRequester(farewell=farewell)
        store = stream_store(_requester=requester)
        try:
            store.start()
            assert wait_until(lambda: requester.connections >= 8)
            assert store.failed is None
            if farewell:
                # A goodbye after an answer is the server recycling the stream,
                # not a failure: it is neither counted nor reported.
                assert store.diagnostics.connection_failures == 0
                assert store.diagnostics.last_error is None
            else:
                # A plain drop counts until the next answer clears it, so the
                # count may read 1 mid-reconnect. What it must never do is climb.
                assert store.diagnostics.connection_failures <= 1
        finally:
            store.close()

    def test_a_goodbye_before_any_answer_is_a_failure(self, caplog: Any) -> None:
        """Only a completed exchange makes a goodbye routine. A server that only
        ever says goodbye has delivered nothing, and must stay visible."""

        class _OnlyGoodbye(_FakeRequester):
            def stream(self, basis: str | None) -> Any:
                return _ScriptedConnection([("goodbye", {"reason": "go away"})])

        store = stream_store(_requester=_OnlyGoodbye())
        with caplog.at_level("DEBUG", logger="launchdarkly_ai_server.skills_fdv2"):
            try:
                store.start()
                assert wait_until(lambda: store.diagnostics.connection_failures >= 3)
            finally:
                store.close()
        assert "goodbye" in (store.diagnostics.last_error or "")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("server said goodbye" in r.getMessage() for r in warnings)

    def test_a_recycled_connection_reconnects_quietly(self, caplog: Any) -> None:
        # A healthy idle stream reconnects for as long as the process runs, so
        # warning on each one would fill a customer's logs with a fault they do
        # not have and teach them to ignore the level that means something.
        requester = _UpToDateRecyclingRequester()
        store = stream_store(_requester=requester)
        with caplog.at_level("DEBUG", logger="launchdarkly_ai_server.skills_fdv2"):
            try:
                store.start()
                assert wait_until(lambda: requester.connections >= 5)
            finally:
                store.close()
        assert store.failed is None
        assert not [r for r in caplog.records if r.levelname == "WARNING"]
        assert [r for r in caplog.records if "reconnecting in" in r.getMessage()]

    def test_a_connection_that_never_answered_still_warns(self, caplog: Any) -> None:
        # The quiet path is earned by answering. A connection that failed before
        # it told us anything is the case the warning exists for.
        store = stream_store(_requester=_ScriptedRequester())
        with caplog.at_level("DEBUG", logger="launchdarkly_ai_server.skills_fdv2"):
            try:
                store.start()
                assert wait_until(lambda: store.diagnostics.connection_failures >= 3)
            finally:
                store.close()
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings
        assert all("Skill delivery failed" in r.getMessage() for r in warnings)

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("timed out"),
            ConnectionResetError(54, "Connection reset by peer"),
            IncompleteRead(b"partial"),
        ],
        ids=["read timeout", "reset", "truncated body"],
    )
    def test_a_stream_that_dies_mid_read_reconnects(self, exc: BaseException) -> None:
        # A stream fails in its body far more often than at its connect, and
        # ``read_timeout`` exists to bound one that has gone quiet. Treating
        # such a failure as unexpected would stop delivery — including
        # revocation — for the process lifetime the first time a socket died.
        requester = _DyingStreamRequester(exc)
        store = stream_store(_requester=requester)
        try:
            store.start()
            assert store.wait_for_skills(timeout=5) is True
            assert wait_until(lambda: requester.connections >= 5)
            assert store.failed is None
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
            _FatalTransportError("401"),
        )
        store = stream_store(_requester=requester)
        try:
            store.start()
            assert store.wait_for_skills(timeout=5)
            assert wait_until(lambda: store.failed is not None)
            # Three failures, then a commit, then the stream's own drop. The
            # count starts again at the commit, so the drop is failure one, not
            # four.
            assert store.diagnostics.connection_failures == 1
            assert len(requester.calls) == 5
        finally:
            store.close()

    def test_an_answer_alone_does_not_reset_the_backoff_delay(
        self, monkeypatch: Any
    ) -> None:
        """A server that answers ``none`` and drops at once is backed off.

        Resetting the delay on every answer would reconnect it about once a
        second, from every process, for as long as it stays degraded. The
        failure count still clears on each answer.
        """
        attempts = _record_backoff_attempts(monkeypatch)
        requester = _UpToDateRecyclingRequester()
        store = stream_store(_requester=requester)
        try:
            store.start()
            assert wait_until(lambda: len(attempts) >= 6)
            assert store.diagnostics.connection_failures <= 1
        finally:
            store.close()
        assert attempts[:6] == [1, 2, 3, 4, 5, 6]

    def test_a_stream_held_past_the_threshold_resets_the_backoff_delay(
        self, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(skills_fdv2, "_BACKOFF_RESET_INTERVAL", 0.05)
        attempts = _record_backoff_attempts(monkeypatch)

        class _HeldThenDropped(_FakeRequester):
            def stream(self, basis: str | None) -> Any:
                def held() -> Any:
                    yield ("server-intent", server_intent("none"))
                    time.sleep(0.1)

                return _ScriptedConnection(held())

        store = stream_store(_requester=_HeldThenDropped())
        try:
            store.start()
            assert wait_until(lambda: len(attempts) >= 3)
        finally:
            store.close()
        assert attempts[:3] == [1, 1, 1]

    def test_a_completed_poll_resets_the_backoff_delay(
        self, endpoint: Any, monkeypatch: Any
    ) -> None:
        """Fail, succeed, fail: the second failure retries at the first step,
        since ``poll_interval`` already spaces the requests."""
        attempts = _record_backoff_attempts(monkeypatch)
        endpoint.queue_poll(status=500)
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(status=500)
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5)
            assert wait_until(lambda: len(attempts) >= 2)
            assert store.diagnostics.connection_failures <= 1
        assert attempts[:2] == [1, 1]

    def test_stream_failures_are_retried_indefinitely(self) -> None:
        """Last known good is served throughout, however long the outage."""
        payload = [
            (e["event"], e["data"]) for e in full_payload(("put-object", put_skill()))
        ]
        store = stream_store(_requester=_ScriptedRequester(payload))
        try:
            store.start()
            assert store.wait_for_skills(timeout=5)
            assert wait_until(lambda: store.diagnostics.connection_failures >= 25)
            assert store.failed is None
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
        finally:
            store.close()

    def test_an_announced_transfer_that_never_completes_is_a_failure(self) -> None:
        """An intent is a promise, not a delivery, so it does not reset the count.

        A server that announces a transfer and drops before
        ``payload-transferred``, every time, has delivered nothing. Counted as
        health, it would be retried forever at the initial backoff.
        """

        class _AnnouncesThenDrops(_FakeRequester):
            def stream(self, basis: str | None) -> Any:
                return _ScriptedConnection(
                    [
                        ("server-intent", server_intent("xfer-full")),
                        ("put-object", put_skill()),
                    ]
                )

        store = stream_store(_requester=_AnnouncesThenDrops())
        try:
            store.start()
            assert wait_until(lambda: store.diagnostics.connection_failures >= 5)
            assert store.failed is None
            assert store.is_initialized() is False
        finally:
            store.close()

    def test_a_retry_after_header_is_honoured(self) -> None:
        requester = _ScriptedRequester(
            _RecoverableTransportError("slow down", retry_after=0.5),
        )
        store = FDv2SkillStore(
            SDK_KEY,
            mode="poll",
            poll_interval=10.0,
            initial_backoff=0.01,
            max_backoff=5.0,
            _requester=requester,
        )
        try:
            started = time.monotonic()
            store.start()
            assert wait_until(lambda: len(requester.calls) >= 2, timeout=5)
            elapsed = time.monotonic() - started
            # The server asked for 0.5s and our own backoff would have been
            # 0.01s, so waiting is the only way the header could have been read.
            # Asked *longer* rather than shorter on purpose: a shorter request
            # is floored at ``initial_backoff``, so it cannot discriminate.
            assert elapsed >= 0.4
        finally:
            store.close()

    def test_a_retry_after_of_zero_still_waits_the_initial_backoff(self) -> None:
        """``Retry-After: 0`` is floored, not taken literally.

        A server — or an intermediate proxy — answering ``0`` would otherwise
        have the loop reconnect as fast as it can schedule, spending the whole
        bounded retry budget in milliseconds and putting needless load on the
        endpoint. The floor is ``initial_backoff``, the same floor this module's
        own backoff starts from.
        """
        requester = _ScriptedRequester(
            _RecoverableTransportError("slow down", retry_after=0.0),
        )
        store = FDv2SkillStore(
            SDK_KEY,
            mode="poll",
            poll_interval=10.0,
            initial_backoff=0.5,
            max_backoff=5.0,
            _requester=requester,
        )
        try:
            started = time.monotonic()
            store.start()
            assert wait_until(lambda: len(requester.calls) >= 2, timeout=5)
            assert time.monotonic() - started >= 0.4
        finally:
            store.close()

    def test_a_retry_after_header_is_parsed_off_the_wire(self, endpoint: Any) -> None:
        endpoint.queue_poll(status=429, retry_after="0.5")
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        started = time.monotonic()
        with poll_store(
            endpoint, initial_backoff=0.01, max_backoff=5.0, poll_interval=10.0
        ) as store:
            assert store.wait_for_skills(timeout=5) is True
        # 0.5s is only obtainable from the header: our own backoff here is 0.01s
        # and the cap is 5s, so neither could have produced this wait.
        assert time.monotonic() - started >= 0.4

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
        assert _backoff_delay(1, base=1.0, maximum=30.0, jitter=0.0) == 1.0
        assert _backoff_delay(2, base=1.0, maximum=30.0, jitter=0.0) == 2.0
        assert _backoff_delay(3, base=1.0, maximum=30.0, jitter=0.0) == 4.0
        assert _backoff_delay(20, base=1.0, maximum=30.0, jitter=0.0) == 30.0

    def test_backoff_stays_finite_at_any_attempt_number(self) -> None:
        """Retries are unbounded, so the attempt number is too.

        Unclamped, ``float(2 ** n)`` raises ``OverflowError`` past about 1024,
        which a long enough outage reaches.
        """
        assert _backoff_delay(10_000, base=1.0, maximum=30.0, jitter=0.0) == 30.0
        delay = _backoff_delay(10_000, base=1.0, maximum=30.0)
        assert math.isfinite(delay)
        assert 0.0 <= delay <= 30.0

    def test_jitter_never_exceeds_the_cap(self) -> None:
        for attempt in range(1, 12):
            for _ in range(50):
                assert 0.0 <= _backoff_delay(attempt, base=1.0, maximum=5.0) <= 5.0

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


# ---------------------------------------------------------------------------
# Redirects are refused
# ---------------------------------------------------------------------------


class _LineSource:
    """A streaming body served from bytes, with the ``readline`` the parser uses."""

    def __init__(self, data: bytes) -> None:
        self._buf = data
        self.closed = False

    def readline(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._buf)
        newline = self._buf.find(b"\n", 0, size)
        end = size if newline < 0 else newline + 1
        line, self._buf = self._buf[:end], self._buf[end:]
        return line

    def close(self) -> None:
        self.closed = True


class TestTransportMemoryBound:
    """
    ``MAX_RESPONSE_BYTES`` bounds what one response may put in memory before
    verification's per-skill content cap can see any of it. Disk was already
    bounded; this is what bounds memory. The cap is patched small here so the
    suite does not have to move 64 MiB to prove it.
    """

    def test_the_bound_is_far_above_any_legitimate_payload(self) -> None:
        assert MAX_RESPONSE_BYTES == 64 * 1024 * 1024

    def test_crossing_the_bound_is_fatal(self) -> None:
        assert issubclass(_ResponseTooLargeError, _FatalTransportError)
        assert not issubclass(_ResponseTooLargeError, _RecoverableTransportError)

    def test_an_over_cap_poll_body_stops_delivery_without_applying_it(
        self, endpoint: Any, monkeypatch: Any
    ) -> None:
        """Fatal, like a 422: the size belongs to the environment, so a retry
        would download it again and be refused the same way."""
        monkeypatch.setattr(skills_fdv2, "MAX_RESPONSE_BYTES", 2048)
        endpoint.queue_poll(full_payload(("put-object", put_skill(content="x" * 8192))))
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
            assert "2048-byte transport bound" in store.failed
            assert "2048-byte transport bound" in (store.diagnostics.last_error or "")
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is None
            assert store.diagnostics.payloads_transferred == 0
            assert store.diagnostics.skill_objects_received == 0
            assert store.diagnostics.connection_failures == 0
            # Fatal means one request, not a retry loop.
            assert len(endpoint.requests) == 1

            # Once the payload is back under the bound, start() resumes.
            endpoint.queue_poll(full_payload(("put-object", put_skill())))
            store.start()
            assert store.wait_for_skills(timeout=5) is True
            assert store.failed is None

    def test_a_poll_body_exactly_at_the_cap_is_accepted(
        self, endpoint: Any, monkeypatch: Any
    ) -> None:
        payload = full_payload(("put-object", put_skill()))
        body = json.dumps({"events": payload}).encode("utf-8")
        monkeypatch.setattr(skills_fdv2, "MAX_RESPONSE_BYTES", len(body))
        endpoint.queue_poll(payload)
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
            assert store.diagnostics.connection_failures == 0
            assert store.diagnostics.last_error is None

    def test_a_poll_body_one_byte_over_the_cap_is_refused(
        self, endpoint: Any, monkeypatch: Any
    ) -> None:
        payload = full_payload(("put-object", put_skill()))
        body = json.dumps({"events": payload}).encode("utf-8")
        monkeypatch.setattr(skills_fdv2, "MAX_RESPONSE_BYTES", len(body) - 1)
        endpoint.queue_poll(payload)
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is None
        assert f"{len(body) - 1}-byte transport bound" in store.failed
        assert f"at least {len(body)} bytes received" in store.failed

    def test_the_default_cap_leaves_ordinary_payloads_alone(
        self, endpoint: Any
    ) -> None:
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert store.wait_for_skills(timeout=5) is True
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
            assert store.diagnostics.connection_failures == 0
            assert store.diagnostics.last_error is None

    def test_an_over_cap_stream_event_abandons_the_payload_in_flight(
        self, endpoint: Any, monkeypatch: Any
    ) -> None:
        """
        The first payload commits. The second starts, then carries an event
        over the cap: delivery stops, the half-received payload is never
        committed, and the committed set is still served.
        """
        monkeypatch.setattr(skills_fdv2, "MAX_RESPONSE_BYTES", 2048)
        endpoint.queue_stream(
            full_payload(("put-object", put_skill()))
            + events(
                ("server-intent", server_intent("xfer-changes")),
                ("put-object", put_skill("oversized", content="x" * 8192)),
                ("payload-transferred", transferred("basis-2")),
            )
        )
        store = FDv2SkillStore(
            SDK_KEY,
            base_uri=endpoint.base_uri,
            mode="stream",
            initial_backoff=0.01,
            max_backoff=0.02,
        )
        try:
            store.start()
            assert store.wait_for_skills(timeout=5) is True
            assert wait_until(lambda: store.failed is not None)
            assert "transport bound" in store.failed
            assert store.get_object(SKILL_OBJECT_KIND, "pdf-extraction") is not None
            assert store.get_object(SKILL_OBJECT_KIND, "oversized") is None
            assert store.diagnostics.payloads_transferred == 1
            assert store.diagnostics.skill_objects_received == 1
            assert len(endpoint.requests) == 1
        finally:
            store.close()

    def test_a_stream_line_that_never_ends_is_refused_and_the_body_closed(
        self, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(skills_fdv2, "MAX_RESPONSE_BYTES", 1024)
        source = _LineSource(b"data: " + b"x" * 4096)
        with pytest.raises(_ResponseTooLargeError, match="1024-byte"):
            list(_iter_sse(source))
        assert source.closed

    def test_an_event_is_measured_across_its_data_lines(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(skills_fdv2, "MAX_RESPONSE_BYTES", 1024)
        lines = b"".join(b"data: " + b"x" * 500 + b"\n" for _ in range(3))
        source = _LineSource(b"event: put-object\n" + lines + b"\n")
        with pytest.raises(_ResponseTooLargeError, match="1024-byte"):
            list(_iter_sse(source))
        assert source.closed

    def test_multi_line_data_under_the_cap_still_decodes(self) -> None:
        source = _LineSource(b'event: put-object\ndata: {"a":\ndata: 1}\n\n')
        assert list(_iter_sse(source)) == [("put-object", {"a": 1})]
        assert source.closed


@pytest.fixture
def second_endpoint() -> Any:
    """A second host, to stand for wherever a ``Location`` header points."""
    server = _FakeFDv2Endpoint()
    yield server
    server.close()


class TestRedirectsAreRefused:
    """
    ``urllib``'s standard redirect handler copies every request header onto the
    redirected request, ``Authorization`` included. The transport's opener
    declines every redirect instead, so a 3xx is a fatal, non-retried failure
    and the SDK key never reaches the host ``Location`` names.
    """

    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    def test_a_poll_redirect_is_fatal_and_not_followed(
        self, endpoint: Any, second_endpoint: Any, status: int
    ) -> None:
        target = second_endpoint.base_uri + "/sdk/poll"
        endpoint.queue_poll(status=status, location=target)
        requester = _Requester(SDK_KEY, endpoint.base_uri, read_timeout=5.0)
        with pytest.raises(_FatalTransportError) as excinfo:
            requester.poll(None, None)
        assert str(status) in str(excinfo.value)
        assert "not followed" in str(excinfo.value)
        assert second_endpoint.requests == []

    def test_a_stream_redirect_is_fatal_and_not_followed(
        self, endpoint: Any, second_endpoint: Any
    ) -> None:
        endpoint.redirect_stream_to = second_endpoint.base_uri + "/sdk/stream"
        requester = _Requester(SDK_KEY, endpoint.base_uri, read_timeout=5.0)
        with pytest.raises(_FatalTransportError) as excinfo:
            requester.stream(None)
        assert "307" in str(excinfo.value)
        assert second_endpoint.requests == []

    def test_a_same_host_redirect_is_refused_too(self, endpoint: Any) -> None:
        """The endpoints do not redirect, so there is nothing legitimate to follow."""
        endpoint.queue_poll(status=302, location=endpoint.base_uri + "/sdk/poll")
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        requester = _Requester(SDK_KEY, endpoint.base_uri, read_timeout=5.0)
        with pytest.raises(_FatalTransportError):
            requester.poll(None, None)
        assert len(endpoint.requests) == 1

    def test_the_sdk_key_never_reaches_the_second_host(
        self, endpoint: Any, second_endpoint: Any
    ) -> None:
        """
        End to end through the store: the redirect stops delivery for good,
        with no retry spent on it, and the second host sees no request at all —
        so no ``Authorization`` header, since that is what following would
        have forwarded.
        """
        endpoint.queue_poll(status=301, location=second_endpoint.base_uri + "/sdk/poll")
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        with poll_store(endpoint) as store:
            assert wait_until(lambda: store.failed is not None)
            assert store.wait_for_skills(timeout=5) is False
        assert store.failed is not None
        assert "301" in store.failed
        assert "never forwarded" in store.failed
        assert store.diagnostics.connection_failures == 0
        assert len(endpoint.requests) == 1
        assert [r["authorization"] for r in second_endpoint.requests] == []

    def test_a_redirect_in_stream_mode_stops_delivery(
        self, endpoint: Any, second_endpoint: Any
    ) -> None:
        endpoint.redirect_stream_to = second_endpoint.base_uri + "/sdk/stream"
        store = FDv2SkillStore(
            SDK_KEY, base_uri=endpoint.base_uri, mode="stream", initial_backoff=0.01
        )
        with store:
            assert wait_until(lambda: store.failed is not None)
        assert store.failed is not None
        assert "307" in store.failed
        assert second_endpoint.requests == []

    def test_a_redirect_with_no_location_is_still_fatal(self, endpoint: Any) -> None:
        endpoint.queue_poll(status=302)
        requester = _Requester(SDK_KEY, endpoint.base_uri, read_timeout=5.0)
        with pytest.raises(_FatalTransportError):
            requester.poll(None, None)

    def test_a_304_is_not_a_redirect(self, endpoint: Any) -> None:
        """The refusal must leave the poll's not-modified path exactly as it was."""
        endpoint.queue_poll(status=304)
        requester = _Requester(SDK_KEY, endpoint.base_uri, read_timeout=5.0)
        result = requester.poll(None, "etag-1")
        assert result.not_modified is True
        assert result.etag == "etag-1"


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

    async def test_a_missed_pin_is_absent_even_beside_a_malformed_sibling(
        self, endpoint: Any
    ) -> None:
        """
        A key can hold a well-formed version and a version-less entry at once —
        a malformed object arrives with no version in its wire key, and is held
        anyway so verification withholds it with a signal.

        A pin that misses is still a plain miss. Answering it with the
        version-less entry would report ``integrity_failure`` for a skill whose
        integrity is not in question, and that is the one reason callers are
        told to fail closed on.
        """
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill(object_version=3)),
                ("put-object", put_skill(object_version=None)),
            )
        )
        with poll_store(endpoint) as store:
            store.wait_for_skills(timeout=5)
            await init_client(options={"skillStore": store}, client=object())

            missed = await get_skill_result("pdf-extraction", version=9)
            assert missed.skill is None
            assert missed.reason == "absent"
            # The well-formed version still resolves, and the malformed sibling
            # is still reachable to be withheld when nothing else answers.
            assert await get_skill("pdf-extraction", version=3) is not None

    async def test_the_payload_version_is_not_resolvable_as_a_skill_version(
        self, endpoint: Any
    ) -> None:
        """
        The end-to-end form of the wire-key/``version`` assertion.

        Asking for the payload version resolves nothing — reported ``absent``,
        because the store answers "I hold no such version" rather than answering
        with the wrong one. The version that *does* resolve is the one after the
        delimiter in the object's wire ``key``.
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


class TestBaseUriScheme:
    """
    Every request carries the SDK key in ``Authorization``, so the base URI is
    ``https://`` only. Plain ``http://`` is allowed to a loopback host and
    nowhere else: that is what this suite's own endpoints listen on, and it
    never leaves the machine.
    """

    def test_a_plain_http_base_uri_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cleartext") as excinfo:
            FDv2SkillStore(SDK_KEY, base_uri="http://sdk.launchdarkly.com")
        assert "https://" in str(excinfo.value)

    def test_a_plain_http_base_uri_is_refused_even_with_a_requester_injected(
        self,
    ) -> None:
        """The check is on the store, not on the socket it happens to open."""
        with pytest.raises(ValueError, match="cleartext"):
            FDv2SkillStore(
                SDK_KEY,
                base_uri="http://relay.internal:8030",
                _requester=_FakeRequester(),
            )

    @pytest.mark.parametrize(
        "base_uri",
        [
            "http://localhost:8030",
            "http://127.0.0.1:8030",
            "http://[::1]:8030",
            "http://LOCALHOST/",
        ],
    )
    def test_plain_http_to_a_loopback_host_is_allowed(self, base_uri: str) -> None:
        assert FDv2SkillStore(SDK_KEY, base_uri=base_uri) is not None

    def test_a_private_address_is_not_loopback(self) -> None:
        """Only the machine itself is exempt; the LAN is not."""
        with pytest.raises(ValueError, match="cleartext"):
            FDv2SkillStore(SDK_KEY, base_uri="http://10.0.0.5:8030")

    @pytest.mark.parametrize(
        "base_uri",
        ["", "   ", "sdk.launchdarkly.com", "ftp://sdk.launchdarkly.com", "https://"],
    )
    def test_anything_but_an_https_url_with_a_host_is_refused(
        self, base_uri: str
    ) -> None:
        with pytest.raises(ValueError, match="https://"):
            FDv2SkillStore(SDK_KEY, base_uri=base_uri)

    def test_https_is_accepted(self) -> None:
        assert FDv2SkillStore(SDK_KEY, base_uri="https://sdk.example.com/") is not None
        assert FDv2SkillStore(SDK_KEY) is not None

    def test_a_plain_http_stream_uri_is_refused_by_name(self) -> None:
        """The streaming host is checked too, and the message names it."""
        with pytest.raises(ValueError, match="cleartext") as excinfo:
            FDv2SkillStore(
                SDK_KEY,
                base_uri="https://sdk.example.com",
                stream_uri="http://stream.example.com",
            )
        assert "stream_uri" in str(excinfo.value)


class TestStreamHostDefaults:
    """
    LaunchDarkly serves ``/sdk/stream`` from a different host than ``/sdk/poll``,
    and ``mode="stream"`` is the default — so a single-host default would have
    the *default* configuration talk to the wrong host on first contact with a
    real environment. Both base server-side SDKs ship the hosts as a pair
    (``ldclient.Config``'s ``stream_uri``, ``js-server-sdk-common``'s
    ``streamUri``), which is what these defaults follow.
    """

    @staticmethod
    def _origins(store: FDv2SkillStore) -> tuple[str, str]:
        requester = store._requester
        return requester._base_uri, requester._stream_uri

    def test_the_two_defaults_are_different_hosts(self) -> None:
        assert DEFAULT_BASE_URI == "https://sdk.launchdarkly.com"
        assert DEFAULT_STREAM_URI == "https://stream.launchdarkly.com"
        assert self._origins(FDv2SkillStore(SDK_KEY)) == (
            DEFAULT_BASE_URI,
            DEFAULT_STREAM_URI,
        )

    def test_a_base_uri_alone_serves_both_endpoints(self) -> None:
        # A relay or a private instance serving both from one host needs one
        # option, not two.
        assert self._origins(
            FDv2SkillStore(SDK_KEY, base_uri="https://relay.example.com")
        ) == ("https://relay.example.com", "https://relay.example.com")

    def test_naming_both_overrides_them_independently(self) -> None:
        assert self._origins(
            FDv2SkillStore(
                SDK_KEY,
                base_uri="https://sdk.example.com/",
                stream_uri="https://stream.example.com/",
            )
        ) == ("https://sdk.example.com", "https://stream.example.com")

    def test_a_stream_uri_alone_leaves_polling_on_its_default(self) -> None:
        assert self._origins(
            FDv2SkillStore(SDK_KEY, stream_uri="https://stream.example.com")
        ) == (DEFAULT_BASE_URI, "https://stream.example.com")

    def test_the_stream_request_goes_to_the_stream_host(
        self, endpoint: Any, second_endpoint: Any
    ) -> None:
        """The split is on the wire, not only in the attributes.

        Polling at one host and streaming at another is the whole point, so
        assert it where it is observable: the streaming request arrives at the
        streaming endpoint and nothing arrives at the polling one.
        """
        requester = _Requester(
            SDK_KEY,
            endpoint.base_uri,
            read_timeout=5.0,
            stream_uri=second_endpoint.base_uri,
        )
        requester.stream(None).close()
        assert [r["path"] for r in second_endpoint.requests] == ["/sdk/stream"]
        assert endpoint.requests == []


# ---------------------------------------------------------------------------
# The eager re-reconcile, end to end over the transport
# ---------------------------------------------------------------------------


class TestWatchSkillsOverTheTransport:
    """
    ``watch_skills`` against a live ``FDv2SkillStore``. The watcher's own
    behaviour — debounce, refusal of a store without ``add_listener``, detaching
    on close — is covered in ``test_skills_watch.py`` against the in-memory
    store; these are the cases that only mean something with a transport
    underneath: a wire-level revocation, a new skill version, and an outage.
    """

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

    async def test_a_full_transfer_that_omits_every_skill_prunes(
        self, endpoint: Any, tmp_path: Any
    ) -> None:
        """
        The environment's last skill revoked. The full transfer that follows
        carries nothing at all, so the only thing that can wake the watcher is
        the revocation the transfer states by omission.
        """
        endpoint.queue_poll(full_payload(("put-object", put_skill())))
        endpoint.queue_poll(full_payload(state="basis-2"))
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
                assert store.diagnostics.objects_revoked == 1
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


# ---------------------------------------------------------------------------
# Listener registration
# ---------------------------------------------------------------------------


class TestListenerRegistration:
    @staticmethod
    def _skill_listeners(store: Any) -> list[Any]:
        return list(store._listeners.get(SKILL_OBJECT_KIND, []))

    def test_fdv2_add_listener_for_a_non_skill_kind_raises(self, endpoint: Any) -> None:
        """The transport refuses the same registration the in-memory store does.

        Only skill objects are ever delivered here, so a listener on any other
        kind would never fire — and a store that accepted it has promised
        something it cannot keep.
        """
        with poll_store(endpoint) as store:
            with pytest.raises(ValueError, match="would never fire") as excinfo:
                store.add_listener("flag", print)
            assert SKILL_OBJECT_KIND in str(excinfo.value)
            assert store._listeners == {}

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

    def test_a_closed_store_does_not_restart(self, endpoint: Any) -> None:
        """``close`` is final, and a restart raises rather than resuming.

        Finality is what gives ``close`` a postcondition a caller can rely on —
        delivery has stopped — including when the join timed out. A store that
        could be restarted from there leaves the caller unable to tell whether
        delivery stopped, and a restart that silently never delivered again is
        the failure this forecloses. To resume, construct a new store.
        """
        store = poll_store(endpoint)
        store.start()
        store.close()
        with pytest.raises(RuntimeError, match="close\\(\\) is final") as excinfo:
            store.start()
        # The remedy is in the message, not only in the docs.
        assert "Construct a new FDv2SkillStore" in str(excinfo.value)

    def test_a_store_closed_before_it_started_also_refuses_to_start(
        self, endpoint: Any
    ) -> None:
        store = poll_store(endpoint)
        store.close()
        with pytest.raises(RuntimeError, match="close\\(\\) is final"):
            store.start()

    def test_reentering_a_closed_store_as_a_context_manager_raises(
        self, endpoint: Any
    ) -> None:
        # ``__enter__`` is ``start``, so finality reaches the ``with`` form too.
        store = poll_store(endpoint)
        with store:
            pass
        with pytest.raises(RuntimeError, match="close\\(\\) is final"):
            with store:
                pass

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

    def test_the_store_satisfies_the_interface_before_it_starts(self) -> None:
        store = FDv2SkillStore(SDK_KEY)
        assert store.get_object(SKILL_OBJECT_KIND, "anything") is None
        assert store.all_objects(SKILL_OBJECT_KIND) == {}


# ---------------------------------------------------------------------------
# Layering and dependencies
# ---------------------------------------------------------------------------


def _module_imports(module: Any) -> set[str]:
    """Every top-level package or module *module* imports, by root name."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # A relative import is a sibling module in this package, and is
            # spelled with its leading dot so the two kinds cannot be confused.
            prefix = "." if node.level else ""
            found.add(f"{prefix}{node.module.split('.')[0]}")
    return found


class TestTransportLayering:
    """The transport sits below the store interface and stays there.

    It is the only module in the feature that opens a socket, and it earns that
    by depending on almost nothing: the standard library, plus the two modules
    holding the interface kind and the version predicate. Nothing in the feature
    depends on it in turn, so the layering cannot invert — a module above the
    interface must not be able to tell which store produced an object.
    """

    def test_it_adds_no_dependency(self) -> None:
        """Standard library only, in a package whose one runtime dependency is
        the OpenTelemetry API.

        Asserted as an allowlist rather than by denying a list of package names:
        a denylist cannot catch the dependency nobody thought of, which is the
        only kind that gets added.
        """
        stdlib = {
            "__future__",
            "collections",
            "dataclasses",
            "json",
            "logging",
            "math",
            "random",
            "re",
            "socket",
            "threading",
            "time",
            "typing",
            "urllib",
        }
        within_the_feature = {".skills_core", ".types_validation"}

        assert _module_imports(skills_fdv2) == stdlib | within_the_feature

    def test_nothing_in_the_feature_imports_it(self) -> None:
        """Only the package's own entry point names it, to re-export two types.

        That re-export is the published surface rather than a dependency: it is
        what makes ``FDv2SkillStore`` reachable without a sub-path import. A
        *feature* module importing the transport would be the layering
        inverting, and the accessors would start being able to tell which store
        answered them.
        """
        feature = [
            "skills.py",
            "skills_core.py",
            "skills_fs.py",
            "skills_watch.py",
            "safe_fs.py",
            "types.py",
            "types_validation.py",
            "lifecycle.py",
        ]
        source_dir = Path(inspect.getfile(skills_fdv2)).parent
        importers = [
            name
            for name in feature
            if "skills_fdv2" in (source_dir / name).read_text(encoding="utf-8")
        ]

        assert importers == []
        # The positive control: the entry point does name it, so an empty result
        # above cannot be a misspelt filename list finding nothing anywhere.
        assert "skills_fdv2" in (source_dir / "__init__.py").read_text(encoding="utf-8")


class TestTransportEmitsNoTelemetry:
    """The three skills signals belong to verification and materialization.

    The transport delivers bytes and counts what it saw; it decides nothing that
    a signal reports. Its own observability is ``diagnostics`` and ``failed``,
    which are assertable facts rather than telemetry a customer may have turned
    off.
    """

    def test_a_full_delivery_cycle_records_nothing(
        self, endpoint: Any, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        # One of everything the transport can see in a payload: a skill, a
        # revocation, an object of a kind it did not ask for, and a skill with no
        # ``contentHash`` — the last two being the cases it counts and warns
        # about, which are the likeliest places for a signal to be reached for.
        endpoint.queue_poll(
            full_payload(
                ("put-object", put_skill()),
                ("put-object", put_skill("to-revoke", object_version=1)),
                ("put-object", put_skill("hashless", object_version=1, omit_hash=True)),
                ("put-object", put_flag()),
                ("delete-object", delete_skill("to-revoke", object_version=1)),
            )
        )
        store = poll_store(endpoint)
        try:
            store.start()
            assert store.wait_for_skills(timeout=5)
            assert wait_until(lambda: store.diagnostics.hashless_objects >= 1)
        finally:
            store.close()

        assert store.diagnostics.objects_revoked == 1
        assert store.diagnostics.objects_ignored == 1
        assert recording_emitter.records == []

    def test_a_failed_delivery_records_nothing_either(
        self, recording_emitter: Any
    ) -> None:
        """Giving up is reported on ``failed``, not as a signal.

        The failure path is the other place a signal would plausibly be added,
        and it is reached without any payload ever committing — so a test of the
        success path alone would not cover it.
        """
        skills_module._set_emitter_for_testing(recording_emitter)
        store = stream_store(_requester=_ScriptedRequester(_FatalTransportError("401")))
        try:
            store.start()
            assert wait_until(lambda: store.failed is not None)
        finally:
            store.close()

        assert recording_emitter.records == []


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


class _StalledBody:
    """
    A listening socket that answers with headers and then stalls the body.

    Distinct from ``_BlackHole``: here the request succeeds far enough to hand
    urllib a response, and the caller then parks in ``read``. That is the state
    ``close`` has to interrupt — and, unlike a request still inside its connect,
    the state an interrupt can actually reach.
    """

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._accepted: list[socket.socket] = []
        self.serving = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()
        host, port = self._listener.getsockname()
        self.base_uri = f"http://{host}:{port}"

    def _serve_forever(self) -> None:
        self._listener.settimeout(0.05)
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                continue
            self._accepted.append(conn)
            try:
                conn.recv(4096)
                # A length far longer than the body that follows, so the read
                # blocks rather than seeing the end of the message.
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 4096\r\n\r\n"
                )
            except OSError:
                continue
            self.serving.set()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        for conn in self._accepted:
            conn.close()
        self._listener.close()


@pytest.fixture
def stalled_body() -> Any:
    server = _StalledBody()
    yield server
    server.close()


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
        requester = _Requester(SDK_KEY, black_hole.base_uri, read_timeout=0.3)
        started = time.monotonic()
        with pytest.raises(_RecoverableTransportError) as excinfo:
            requester.poll(None, None)
        elapsed = time.monotonic() - started
        assert 0.2 <= elapsed < 2.0
        assert "timed out" in str(excinfo.value)

    def test_a_stream_against_an_unresponsive_host_fails_within_read_timeout(
        self, black_hole: Any
    ) -> None:
        requester = _Requester(SDK_KEY, black_hole.base_uri, read_timeout=0.3)
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

    @pytest.mark.parametrize(
        "value",
        [0.0, -1.0, float("inf"), float("nan")],
        ids=["zero", "negative", "inf", "nan"],
    )
    def test_a_non_positive_poll_interval_is_rejected(self, value: float) -> None:
        """The same rule as ``read_timeout`` above, and for a sharper reason.

        ``poll_interval`` is the argument to the wait between polls, so the two
        non-finite values do not merely slip through a ``<= 0`` guard — they
        mean opposite things once they arrive. ``Event.wait(nan)`` returns at
        once, so the loop polls as fast as the endpoint will answer;
        ``Event.wait(inf)`` never returns, so the store polls once and then
        never again while reporting itself healthy.
        """
        with pytest.raises(ValueError, match="poll_interval"):
            FDv2SkillStore(SDK_KEY, mode="poll", poll_interval=value)

    def test_a_finite_positive_poll_interval_is_accepted(self) -> None:
        """Positive control, so the guard above cannot reject every value."""
        store = FDv2SkillStore(SDK_KEY, mode="poll", poll_interval=0.5)
        assert store.failed is None

    def test_there_is_no_separate_connect_timeout(self) -> None:
        # ``urllib`` cannot bound the connect separately from the reads, so the
        # constructor does not offer a parameter that would only pretend to.
        with pytest.raises(TypeError):
            FDv2SkillStore(SDK_KEY, connect_timeout=2.0)  # type: ignore[call-arg]


class TestWaitingForSkills:
    """
    ``wait_for_skills`` answers with what happened, and never outlives it.

    Its budget is a boot-ordering allowance, not a delay to spend: a store that
    already knows no payload is coming owes the caller that answer immediately.
    """

    def test_close_releases_a_waiter_rather_than_leaving_it_parked(self) -> None:
        # A shutdown racing a waiter is the ordinary case, not an exotic one:
        # ``close`` on the main thread while a worker is still waiting for its
        # first payload. Parking that worker for the rest of its timeout adds
        # the whole budget to a process that has already decided to stop.
        store = stream_store(_requester=_SilentStreamRequester())
        store.start()
        answers: list[bool] = []
        waiter = threading.Thread(
            target=lambda: answers.append(store.wait_for_skills(timeout=10)),
            daemon=True,
        )
        waiter.start()
        time.sleep(0.2)
        started = time.monotonic()
        store.close()
        waiter.join(timeout=5)
        assert not waiter.is_alive()
        assert time.monotonic() - started < 2.0
        assert answers == [False]

    def test_is_initialized_tracks_the_first_payload(self) -> None:
        """The probe ``write_skills("*")`` reads to decide whether it may prune.

        Before the first payload, an empty store and an environment with no
        skills are the same answer through ``all_objects``; this is what tells
        them apart.
        """
        store = stream_store(_requester=_SilentStreamRequester())
        try:
            assert store.is_initialized() is False
            store.start()
            assert store.wait_for_skills(timeout=0.2) is False
            assert store.is_initialized() is False
        finally:
            store.close()

        delivering = stream_store(_requester=_RecyclingRequester())
        try:
            delivering.start()
            assert delivering.wait_for_skills(timeout=5) is True
            assert delivering.is_initialized() is True
        finally:
            delivering.close()
        # Content outlives the connection, so the fact about it does too.
        assert delivering.is_initialized() is True

    def test_a_payload_already_held_still_answers_true_after_close(self) -> None:
        # ``close`` does not drop content, so it must not turn the answer about
        # that content into a lie either.
        requester = _RecyclingRequester()
        store = stream_store(_requester=requester)
        try:
            store.start()
            assert store.wait_for_skills(timeout=5) is True
        finally:
            store.close()
        assert store.wait_for_skills(timeout=5) is True

    def test_a_store_restarted_after_giving_up_waits_again(self) -> None:
        # The released flag is sticky by design, so a store that gave up before
        # any payload and is then started again has to re-arm: otherwise the
        # next waiter is let go before delivery has had a chance to begin.
        # Restarting after a *close* is not available — see
        # ``TestCloseIsFinal`` — so the give-up path is what exercises this.
        class _FailsThenGoesQuiet(_FakeRequester):
            """One fatal failure; silent on every run after."""

            def __init__(self) -> None:
                self.attempts = 0

            def stream(self, basis: str | None) -> Any:
                self.attempts += 1
                if self.attempts == 1:
                    raise _FatalTransportError("x")
                return _BlockingConnection()

        store = stream_store(_requester=_FailsThenGoesQuiet())
        store.start()
        assert wait_until(lambda: store.failed is not None)
        assert store.wait_for_skills(timeout=0.1) is False

        store.start()
        try:
            started = time.monotonic()
            assert store.wait_for_skills(timeout=0.5) is False
            # Waited, rather than being released by the previous give-up.
            assert time.monotonic() - started >= 0.4
        finally:
            store.close()


class TestPollShutdown:
    """
    ``close`` has to interrupt a poll in flight, as it already does a stream.

    Without it the delivery thread stays parked in its request and ``close``
    returns only when the join times out — on a 300s-class request, long after
    the process meant to exit. The bound is loose on purpose: the point is
    promptly rather than a particular number of milliseconds.
    """

    def test_interrupt_unblocks_a_poll_stalled_in_its_body(
        self, stalled_body: Any
    ) -> None:
        requester = _Requester(SDK_KEY, stalled_body.base_uri, read_timeout=30.0)
        raised: list[BaseException] = []

        def poll_until_interrupted() -> None:
            try:
                requester.poll(None, None)
            except BaseException as exc:
                raised.append(exc)

        thread = threading.Thread(target=poll_until_interrupted, daemon=True)
        thread.start()
        assert stalled_body.serving.wait(timeout=5)
        # The response is in hand; give the read a moment to park in it.
        time.sleep(0.2)
        started = time.monotonic()
        requester.interrupt()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert time.monotonic() - started < 2.0
        assert raised and isinstance(raised[0], _RecoverableTransportError)

    def test_close_during_a_stalled_poll_returns_promptly(
        self, stalled_body: Any
    ) -> None:
        store = FDv2SkillStore(
            SDK_KEY,
            base_uri=stalled_body.base_uri,
            mode="poll",
            poll_interval=0.05,
            read_timeout=30.0,
        )
        store.start()
        assert stalled_body.serving.wait(timeout=5)
        time.sleep(0.2)
        started = time.monotonic()
        store.close(timeout=5.0)
        assert time.monotonic() - started < 2.0
        assert store._thread is not None and not store._thread.is_alive()

    def test_a_poll_we_interrupted_is_not_a_delivery_failure(
        self, stalled_body: Any
    ) -> None:
        # Our own shutdown is not an outage: counting it would spend a retry
        # from the bounded budget and leave a misleading ``last_error`` behind
        # on a store whose content is still perfectly good.
        store = FDv2SkillStore(
            SDK_KEY,
            base_uri=stalled_body.base_uri,
            mode="poll",
            poll_interval=0.05,
            read_timeout=30.0,
        )
        store.start()
        assert stalled_body.serving.wait(timeout=5)
        time.sleep(0.2)
        store.close(timeout=5.0)
        assert store.diagnostics.connection_failures == 0
        assert store.diagnostics.last_error is None
        assert store.failed is None

    def test_a_close_that_timed_out_is_still_final(self) -> None:
        # A request blocked inside its connect is beyond any interrupt, so
        # ``close`` can still return with the thread alive. This is the case
        # finality exists for: the caller cannot tell whether delivery stopped,
        # and a ``start`` that adopted the dying thread would leave a store
        # reporting itself started and never delivering. Raising says so.
        requester = _SlowPollRequester()
        store = FDv2SkillStore(
            SDK_KEY, mode="poll", poll_interval=0.01, _requester=requester
        )
        try:
            store.start()
            assert requester.entered.wait(timeout=5)
            store.close(timeout=0.2)
            assert store._thread is not None and store._thread.is_alive()
            with pytest.raises(RuntimeError, match="close\\(\\) is final"):
                store.start()
        finally:
            requester.release.set()
            store.close(timeout=2)
