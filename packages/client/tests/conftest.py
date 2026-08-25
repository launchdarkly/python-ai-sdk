import hashlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
import launchdarkly_ai_server.skills as skills_module
from launchdarkly_ai_server import InMemorySkillStore


@pytest.fixture
def mock_ld_client() -> MagicMock:
    """Stub LaunchDarkly client with variation/track/flush/close."""
    client = MagicMock()
    client.variation = AsyncMock(return_value=None)
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_span() -> MagicMock:
    """OTel span stub with add_event/set_status/end/record_exception spies."""
    span = MagicMock()
    span.add_event = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.end = MagicMock()
    span.record_exception = MagicMock()
    return span


@pytest.fixture
def mock_tracer(mock_span: MagicMock) -> MagicMock:
    """OTel tracer that returns mock_span from start_as_current_span and start_span."""
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=mock_span
    )
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    tracer.start_span.return_value = mock_span
    return tracer


# ---------------------------------------------------------------------------
# Agent Skills helpers
#
# Exposed as fixtures rather than importable module-level helpers: pytest runs
# with --import-mode=importlib and the tests directory is not a package, so
# sibling imports from conftest are not reliable.
# ---------------------------------------------------------------------------

SKILL_BODY = "---\nname: Test Skill\n---\nDo the thing.\n"


class _RecordingEmitter:
    """Telemetry seam double — records (signal, properties) pairs."""

    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def record(self, signal: str, properties: dict[str, Any]) -> None:
        self.records.append((signal, properties))

    def signals(self, name: str) -> list[dict[str, Any]]:
        return [props for sig, props in self.records if sig == name]


class _ThrowingEmitter:
    """Telemetry seam double whose record() always raises."""

    def record(self, signal: str, properties: dict[str, Any]) -> None:
        raise RuntimeError("emitter exploded")


@pytest.fixture
def make_raw_skill() -> Any:
    """Factory for wire-shaped raw store objects with a correct contentHash."""

    def _make(
        key: str = "test-skill",
        version: int = 1,
        content: str = SKILL_BODY,
        **overrides: Any,
    ) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "key": key,
            "version": version,
            "content": content,
            "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "name": "Test Skill",
            "description": "A skill used in tests.",
        }
        obj.update(overrides)
        return obj

    return _make


@pytest.fixture
def store() -> InMemorySkillStore:
    """An in-memory store, wired in as the configured store for the test."""
    s = InMemorySkillStore()
    skills_module._set_store(s)
    return s


class _ExplodingStore:
    """Store double whose every read raises — the "transport is down" case."""

    def get_object(self, kind: str, key: str) -> dict[str, Any] | None:
        raise RuntimeError("transport failure")

    def all_objects(self, kind: str) -> dict[str, dict[str, Any]]:
        raise RuntimeError("transport failure")


@pytest.fixture
def exploding_store() -> _ExplodingStore:
    """A raising store, wired in as the configured store for the test."""
    s = _ExplodingStore()
    skills_module._set_store(s)
    return s


@pytest.fixture
def reset_skill_state() -> Iterator[None]:
    """Clears client, store, and emitter module state around one test.

    Opted into per module with ``pytestmark = pytest.mark.usefixtures(...)``
    rather than being autouse here: autouse would newly reset lifecycle state
    for every test in every module in this directory, which is a behaviour change
    well outside the skills tests.
    """
    lifecycle_module._reset_for_testing()
    yield
    lifecycle_module._reset_for_testing()


@pytest.fixture
def recording_emitter() -> _RecordingEmitter:
    return _RecordingEmitter()


@pytest.fixture
def throwing_emitter() -> _ThrowingEmitter:
    return _ThrowingEmitter()
