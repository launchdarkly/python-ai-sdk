"""
Tests for ``watch_skills`` / ``SkillWatcher`` — the eager re-reconcile.

The watcher is wired to the ``SkillStore`` interface, not to any one transport:
it needs a store that implements ``add_listener``, and nothing more. These tests
therefore drive it from ``InMemorySkillStore``, whose ``put`` notifies its
listeners synchronously, and from small hand-written store doubles.

Every test writes only inside pytest's ``tmp_path``. The watcher runs a real
worker thread, so tests wait on observable outcomes rather than on fixed sleeps
wherever the outcome is something that *does* happen; a fixed sleep is used only
to assert that something does *not*.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from launchdarkly_ai_server import InMemorySkillStore, init_client, watch_skills
from launchdarkly_ai_server.skills_core import SKILL_OBJECT_KIND

pytestmark = pytest.mark.usefixtures("reset_skill_state")


def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# Starting a watch, and what it refuses
# ---------------------------------------------------------------------------


class TestWatchSkills:
    async def test_the_in_memory_store_can_also_drive_a_watch(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        """The watcher is wired to the ``SkillStore`` interface, not to the FDv2
        store."""
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a", version=1, content="body"))
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
        try:
            written = tmp_path / "s" / "a" / "SKILL.md"
            assert written.read_text() == "body"
            store.put(make_raw_skill(key="a", version=2, content="new body"))
            assert wait_until(lambda: written.read_text() == "new body", timeout=10)
        finally:
            watcher.close()

    async def test_a_burst_of_changes_coalesces_into_few_reconciles(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.1)
        try:
            for i in range(12):
                store.put(make_raw_skill(key=f"skill-{i}"))
            time.sleep(0.5)
            # Twelve objects put back to back fire twelve listener calls; without
            # coalescing that is twelve reconciles of one root.
            assert watcher.reconciles <= 2
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


# ---------------------------------------------------------------------------
# Changes that land while the initial reconcile is running
# ---------------------------------------------------------------------------


class TestChangesDuringTheInitialReconcile:
    """The listener attaches before the initial reconcile, so a change delivered
    while that reconcile is still running is acted on rather than lost."""

    async def test_a_revocation_landing_mid_reconcile_is_not_missed(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        class RevokesAfterSnapshot(InMemorySkillStore):
            """Revokes everything the moment the reconcile has taken its
            snapshot — where a ``delete-object`` lands when it arrives a fraction
            of a second into startup, with nothing after it."""

            def __init__(self) -> None:
                super().__init__()
                self.snapshots = 0

            def all_objects(self, kind: str) -> dict[str, dict[str, Any]]:
                objects = super().all_objects(kind)
                self.snapshots += 1
                if self.snapshots == 1:
                    self._versions.clear()
                    self._loose.clear()
                    for listener in self._listeners.get(SKILL_OBJECT_KIND, []):
                        listener({"key": "pdf-extraction"})
                return objects

        store = RevokesAfterSnapshot()
        store.put(make_raw_skill(key="pdf-extraction", version=1, content="body"))
        await init_client(options={"skillStore": store}, client=object())

        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
        try:
            written = tmp_path / "s" / "pdf-extraction" / "SKILL.md"
            # The initial reconcile wrote what its snapshot held, so the file is
            # on disk and the revocation that followed it is the only change
            # left to act on.
            assert written.read_text() == "body"
            assert wait_until(lambda: not written.exists(), timeout=10)
        finally:
            watcher.close()

    async def test_a_failed_initial_reconcile_leaves_no_listener_behind(
        self, tmp_path: Any
    ) -> None:
        """Registering first means a reconcile that raises has to detach: the
        caller is handed an exception, not a watcher to close."""
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        not_a_directory = tmp_path / "file"
        not_a_directory.write_text("")

        with pytest.raises(ValueError, match="not a directory"):
            await watch_skills("*", not_a_directory)

        assert store._listeners.get(SKILL_OBJECT_KIND, []) == []


# ---------------------------------------------------------------------------
# Closing a watch
# ---------------------------------------------------------------------------


class TestWatcherDetachesOnClose:
    """``SkillWatcher.close`` unregisters ``notify``, so a closed watcher is
    neither called nor kept alive by the store."""

    @staticmethod
    def _skill_listeners(store: Any) -> list[Any]:
        return list(store._listeners.get(SKILL_OBJECT_KIND, []))

    async def test_a_closed_watcher_is_no_longer_notified(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="pdf-extraction", version=1, content="first"))
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
        assert watcher.notify in self._skill_listeners(store)

        watcher.close()

        assert watcher.notify not in self._skill_listeners(store)
        store.put(make_raw_skill(key="pdf-extraction", version=4, content="second"))
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
