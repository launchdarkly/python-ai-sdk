"""
Tests for ``watch_skills`` / ``SkillWatcher`` — the eager re-reconcile.

The watcher is wired to the ``SkillStore`` interface, not to any one transport:
it needs a store that implements ``add_listener``, and nothing more. These tests
therefore drive it from ``InMemorySkillStore``, whose ``put`` notifies its
listeners synchronously, and from small hand-written store doubles. The
end-to-end path — a ``delete-object`` arriving over a live FDv2 connection and
pruning a skill's files — is exercised in ``test_skills_fdv2.py``, where the fake
endpoint lives.

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

    async def test_a_burst_of_changes_coalesces_into_a_debounce_window(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        """Six notifications spread across one window produce two reconciles.

        Three things are needed for this to discriminate rather than pass
        vacuously, and this test does all three:

        * The puts happen **after** ``watch_skills`` has attached its listener.
          A payload committed before that reaches nobody, the counter stays at
          zero, and an upper bound then passes against an implementation with
          the debouncing deleted.
        * The counter is asserted to have **moved**, and against an exact
          figure. A run in which nothing was ever notified is not a test of
          coalescing.
        * The notifications are **spread over time**. A burst arriving inside
          one synchronous pass of the listener collapses to a single reconcile
          whether or not the debounce exists, because the worker had not woken
          yet — so a payload of twelve objects put back to back proves nothing.
          Each put here lands in its own scheduler pass.

        **Two, not one.** This watcher clears its wake flag before the window
        rather than after, so a change arriving *inside* the window schedules a
        further pass instead of being merged into the reconcile that is about to
        run: six puts across one window are one reconcile for the window plus
        one for the spill. Six separate reconciles is what the debounce
        prevents, and is what this figure discriminates against.
        """
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.6)
        try:
            assert watcher.reconciles == 0
            for i in range(6):
                store.put(make_raw_skill(key=f"skill-{i}"))
                time.sleep(0.1)
            assert wait_until(lambda: watcher.reconciles >= 2, timeout=10)
            # Settle well past two windows, so a third would have landed by now.
            time.sleep(1.5)
            assert watcher.reconciles == 2
        finally:
            watcher.close()

    async def test_changes_spread_beyond_the_window_do_not_coalesce(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        """The control for the test above: the window is what merges them.

        Without it the two tests could both be satisfied by a watcher that
        reconciles once and then never again.
        """
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        _report, watcher = await watch_skills("*", tmp_path / "s", debounce=0.05)
        try:
            store.put(make_raw_skill(key="one"))
            assert wait_until(lambda: watcher.reconciles >= 1, timeout=10)
            store.put(make_raw_skill(key="two"))
            assert wait_until(lambda: watcher.reconciles >= 2, timeout=10)
        finally:
            watcher.close()

    @pytest.mark.parametrize(
        "debounce", [-1.0, -0.001, float("nan"), float("inf"), float("-inf")]
    )
    async def test_a_negative_or_non_finite_debounce_raises(
        self, tmp_path: Any, debounce: float
    ) -> None:
        """``NaN`` is the case a bare ``< 0`` guard misses.

        ``nan < 0`` is ``False``, so it would pass validation and then collapse
        the window to nothing — ``Event.wait(nan)`` returns immediately, so
        every delivered object reconciles on its own with no coalescing at all.
        ``write_skills`` already guards its ``timeout`` this way.
        """
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        with pytest.raises(ValueError, match="debounce"):
            await watch_skills("*", tmp_path / "s", debounce=debounce)
        # Refused before anything was attached, so there is no watcher to close.
        assert store._listeners.get(SKILL_OBJECT_KIND, []) == []

    async def test_on_reconcile_receives_each_subsequent_report(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        """Not the initial report: that one is returned to the caller directly."""
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a", version=1, content="body"))
        await init_client(options={"skillStore": store}, client=object())
        seen: list[Any] = []
        report, watcher = await watch_skills(
            "*", tmp_path / "s", debounce=0.05, on_reconcile=seen.append
        )
        try:
            assert seen == []
            store.put(make_raw_skill(key="a", version=2, content="new body"))
            assert wait_until(lambda: len(seen) >= 1, timeout=10)
            assert seen[0] is not report
            assert seen[0].ok is True
            assert [a.key for a in seen[0].actions] == ["a"]
        finally:
            watcher.close()

    async def test_a_reconcile_that_raises_does_not_kill_the_watcher(
        self, tmp_path: Any, make_raw_skill: Any, caplog: Any
    ) -> None:
        """A watcher that died on one bad run would silently stop pruning."""
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        calls: list[int] = []

        def explodes_once(_report: Any) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")

        _report, watcher = await watch_skills(
            "*", tmp_path / "s", debounce=0.05, on_reconcile=explodes_once
        )
        try:
            with caplog.at_level("ERROR"):
                store.put(make_raw_skill(key="a", version=1, content="one"))
                assert wait_until(lambda: len(calls) >= 1, timeout=10)
                store.put(make_raw_skill(key="a", version=2, content="two"))
                assert wait_until(lambda: len(calls) >= 2, timeout=10)
            written = tmp_path / "s" / "a" / "SKILL.md"
            assert wait_until(lambda: written.read_text() == "two", timeout=10)
        finally:
            watcher.close()

    async def test_every_write_skills_option_is_passed_straight_through(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        """``prune``, ``timeout`` and ``on_unavailable`` are not reinterpreted.

        Driven through one of ``write_skills``' own cases: a store that cannot
        answer, with ``on_unavailable="raise"``, raises out of the initial
        reconcile exactly as it would out of ``write_skills``.
        """

        class Unavailable(InMemorySkillStore):
            def all_objects(self, _kind: str) -> dict[str, Any]:
                raise RuntimeError("store is down")

        store = Unavailable()
        await init_client(options={"skillStore": store}, client=object())
        with pytest.raises(RuntimeError, match="store is down"):
            await watch_skills(
                "*", tmp_path / "s", on_unavailable="raise", debounce=0.05
            )
        assert store._listeners.get(SKILL_OBJECT_KIND, []) == []

    async def test_prune_is_passed_through_and_can_be_turned_off(
        self, tmp_path: Any, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a", version=1, content="body"))
        await init_client(options={"skillStore": store}, client=object())
        report, watcher = await watch_skills(
            "*", tmp_path / "s", prune=False, debounce=0.05
        )
        watcher.close()
        written = tmp_path / "s" / "a" / "SKILL.md"
        assert written.read_text() == "body"
        assert report.ok is True
        assert all(a.action != "removed" for a in report.actions)

    async def test_an_invalid_root_raises_out_of_watch_skills(
        self, tmp_path: Any
    ) -> None:
        """The initial reconcile runs on the caller's thread, so a bad root is
        the caller's exception rather than a line in a worker thread's log."""
        store = InMemorySkillStore()
        await init_client(options={"skillStore": store}, client=object())
        not_a_directory = tmp_path / "file"
        not_a_directory.write_text("")
        with pytest.raises(ValueError, match="not a directory"):
            await watch_skills("*", not_a_directory, debounce=0.05)

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
