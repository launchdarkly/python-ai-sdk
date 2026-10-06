"""
Agent Skills — keep skills on disk in sync as delivery changes.

``write_skills`` is a one-shot reconcile of what the store holds now.
``watch_skills`` re-runs it whenever the store reports a change. With ``"*"``,
a skill revoked in LaunchDarkly is removed from disk within a debounce interval
rather than at the next restart. An explicit list is fixed: a skill the store
answers ``absent`` for keeps its files and reports an error, and a config change
that unpins a skill or moves it to a new version is not seen.

``on_unavailable="keep"`` remains the default, so an outage never deletes the
application's skill files.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .skills_core import SKILL_OBJECT_KIND, get_store
from .skills_fs import OnUnavailable, write_skills
from .types import ReconcileReport, Skill, SkillReference

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECONDS = 0.5
"""
Default debounce, in seconds: how long to wait after a change before
reconciling, so a payload of many skills triggers one reconcile, not one each.
"""


class SkillWatcher:
    """
    A running watch. Returned by ``watch_skills``; stop it with ``close`` (or
    use it as a context manager).

    **One watcher per root.** Don't point two watchers at the same root or run
    ``write_skills`` on a watched root concurrently: interleaved reconciles can
    lose manifest entries. The watcher serialises its own reconciles only.
    """

    def __init__(
        self,
        request: Sequence[Skill | SkillReference | str] | str,
        root: str | os.PathLike[str],
        store: Any,
        *,
        prune: bool,
        timeout: float,
        on_unavailable: OnUnavailable,
        debounce: float,
        on_reconcile: Callable[[ReconcileReport], Any] | None,
    ) -> None:
        self._request = request
        self._root = root
        self._prune = prune
        self._timeout = timeout
        self._on_unavailable = on_unavailable
        self._debounce = debounce
        self._on_reconcile = on_reconcile

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._reconciles = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="ld-ai-skills-reconcile", daemon=True
        )

        # Register now but start the worker later (``_start``): a change during
        # the initial reconcile is recorded, not lost, and never reconciled
        # concurrently with it. If ``add_listener`` raises, no thread is left.
        self._store = store
        self._registered = False
        store.add_listener(SKILL_OBJECT_KIND, self.notify)
        self._registered = True

    def _start(self) -> None:
        """Starts the worker. Called once by ``watch_skills`` after the initial reconcile."""
        self._thread.start()

    # -- the listener the store calls -------------------------------------

    def notify(self, _raw: Any = None) -> None:
        """
        The store's change listener. Records that something changed; runs nothing.

        Called on the delivery thread, so it must not block on filesystem work.
        The argument is ignored: any change triggers a full reconcile.
        """
        self._wake.set()

    # -- the worker --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.5):
                continue
            if self._stop.is_set():
                return
            # Clear before the debounce so a change arriving mid-debounce
            # triggers another pass instead of being swallowed by this one.
            self._wake.clear()
            if self._stop.wait(self._debounce):
                return
            self._reconcile_once()

    def _reconcile_once(self) -> None:
        try:
            report = asyncio.run(
                write_skills(
                    self._request,
                    self._root,
                    prune=self._prune,
                    timeout=self._timeout,
                    on_unavailable=self._on_unavailable,
                )
            )
        except Exception:
            # Log and keep watching; dying here would silently stop revocations.
            logger.error(
                "A skill re-reconcile raised; the watcher continues", exc_info=True
            )
            return

        with self._lock:
            self._reconciles += 1
        changed = [
            action
            for action in report.actions
            if action.action in ("written", "updated", "removed", "error")
        ]
        if changed:
            logger.info(
                "Re-reconciled skills after a delivery change: %d action(s) of note",
                len(changed),
            )
        if self._on_reconcile is not None:
            try:
                self._on_reconcile(report)
            except Exception:
                logger.error("A watch_skills callback raised", exc_info=True)

    # -- lifecycle ---------------------------------------------------------

    @property
    def reconciles(self) -> int:
        """Number of re-reconciles completed, excluding the initial one."""
        with self._lock:
            return self._reconciles

    def close(self, timeout: float = 15.0) -> None:
        """
        Stops watching. Idempotent; leaves files on disk as they are.

        Detaches from the store (when it has ``remove_listener``), then waits up
        to *timeout* seconds for an in-flight reconcile to finish rather than
        interrupting it mid-write.
        """
        self._detach()
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)

    def _detach(self) -> None:
        with self._lock:
            if not self._registered:
                return
            self._registered = False
        remove_listener = getattr(self._store, "remove_listener", None)
        if callable(remove_listener):
            remove_listener(SKILL_OBJECT_KIND, self.notify)

    def __enter__(self) -> SkillWatcher:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


async def watch_skills(
    skills: Sequence[Skill | SkillReference | str] | str,
    root: str | os.PathLike[str],
    *,
    prune: bool = True,
    timeout: float = 10.0,
    on_unavailable: OnUnavailable = "keep",
    debounce: float = DEFAULT_DEBOUNCE_SECONDS,
    on_reconcile: Callable[[ReconcileReport], Any] | None = None,
) -> tuple[ReconcileReport, SkillWatcher]:
    """
    Reconciles now, then re-reconciles whenever delivery changes.

    Takes the same arguments as ``write_skills``, plus the two below. Errors
    from the initial ``write_skills`` (e.g. a bad root) propagate, so you can
    fail fast exactly as with ``write_skills``. If the store has no
    ``remove_listener``, a closed watcher stays registered with the store for
    the store's lifetime.

    Args:
        debounce: Seconds to wait after a change before reconciling. Must be a
            non-negative finite number.
        on_reconcile: Called with each re-reconcile's report (not the initial
            one). Exceptions it raises are logged and do not stop the watcher.

    Returns:
        The initial reconcile's report and a ``SkillWatcher`` to close when done::

            report, watcher = await watch_skills("*", "/etc/agent/skills")
            try:
                ...
            finally:
                watcher.close()

    Raises:
        RuntimeError: If no store is configured, or the store has no
            ``add_listener`` (use ``write_skills`` for a one-shot reconcile).
        ValueError: If *debounce* is negative, ``NaN``, or infinite.
    """
    store = get_store()
    if store is None:
        raise RuntimeError(
            "watch_skills needs a configured skill store. Configure one with "
            "set_skill_store(store) from launchdarkly_ai_server.experimental.skills."
        )
    add_listener = getattr(store, "add_listener", None)
    if not callable(add_listener):
        raise RuntimeError(
            "watch_skills needs a skill store that implements add_listener(kind, "
            "fn); the configured store does not, so delivery changes cannot be "
            "observed. Use write_skills for a one-shot reconcile, or configure a "
            "store with a delivery transport (FDv2SkillStore)."
        )
    # A bare ``< 0`` check misses both non-finite values: ``Event.wait(nan)``
    # returns immediately (no debouncing) and ``Event.wait(inf)`` never returns.
    if not math.isfinite(debounce) or debounce < 0:
        raise ValueError(
            f"debounce must be a non-negative finite number of seconds, got "
            f"{debounce!r}"
        )

    # Attach the listener before the initial reconcile, so a change delivered
    # after its store snapshot is still seen (nothing re-reconciles on a timer).
    watcher = SkillWatcher(
        skills,
        root,
        store,
        prune=prune,
        timeout=timeout,
        on_unavailable=on_unavailable,
        debounce=debounce,
        on_reconcile=on_reconcile,
    )
    try:
        # Run on the caller's task, so a bad root raises to the caller.
        report = await write_skills(
            skills, root, prune=prune, timeout=timeout, on_unavailable=on_unavailable
        )
    except BaseException:
        # The caller gets no watcher to close, so detach the listener here.
        watcher.close()
        raise

    # Any change seen during the initial reconcile is already pending.
    watcher._start()
    return report, watcher
