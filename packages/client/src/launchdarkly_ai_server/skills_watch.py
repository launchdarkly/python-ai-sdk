"""
Agent Skills — re-reconcile on delivery, so revocation does not wait for a restart.

``write_skills`` is a one-shot reconcile: it materializes what the store holds
now. That was the whole story while the only transport was a hand-populated
store, and the design accordingly deferred an eager re-reconcile — revocation
would take effect at the next process restart, which the security review filed
as AV-1.

A streaming FDv2 connection changes the premise. A ``delete-object`` reaches a
live connection in **seconds**, and the store already publishes a change
listener, so the gap between "LaunchDarkly revoked this skill" and "its
``SKILL.md`` is off the agent's disk" collapses from a process lifetime to a
debounce interval. That is the single largest resilience improvement available
at this layer, which is why it is here rather than in a later phase.

``on_unavailable="keep"`` stays the default, deliberately and per the review: an
outage must not read as "everything was revoked". A watcher that pruned on a
failed retrieval would convert every transport blip into deletion of a
customer's skill files.

Layering: this module sits *above* ``skills_fs`` and calls ``write_skills``
without modifying it. Nothing in the reconcile, the accessors, or verification
knows this file exists.
"""

from __future__ import annotations

import asyncio
import logging
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
How long a change waits for its neighbours before a reconcile runs.

A full payload transfer commits many objects at once and the listener fires per
object, so without coalescing a payload of forty skills would run forty
reconciles against one root. Half a second is far below the seconds-scale
latency this feature is trying to achieve and far above the microseconds a
commit's listener calls take.
"""


class SkillWatcher:
    """
    A running re-reconcile. Returned by ``watch_skills``; stop it with ``close``.

    One watcher owns one root. **Do not point two watchers at the same root**,
    and do not run ``write_skills`` against a watched root concurrently: the
    reconcile's own contract is one root, one reconcile at a time, because two
    interleaved runs lose the loser's manifest entries and leave the files it
    wrote unmanaged. This class enforces that for its *own* reconciles — they run
    on a single worker thread, serialized — and cannot enforce it against a
    caller who reconciles the same root by hand.
    """

    def __init__(
        self,
        request: Sequence[Skill | SkillReference | str] | str,
        root: str | os.PathLike[str],
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
        self._thread.start()

    # -- the listener the store calls -------------------------------------

    def notify(self, _raw: Any = None) -> None:
        """
        The store's change listener. Records that something changed; runs nothing.

        Deliberately trivial. It is called on the delivery thread, where a
        reconcile — which does synchronous filesystem I/O, an fsync per file, and
        a manifest rewrite — would stall event processing for the duration and,
        on a stream, let the connection's read buffer back up behind a disk write.
        The argument is ignored: a put's raw object and a revocation's tombstone
        both mean the same thing here, which is "the store is not what it was".
        """
        self._wake.set()

    # -- the worker --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.5):
                continue
            if self._stop.is_set():
                return
            # Coalesce the rest of the burst. Clearing *before* the sleep rather
            # than after is what makes a change arriving mid-debounce trigger the
            # next pass instead of being swallowed by this one.
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
            # A watcher that died on one bad reconcile would silently stop
            # tracking revocations, which is worse than a noisy one.
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
        """How many re-reconciles have completed since the watcher started.

        Excludes the initial reconcile ``watch_skills`` awaits, which is the
        caller's own result."""
        with self._lock:
            return self._reconciles

    def close(self, timeout: float = 15.0) -> None:
        """
        Stops watching. Idempotent. Does not undo anything already on disk.

        Waits out an in-flight reconcile rather than interrupting one, because a
        reconcile killed between its content writes and its manifest rewrite is
        the one case the manifest format has to recover from — worth avoiding when
        we control the timing.
        """
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)

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

    Every argument that ``write_skills`` takes means the same thing here and is
    passed straight through; the reconcile's semantics are untouched. Returns the
    initial reconcile's report — so a caller can fail fast on a bad root or a
    corrupt manifest exactly as they would with ``write_skills`` — paired with a
    ``SkillWatcher`` to close when the process is done::

        report, watcher = await watch_skills("*", "/etc/agent/skills")
        try:
            ...
        finally:
            watcher.close()

    A revocation delivered over a streaming connection then prunes the skill's
    files within ``debounce`` of arriving, rather than at the next restart.

    Requires a store that implements the optional ``add_listener`` half of the
    seam. Raises ``RuntimeError`` when no store is configured, and when the
    configured store has no ``add_listener`` — the second case failing loudly
    rather than degrading to a one-shot reconcile, because a watcher that
    silently never fires looks exactly like a watcher whose skills never changed.
    """
    store = get_store()
    if store is None:
        raise RuntimeError(
            "watch_skills needs a configured skill store. Configure one with "
            'init_client(options={"skillStore": store}).'
        )
    add_listener = getattr(store, "add_listener", None)
    if not callable(add_listener):
        raise RuntimeError(
            "watch_skills needs a skill store that implements add_listener(kind, "
            "fn); the configured store does not, so delivery changes cannot be "
            "observed. Use write_skills for a one-shot reconcile, or configure a "
            "store with a delivery transport (FDv2SkillStore)."
        )
    if debounce < 0:
        raise ValueError(f"debounce must not be negative, got {debounce!r}")

    # The initial reconcile runs first and on the caller's thread, so its report
    # is the caller's to inspect and a bad root raises out of `watch_skills`
    # rather than into a worker thread's log.
    report = await write_skills(
        skills, root, prune=prune, timeout=timeout, on_unavailable=on_unavailable
    )

    watcher = SkillWatcher(
        skills,
        root,
        prune=prune,
        timeout=timeout,
        on_unavailable=on_unavailable,
        debounce=debounce,
        on_reconcile=on_reconcile,
    )
    add_listener(SKILL_OBJECT_KIND, watcher.notify)
    return report, watcher
