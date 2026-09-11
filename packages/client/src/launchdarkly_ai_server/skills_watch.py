"""
Agent Skills — re-reconcile on delivery, so revocation does not wait for a restart.

``write_skills`` is a one-shot reconcile: it materializes what the store holds
now. With a hand-populated store that is sufficient, and a revocation takes
effect at the next process restart.

A streaming FDv2 connection changes the premise. A ``delete-object`` reaches a
live connection in **seconds**, and the store publishes a change listener, so
wiring the two together collapses the gap between "LaunchDarkly revoked this
skill" and "its ``SKILL.md`` is off the agent's disk" from a process lifetime to
a debounce interval.

``on_unavailable="keep"`` stays the default: an outage must not read as
"everything was revoked". A watcher that pruned on a failed retrieval would
convert every transport failure into deletion of a customer's skill files.

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
    on a single worker thread, serialised — and cannot enforce it against a
    caller who reconciles the same root by hand.

    The watcher owns its registration on *store*: it registers ``notify`` when
    constructed and unregisters it in ``close``, so a closed watcher is no longer
    reachable from the store and can be collected. *store* must implement
    ``add_listener``; ``remove_listener`` is probed for and, when the store does
    not offer it, the listener stays registered for the store's lifetime.
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

        # Register before the initial reconcile, and leave the worker unstarted
        # until ``start``. ``notify`` only sets an event, so a change that lands
        # while that reconcile is still running is recorded rather than lost, and
        # the worker cannot reconcile the root while the caller's own reconcile is
        # in flight. A store whose ``add_listener`` raises leaves no thread behind.
        self._store = store
        self._registered = False
        store.add_listener(SKILL_OBJECT_KIND, self.notify)
        self._registered = True

    def _start(self) -> None:
        """
        Starts the worker. ``watch_skills`` calls this once, after the initial
        reconcile; it is not part of the caller-facing interface.

        Split from construction so registration and reconciling can be ordered
        independently: the listener attaches first, so no change is missed, while
        the first re-reconcile waits for the initial one to finish, so a root only
        ever has one reconcile running at a time.
        """
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

        Detaches ``notify`` from the store first, so no further change reaches a
        watcher that is shutting down and the store no longer holds a reference to
        it. A store without the optional ``remove_listener`` is left as it is
        rather than failing the close.
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
    ``SkillStore`` interface. Raises ``RuntimeError`` when no store is configured,
    and when the configured store has no ``add_listener`` — the second case
    failing loudly rather than degrading to a one-shot reconcile, because a
    watcher that silently never fires looks exactly like a watcher whose skills
    never changed. The optional ``remove_listener`` lets ``SkillWatcher.close``
    detach from the store; a store without it still works, but each closed
    watcher then stays registered for the store's lifetime.
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
            "store with a delivery transport."
        )
    if debounce < 0:
        raise ValueError(f"debounce must not be negative, got {debounce!r}")

    # The watcher attaches its listener before the initial reconcile, not after.
    # The reconcile snapshots the store as its first step and then spends the
    # rest of its time on the filesystem — a write and an fsync per skill, the
    # prune, the manifest rewrite — so a change delivered after that snapshot
    # needs something already listening to be seen at all. Nothing re-reconciles
    # on a timer, so a revocation that landed unobserved would wait for the next
    # unrelated change, which on a quiet root means the next restart.
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
        # The initial reconcile runs on the caller's thread, so its report is the
        # caller's to inspect and a bad root raises out of `watch_skills` rather
        # than into a worker thread's log.
        report = await write_skills(
            skills, root, prune=prune, timeout=timeout, on_unavailable=on_unavailable
        )
    except BaseException:
        # The listener is already attached, so a reconcile that raises must not
        # leave it on the store: the caller has no watcher to close.
        watcher.close()
        raise

    # Only now start the worker. A change that arrived during the reconcile has
    # already set the wake event, so the worker's first pass picks it up; one that
    # arrived before the reconcile's snapshot is already on disk, and the
    # redundant pass it triggers converges on the same state.
    watcher._start()
    return report, watcher
