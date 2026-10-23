"""
Agent Skills — keep skills on disk in sync as delivery changes.

``write_skills`` is a one-shot reconcile of what the store holds now.
``watch_skills`` re-runs it whenever the store reports a change and, for a
``SkillSource``, whenever one of its AI Configs changes. With ``"*"``, a skill
revoked in LaunchDarkly is removed from disk within a debounce interval rather
than at the next restart. With ``config_skills(key, context)``, a config moving
a skill to a new version replaces the file, and a skill it stops referencing is
pruned. An explicit list is fixed: a skill the store answers ``absent`` for
keeps its files and reports an error, and a config change that unpins a skill
or moves it to a new version is not seen.

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

from .lifecycle import get_client
from .skills_core import SKILL_OBJECT_KIND, get_store
from .skills_fs import OnUnavailable, write_skills
from .types import ReconcileReport, Skill, SkillReference, SkillSource

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
        request: Sequence[Skill | SkillReference | str] | str | SkillSource,
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

        # Set by ``_follow_configs``; detached by ``close``.
        self._config_keys: frozenset[str] = frozenset()
        self._flag_tracker: Any = None

    def _follow_configs(self, flag_tracker: Any, config_keys: Sequence[str]) -> None:
        """Subscribes to the base client's flag changes, filtered to *config_keys*."""
        self._config_keys = frozenset(config_keys)
        flag_tracker.add_listener(self._on_flag_change)
        self._flag_tracker = flag_tracker

    def _start(self) -> None:
        """Starts the worker. Called once by ``watch_skills`` after the initial reconcile."""
        self._thread.start()

    # -- the listener the store calls -------------------------------------

    def notify(self, _raw: Any = None) -> None:
        """
        Schedules a reconcile, through the same debounce as a delivery change.

        Also the store's change listener. Call it yourself from your own
        change listener when the watched ``SkillSource`` has no
        ``config_keys``, or when the client cannot report config changes. Runs
        nothing itself, so it is safe to call from any thread. The argument is
        ignored: any change triggers a full reconcile.
        """
        self._wake.set()

    def _on_flag_change(self, change: Any) -> None:
        """
        The base client's flag-change listener.

        Only signals. The base SDK calls it on the thread applying flag data,
        while holding the store lock every evaluation takes, so reading the
        source here would stall flag delivery and every ``variation`` call.
        """
        if getattr(change, "key", None) in self._config_keys:
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
            flag_tracker, self._flag_tracker = self._flag_tracker, None
        remove_listener = getattr(self._store, "remove_listener", None)
        if callable(remove_listener):
            remove_listener(SKILL_OBJECT_KIND, self.notify)
        remove_flag_listener = getattr(flag_tracker, "remove_listener", None)
        if callable(remove_flag_listener):
            remove_flag_listener(self._on_flag_change)

    def __enter__(self) -> SkillWatcher:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


async def watch_skills(
    skills: Sequence[Skill | SkillReference | str] | str | SkillSource,
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

    What reaches disk depends on the form of *skills*:

    - ``"*"`` follows the whole library: a deleted skill is pruned and a new
      version is written.
    - A fixed list never re-reads a config. A version it pins that leaves the
      store stays on disk, reported ``error``, and the newer one is never
      written.
    - A ``SkillSource``, such as ``config_skills(key, context)``, is read again
      on every reconcile, and the watcher also reconciles when one of its
      ``config_keys`` changes in the base client. A version bump replaces the
      file, and a skill the config stops referencing is pruned on the next
      reconcile whose retrieval is available. A skill the store answers
      ``absent`` for beside it keeps its own file and does not hold that prune;
      a store that raises, has no initial data, or runs out of time does.

    Following config changes needs the initialized base LaunchDarkly client's
    ``flag_tracker``; it is never read for a fixed list or ``"*"``. Under daemon
    mode (``use_ldd`` or ``ldclient.datasystem.daemon``) the base client has no
    data source, so its ``flag_tracker`` never fires: the watcher follows
    skill-store changes only. Call ``watcher.notify()`` yourself if you can
    detect config changes another way.

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
            For a ``SkillSource`` with ``config_keys``, also if no AI client is
            initialized, the client has no ``flag_tracker``, or the client is
            offline: such a watcher would never hear about a config change.
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
        # Subscribed before the initial reconcile, like the store listener, so
        # a config change landing during it is followed by one more.
        if isinstance(skills, SkillSource) and skills.config_keys:
            watcher._follow_configs(_config_flag_tracker(), skills.config_keys)
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


_NO_CONFIG_SIGNAL_REMEDY = (
    "Initialize the AI client with a base LaunchDarkly client "
    "(init_client() with an SDK key, or init_client(client=LDClient(...))), or "
    "build the SkillSource with config_keys=() and call watcher.notify() from "
    "your own change listener."
)


def _config_flag_tracker() -> Any:
    """
    The base client's ``flag_tracker``, or raises naming the remedy.

    A watcher that never hears about config changes looks exactly like one
    whose config never changed, so a missing signal is an error, not a no-op.
    """
    try:
        client = get_client()
    except RuntimeError:
        raise RuntimeError(
            "watch_skills cannot follow config changes for this SkillSource: no "
            "AI client is initialized. " + _NO_CONFIG_SIGNAL_REMEDY
        ) from None
    flag_tracker = getattr(client, "flag_tracker", None)
    if not callable(getattr(flag_tracker, "add_listener", None)):
        raise RuntimeError(
            "watch_skills cannot follow config changes for this SkillSource: "
            "the AI client has no flag_tracker.add_listener, so it cannot report "
            "flag changes. " + _NO_CONFIG_SIGNAL_REMEDY
        )
    # An offline client still has a flag_tracker, but no data source to fire it.
    # One that raises is treated as offline, so the caller still gets the remedy.
    is_offline = getattr(client, "is_offline", None)
    try:
        offline = callable(is_offline) and is_offline() is True
    except Exception:
        offline = True
    if offline:
        raise RuntimeError(
            "watch_skills cannot follow config changes for this SkillSource: "
            "the AI client is offline, so it never reports flag changes. "
            + _NO_CONFIG_SIGNAL_REMEDY
        )
    return flag_tracker
