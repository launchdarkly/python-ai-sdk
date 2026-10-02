"""
Agent Skills — filesystem materialization.

Writes already-verified skill content to disk under a managed root. Retrieval
and verification live in ``skills.py``; the descriptor-pinned primitives live
in ``safe_fs.py``.

Safety invariants:

- The root is pinned to a descriptor once per reconcile, and every operation
  under it (including the reads that decide what to do) runs relative to that
  descriptor, so a directory swapped mid-run cannot redirect anything.
- Destructive operations only touch paths ``<root>/.launchdarkly-skills.json``
  records under a matching key.
- A corrupt manifest suppresses every destructive action; an incomplete
  retrieval suppresses pruning.
- Content is re-verified immediately before the write.

These checks are non-relaxable; see ``agents.md``, *Security posture*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .safe_fs import (
    DirectoryMissing,
    SymlinkRefused,
    atomic_write,
    is_temp_name,
    open_directory_nofollow,
    pinned_directory,
    unlink_file,
)
from .skills_core import (
    NO_STORE_MESSAGE,
    Resolution,
    SkillStore,
    VerificationFailure,
    get_store,
    list_raw_objects,
    log_withholding_summary,
    newest_by_key,
    record_materialized,
    record_revoked,
    reference_target,
    resolve_from_store,
    store_is_initialized,
    verified_bytes,
    verify_raw_skill,
)
from .types import (
    ReconcileAction,
    ReconcileActionKind,
    ReconcileReport,
    Skill,
    SkillReference,
)
from .types_validation import (
    is_valid_skill_key,
    is_valid_skill_version,
    skill_key_rejection_reason,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".launchdarkly-skills.json"
"""The SDK's record of what it has written under a managed root."""

MANIFEST_VERSION = 1
"""Manifest schema version this release writes, and the highest it can read."""

_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
"""
Hard cap on the manifest read, far above any real manifest. A larger file is
treated as corrupt rather than read into memory.
"""

SKILL_FILENAME = "SKILL.md"
"""The single file each skill materializes to, under ``<root>/<key>/``."""

OnUnavailable = Literal["keep", "raise"]
"""How ``write_skills`` reacts to content it could not retrieve."""

_UNAVAILABLE_PREFIX = "skill retrieval unavailable: "
"""Prefix on every error describing content that could not be retrieved."""

_MAX_PATH_COMPONENT_BYTES = 255
"""
The single path-component limit on Linux, macOS and Windows. Skill keys may be
longer, so an over-long key is reported rather than failing with ENAMETOOLONG.
"""


_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in range(1, 10)}
    | {f"lpt{digit}" for digit in range(1, 10)}
)
"""
Windows reserved device names, which cannot be directory names there. Rejected
on every platform so a managed root is usable on any OS. The key grammar
(lowercase, no ``.`` or ``$``) makes forms like ``con.txt`` unreachable.
"""


# -------------------------------------------------------------------------
# The reconcile entry point
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class _PendingWrite:
    """One skill queued for the reconcile: resolved content, or why there is none."""

    key: str
    skill: Skill | None = None
    error: str | None = None


async def write_skills(
    skills: Sequence[Skill | SkillReference | str] | str,
    root: str | os.PathLike[str],
    *,
    prune: bool = True,
    timeout: float = 10.0,
    on_unavailable: OnUnavailable = "keep",
) -> ReconcileReport:
    """
    Materializes skills under a managed root at ``<root>/<key>/SKILL.md``.

    *skills* is a sequence of ``Skill`` / ``SkillReference`` / key strings, or
    ``"*"`` for every skill the store holds. ``Skill`` values are written as-is;
    references and keys resolve through the configured store.

    The reconcile is driven by a manifest, ``<root>/.launchdarkly-skills.json``:
    it only overwrites or deletes files the manifest records as written by the
    SDK.

    - ``prune``: remove previously managed skills that are no longer requested.
      This is how revocation takes effect.
    - ``timeout``: seconds, non-negative and finite. Bounds retrieval, writes and
      pruning; checked between steps, not mid-operation. The manifest rewrite
      always runs.
    - ``on_unavailable``: ``"keep"`` reports content that could not be retrieved
      and leaves existing files alone; ``"raise"`` raises ``RuntimeError``.

    Returns a ``ReconcileReport`` listing every outcome. Raises ``ValueError``
    for an invalid argument or an unusable root.

    Run at most one reconcile per root at a time: concurrent runs can lose each
    other's manifest entries.

    **This call performs synchronous filesystem I/O and does not yield**, so a
    large reconcile blocks the event loop. Wrap it in ``asyncio.to_thread`` if
    that matters. Descriptor pinning of the root requires POSIX; see ``safe_fs``.
    """
    # Typed as closed sets, but untyped callers can pass anything.
    if on_unavailable not in ("keep", "raise"):
        raise ValueError(
            f'on_unavailable must be "keep" or "raise", got {on_unavailable!r}'
        )
    # ``nan`` and ``inf`` both pass ``< 0`` and would leave the deadline
    # unbounded (or, for ``nan``, inconsistently expired).
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(
            f"timeout must be a non-negative finite number of seconds, got {timeout!r}"
        )

    deadline = time.monotonic() + timeout
    root_path = _resolve_root(root)

    # Pinned once and held for the whole reconcile. None where the *at() family
    # is absent; the path-based checks are then the only defense.
    try:
        root_fd = open_directory_nofollow(root_path)
    except ValueError as exc:
        # The root was valid a moment ago, so this is a swap being refused:
        # reported, not raised. Nothing has been touched.
        return ReconcileReport(
            actions=[
                _run_error(
                    f"the skills root {root_path} could not be pinned for the "
                    f"reconcile: {exc}; no action was taken"
                )
            ]
        )

    try:
        manifest, manifest_error = _load_manifest(root_path, root_fd)
        entries: dict[str, Any] = manifest.get("entries", {})

        actions: list[ReconcileAction] = []
        if manifest_error is not None:
            # Run-level failure: there is no single skill key to hang it off.
            actions.append(_run_error(manifest_error))

        requests, incomplete = _resolve_requests(skills, deadline, on_unavailable)

        written, write_timed_out = _write_all(
            root_path, root_fd, requests, entries, deadline
        )
        actions.extend(written)
        incomplete = incomplete or write_timed_out

        # Prune only when the SDK knows both what it owns (manifest intact) and
        # what is current (retrieval and writes completed).
        if prune and manifest_error is None and not incomplete:
            actions.extend(
                _prune(
                    root_path,
                    root_fd,
                    entries,
                    {request.key for request in requests},
                    deadline,
                )
            )

        if manifest_error is None:
            actions.extend(_rewrite_manifest(root_path, root_fd, manifest, entries))

        return ReconcileReport(actions=actions)
    finally:
        if root_fd is not None:
            os.close(root_fd)


_RUN_LEVEL_KEY = ""
"""The ``ReconcileAction`` key for a failure that belongs to no single skill."""


def _run_error(message: str) -> ReconcileAction:
    """A failure belonging to the run rather than to one skill."""
    return ReconcileAction(key=_RUN_LEVEL_KEY, action="error", error=message)


def _write_all(
    root: Path,
    root_fd: int | None,
    requests: list[_PendingWrite],
    entries: dict[str, Any],
    deadline: float,
) -> tuple[list[ReconcileAction], bool]:
    """
    Reconciles every pending write. Returns ``(actions, timed out mid-run)``.

    Never aborts: a per-skill failure becomes an ``error`` action, so the
    manifest rewrite still records every file already written.
    """
    actions: list[ReconcileAction] = []
    timed_out = False

    for request in requests:
        if request.skill is None:
            actions.append(
                ReconcileAction(
                    key=request.key,
                    action="error",
                    error=request.error
                    or f"skill '{request.key}' could not be resolved",
                )
            )
            continue
        if time.monotonic() >= deadline:
            timed_out = True
            actions.append(
                ReconcileAction(
                    key=request.key,
                    action="error",
                    error=(
                        "the timeout was exhausted before skill "
                        f"'{request.key}' could be written"
                    ),
                )
            )
            continue
        try:
            actions.append(_write_one(root, root_fd, request.skill, entries))
        except OSError as exc:
            # Safety net: an unexpected filesystem error must not abort the loop.
            actions.append(
                ReconcileAction(
                    key=request.skill.key,
                    action="error",
                    version=request.skill.version,
                    error=f"skill '{request.skill.key}' could not be reconciled: {exc}",
                )
            )

    return actions, timed_out


def _rewrite_manifest(
    root: Path,
    root_fd: int | None,
    manifest: dict[str, Any],
    entries: dict[str, Any],
) -> list[ReconcileAction]:
    """Writes the updated manifest. Returns an error action, or nothing."""
    manifest["manifestVersion"] = MANIFEST_VERSION
    manifest["entries"] = entries
    try:
        # Inside the guard: unknown fields are round-tripped, so a deeply nested
        # planted field can raise RecursionError here.
        serialized = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        atomic_write(root, MANIFEST_FILENAME, serialized, dir_fd=root_fd)
    except Exception as exc:
        return [_run_error(f"the skills manifest could not be written: {exc}")]
    return []


# -------------------------------------------------------------------------
# Request resolution — content in, or a reason there is none
# -------------------------------------------------------------------------


def _unavailable(reason: str) -> str:
    """Wraps *reason* as a retrieval-unavailable message."""
    return f"{_UNAVAILABLE_PREFIX}{reason}"


@dataclass(frozen=True)
class _RetrievalBlocked:
    """Why retrieval must not be attempted. The reason is caller-facing."""

    reason: str


def _available_store(deadline: float, subject: str) -> SkillStore | _RetrievalBlocked:
    """
    The configured store, or why retrieval must not be attempted.

    The single gate for both single references and ``"*"``. It blocks on an
    exhausted deadline, an absent store, or a store without its initial data;
    each marks the run incomplete, which suppresses pruning.
    """
    if time.monotonic() >= deadline:
        return _RetrievalBlocked(
            _unavailable(
                f"the timeout was exhausted before {subject} could be retrieved"
            )
        )
    store = get_store()
    if store is None:
        return _RetrievalBlocked(_unavailable(NO_STORE_MESSAGE))
    if not store_is_initialized(store):
        # Before its first delivery a store answers "nothing", which "*" would
        # read as every skill revoked.
        return _RetrievalBlocked(
            _unavailable(
                "the skill store has not received its initial data, so "
                f"{subject} could not be retrieved and nothing on disk was "
                "changed. Wait for delivery before reconciling: "
                "FDv2SkillStore.wait_for_skills(timeout) returns True once the "
                "first payload has arrived."
            )
        )
    return store


def _resolve_requests(
    skills: Sequence[Skill | SkillReference | str] | str,
    deadline: float,
    on_unavailable: OnUnavailable,
) -> tuple[list[_PendingWrite], bool]:
    """
    Turns the caller's input into one request per skill.

    Also returns whether any retrieval was incomplete (absent, uninitialized or
    raising store, or an exhausted timeout). That flag suppresses pruning, so an
    outage never deletes managed files.
    """
    if isinstance(skills, str):
        if skills != "*":
            raise ValueError(
                'write_skills takes a sequence of skills or the literal "*"; '
                f"got {skills!r}"
            )
        return _resolve_all(deadline, on_unavailable)

    requests: list[_PendingWrite] = []
    incomplete = False
    for item in skills:
        if isinstance(item, Skill):
            requests.append(_PendingWrite(key=item.key, skill=item))
            continue

        key, wanted = reference_target(item)
        resolved = _resolve_reference(key, wanted, deadline)
        if resolved.unavailable:
            incomplete = True
            if on_unavailable == "raise":
                raise RuntimeError(resolved.error)
        requests.append(
            _PendingWrite(key=key, skill=resolved.skill, error=resolved.error)
        )

    return requests, incomplete


def _resolve_reference(
    key: str, wanted_version: int | None, deadline: float
) -> Resolution:
    """
    Resolves one reference for the materialization path.

    Same core as the accessors, but a blocked store (see ``_available_store``)
    is reported as unavailable rather than raised.
    """
    store = _available_store(deadline, f"'{key}'")
    if isinstance(store, _RetrievalBlocked):
        return Resolution(
            reason="store_unavailable", error=store.reason, unavailable=True
        )

    resolved = resolve_from_store(store, key, wanted_version)
    if resolved.unavailable and resolved.error is not None:
        return Resolution(
            reason="store_unavailable",
            error=_unavailable(resolved.error),
            unavailable=True,
        )
    return resolved


def _unavailable_run(
    error: str, on_unavailable: OnUnavailable
) -> tuple[list[_PendingWrite], bool]:
    """
    One run-level retrieval failure — raised, or reported against the empty key.

    Always marks the run incomplete, so nothing is pruned.
    """
    if on_unavailable == "raise":
        raise RuntimeError(error)
    return [_PendingWrite(key="", error=error)], True


def _pending_for_raw(object_key: str, raw: Any) -> _PendingWrite:
    """
    One raw store object as a pending write — verified, or reported as failed.

    Unverifiable is not revoked: a failed object keeps its key in the requested
    set, so prune leaves the last known-good copy on disk.
    """
    skill = verify_raw_skill(raw)
    if skill is not None:
        return _PendingWrite(key=skill.key, skill=skill)
    # Prefer the object's own key (the on-disk directory name); a custom store
    # may use a different map key.
    raw_key = raw.get("key") if isinstance(raw, dict) else None
    key = raw_key if is_valid_skill_key(raw_key) else object_key
    if not is_valid_skill_key(key):
        # No usable key: report at run level, which ``_resolve_all`` treats as
        # an incomplete run.
        return _PendingWrite(
            key=_RUN_LEVEL_KEY,
            error="the skill store served an object under an invalid key; "
            "it was withheld",
        )
    return _PendingWrite(
        key=key,
        error=f"skill '{key}' failed integrity verification and was "
        "withheld; the copy already on disk was left alone",
    )


def _resolve_all(
    deadline: float, on_unavailable: OnUnavailable
) -> tuple[list[_PendingWrite], bool]:
    """Resolves the ``"*"`` form — everything the store currently holds."""
    store = _available_store(deadline, "the skill set")
    if isinstance(store, _RetrievalBlocked):
        return _unavailable_run(store.reason, on_unavailable)

    # Not via all_skills(), which reports a raising store as empty — that would
    # read as every skill revoked.
    objects, error = list_raw_objects(store)
    if error is not None:
        return _unavailable_run(_unavailable(error), on_unavailable)

    # One object per key, at its newest version: each key has a single path.
    candidates = newest_by_key(objects)
    requests = [_pending_for_raw(key, raw) for key, raw in candidates]
    log_withholding_summary(
        "skills held by the store",
        len(requests),
        sum(1 for request in requests if request.skill is not None),
    )
    # A failure with no key cannot protect its copy on disk, so it suppresses
    # pruning for the whole run.
    unattributed = any(
        request.skill is None and request.key == _RUN_LEVEL_KEY for request in requests
    )
    return requests, unattributed


# -------------------------------------------------------------------------
# The managed root and its manifest
# -------------------------------------------------------------------------


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    """
    Resolves the managed root once, up front.

    Raises ``ValueError`` for an unusable root. Only the leaf directory is
    created, so a typo cannot create a directory tree.

    A caller-error check, not a security boundary: the pin that ``write_skills``
    takes immediately afterwards is what guards against a swap.
    """
    path = Path(os.fspath(root))

    # pathlib can raise e.g. PermissionError here; surface it as ValueError.
    try:
        is_symlink = path.is_symlink()
        exists = path.exists()
        is_dir = path.is_dir()
    except OSError as exc:
        raise ValueError(f"the skills root could not be inspected: {exc}") from exc

    if is_symlink:
        raise ValueError(
            f"the skills root must be a real directory, not a symlink: {path}"
        )

    if exists:
        if not is_dir:
            raise ValueError(f"the skills root is not a directory: {path}")
    else:
        parent = path.parent
        try:
            parent_is_dir = parent.is_dir()
        except OSError as exc:
            raise ValueError(
                f"the parent of the skills root could not be inspected: {exc}"
            ) from exc
        if not parent_is_dir:
            raise ValueError(
                f"the parent of the skills root does not exist: {parent}. "
                "write_skills creates only the leaf directory."
            )
        try:
            path.mkdir()
        except OSError as exc:
            raise ValueError(f"the skills root could not be created: {exc}") from exc

    return Path(os.path.realpath(path))


def _load_manifest(
    root: Path, root_fd: int | None
) -> tuple[dict[str, Any], str | None]:
    """
    Loads the manifest. Returns ``(manifest, error)``.

    A manifest that cannot be read or parsed, is not an object, is over the size
    cap, has a ``manifestVersion`` outside ``[1, MANIFEST_VERSION]``, or has a
    malformed ``entries`` map is **corrupt**: the caller then takes no
    destructive action and leaves the file alone. An absent manifest is a fresh
    root.

    Read relative to the root descriptor; a symlink or FIFO at the manifest's
    name is treated as corrupt rather than followed.
    """
    fresh: dict[str, Any] = {"manifestVersion": MANIFEST_VERSION, "entries": {}}

    try:
        raw = _read_regular_file(
            MANIFEST_FILENAME if root_fd is not None else root / MANIFEST_FILENAME,
            max_bytes=_MAX_MANIFEST_BYTES,
            dir_fd=root_fd,
        )
    except FileNotFoundError:
        return fresh, None
    except OSError as exc:
        return {}, f"the skills manifest {MANIFEST_FILENAME} could not be read: {exc}"

    # The read stops one byte past the cap, which is enough to detect an overage.
    if len(raw) > _MAX_MANIFEST_BYTES:
        return {}, (
            f"the skills manifest {MANIFEST_FILENAME} is larger than the "
            f"{_MAX_MANIFEST_BYTES} byte cap; refusing every destructive action"
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Non-UTF-8 bytes are corruption like any other.
        return {}, f"the skills manifest {MANIFEST_FILENAME} could not be read: {exc}"

    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        return {}, (
            f"the skills manifest {MANIFEST_FILENAME} is not valid JSON ({exc}); "
            "refusing every destructive action"
        )

    if not isinstance(data, dict):
        return {}, (
            f"the skills manifest {MANIFEST_FILENAME} is not a JSON object; "
            "refusing every destructive action"
        )

    version = data.get("manifestVersion")
    # Bounded at both ends: a future schema cannot be interpreted, and versions
    # below 1 were never written.
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not 1 <= version <= MANIFEST_VERSION
    ):
        return {}, (
            f"the skills manifest {MANIFEST_FILENAME} declares manifestVersion "
            f"{version!r}, which this SDK cannot read; refusing every destructive "
            "action"
        )

    if not isinstance(data.get("entries"), dict):
        return {}, (
            f"the skills manifest {MANIFEST_FILENAME} has a malformed 'entries' "
            "map; refusing every destructive action"
        )

    return data, None


# -------------------------------------------------------------------------
# Per-skill reconcile
# -------------------------------------------------------------------------


def _unsafe_path_reason(
    root: Path, skill_dir: Path, target: Path, key: str, *, require_directory: bool
) -> str | None:
    """
    Why ``<root>/<key>/SKILL.md`` must not be touched, or ``None``.

    Shared by the write and prune paths. Defense in depth: these are path-based
    checks, so the descriptor pin is the real boundary where available.
    *require_directory* is set for writes, which need a real directory; a prune
    only needs to not follow a link.
    """
    if skill_dir.is_symlink():
        return f"{key} is a symlink"
    if require_directory and skill_dir.exists() and not skill_dir.is_dir():
        return f"{key} exists and is not a directory"
    if target.is_symlink():
        return "the target file is a symlink"
    if Path(os.path.realpath(skill_dir)).parent != root:
        return f"it resolves outside the managed root {root}"
    return None


def _key_rejection_reason(key: Any) -> str | None:
    """
    Why *key* must not become a directory name under the managed root, or ``None``.

    Always re-validated before any filesystem call, since a key becomes a path
    component. Shared by the write and prune paths.
    """
    if not is_valid_skill_key(key):
        return f"{key!r} is not a valid skill key: it {skill_key_rejection_reason(key)}"
    # Safe to encode only after the pattern check, which admits no surrogate.
    key_bytes = len(key.encode("utf-8"))
    if key_bytes > _MAX_PATH_COMPONENT_BYTES:
        return (
            f"skill key '{key[:32]}...' is {key_bytes} bytes, over the "
            f"{_MAX_PATH_COMPONENT_BYTES}-byte limit for a single directory name"
        )
    # Checked here rather than in the key grammar, so one such key fails only
    # its own write rather than the whole AI Config.
    if key in _WINDOWS_RESERVED_NAMES:
        return (
            f"skill key '{key}' is a name Windows reserves for a device and "
            "cannot be a directory name there; it is rejected on every platform "
            "so a managed root written on one OS is usable on the other"
        )
    return None


def _write_one(
    root: Path, root_fd: int | None, skill: Skill, entries: dict[str, Any]
) -> ReconcileAction:
    """Reconciles one verified skill against the managed root."""
    key = skill.key

    def failed(message: str) -> ReconcileAction:
        return ReconcileAction(
            key=key, action="error", version=skill.version, error=message
        )

    rejection = _key_rejection_reason(key)
    if rejection is not None:
        return failed(f"{rejection}; nothing was written")
    if not is_valid_skill_version(skill.version):
        return failed(
            f"skill '{key}' has version {skill.version!r}, which is not an "
            "integer >= 1; nothing was written"
        )

    skill_dir = root / key
    target = skill_dir / SKILL_FILENAME
    relative = f"{key}/{SKILL_FILENAME}"

    unsafe = _unsafe_path_reason(root, skill_dir, target, key, require_directory=True)
    if unsafe is not None:
        return failed(f"'{relative}' was refused: {unsafe}; nothing was written")

    # Re-verify: a Skill can also be constructed directly by a caller.
    verified = verified_bytes(key, skill.content, skill.content_hash, skill.version)
    if isinstance(verified, VerificationFailure):
        return failed(
            f"skill '{key}' failed verification immediately before writing: "
            f"{verified.reason}; nothing was written"
        )
    encoded, content_hash = verified.encoded, verified.content_hash

    # Before writing, so this run's own temp file is never a candidate.
    _sweep_orphan_temp_files(root, root_fd, key)

    # Overwrite only what the manifest records as the SDK's under this key.
    entry = entries.get(relative)
    managed = isinstance(entry, dict) and entry.get("key") == key

    # Pin (or create) the directory before deciding anything, and hold it
    # through the write, so the probe, compare read and rename all see the same
    # directory.
    try:
        with pinned_directory(skill_dir, create=True, dir_fd=root_fd) as skill_fd:
            try:
                on_disk = _read_skill_file(skill_dir, skill_fd, max_bytes=len(encoded))
            except OSError as exc:
                if not managed:
                    # A failed read must never become an overwrite.
                    return failed(
                        f"'{relative}' exists, the manifest does not record it as "
                        f"managed under key '{key}', and it could not be read to "
                        f"compare against the resolved content: {exc}; refusing to "
                        "overwrite a file this SDK may not have written"
                    )
                return failed(f"'{relative}' could not be read: {exc}")

            if on_disk is None:
                action: ReconcileActionKind = "written"
            elif hashlib.sha256(on_disk).hexdigest() == content_hash:
                # Adoption: a file byte-identical to the resolved content is
                # recorded as managed even without a manifest entry (e.g. the
                # process was killed before the manifest was written). Only
                # exact matches are adopted.
                _update_entry(entries, relative, skill, content_hash)
                record_materialized(key, len(encoded), content_hash, "skipped_current")
                return ReconcileAction(
                    key=key,
                    action="skipped_current",
                    version=skill.version,
                    path=str(target),
                )
            elif not managed:
                return failed(
                    f"'{relative}' exists but the manifest does not record it as "
                    f"managed under key '{key}'; refusing to overwrite a file this "
                    "SDK did not write"
                )
            else:
                # Stale version or local tampering — LD-resolved content wins.
                action = "updated"

            try:
                atomic_write(skill_dir, SKILL_FILENAME, encoded, dir_fd=skill_fd)
            except OSError as exc:
                return failed(f"'{relative}' could not be written: {exc}")
    except OSError as exc:
        return failed(f"the directory for skill '{key}' could not be created: {exc}")
    except ValueError as exc:
        return failed(f"'{relative}' was refused: {exc}")

    _update_entry(entries, relative, skill, content_hash)
    record_materialized(key, len(encoded), content_hash, action)
    return ReconcileAction(
        key=key, action=action, version=skill.version, path=str(target)
    )


def _skill_file_present(skill_dir: Path, skill_fd: int | None) -> bool:
    """
    Whether ``SKILL.md`` is present in the pinned *skill_dir*.

    Only ``ENOENT`` means absent; any other error reports present so the read or
    unlink that follows fails with the real reason. Path-based when there is no
    descriptor.
    """
    if skill_fd is None:
        return (skill_dir / SKILL_FILENAME).exists()
    try:
        os.stat(SKILL_FILENAME, dir_fd=skill_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _read_skill_file(
    skill_dir: Path, skill_fd: int | None, *, max_bytes: int
) -> bytes | None:
    """
    The bytes at ``SKILL.md`` in the pinned *skill_dir*, or ``None`` if absent.

    Other failures propagate as ``OSError`` for the caller to report.
    """
    if not _skill_file_present(skill_dir, skill_fd):
        return None
    if skill_fd is None:
        return _read_regular_file(skill_dir / SKILL_FILENAME, max_bytes=max_bytes)
    return _read_regular_file(SKILL_FILENAME, max_bytes=max_bytes, dir_fd=skill_fd)


def _read_regular_file(
    target: Path | str, *, max_bytes: int, dir_fd: int | None = None
) -> bytes:
    """
    Reads *target*, refusing anything that is not a regular file.

    - ``O_NONBLOCK``: opening a FIFO with no writer would otherwise hang forever.
    - ``O_NOFOLLOW``: refuses a trailing symlink.
    - ``O_BINARY``: no CRLF translation on Windows.
    - The type check uses ``fstat`` on the descriptor, not the path.

    Reads at most ``max_bytes + 1`` bytes; the extra byte shows the file is over
    the bound. With *dir_fd*, *target* is a bare name inside that directory.
    """
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(target, flags, dir_fd=dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("the target file is not a regular file")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sweep_orphan_temp_files(root: Path, root_fd: int | None, key: str) -> None:
    """
    Removes temp files a killed reconcile left behind under ``<root>/<key>/``.

    Such orphans would otherwise keep prune's ``rmdir`` from ever succeeding.
    This is the only removal of a file the manifest does not list, so it is
    tightly bounded: a valid key's directory only, only names
    ``safe_fs.is_temp_name`` recognizes, only regular files, all via the pinned
    descriptor.

    Never raises: a failed sweep is logged as a warning.
    """
    if _key_rejection_reason(key) is not None:
        return
    skill_dir = root / key

    try:
        with pinned_directory(skill_dir, dir_fd=root_fd) as dir_fd:
            listed = os.listdir(skill_dir if dir_fd is None else dir_fd)
            for name in sorted(listed):
                if is_temp_name(name, SKILL_FILENAME):
                    _remove_orphan_temp_file(skill_dir, name, dir_fd)
    except DirectoryMissing:
        return
    except (OSError, ValueError) as exc:
        logger.warning(
            "Orphaned temp files under skill '%s' could not be swept: %s", key, exc
        )


def _remove_orphan_temp_file(skill_dir: Path, name: str, dir_fd: int | None) -> None:
    """
    Removes one recognized orphan. A per-file failure warns and moves on.

    Only regular files are removed, so a symlink or FIFO with a temp-file name
    is left alone.
    """
    try:
        if dir_fd is not None:
            mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        else:
            mode = os.lstat(skill_dir / name).st_mode
        if not stat.S_ISREG(mode):
            return
        unlink_file(skill_dir, name, dir_fd=dir_fd)
    except (OSError, ValueError) as exc:
        logger.warning("an orphaned temp file could not be removed: %s", exc)


def _update_entry(
    entries: dict[str, Any], relative: str, skill: Skill, content_hash: str
) -> None:
    """
    Records a managed path in the manifest.

    Merges into any existing entry so fields written by a newer SDK release
    survive. ``sha256`` and ``writtenAt`` are informational; currency is always
    decided by hashing the bytes on disk.
    """
    existing = entries.get(relative)
    entry = dict(existing) if isinstance(existing, dict) else {}
    entry["key"] = skill.key
    entry["version"] = skill.version
    entry["sha256"] = content_hash
    entry["writtenAt"] = _utc_timestamp()
    entries[relative] = entry


# -------------------------------------------------------------------------
# Pruning — how revocation takes effect
# -------------------------------------------------------------------------


def _prune_error(key: str, message: str, version: Any = None) -> ReconcileAction:
    """
    A prune refusal.

    *version* comes from the untrusted manifest, so an invalid one is reported
    as ``None``, the same as for a ``removed`` action.
    """
    return ReconcileAction(
        key=key,
        action="error",
        version=version if is_valid_skill_version(version) else None,
        error=message,
    )


def _prune(
    root: Path,
    root_fd: int | None,
    entries: dict[str, Any],
    requested: set[str],
    deadline: float,
) -> list[ReconcileAction]:
    """
    Removes managed skills that are no longer requested.

    This is how revocation takes effect. The deadline is checked per entry; an
    entry left unpruned is reported as an error and stays in the manifest for
    the next reconcile.
    """
    actions: list[ReconcileAction] = []

    for relative, entry in list(entries.items()):
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not isinstance(key, str) or key in requested:
            continue

        if time.monotonic() >= deadline:
            actions.append(
                _prune_error(
                    key,
                    f"the timeout was exhausted before '{relative}' could be "
                    "pruned; it was left in place",
                    entry.get("version"),
                )
            )
            continue

        # Only a manifest path this SDK could have written is removable.
        if (
            _key_rejection_reason(key) is not None
            or relative != f"{key}/{SKILL_FILENAME}"
        ):
            actions.append(
                _prune_error(
                    key,
                    f"manifest entry '{relative}' does not name a path this SDK "
                    f"could own under key '{key}'; it was left in place",
                    entry.get("version"),
                )
            )
            continue

        try:
            actions.append(_prune_one(root, root_fd, relative, key, entries))
        except OSError as exc:
            actions.append(
                _prune_error(
                    key,
                    f"'{relative}' could not be removed: {exc}",
                    entry.get("version"),
                )
            )

    return actions


def _unlink_skill_file(
    skill_dir: Path, skill_fd: int | None, relative: str
) -> str | None:
    """
    Unlinks ``SKILL.md``. Returns a failure reason, or ``None`` on success.

    Uses the same pinned *skill_fd* as the existence probe, so it removes the
    file the probe found.
    """
    try:
        unlink_file(skill_dir, SKILL_FILENAME, dir_fd=skill_fd)
    except SymlinkRefused:
        return f"'{relative}' was not removed: the target file is a symlink"
    except OSError as exc:
        return f"'{relative}' could not be removed: {exc}"
    return None


def _prune_one(
    root: Path,
    root_fd: int | None,
    relative: str,
    key: str,
    entries: dict[str, Any],
) -> ReconcileAction:
    """Removes one managed skill file, and its directory when that empties it."""
    skill_dir = root / key
    target = skill_dir / SKILL_FILENAME
    version = entries[relative].get("version")

    unsafe = _unsafe_path_reason(root, skill_dir, target, key, require_directory=False)
    if unsafe is not None:
        return _prune_error(key, f"'{relative}' was not removed: {unsafe}", version)

    # Before the removal, so an orphaned temp file cannot block the ``rmdir``.
    _sweep_orphan_temp_files(root, root_fd, key)

    # Pinned before the existence probe and held through the unlink, so a
    # swapped directory cannot make a still-present file look already removed.
    # A missing directory just means the file is already gone.
    removed_from_disk = False
    try:
        with pinned_directory(skill_dir, dir_fd=root_fd) as skill_fd:
            if _skill_file_present(skill_dir, skill_fd):
                failure = _unlink_skill_file(skill_dir, skill_fd, relative)
                if failure is not None:
                    return _prune_error(key, failure, version)
                removed_from_disk = True
    except DirectoryMissing:
        pass
    except ValueError as exc:
        return _prune_error(key, f"'{relative}' was not removed: {exc}", version)

    if removed_from_disk:
        try:
            # Relative to the root descriptor; rmdir refuses a symlink and a
            # non-empty directory.
            if root_fd is not None:
                os.rmdir(key, dir_fd=root_fd)
            else:
                skill_dir.rmdir()
        except OSError:
            pass  # the directory is not empty: other files live here too

    entries.pop(relative, None)

    if removed_from_disk:
        record_revoked(key, version)

    return ReconcileAction(
        key=key,
        action="removed",
        version=version if is_valid_skill_version(version) else None,
        path=str(target),
    )


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
