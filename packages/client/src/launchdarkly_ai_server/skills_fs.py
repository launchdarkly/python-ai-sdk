"""
Agent Skills — filesystem materialization.

The highest-blast-radius layer of the feature: this is the part that writes to a
customer's disk. Split out of ``skills.py`` on that boundary — everything here
takes already-verified content and reconciles it against a managed root, while
``skills.py`` owns retrieval and verification and knows nothing about the
filesystem. The dependency runs one way only, and the descriptor-pinned
primitives every destructive step goes through live in ``safe_fs.py``.

The reconcile is manifest-driven and fails closed: destructive operations only
ever touch paths ``<root>/.launchdarkly-skills.json`` records under a matching
key, a corrupt manifest suppresses every destructive action, and an incomplete
retrieval suppresses pruning. Content is re-verified immediately before the
write, because a ``Skill`` can also be constructed directly by a caller.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .safe_fs import (
    SymlinkRefused,
    atomic_write,
    atomic_write_in,
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
    record_materialized,
    record_revoked,
    reference_target,
    resolve_from_store,
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

SKILL_FILENAME = "SKILL.md"
"""The single file each skill materializes to, under ``<root>/<key>/``."""

OnUnavailable = Literal["keep", "raise"]
"""How ``write_skills`` reacts to content it could not retrieve."""

_UNAVAILABLE_PREFIX = "skill retrieval unavailable: "
"""
Prefix on every error describing content that could not be retrieved. Callers
assert on it, so it lives in one place.
"""

_MAX_PATH_COMPONENT_BYTES = 255
"""
NAME_MAX on Linux and macOS, and the component limit on Windows. A skill key
becomes a single directory name, and the data model permits keys up to 256
characters — one byte longer than any of those filesystems can represent. Such a
key is rejected before any filesystem call so the caller gets a reported action
rather than an ENAMETOOLONG escaping from a stat deep inside the reconcile.
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
    the literal ``"*"`` meaning everything ``all_skills()`` returns. ``Skill``
    values are used as-is; references and strings resolve through the accessors,
    so they need a configured store.

    The reconcile is manifest-driven (``<root>/.launchdarkly-skills.json``):
    destructive operations only ever touch paths the manifest records under a
    matching key, so a file the SDK did not write is never overwritten or
    deleted. ``prune`` removes formerly-managed skills that are no longer in the
    requested set — which is also how revocation takes effect. ``timeout``
    bounds retrieval, the writes, and pruning; the final manifest rewrite
    always runs, so files already written are never orphaned. ``on_unavailable``
    chooses between reporting a failed retrieval (``"keep"``, leaving existing
    managed files alone) and raising (``"raise"``).

    Returns a ``ReconcileReport`` in which every outcome is visible; raises
    ``ValueError`` for a caller error such as an unusable root.

    **This call performs synchronous filesystem I/O and does not yield.** It is
    ``async`` for signature parity with the other accessors and with the
    TypeScript SDK, not because it awaits anything: every read, write, ``fsync``
    and rename runs inline, so a large reconcile blocks the event loop for its
    duration. Wrap it in ``asyncio.to_thread`` if that matters on your loop.
    ``timeout`` is checked between steps rather than interrupting one in
    progress, for the same reason.

    **One root, one reconcile at a time.** Because nothing here yields, a whole
    reconcile is atomic against every other task on the loop today. Wrapping it
    to run concurrently makes that the caller's problem instead: two runs
    against the same root interleave on the manifest, and the loser's entries
    are lost — which leaves the files it wrote unmanaged, and a later reconcile
    then refuses them as files the SDK did not write.
    """
    # Both of these are annotated as closed sets, but the values can still arrive
    # from untyped code, so they are checked rather than assumed.
    if on_unavailable not in ("keep", "raise"):
        raise ValueError(
            f'on_unavailable must be "keep" or "raise", got {on_unavailable!r}'
        )
    if timeout < 0:
        raise ValueError(f"timeout must not be negative, got {timeout!r}")

    deadline = time.monotonic() + timeout
    root_path = _resolve_root(root)
    manifest, manifest_error = _load_manifest(root_path)
    entries: dict[str, Any] = manifest.get("entries", {})

    actions: list[ReconcileAction] = []
    if manifest_error is not None:
        # Run-level failure: there is no single skill key to hang it off.
        actions.append(_run_error(manifest_error))

    requests, incomplete = _resolve_requests(skills, deadline, on_unavailable)

    written, write_timed_out = _write_all(root_path, requests, entries, deadline)
    actions.extend(written)
    incomplete = incomplete or write_timed_out

    # Pruning is destructive, so it needs a trustworthy picture of both sides: a
    # corrupt manifest means we do not know what we own, and an incomplete run —
    # a retrieval that failed, or a deadline that expired mid-write — means we do
    # not know what is still current. Either way, deleting would be a guess.
    if prune and manifest_error is None and not incomplete:
        actions.extend(
            _prune(
                root_path,
                entries,
                {request.key for request in requests},
                deadline,
            )
        )

    if manifest_error is None:
        actions.extend(_rewrite_manifest(root_path, manifest, entries))

    return ReconcileReport(actions=actions)


_RUN_LEVEL_KEY = ""
"""
The documented sentinel for a failure that belongs to no single skill (see
``ReconcileAction``). Spelled once so every path that cannot attribute a
failure to a key agrees with the others.
"""


def _run_error(message: str) -> ReconcileAction:
    """
    A failure belonging to the run rather than to one skill.

    Uses the run-level sentinel key; it is constructed here so every run-level
    error agrees.
    """
    return ReconcileAction(key=_RUN_LEVEL_KEY, action="error", error=message)


def _write_all(
    root: Path,
    requests: list[_PendingWrite],
    entries: dict[str, Any],
    deadline: float,
) -> tuple[list[ReconcileAction], bool]:
    """
    Reconciles every pending write. Returns ``(actions, timed out mid-run)``.

    The loop never aborts: a per-skill failure becomes an ``error`` action and the
    next skill is attempted, because returning early would skip the caller's
    manifest rewrite and orphan every file already written in this run.
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
            actions.append(_write_one(root, request.skill, entries))
        except OSError as exc:
            # A safety net, not the primary defense. pathlib's stat probes swallow
            # only ENOENT/ENOTDIR/EBADF/ELOOP and re-raise every other errno, so an
            # unexpected filesystem condition must not abort the loop.
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
    root: Path, manifest: dict[str, Any], entries: dict[str, Any]
) -> list[ReconcileAction]:
    """Writes the updated manifest. Returns an error action, or nothing."""
    manifest["manifestVersion"] = MANIFEST_VERSION
    manifest["entries"] = entries
    try:
        # json.dumps is inside the guard: indent= selects the pure-Python encoder,
        # and unknown fields must be round-tripped, so a deeply nested
        # planted field can raise RecursionError here — after every skill file is
        # already on disk.
        serialized = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        atomic_write_in(root, MANIFEST_FILENAME, serialized)
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

    Written once because this gate is what sets ``unavailable`` and therefore
    suppresses pruning. If it were maintained in two places, a condition added
    to one and not the other would not merely produce a wrong message — it
    would delete the user's files.
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
    return store


def _resolve_requests(
    skills: Sequence[Skill | SkillReference | str] | str,
    deadline: float,
    on_unavailable: OnUnavailable,
) -> tuple[list[_PendingWrite], bool]:
    """
    Turns the caller's input into one request per skill.

    Returns the requests plus whether any retrieval was left incomplete — an
    absent store, a raising store, or an exhausted timeout. That flag suppresses
    pruning: deleting managed files because retrieval failed would turn a
    transport outage into data loss.
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

    Same core as the accessors, plus the two conditions only this path treats as
    data rather than as an exception: an exhausted deadline and an absent store.
    """
    store = _available_store(deadline, f"'{key}'")
    if isinstance(store, _RetrievalBlocked):
        return Resolution(error=store.reason, unavailable=True)

    resolved = resolve_from_store(store, key, wanted_version)
    if resolved.unavailable and resolved.error is not None:
        return Resolution(error=_unavailable(resolved.error), unavailable=True)
    return resolved


def _unavailable_run(
    error: str, on_unavailable: OnUnavailable
) -> tuple[list[_PendingWrite], bool]:
    """
    One run-level retrieval failure — raised, or reported against the empty key.

    Always reports the run incomplete, which is what suppresses pruning: nothing
    was retrieved, so every managed file on disk has to be assumed current.
    """
    if on_unavailable == "raise":
        raise RuntimeError(error)
    return [_PendingWrite(key="", error=error)], True


def _pending_for_raw(object_key: str, raw: Any) -> _PendingWrite:
    """
    One raw store object as a pending write — verified, or reported as failed.

    Present but unverifiable is NOT the same as revoked. Dropping it silently
    would leave the key out of the requested set, so prune would delete the last
    known-good copy already on disk and report a routine "removed" with
    report.ok still true. A failed request instead gets the same treatment the
    reference path already gives (see ``_resolve_reference``): the outcome is
    surfaced, and the key stays in the requested set so nothing is pruned.
    """
    skill = verify_raw_skill(raw)
    if skill is not None:
        return _PendingWrite(key=skill.key, skill=skill)
    # The on-disk copy lives under the object's *own* key, which a custom store
    # may key differently in ``all_objects``. The failure must be recorded under
    # the object's key, or the copy written under it on an earlier run would
    # fall out of the requested set and be pruned — the very deletion this
    # function exists to prevent.
    raw_key = raw.get("key") if isinstance(raw, dict) else None
    key = raw_key if is_valid_skill_key(raw_key) else object_key
    if not is_valid_skill_key(key):
        # Neither key is usable, so this failure cannot be attributed to a skill
        # — the run-level sentinel is the honest report.
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

    # Deliberately not via all_skills(), which reports a raising store as an
    # empty result — that would look like "every skill was revoked" and let
    # prune delete the lot.
    objects, error = list_raw_objects(store)
    if error is not None:
        return _unavailable_run(_unavailable(error), on_unavailable)

    return [_pending_for_raw(key, raw) for key, raw in objects.items()], False


# -------------------------------------------------------------------------
# The managed root and its manifest
# -------------------------------------------------------------------------


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    """
    Resolves the managed root once, up front.

    An unusable root is a caller error rather than a per-skill outcome, so this
    raises. Only the leaf directory is ever created — recursively creating
    missing ancestors would let a typo scatter a directory tree.
    """
    path = Path(os.fspath(root))

    # pathlib re-raises any errno outside ENOENT/ENOTDIR/EBADF/ELOOP, so an
    # unreadable parent would surface as PermissionError where the docs
    # promise ValueError.
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


def _load_manifest(root: Path) -> tuple[dict[str, Any], str | None]:
    """
    Loads the manifest. Returns ``(manifest, error)``.

    A manifest that cannot be read, cannot be parsed, is not an object, carries a
    ``manifestVersion`` this release does not understand, or has a malformed
    ``entries`` map is **corrupt**. The caller then performs no destructive
    action and leaves the file itself alone: rewriting it would destroy the only
    record of what the SDK owns, and acting on a manifest we cannot read would
    mean guessing at which of the customer's files are ours.

    An absent manifest is not corrupt — that is simply a fresh root.
    """
    path = root / MANIFEST_FILENAME
    fresh: dict[str, Any] = {"manifestVersion": MANIFEST_VERSION, "entries": {}}

    if not path.exists():
        return fresh, None

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError is a ValueError, not an OSError: non-UTF-8 bytes in
        # the manifest are corruption, and must fail closed like any other.
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
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > MANIFEST_VERSION
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
    The path defenses, in one place.

    Returns why ``<root>/<key>/SKILL.md`` must not be touched, or ``None``.
    Shared by the write and prune paths: ``agents.md`` marks these checks
    non-relaxable, and maintaining them twice is how they drift.

    *require_directory* is the one genuine difference between the two callers. A
    write needs a real directory to write into. A prune only needs to not follow
    a link — an entry whose directory has been replaced by a plain file has
    already lost the file this SDK owned, so reporting ``removed`` is what lets
    the stale manifest entry be dropped rather than pinned forever.

    Note that the containment check is unconditional even though ``skill_dir``
    may not exist yet: ``realpath`` resolves the existing prefix and appends the
    rest, so a fresh key under a valid root passes.
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

    Re-validated locally whatever any upstream layer already did, and
    before any filesystem call, because a key becomes a path component. Shared by
    the write and the prune paths so the two cannot disagree about which keys
    this SDK could own; ``agents.md`` marks these checks non-relaxable, and
    maintaining them twice is how they drift.

    ``key.encode`` is safe here only because it runs *after* the pattern check:
    the key grammar admits no surrogate, so there is no unencodable key left to
    raise on. Do not reorder these two.
    """
    if not is_valid_skill_key(key):
        return f"{key!r} is not a valid skill key: it {skill_key_rejection_reason(key)}"
    # The data model allows 256 characters; no mainstream filesystem allows a
    # 256-byte path component. Catch it here so it is a reported action rather
    # than an ENAMETOOLONG raised from the first stat in the caller.
    key_bytes = len(key.encode("utf-8"))
    if key_bytes > _MAX_PATH_COMPONENT_BYTES:
        return (
            f"skill key '{key[:32]}...' is {key_bytes} bytes, over the "
            f"{_MAX_PATH_COMPONENT_BYTES}-byte limit for a single directory name"
        )
    return None


def _write_one(root: Path, skill: Skill, entries: dict[str, Any]) -> ReconcileAction:
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

    # Re-verify immediately before writing, through the same core the accessors
    # use: a Skill can also be constructed directly by a caller.
    verified = verified_bytes(key, skill.content, skill.content_hash, skill.version)
    if isinstance(verified, VerificationFailure):
        return failed(
            f"skill '{key}' failed verification immediately before writing: "
            f"{verified.reason}; nothing was written"
        )
    encoded, content_hash = verified.encoded, verified.content_hash

    # Overwrite only what the manifest records as ours under this key.
    entry = entries.get(relative)
    managed = isinstance(entry, dict) and entry.get("key") == key
    exists = target.exists()

    if exists and not managed:
        return failed(
            f"'{relative}' exists but the manifest does not record it as managed "
            f"under key '{key}'; refusing to overwrite a file this SDK did not write"
        )

    if exists:
        try:
            on_disk = _read_regular_file(target)
        except OSError as exc:
            return failed(f"'{relative}' could not be read: {exc}")

        if hashlib.sha256(on_disk).hexdigest() == content_hash:
            _update_entry(entries, relative, skill, content_hash)
            record_materialized(key, len(encoded), content_hash, "skipped_current")
            return ReconcileAction(
                key=key,
                action="skipped_current",
                version=skill.version,
                path=str(target),
            )
        # Stale version or local tampering — LD-resolved content wins.
        action: ReconcileActionKind = "updated"
    else:
        action = "written"

    write_error = _write_through_descriptor(skill_dir, encoded, key, relative)
    if write_error is not None:
        return failed(write_error)

    _update_entry(entries, relative, skill, content_hash)
    record_materialized(key, len(encoded), content_hash, action)
    return ReconcileAction(
        key=key, action=action, version=skill.version, path=str(target)
    )


def _read_regular_file(target: Path) -> bytes:
    """
    Reads *target*, refusing anything that is not a regular file.

    A plain ``Path.read_bytes`` would ``open()`` by name — and opening a FIFO
    with no writer blocks forever, so an attacker who can swap the managed file
    for one (the same capability the symlink checks defend against) could hang
    the whole reconcile, and the event loop with it. ``O_NONBLOCK`` makes that
    open return immediately (it is a no-op for regular files), ``O_NOFOLLOW``
    refuses a trailing symlink, and the ``fstat`` on the descriptor — not the
    path — is what the type check trusts. ``O_BINARY`` is what keeps these
    bytes the *verbatim* bytes: it is 0 on POSIX, but on Windows a descriptor
    without it translates CRLF on read, which would fail the hash comparison
    against content that is actually current.
    """
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(target, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("the target file is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _write_through_descriptor(
    skill_dir: Path, encoded: bytes, key: str, relative: str
) -> str | None:
    """
    Performs the write itself. Returns a failure reason, or ``None`` on success.

    Split out of ``_write_one`` because everything above it decides *whether* to
    write and this decides nothing: the directory is pinned to a descriptor and
    every remaining step is relative to it, so none of the checks above can be
    invalidated by a swap between here and the rename.
    """
    try:
        with pinned_directory(skill_dir, create=True) as dir_fd:
            try:
                atomic_write(skill_dir, SKILL_FILENAME, encoded, dir_fd=dir_fd)
            except OSError as exc:
                return f"'{relative}' could not be written: {exc}"
    except OSError as exc:
        return f"the directory for skill '{key}' could not be created: {exc}"
    except ValueError as exc:
        return f"'{relative}' was refused: {exc}"
    return None


def _update_entry(
    entries: dict[str, Any], relative: str, skill: Skill, content_hash: str
) -> None:
    """
    Records a managed path in the manifest.

    Merges into any existing entry rather than replacing it, so fields written by
    a future SDK release survive this one's rewrite.

    ``sha256`` and ``writtenAt`` are recorded for forensics only: the reconcile
    decides currency by hashing the bytes on disk, precisely because the
    manifest is untrusted, so neither field is ever read back as a decision
    input.
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
    A prune refusal. Mirrors ``_write_one``'s local ``failed`` helper.

    *version* comes off the manifest, which is untrusted, so it is validated here
    rather than at each call site — the same guard the ``removed`` action applies,
    so a refusal and a removal report the field identically.
    Callers that genuinely do not know a version pass nothing; none of them may
    invent one.
    """
    return ReconcileAction(
        key=key,
        action="error",
        version=version if is_valid_skill_version(version) else None,
        error=message,
    )


def _prune(
    root: Path, entries: dict[str, Any], requested: set[str], deadline: float
) -> list[ReconcileAction]:
    """
    Removes managed skills that are no longer requested.

    This is also how revocation takes effect: a revoked skill is simply absent
    from the resolved set, so the next reconcile removes it. There is
    deliberately no opt-out.

    The deadline applies here just as it does to the writes: a skill left
    unpruned is reported as an error and stays in the manifest, so the next
    reconcile picks it up.
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
            actions.append(_prune_one(root, relative, key, entries))
        except OSError as exc:
            actions.append(
                _prune_error(
                    key,
                    f"'{relative}' could not be removed: {exc}",
                    entry.get("version"),
                )
            )

    return actions


def _unlink_through_descriptor(skill_dir: Path, relative: str) -> str | None:
    """
    Performs the removal itself. Returns a failure reason, or ``None`` on success.

    The mirror of ``_write_through_descriptor``, and split out for the same
    reason: everything above it decides *whether* to remove, and this decides
    nothing. The directory is pinned before the unlink because unlink never
    follows a trailing symlink but does resolve the directory above it, so a
    ``<root>/<key>`` swapped for a symlink between the checks and here would
    otherwise delete a file outside the root.
    """
    try:
        with pinned_directory(skill_dir) as dir_fd:
            try:
                unlink_file(skill_dir, SKILL_FILENAME, dir_fd=dir_fd)
            except SymlinkRefused:
                return f"'{relative}' was not removed: the target file is a symlink"
            except OSError as exc:
                return f"'{relative}' could not be removed: {exc}"
    except ValueError as exc:
        return f"'{relative}' was not removed: {exc}"
    return None


def _prune_one(
    root: Path, relative: str, key: str, entries: dict[str, Any]
) -> ReconcileAction:
    """Removes one managed skill file, and its directory when that empties it."""
    skill_dir = root / key
    target = skill_dir / SKILL_FILENAME
    version = entries[relative].get("version")

    unsafe = _unsafe_path_reason(root, skill_dir, target, key, require_directory=False)
    if unsafe is not None:
        return _prune_error(key, f"'{relative}' was not removed: {unsafe}", version)

    removed_from_disk = False
    if target.exists():
        failure = _unlink_through_descriptor(skill_dir, relative)
        if failure is not None:
            return _prune_error(key, failure, version)
        removed_from_disk = True
        try:
            # Path-based, and safe that way: rmdir never follows a trailing
            # symlink (it fails ENOTDIR) and only ever succeeds on an empty
            # directory.
            skill_dir.rmdir()
        except OSError:
            pass  # the customer keeps their own files here too

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
