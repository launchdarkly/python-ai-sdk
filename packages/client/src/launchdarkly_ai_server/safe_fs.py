"""
Descriptor-pinned filesystem primitives.

Nothing here knows what a skill is: this is the "write a file under a directory
an attacker may be racing you for" problem, solved once. ``skills_fs.py`` is the
only caller today.

**The invariant:** every operation runs relative to a descriptor pinned to a
directory the caller already verified, never against a re-resolved path. A path
check is only as good as the last resolution after it.

**POSIX only.** Windows has no ``*at()`` family, so only the per-component
``lstat`` floor runs there — a check-then-use race rather than a closed window.
Write permission on the managed root is therefore the only boundary on Windows,
which is why the README documents a privilege-separated deployment as the
mitigation rather than as advice.

The threat model, the platform decision behind it, and what must not be relaxed
are in ``agents.md`` under *Descriptor-pinned filesystem access*.
"""

from __future__ import annotations

import errno
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_FILE_MODE = 0o644
"""Mode set explicitly on every written file — never inherited from the umask,
and never executable."""

_SUPPORTS_FCHMOD = hasattr(os, "fchmod")
"""
Whether the mode can be set on the descriptor rather than on a path.

POSIX always has ``os.fchmod``; Windows only gained it in CPython 3.13, and this
package supports 3.12. Probing for it rather than assuming it is what keeps the
documented Windows fallback a fallback instead of an ``AttributeError`` raised
after the temp file is already open.
"""

SUPPORTS_DIR_FD = os.supports_dir_fd.issuperset(
    # renameat, openat, unlinkat, fstatat, mkdirat, and unlinkat's AT_REMOVEDIR
    # form — the six this module and its caller need.
    {os.rename, os.open, os.unlink, os.stat, os.mkdir, os.rmdir}
)
"""
Whether the ``*at()`` syscall family is available. Gates every descriptor-pinned
operation in this module; ``False`` falls back to the ``lstat`` floor.

**Do not "correct" the names in this probe.** ``os.supports_dir_fd`` is
populated per underlying syscall, and CPython registers ``renameat`` under
``rename`` only and ``fstatat`` under ``stat`` only — so probing the
``os.replace`` and ``os.lstat`` this module actually calls reports
"unsupported" on every POSIX platform and silently disables the defense.
"""


def _at(directory: Path, dir_fd: int | None) -> str | Path:
    """
    What to name *directory* by, given a descriptor for its parent.

    With a *dir_fd*, the bare final component, so the kernel resolves it inside
    the pinned parent; without one, the full path. Spelled once because a single
    call site left on the full path would silently reopen the window the
    descriptor closes.
    """
    return directory.name if dir_fd is not None else directory


def open_directory_nofollow(
    directory: Path, *, dir_fd: int | None = None
) -> int | None:
    """
    Opens *directory* without following a final symlink, and pins it.

    *dir_fd* is a descriptor for the *parent*. Passing one extends the guarantee
    past the final component: ``O_NOFOLLOW`` refuses a link at *directory*
    itself, but without a parent descriptor every ancestor is re-resolved on
    each open.

    Returns ``None`` where the ``*at()`` family is absent, after an ``lstat``
    check for a real non-symlink directory. It must not attempt the descriptor
    open there: ``os.open`` cannot open a directory on Windows, so that path
    would fail every operation rather than fall back.

    Raises ``ValueError`` when the path will not open, or inspect, as a real
    directory.
    """
    if not SUPPORTS_DIR_FD:
        try:
            mode = os.lstat(directory).st_mode
        except OSError as exc:
            raise ValueError(f"the directory could not be inspected: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError("the directory is a symlink")
        if not stat.S_ISDIR(mode):
            raise ValueError("the path is not a directory")
        return None

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(_at(directory, dir_fd), flags, dir_fd=dir_fd)
    except OSError as exc:
        raise ValueError(
            f"the directory could not be opened without following links: {exc}"
        ) from exc
    try:
        # O_DIRECTORY already guarantees this wherever the platform defines it;
        # the explicit check is what covers the platforms that do not.
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError("the path is not a directory")
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_or_create_directory(
    directory: Path, *, dir_fd: int | None = None
) -> int | None:
    """
    Creates *directory* if absent and returns a descriptor pinned to it.

    ``os.mkdir`` plus an ``lstat`` on the ``FileExistsError`` path, never
    ``Path.mkdir(exist_ok=True)``: that accepts an existing
    symlink-to-directory as "already there", reopening the hole the caller's
    check just closed.

    *dir_fd* is a descriptor for the parent, and the ``mkdir`` needs it as much
    as the open does — ``mkdir`` follows a symlink at its parent, so a create
    against the full path is how a directory gets made, and then written into,
    outside the root.
    """
    # A parent descriptor is only usable where the ``*at()`` family is, and the
    # mkdir below would raise rather than take the floor without this.
    if not SUPPORTS_DIR_FD:
        dir_fd = None
    try:
        os.mkdir(_at(directory, dir_fd), 0o755, dir_fd=dir_fd)
    except FileExistsError:
        # os.stat(follow_symlinks=False) on the descriptor path, os.lstat off it:
        # identical results, and the former is the spelling os.supports_dir_fd
        # advertises.
        if dir_fd is not None:
            mode = os.stat(directory.name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        else:
            mode = os.lstat(directory).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("the directory is a symlink") from None
        if not stat.S_ISDIR(mode):
            raise ValueError("the path is not a directory") from None
    return open_directory_nofollow(directory, dir_fd=dir_fd)


@contextmanager
def pinned_directory(
    directory: Path, *, create: bool = False, dir_fd: int | None = None
) -> Iterator[int | None]:
    """
    Holds *directory* pinned for the duration of the block, then releases it.

    Yields what the two openers above return — a descriptor, or ``None`` on the
    ``lstat`` floor — so a caller states the platform split once and cannot
    forget the ``os.close``. Raises ``ValueError`` for a directory that will not
    pin.

    Note which descriptor is which: *dir_fd* pins the parent, and the yielded
    one pins *directory* itself.
    """
    dir_fd = (
        open_or_create_directory(directory, dir_fd=dir_fd)
        if create
        else open_directory_nofollow(directory, dir_fd=dir_fd)
    )
    try:
        yield dir_fd
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


class SymlinkRefused(OSError):
    """
    Raised instead of removing a symlink found where a real file was expected.

    An ``OSError`` subclass so a caller that only cares that the removal failed
    keeps its single ``except``; a distinct type so one that must report *this*
    refusal specifically does not have to match on a message.
    """


def unlink_file(directory: Path, name: str, *, dir_fd: int | None) -> None:
    """
    Removes ``<directory>/<name>``, refusing to follow a symlink at *name*.

    Descriptor-relative for the same reason as ``atomic_write``: ``unlink``
    never follows a *trailing* symlink, but it does resolve the directory above
    it, so a swapped ``<directory>`` would turn this into a delete of an
    attacker-chosen file.

    Raises ``SymlinkRefused`` when *name* is a symlink — refusing rather than
    removing, because a link where the SDK expects its own file means the state
    on disk is not what the manifest describes, and that is the caller's to
    report rather than to tidy away.
    """
    if dir_fd is None:
        # No ``*at()`` family: the trailing-symlink check and the unlink are both
        # path-based, the per-component floor.
        target = directory / name
        if target.is_symlink():
            raise SymlinkRefused(f"{name} is a symlink")
        target.unlink()
        return

    # os.stat(follow_symlinks=False), not os.lstat: identical result, and it is
    # the spelling os.supports_dir_fd actually advertises.
    probe = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    if stat.S_ISLNK(probe.st_mode):
        raise SymlinkRefused(f"{name} is a symlink")
    os.unlink(name, dir_fd=dir_fd)


_TEMP_SUFFIX = ".tmp"
"""Suffix on every temp file this module creates."""

_TEMP_TOKEN_BYTES = 8
"""Bytes of randomness in a temp name, as ``secrets.token_hex`` takes them."""

_TEMP_TOKEN_PATTERN = re.compile(
    # Two producers, one recognizer: ``secrets.token_hex`` on the descriptor
    # path, ``tempfile.mkstemp``'s eight ``[a-z0-9_]`` characters on the
    # fallback. Used with ``fullmatch``, so both branches are anchored.
    rf"[0-9a-f]{{{_TEMP_TOKEN_BYTES * 2}}}|[a-z0-9_]{{8}}"
)


def temp_name_prefix(name: str) -> str:
    """
    The prefix every temp file for *name* is created under.

    Spelled once because two callers must agree: ``atomic_write`` creates the
    name and the orphan sweep recognizes it, and a second copy of the format
    would drift from the writer.
    """
    return f".{name}."


def is_temp_name(candidate: str, name: str) -> bool:
    """
    Whether *candidate* is a name this module could have created for *name*.

    Deliberately narrow — prefix, random token and suffix must all match, with
    nothing before or after — because the only thing a caller does with a
    ``True`` here is delete the file.
    """
    prefix = temp_name_prefix(name)
    if not candidate.startswith(prefix) or not candidate.endswith(_TEMP_SUFFIX):
        return False
    token = candidate[len(prefix) : -len(_TEMP_SUFFIX)]
    return _TEMP_TOKEN_PATTERN.fullmatch(token) is not None


def _mkstemp_at(dir_fd: int, prefix: str) -> tuple[int, str]:
    """
    ``tempfile.mkstemp`` for a directory descriptor.

    ``tempfile`` has no ``dir_fd`` form, so this reproduces the part that
    matters: ``O_CREAT | O_EXCL`` against an unpredictable name, retried on
    collision, so a planted temp path is never written through.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(tempfile.TMP_MAX):
        name = f"{prefix}{secrets.token_hex(_TEMP_TOKEN_BYTES)}{_TEMP_SUFFIX}"
        try:
            return os.open(name, flags, 0o600, dir_fd=dir_fd), name
        except FileExistsError:
            continue
    raise OSError(errno.EEXIST, "no usable temporary file name was found")


def atomic_write(
    directory: Path, name: str, data: bytes, *, dir_fd: int | None = None
) -> None:
    """
    Writes *data* to ``<directory>/<name>`` so no partial file is ever
    observable.

    The temp file is created exclusively in the target's *own* directory — one
    anywhere else would make the rename cross-device, and so not atomic —
    written, fsynced, renamed over the target, and the directory fsynced so the
    rename survives a crash. Mode is set explicitly rather than left to the
    umask, and the execute bit is never set.

    Given a *dir_fd*, every one of those steps runs relative to that descriptor
    and both names are bare filenames; without one, the identical sequence runs
    against full paths.

    ``os.replace`` is the one and only rename call site. ``os.rename`` must not
    be substituted for it: only ``os.replace`` has defined overwrite semantics
    on Windows.
    """
    at_fd = dir_fd if dir_fd is not None and SUPPORTS_DIR_FD else None
    prefix = temp_name_prefix(name)
    target: str | Path

    if at_fd is not None:
        fd, temp = _mkstemp_at(at_fd, prefix)
        target = name
    else:
        # mkstemp opens with O_CREAT|O_EXCL, so an existing temp path is never
        # reused.
        fd, temp = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=_TEMP_SUFFIX)
        target = directory / name

    try:
        try:
            # fchmod, not chmod: operating on the descriptor cannot be redirected
            # by a swap of the temp path, and it makes the mode independent of
            # the process umask (both creation paths open 0600).
            if _SUPPORTS_FCHMOD:
                os.fchmod(fd, _FILE_MODE)
            elif at_fd is None:
                # Windows before 3.13 has no fchmod — and no ``*at()`` family
                # either, so ``temp`` is a full path here and the mode goes on it.
                # That concedes nothing this platform was getting: it is already
                # on the path-based floor, the temp name is unguessable, and the
                # only bit Windows takes from a POSIX mode is read-only. A
                # platform with descriptors but no fchmod does not exist; were
                # there one it would keep the 0600 the exclusive open already set,
                # rather than have a bare name chmoded relative to the cwd.
                os.chmod(temp, _FILE_MODE)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        if at_fd is not None:
            os.replace(temp, target, src_dir_fd=at_fd, dst_dir_fd=at_fd)
        else:
            os.replace(temp, target)
    except BaseException:
        try:
            if at_fd is not None:
                os.unlink(temp, dir_fd=at_fd)
            else:
                os.unlink(temp)
        except OSError:
            pass
        raise

    if at_fd is not None:
        _fsync_directory_fd(at_fd)
    else:
        _fsync_directory(directory)


def atomic_write_in(directory: Path, name: str, data: bytes) -> None:
    """
    ``atomic_write`` against a directory this module does not already hold open.

    The descriptor is taken with ``O_NOFOLLOW``, so a directory swapped for a
    symlink between the caller's checks and the write fails it rather than
    redirecting it. That still leaves the directory's *ancestors* re-resolved on
    the open, so prefer ``atomic_write`` with a descriptor the caller already
    holds; this is for callers that hold nothing better.
    """
    with pinned_directory(directory) as dir_fd:
        atomic_write(directory, name, data, dir_fd=dir_fd)


def _fsync_directory_fd(fd: int) -> None:
    """Best effort — not every platform allows fsync on a directory descriptor."""
    try:
        os.fsync(fd)
    except OSError:
        pass


def _fsync_directory(directory: Path) -> None:
    """Best effort — not every platform lets a directory be opened for fsync."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        _fsync_directory_fd(fd)
    finally:
        os.close(fd)
