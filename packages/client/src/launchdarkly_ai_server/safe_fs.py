"""
Descriptor-pinned filesystem primitives for writing files under a directory
another process may be racing to replace.

**The invariant:** every operation runs relative to a descriptor pinned to a
directory the caller already verified, never against a re-resolved path. A path
check is only as good as the last resolution after it.

**POSIX only.** Windows has no ``*at()`` family, so only a per-component
``lstat`` check runs there — a check-then-use race, not a closed window. On
Windows, write permission on the managed root is the only boundary; the README's
privilege-separated deployment is the mitigation.

Full threat model: ``agents.md``, *Descriptor-pinned filesystem access*.
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
"""Mode set explicitly on every written file: never from the umask, never
executable."""

_DIR_MODE = 0o755
"""Mode set explicitly on every directory this module creates, because
``mkdir``'s mode argument is masked by the umask: under ``0077`` a separate agent
identity could not traverse the directory to read what is inside."""

_SUPPORTS_FCHMOD = hasattr(os, "fchmod")
"""Whether the mode can be set on the descriptor. Probed because Windows only
has ``os.fchmod`` from CPython 3.13, and this package supports 3.12."""

SUPPORTS_DIR_FD = os.supports_dir_fd.issuperset(
    # renameat, openat, unlinkat, fstatat, mkdirat, and unlinkat(AT_REMOVEDIR).
    {os.rename, os.open, os.unlink, os.stat, os.mkdir, os.rmdir}
)
"""
Whether the ``*at()`` syscall family is available. Gates every descriptor-pinned
operation; ``False`` falls back to the ``lstat`` check.

**Do not "correct" the names in this probe.** CPython registers ``renameat``
under ``os.rename`` only and ``fstatat`` under ``os.stat`` only, so probing
``os.replace`` / ``os.lstat`` reports "unsupported" everywhere and silently
disables the defense.
"""


class DirectoryMissing(ValueError):
    """
    Raised by ``open_directory_nofollow`` when the path is absent (``ENOENT``)
    or is not a directory (``ENOTDIR``).

    A ``ValueError`` subclass, so callers can catch every unpinnable directory
    alike, or catch this alone to treat "not there" as an ordinary outcome. A
    symlink never raises this; it is refused as a symlink.
    """


def _at(directory: Path, dir_fd: int | None) -> str | Path:
    """
    How to name *directory* given a descriptor for its parent: the bare final
    component with a *dir_fd*, the full path without. One helper, because a
    single call site left on the full path would reopen the race.
    """
    return directory.name if dir_fd is not None else directory


def open_directory_nofollow(
    directory: Path, *, dir_fd: int | None = None
) -> int | None:
    """
    Opens *directory* without following a final symlink, and pins it.

    *dir_fd* is a descriptor for the *parent*. Without one, ``O_NOFOLLOW`` only
    protects the final component; every ancestor is re-resolved on the open.

    Returns ``None`` where the ``*at()`` family is absent, after an ``lstat``
    check for a real, non-symlink directory (``os.open`` cannot open a directory
    on Windows).

    Raises ``ValueError`` when the path will not open as a real directory, or
    ``DirectoryMissing`` when nothing is there or it is not a directory. Both are
    decided by the open (or ``lstat``) itself, never by a separate existence check.
    """
    if not SUPPORTS_DIR_FD:
        try:
            mode = os.lstat(directory).st_mode
        except FileNotFoundError as exc:
            raise DirectoryMissing(f"the directory does not exist: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"the directory could not be inspected: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError("the directory is a symlink")
        if not stat.S_ISDIR(mode):
            raise DirectoryMissing("the path is not a directory")
        return None

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(_at(directory, dir_fd), flags, dir_fd=dir_fd)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise DirectoryMissing(f"the directory does not exist: {exc}") from exc
    except OSError as exc:
        raise ValueError(
            f"the directory could not be opened without following links: {exc}"
        ) from exc
    try:
        # Covers platforms without O_DIRECTORY.
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

    Uses ``os.mkdir`` plus an ``lstat`` on ``FileExistsError``, because
    ``Path.mkdir(exist_ok=True)`` accepts an existing symlink-to-directory.

    *dir_fd* is a descriptor for the parent. The ``mkdir`` needs it too:
    ``mkdir`` follows a symlinked parent, so creating against the full path
    could create (and then write into) a directory outside the root.
    """
    # Without the *at() family the mkdir below would raise on a dir_fd.
    if not SUPPORTS_DIR_FD:
        dir_fd = None
    created = False
    try:
        os.mkdir(_at(directory, dir_fd), _DIR_MODE, dir_fd=dir_fd)
        created = True
    except FileExistsError:
        # os.stat(follow_symlinks=False) is the spelling os.supports_dir_fd
        # advertises; it is equivalent to os.lstat.
        if dir_fd is not None:
            mode = os.stat(directory.name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        else:
            mode = os.lstat(directory).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("the directory is a symlink") from None
        if not stat.S_ISDIR(mode):
            raise ValueError("the path is not a directory") from None
    fd = open_directory_nofollow(directory, dir_fd=dir_fd)
    # Only a directory this call created: an existing one keeps the mode its
    # owner gave it. fchmod on the pinned descriptor, never chmod on the path,
    # which a swap between mkdir and open could redirect. Without a descriptor
    # (no *at() family, i.e. Windows) there are no POSIX modes to correct.
    if created and fd is not None and _SUPPORTS_FCHMOD:
        try:
            os.fchmod(fd, _DIR_MODE)
        except BaseException:
            os.close(fd)
            raise
    return fd


@contextmanager
def pinned_directory(
    directory: Path, *, create: bool = False, dir_fd: int | None = None
) -> Iterator[int | None]:
    """
    Holds *directory* pinned for the duration of the block, then closes it.

    Yields a descriptor for *directory*, or ``None`` where the ``*at()`` family
    is absent. *dir_fd*, if given, is the parent's descriptor. Raises
    ``ValueError`` for a directory that cannot be pinned.
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

    An ``OSError`` subclass, so callers that only care that removal failed need
    no extra ``except``.
    """


def unlink_file(directory: Path, name: str, *, dir_fd: int | None) -> None:
    """
    Removes ``<directory>/<name>``, refusing to follow a symlink at *name*.

    Descriptor-relative because ``unlink`` resolves the directory above the
    name: a swapped ``<directory>`` would otherwise delete an attacker-chosen file.

    Raises ``SymlinkRefused`` when *name* is a symlink: a link where the SDK
    expects its own file means the disk does not match the manifest, which the
    caller should report rather than silently tidy.
    """
    if dir_fd is None:
        # No *at() family: path-based check and unlink.
        target = directory / name
        if target.is_symlink():
            raise SymlinkRefused(f"{name} is a symlink")
        target.unlink()
        return

    probe = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    if stat.S_ISLNK(probe.st_mode):
        raise SymlinkRefused(f"{name} is a symlink")
    os.unlink(name, dir_fd=dir_fd)


_TEMP_SUFFIX = ".tmp"
"""Suffix on every temp file this module creates."""

_TEMP_TOKEN_BYTES = 8
"""Bytes of randomness in a temp name."""

_TEMP_TOKEN_PATTERN = re.compile(
    # secrets.token_hex on the descriptor path; tempfile.mkstemp's eight
    # [a-z0-9_] characters otherwise. Used with fullmatch.
    rf"[0-9a-f]{{{_TEMP_TOKEN_BYTES * 2}}}|[a-z0-9_]{{8}}"
)


def temp_name_prefix(name: str) -> str:
    """
    The prefix every temp file for *name* is created under. Shared by
    ``atomic_write`` and the orphan sweep so the two cannot drift.
    """
    return f".{name}."


def is_temp_name(candidate: str, name: str) -> bool:
    """
    Whether *candidate* is a temp name this module could have created for *name*.

    Deliberately strict (prefix, token and suffix must match exactly), because
    callers delete the file when this returns ``True``.
    """
    prefix = temp_name_prefix(name)
    if not candidate.startswith(prefix) or not candidate.endswith(_TEMP_SUFFIX):
        return False
    token = candidate[len(prefix) : -len(_TEMP_SUFFIX)]
    return _TEMP_TOKEN_PATTERN.fullmatch(token) is not None


def _mkstemp_at(dir_fd: int, prefix: str) -> tuple[int, str]:
    """
    ``tempfile.mkstemp`` relative to a directory descriptor: ``O_CREAT | O_EXCL``
    on an unpredictable name, retried on collision, so a planted temp path is
    never written through.
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

    The temp file is created exclusively in the target's own directory (so the
    rename is not cross-device), written, fsynced, renamed over the target, and
    the directory fsynced so the rename survives a crash. The mode is set
    explicitly, never executable.

    With a *dir_fd*, every step runs relative to that descriptor; without one,
    against full paths.

    Uses ``os.replace``, not ``os.rename``: only ``os.replace`` overwrites on
    Windows.
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
            # fchmod on the descriptor cannot be redirected by a swap of the
            # temp path, and overrides the 0600 both creation paths use.
            if _SUPPORTS_FCHMOD:
                os.fchmod(fd, _FILE_MODE)
            elif at_fd is None:
                # Windows before 3.13: no fchmod and no *at() family, so temp is
                # a full path. Guarded on at_fd so a platform with descriptors
                # but no fchmod keeps 0600 rather than chmod a bare relative name.
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
    ``atomic_write`` for a directory the caller does not already hold open.

    The directory is opened with ``O_NOFOLLOW``, so one swapped for a symlink
    fails the write. Its ancestors are still re-resolved, so prefer
    ``atomic_write`` with a descriptor you already hold.
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
