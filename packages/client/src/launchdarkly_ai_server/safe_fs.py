"""
Descriptor-pinned filesystem primitives.

Split out because none of this knows what a skill is: it is the "write a file
under a directory an attacker may be racing you for" problem, solved once.
``skills_fs.py`` is the only caller today.

The whole point is that a path check is only as good as the last path
resolution after it. Every operation here therefore runs relative to a
descriptor pinned to a directory the caller has already validated, rather than
re-resolving a name — which is what closes the swap window rather than merely
narrowing it. Where the platform has no ``*at()`` syscall family (Windows) the
identical sequence runs against full paths, the per-component ``lstat`` floor.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_FILE_MODE = 0o644
"""Mode set explicitly on every written file — never inherited from the umask,
and never executable."""

SUPPORTS_DIR_FD = os.supports_dir_fd.issuperset(
    # renameat, openat, unlinkat, fstatat — the four this module needs.
    {os.rename, os.open, os.unlink, os.stat}
)
"""
Whether the ``*at()`` syscall family is available, so every operation under the
managed root can be performed relative to a descriptor pinned to a directory
this module has already verified rather than re-resolved from its path.

That is what closes the swap window rather than merely narrowing it:
a descriptor refers to the inode that was checked, so replacing ``<root>/<key>``
with a symlink after the check cannot redirect a write or an unlink out of the
root. POSIX has these calls; Windows does not, and there the per-component
``lstat`` floor the spec permits applies instead.

The probe deliberately names ``os.rename`` and ``os.stat`` rather than the
``os.replace`` and ``os.lstat`` this module actually calls. ``os.supports_dir_fd``
is populated per underlying syscall, and CPython registers ``renameat`` under
``rename`` only and ``fstatat`` under ``stat`` only — even though ``os.replace``
is the same ``renameat``-backed function and ``os.lstat`` is ``fstatat`` with
``AT_SYMLINK_NOFOLLOW``, and both accept the descriptor keywords wherever their
advertised twin does (verified on CPython 3.12 and 3.13, macOS). Probing the
names this module calls would report "unsupported" on every POSIX platform and
silently disable the defense.
"""


def open_directory_nofollow(directory: Path) -> int | None:
    """
    Opens *directory* without following a final symlink, and pins it.

    Everything the caller does afterwards goes through the returned descriptor
    instead of the path, which is what turns the "narrow window" into no
    window at all: the descriptor names the inode that was checked, so swapping
    the path for a symlink between the check and the write cannot redirect the
    write out of the managed root.

    On a platform without the ``*at()`` family (Windows) this returns ``None``
    after verifying via ``lstat`` that the path is a real, non-symlink
    directory — the per-component floor. It must not attempt the descriptor
    open there: ``os.open`` goes through the CRT on Windows, which cannot open
    a directory at all, so the descriptor path would fail every operation
    rather than fall back.

    Raises ``ValueError`` when the path will not open (or inspect) as a real
    directory — the caller reports that as a refusal rather than letting it
    escape.
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
        fd = os.open(directory, flags)
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


def open_or_create_directory(directory: Path) -> int | None:
    """
    Creates *directory* if absent and returns a descriptor pinned to it.

    ``Path.mkdir(exist_ok=True)`` treats an existing symlink-to-directory as
    "already there", which would re-open the very hole the caller's ``lstat``
    check just closed. ``os.mkdir`` plus an ``lstat`` on the ``FileExistsError``
    path does not: a link reports as a link, and is refused.
    """
    try:
        os.mkdir(directory, 0o755)
    except FileExistsError:
        mode = os.lstat(directory).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("the directory is a symlink") from None
        if not stat.S_ISDIR(mode):
            raise ValueError("the path is not a directory") from None
    return open_directory_nofollow(directory)


@contextmanager
def pinned_directory(directory: Path, *, create: bool = False) -> Iterator[int | None]:
    """
    Holds *directory* pinned for the duration of the block, then releases it.

    Yields what the two openers above return — a descriptor, or ``None`` on the
    ``lstat`` floor — so the caller states the platform split once, as
    ``if dir_fd is not None``, and cannot forget the ``os.close``. Raises
    ``ValueError`` for a directory that will not pin, exactly as they do.
    """
    dir_fd = (
        open_or_create_directory(directory)
        if create
        else open_directory_nofollow(directory)
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

    The mirror of ``atomic_write``, and descriptor-relative for the same reason:
    ``unlink`` never follows a *trailing* symlink, but it does resolve the
    directory above it, so a ``<directory>`` swapped for a symlink after the
    caller's checks would otherwise turn this into a delete of an
    attacker-chosen file. Given a *dir_fd* the probe and the unlink both run
    against it; without one the identical sequence runs against full paths.

    Raises ``SymlinkRefused`` when *name* is a symlink. Note that this refuses
    rather than removes: ``unlink`` would happily delete the link itself, but a
    link where this SDK expects its own file means the state on disk is not what
    the manifest describes, and that is the caller's to report rather than to
    tidy away.
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


def _mkstemp_at(dir_fd: int, prefix: str) -> tuple[int, str]:
    """
    ``tempfile.mkstemp`` for a directory descriptor.

    ``tempfile`` has no ``dir_fd`` form, so this reproduces the part that
    matters: ``O_CREAT | O_EXCL`` against an unpredictable name, retried on
    collision, so an existing temp path is never reused and a planted one is
    never written through.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(tempfile.TMP_MAX):
        name = f"{prefix}{secrets.token_hex(8)}.tmp"
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
    anywhere else would make the rename cross-device, and therefore not atomic —
    written, fsynced, renamed over the target, and the directory fsynced so the
    rename itself survives a crash. Mode is set explicitly rather than left to
    the process umask, and the execute bit is never set.

    Given a *dir_fd* on a platform with the ``*at()`` family, every one of those
    steps runs relative to that descriptor and both names are bare filenames.
    Without one (Windows) the identical sequence runs against full paths, which
    is the per-component ``lstat`` floor.

    ``os.replace`` is the one and only rename call site, reached by attribute
    lookup on the ``os`` module so tests can intercept it; ``os.rename`` must
    not be substituted for it (it is also the only one with defined overwrite
    semantics on Windows).
    """
    at_fd = dir_fd if dir_fd is not None and SUPPORTS_DIR_FD else None
    prefix = f".{name}."
    target: str | Path

    if at_fd is not None:
        fd, temp = _mkstemp_at(at_fd, prefix)
        target = name
    else:
        # mkstemp opens with O_CREAT|O_EXCL, so an existing temp path is never
        # reused.
        fd, temp = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
        target = directory / name

    try:
        try:
            # fchmod, not chmod: operating on the descriptor cannot be redirected
            # by anything that swaps the temp path underneath us, and it makes the
            # mode independent of the process umask (both creation paths open 0600).
            os.fchmod(fd, _FILE_MODE)
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

    Used for the skills manifest, whose directory is the managed root. The
    descriptor is taken with ``O_NOFOLLOW``, so a root swapped for a symlink after
    ``_resolve_root`` validated it fails the write instead of redirecting it —
    the caller turns that into a run-level ``error`` action.
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
