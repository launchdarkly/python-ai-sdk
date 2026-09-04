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

**Platform bound — this guarantee is POSIX-only, deliberately.** On POSIX the
descriptor walk closes the swap window. On Windows it does not exist: there is no
``*at()`` family, so the ``lstat`` floor is all that runs, and a floor is a
check-then-use race rather than a closed window. The remedy would be
reparse-point checks (``GetFileAttributesW``, or opening with
``FILE_FLAG_OPEN_REPARSE_POINT``) and it is **not implemented, by decision rather
than by oversight**: Windows is not a supported or tested platform for this
release, and neither SDK repository has a Windows CI runner, so the checks would
ship untested — and the TypeScript SDK could not match them in any case, because
Node exposes no ``*at()`` family on *any* platform. Shipping them in Python alone
would break the cross-language parity the two SDKs are held to and would trade a
documented bound for an unverified one.

Two consequences worth stating plainly rather than discovering later. First, on
Windows write permission on the managed root is the *only* boundary, so the
privilege-separated deployment the README documents is not advice there but the
mitigation. Second, this bound retroactively lowers the priority of the Windows
reserved-device-name work in ``skills_fs.py`` (``_WINDOWS_RESERVED_NAMES``): that
code stays, because it is cheap and it keeps a managed root written on Linux
usable when read from Windows, but it should not be read as evidence that Windows
is a hardened target. It is not. Revisit both together if Windows becomes
supported.
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

SUPPORTS_DIR_FD = os.supports_dir_fd.issuperset(
    # renameat, openat, unlinkat, fstatat, mkdirat, and unlinkat's AT_REMOVEDIR
    # form — the six this module and its caller need.
    {os.rename, os.open, os.unlink, os.stat, os.mkdir, os.rmdir}
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

``os.mkdir`` and ``os.rmdir`` are in the set because the managed root is now
pinned for a whole reconcile and the per-skill directory is created and removed
relative to that descriptor. Both are registered by CPython under the names
called here, so unlike the two below they need no indirection.

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


def _at(directory: Path, dir_fd: int | None) -> str | Path:
    """
    What to name *directory* by, given a descriptor for its parent.

    With a *dir_fd* the call must pass the bare final component, so the kernel
    resolves it inside the pinned parent and no ancestor is re-resolved from its
    path; without one the full path is the only thing there is to pass. Spelled
    once because every operation in this module that accepts a parent
    descriptor has to make the same choice, and one call site left on the full
    path would silently re-open the window the descriptor closes.
    """
    return directory.name if dir_fd is not None else directory


def open_directory_nofollow(
    directory: Path, *, dir_fd: int | None = None
) -> int | None:
    """
    Opens *directory* without following a final symlink, and pins it.

    *dir_fd* is a descriptor for the *parent*, and passing one is what extends
    the guarantee past the final component: ``O_NOFOLLOW`` refuses a link at
    *directory* itself, but every ancestor above it is re-resolved from its path
    on each open, so a parent swapped for a symlink after it was checked
    redirects the open. Given a parent descriptor the bare name is resolved
    inside the inode that was checked instead, and there is nothing left to swap.

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

    ``Path.mkdir(exist_ok=True)`` treats an existing symlink-to-directory as
    "already there", which would re-open the very hole the caller's ``lstat``
    check just closed. ``os.mkdir`` plus an ``lstat`` on the ``FileExistsError``
    path does not: a link reports as a link, and is refused.

    *dir_fd* is a descriptor for the parent, as in ``open_directory_nofollow``,
    and the ``mkdir`` needs it every bit as much as the open does: ``mkdir``
    follows a symlink at the parent, so a create issued against the full path is
    how a directory gets made — and then written into — outside the root.
    """
    # As in ``atomic_write``: a parent descriptor is only usable where the
    # ``*at()`` family is. ``open_directory_nofollow`` returns ``None`` on the
    # platforms without it, so no caller here can hold one — stated rather than
    # left to that invariant, because the ``mkdir`` below would otherwise raise
    # instead of taking the full-path floor the openers fall back to.
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
    ``lstat`` floor — so the caller states the platform split once, as
    ``if dir_fd is not None``, and cannot forget the ``os.close``. Raises
    ``ValueError`` for a directory that will not pin, exactly as they do.

    *dir_fd* is a descriptor for the parent and is passed straight through. Note
    which descriptor is which: the one passed *in* pins the parent, and the one
    yielded pins *directory* itself.
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


_TEMP_SUFFIX = ".tmp"
"""Suffix on every temp file this module creates."""

_TEMP_TOKEN_BYTES = 8
"""Bytes of randomness in a temp name, as ``secrets.token_hex`` takes them."""

_TEMP_TOKEN_PATTERN = re.compile(
    # Two producers, one recognizer. The descriptor path below names its temp
    # file with ``secrets.token_hex(_TEMP_TOKEN_BYTES)`` — twice that many
    # lowercase hex characters. The fallback path hands naming to
    # ``tempfile.mkstemp``, whose sequence is eight characters drawn from
    # ``[a-z0-9_]``. Matched with ``fullmatch``, which anchors both branches at
    # both ends, so nothing longer or otherwise-shaped is ever recognized.
    rf"[0-9a-f]{{{_TEMP_TOKEN_BYTES * 2}}}|[a-z0-9_]{{8}}"
)


def temp_name_prefix(name: str) -> str:
    """
    The prefix every temp file for *name* is created under.

    Spelled once because two callers need to agree on it: ``atomic_write``
    creates the name, and a caller sweeping orphaned temp files left by a crash
    has to recognize it. A copy of the format string in the sweeper would be a
    copy that can drift out of step with the writer.
    """
    return f".{name}."


def is_temp_name(candidate: str, name: str) -> bool:
    """
    Whether *candidate* is a name this module could have created for *name*.

    The recognizer for the orphan sweep: ``atomic_write`` unlinks its temp file
    on any exception, but a ``SIGKILL`` between the create and the rename leaves
    it behind, and nothing else on disk records that it exists. Deliberately
    narrow — prefix, random token, and suffix must all match, with nothing
    before or after — because the only thing a caller does with a ``True`` here
    is delete the file.
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
    collision, so an existing temp path is never reused and a planted one is
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

    The descriptor is taken with ``O_NOFOLLOW``, so a directory swapped for a
    symlink between the caller's checks and the write fails it rather than
    redirecting it. That still leaves the directory's *ancestors* re-resolved on
    the open, so this is the right primitive only where the caller holds nothing
    better. ``skills_fs`` no longer does — it pins the managed root for the whole
    reconcile and writes the manifest with ``atomic_write(..., dir_fd=root_fd)``
    — so prefer passing a descriptor over reaching for this.
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
