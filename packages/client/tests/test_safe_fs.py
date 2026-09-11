"""
Tests for the descriptor-pinned filesystem primitives.

These exercise ``safe_fs`` directly, on its own terms — the module knows nothing
about skills, and its guarantees are worth asserting without a caller in the way.
The TOCTOU races these primitives exist to close are proved through the
materialization layer, which is what actually holds a descriptor across a
sequence of operations.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import launchdarkly_ai_server.safe_fs as safe_fs_module
from launchdarkly_ai_server.safe_fs import (
    SymlinkRefused,
    atomic_write,
    atomic_write_in,
    open_directory_nofollow,
    open_or_create_directory,
    pinned_directory,
    unlink_file,
)


class TestOpenDirectory:
    """Pinning a directory, and refusing anything that is not one."""

    def test_opens_a_real_directory(self, tmp_path: Path) -> None:
        with pinned_directory(tmp_path) as dir_fd:
            if dir_fd is None:
                pytest.skip("no *at() family on this platform")
            assert stat.S_ISDIR(os.fstat(dir_fd).st_mode)

    def test_refuses_a_symlink_to_a_directory(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        with pytest.raises(ValueError):
            open_directory_nofollow(link)

    def test_refuses_a_regular_file(self, tmp_path: Path) -> None:
        target = tmp_path / "file"
        target.write_text("not a directory")
        with pytest.raises(ValueError):
            open_directory_nofollow(target)

    def test_refuses_an_absent_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            open_directory_nofollow(tmp_path / "nope")

    def test_create_makes_the_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "new"
        fd = open_or_create_directory(target)
        try:
            assert target.is_dir()
        finally:
            if fd is not None:
                os.close(fd)

    def test_create_refuses_an_existing_symlink(self, tmp_path: Path) -> None:
        """``Path.mkdir(exist_ok=True)`` would accept this and reopen the hole.

        A symlink-to-directory already present reads as "already there" to
        ``exist_ok``, so the caller's containment check would be bypassed by
        something that was never checked. ``os.mkdir`` plus an ``lstat`` on the
        ``FileExistsError`` path refuses it.
        """
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        with pytest.raises(ValueError):
            open_or_create_directory(link)

    def test_pinned_directory_closes_the_descriptor(self, tmp_path: Path) -> None:
        with pinned_directory(tmp_path) as dir_fd:
            if dir_fd is None:
                pytest.skip("no *at() family on this platform")
            held = dir_fd
        with pytest.raises(OSError):
            os.fstat(held)


class TestAtomicWrite:
    """Explicit mode, no observable partial file, and one rename call site."""

    def test_writes_the_bytes(self, tmp_path: Path) -> None:
        with pinned_directory(tmp_path) as dir_fd:
            atomic_write(tmp_path, "f.txt", b"hello", dir_fd=dir_fd)
        assert (tmp_path / "f.txt").read_bytes() == b"hello"

    def test_mode_is_0644_and_never_executable(self, tmp_path: Path) -> None:
        """Set explicitly on the descriptor, so the process umask cannot widen or
        narrow it and the execute bit is never inherited."""
        previous = os.umask(0o077)
        try:
            with pinned_directory(tmp_path) as dir_fd:
                atomic_write(tmp_path, "f.txt", b"x", dir_fd=dir_fd)
        finally:
            os.umask(previous)
        mode = (tmp_path / "f.txt").stat().st_mode
        assert stat.S_IMODE(mode) == 0o644
        assert not mode & stat.S_IXUSR

    def test_overwrites_an_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_bytes(b"old")
        with pinned_directory(tmp_path) as dir_fd:
            atomic_write(tmp_path, "f.txt", b"new", dir_fd=dir_fd)
        assert (tmp_path / "f.txt").read_bytes() == b"new"

    def test_leaves_no_temp_file_behind(self, tmp_path: Path) -> None:
        with pinned_directory(tmp_path) as dir_fd:
            atomic_write(tmp_path, "f.txt", b"x", dir_fd=dir_fd)
        assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]

    def test_a_failed_rename_removes_the_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash between write and rename must not leave a partial file, and
        must not leave the temp file either."""

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("injected rename failure")

        monkeypatch.setattr(os, "replace", _boom)
        with pinned_directory(tmp_path) as dir_fd:
            with pytest.raises(OSError, match="injected"):
                atomic_write(tmp_path, "f.txt", b"x", dir_fd=dir_fd)
        assert list(tmp_path.iterdir()) == []

    def test_rename_goes_through_os_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``os.replace`` is the single rename call site.

        ``os.rename`` must not be substituted for it: it is the only one with
        defined overwrite semantics on Windows, and it is the seam the
        materialization tests intercept to prove atomicity.
        """
        calls: list[object] = []
        real = os.replace

        def _spy(src: object, dst: object, **kwargs: object) -> None:
            calls.append(dst)
            real(src, dst, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "replace", _spy)
        with pinned_directory(tmp_path) as dir_fd:
            atomic_write(tmp_path, "f.txt", b"x", dir_fd=dir_fd)
        assert len(calls) == 1
        assert os.path.basename(str(calls[0])) == "f.txt"

    def test_write_without_a_descriptor_uses_the_path_fallback(
        self, tmp_path: Path
    ) -> None:
        """The no-``*at()`` shape must produce an identical result.

        Windows takes this path for every write, so it is not a degenerate case —
        the file, its mode, and the absence of a temp file all have to match.
        """
        atomic_write(tmp_path, "f.txt", b"fallback", dir_fd=None)
        assert (tmp_path / "f.txt").read_bytes() == b"fallback"
        assert stat.S_IMODE((tmp_path / "f.txt").stat().st_mode) == 0o644
        assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]

    def test_write_in_pins_the_directory_itself(self, tmp_path: Path) -> None:
        atomic_write_in(tmp_path, "f.txt", b"x")
        assert (tmp_path / "f.txt").read_bytes() == b"x"

    def test_write_in_refuses_a_symlinked_directory(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        with pytest.raises(ValueError):
            atomic_write_in(link, "f.txt", b"x")


class TestUnlinkFile:
    """Removing only a real file, and refusing a link found in its place."""

    def test_removes_a_regular_file(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("x")
        with pinned_directory(tmp_path) as dir_fd:
            unlink_file(tmp_path, "f.txt", dir_fd=dir_fd)
        assert not (tmp_path / "f.txt").exists()

    def test_refuses_a_symlink_rather_than_removing_it(self, tmp_path: Path) -> None:
        """It refuses rather than tidies.

        ``unlink`` would happily delete the link itself, but a link where this SDK
        expects its own file means the state on disk is not what the manifest
        describes — the caller's to report, not this module's to clean up.
        """
        outside = tmp_path / "outside.txt"
        outside.write_text("do not touch")
        link = tmp_path / "f.txt"
        link.symlink_to(outside)

        with pinned_directory(tmp_path) as dir_fd:
            with pytest.raises(SymlinkRefused):
                unlink_file(tmp_path, "f.txt", dir_fd=dir_fd)

        assert link.is_symlink()
        assert outside.read_text() == "do not touch"

    def test_refuses_a_symlink_on_the_path_fallback(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("do not touch")
        (tmp_path / "f.txt").symlink_to(outside)
        with pytest.raises(SymlinkRefused):
            unlink_file(tmp_path, "f.txt", dir_fd=None)
        assert outside.exists()

    def test_symlink_refused_is_an_oserror(self) -> None:
        """A caller that only cares the removal failed keeps its single
        ``except OSError``; one that must report this refusal specifically does
        not have to match on a message."""
        assert issubclass(SymlinkRefused, OSError)


class TestDirFdProbe:
    """The capability probe names the advertised twins, not the calls made."""

    def test_probe_names_the_syscalls_python_advertises(self) -> None:
        """``os.supports_dir_fd`` is populated per underlying syscall, and CPython
        registers ``renameat`` under ``os.rename`` and ``fstatat`` under
        ``os.stat``. Probing ``os.replace`` and ``os.lstat`` — the names this
        module actually calls — reports "unsupported" on every POSIX platform and
        would silently disable the defense.
        """
        expected = os.supports_dir_fd.issuperset(
            {os.rename, os.open, os.unlink, os.stat}
        )
        assert safe_fs_module.SUPPORTS_DIR_FD is expected

    @pytest.mark.skipif(os.name == "nt", reason="POSIX advertises the *at() family")
    def test_posix_has_the_family(self) -> None:
        assert safe_fs_module.SUPPORTS_DIR_FD is True
