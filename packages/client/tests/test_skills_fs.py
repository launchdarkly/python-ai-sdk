"""
Tests for ``write_skills`` — filesystem materialization, manifest reconcile
semantics, and the full security abuse matrix.

Every test writes only inside pytest's ``tmp_path``. No network, no real
LaunchDarkly client, no real skill transport.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import launchdarkly_ai_server.safe_fs as safe_fs_module
import launchdarkly_ai_server.skills as skills_module
import launchdarkly_ai_server.skills_fs as skills_fs_module
from launchdarkly_ai_server import (
    InMemorySkillStore,
    Skill,
    SkillReference,
    get_skill,
    init_client,
    parse_ai_config,
    skill_refs,
    write_skills,
)
from launchdarkly_ai_server.types_validation import is_valid_skill_key

MANIFEST_NAME = ".launchdarkly-skills.json"
SKILL_BODY = "---\nname: Test Skill\n---\nDo the thing.\n"


MATERIALIZED_SIGNAL = "AgentControl Skill Materialized"
REVOKED_SIGNAL = "AgentControl Skill Revoked Received"
INTEGRITY_SIGNAL = "AgentControl Skill Integrity Failure"

# The three signal names are an allowlist, not a floor.
APPROVED_SIGNALS = frozenset({MATERIALIZED_SIGNAL, REVOKED_SIGNAL, INTEGRITY_SIGNAL})

# Considered and deliberately excluded from SDK emission — named explicitly
# so the regression is unmissable.
REMOVED_SIGNALS = frozenset(
    {
        "AgentControl Skill SDK Reference Returned",
        "AgentControl Skill Content Retrieved",
    }
)


pytestmark = pytest.mark.usefixtures("reset_skill_state")


_INJECTED = "simulated crash between write and rename"


def _dir_id(path: Path) -> tuple[int, int]:
    """``(st_dev, st_ino)`` — a directory's identity, independent of its name."""
    info = os.stat(path)
    return (info.st_dev, info.st_ino)


class _RenameCall(NamedTuple):
    """One intercepted ``os.replace`` of a ``SKILL.md``.

    ``src``/``dst`` are exactly what the implementation passed. Where the rename
    is ``dir_fd``-relative they are bare filenames and the location lives in the
    descriptors, so ``*_dir_id`` carries each descriptor's ``(st_dev, st_ino)``
    resolved *at call time* — the implementation closes the descriptors as soon
    as the write returns, so they cannot be resolved from the assertions.
    """

    src: str
    dst: str
    src_dir_fd: int | None
    dst_dir_fd: int | None
    src_dir_id: tuple[int, int] | None
    dst_dir_id: tuple[int, int] | None


class _ReplaceSpy:
    """Records — and optionally fails — every atomic rename of a ``SKILL.md``.

    Write/rename interception hook: the implementation performs
    the final rename through a single ``os.replace`` call site, so patching the
    attribute on the ``os`` module observes it. Destinations other than
    ``SKILL.md`` (i.e. the manifest's own atomic write) pass straight through —
    the filter holds for both call shapes, since the ``dir_fd``-relative form
    passes ``"SKILL.md"`` itself as ``dst``.

    Used two ways: to prove an injected failure is what produced an ``error``
    action (atomicity), and to prove no write was *attempted* for a
    rejected key — the OS would reject several hostile keys on its
    own, so a failed write is not evidence of a defense.
    """

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[_RenameCall] = []
        self._fail = fail
        self._real = os.replace

    def __call__(self, src: Any, dst: Any, **kwargs: Any) -> None:
        if str(dst).endswith("SKILL.md"):
            src_dir_fd = kwargs.get("src_dir_fd")
            dst_dir_fd = kwargs.get("dst_dir_fd")
            self.calls.append(
                _RenameCall(
                    src=str(src),
                    dst=str(dst),
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                    src_dir_id=None if src_dir_fd is None else _fd_id(src_dir_fd),
                    dst_dir_id=None if dst_dir_fd is None else _fd_id(dst_dir_fd),
                )
            )
            if self._fail:
                raise OSError(_INJECTED)
        self._real(src, dst, **kwargs)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _ReplaceSpy:
        # The attribute is set on the shared ``os`` module, so the
        # single ``os.replace`` call site in safe_fs is intercepted wherever it is
        # reached from. Named through the calling module rather than an arbitrary
        # one so the hook documents which code it covers.
        monkeypatch.setattr(safe_fs_module.os, "replace", self)
        return self


def _fd_id(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino)


def _assert_atomic_rename_of(spy: _ReplaceSpy, skill_dir: Path) -> None:
    """Assert the one recorded rename put ``SKILL.md`` into *skill_dir*.

    The temp file must be created in the target's own
    directory, so the rename is atomic rather than cross-device. Two call
    shapes prove it. Where the platform has ``renameat``
    the rename is ``dir_fd``-relative and the property is asserted by descriptor
    identity — one descriptor for both sides, resolving to *skill_dir*'s inode —
    which is stronger than comparing path strings, because it also rules out the
    descriptor having been redirected between the check and the rename. On the
    ``lstat`` floor (Windows) the names are full paths and share a parent.
    """
    assert len(spy.calls) == 1
    call = spy.calls[0]

    if safe_fs_module.SUPPORTS_DIR_FD:
        assert call.dst == "SKILL.md"
        assert call.src != "SKILL.md"
        assert call.src_dir_fd is not None
        assert call.src_dir_fd == call.dst_dir_fd
        assert call.dst_dir_id == _dir_id(skill_dir)
    else:
        assert Path(call.dst) == skill_dir / "SKILL.md"
        assert Path(call.src).parent == skill_dir
        assert Path(call.src).name != "SKILL.md"


class _SwapDirectoryDuring:
    """Fires the directory-swap race at the exact instant of an operation.

    Renames ``<root>/<key>`` aside and leaves a symlink to *outside* in its
    place, then lets the intercepted call proceed — the narrowest possible
    version of the window an attacker with write access to the managed root
    would otherwise have to hit by timing. Both hooks are the
    interception points (``os.replace`` for the write, ``os.unlink`` for the
    prune), so no implementation internals are touched.
    """

    def __init__(self, attribute: str, skill_dir: Path, outside: Path) -> None:
        self.attribute = attribute
        self.skill_dir = skill_dir
        self.moved_to = skill_dir.parent / f"{skill_dir.name}.real"
        self.outside = outside
        self.swapped = False
        self._real = getattr(os, attribute)

    def __call__(self, first: Any, *args: Any, **kwargs: Any) -> Any:
        named = args[0] if args else first
        if str(named).endswith("SKILL.md") and not self.swapped:
            os.rename(self.skill_dir, self.moved_to)
            os.symlink(self.outside, self.skill_dir, target_is_directory=True)
            self.swapped = True
        return self._real(first, *args, **kwargs)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _SwapDirectoryDuring:
        # ``os.replace`` is called from safe_fs, ``os.unlink`` from skills_fs; both
        # resolve to the same module object, so either name reaches both.
        module = safe_fs_module if self.attribute == "replace" else skills_fs_module
        monkeypatch.setattr(module.os, self.attribute, self)
        return self


class _SwapRootDuring:
    """Fires the *root*-swap race at the exact instant of an operation.

    ``_SwapDirectoryDuring`` one level up: renames the managed root itself aside
    and leaves a symlink to *outside* in its place, then lets the intercepted
    call proceed. ``O_NOFOLLOW`` guards only the final path component, so an
    ``os.mkdir`` or ``os.open`` of ``<root>/<key>`` issued after the swap is
    resolved by the kernel *through* the link: the descriptor that comes back is
    pinned to ``<outside>/<key>``, and every descriptor-relative step that
    follows is relative to the wrong directory.

    The attacker capability is write permission on the root's *parent*, not on
    the root. The README's privilege-separation checklist denies the agent
    identity the root, the skill directories, the files and the manifest, and
    says nothing about the parent; in the documented layout
    (``<app>/.claude/skills``) that parent is ``.claude``, which the agent
    identity typically owns.

    Fires once, on the first call whose path argument is ``<root>/<key>``; every
    other call passes straight through.
    """

    def __init__(self, attribute: str, root: Path, key: str, outside: Path) -> None:
        self.attribute = attribute
        self.root = Path(os.path.realpath(root))
        self.trigger = self.root / key
        self.moved_to = self.root.parent / f"{self.root.name}.real"
        self.outside = outside
        self.swapped = False
        self._real = getattr(os, attribute)

    def __call__(self, first: Any, *args: Any, **kwargs: Any) -> Any:
        # ``os.mkdir(path, mode)`` and ``os.open(path, flags, ...)`` both take the
        # path first; descriptor-relative calls pass a bare name, which never
        # compares equal to the absolute trigger.
        if (
            not self.swapped
            and isinstance(first, (str, os.PathLike))
            and Path(os.fspath(first)) == self.trigger
        ):
            os.rename(self.root, self.moved_to)
            os.symlink(self.outside, self.root, target_is_directory=True)
            self.swapped = True
        return self._real(first, *args, **kwargs)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _SwapRootDuring:
        monkeypatch.setattr(safe_fs_module.os, self.attribute, self)
        return self


_needs_dir_fd = pytest.mark.skipif(
    not safe_fs_module.SUPPORTS_DIR_FD,
    reason="no *at() family on this platform; the per-component lstat floor applies",
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "skills"
    r.mkdir()
    return r


pytestmark = pytest.mark.usefixtures("reset_skill_state")
"""Every test in this module runs against freshly cleared module state."""


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _skill(
    key: str = "test-skill", version: int = 1, content: str = SKILL_BODY
) -> Skill:
    return Skill(
        key=key,
        version=version,
        content=content.encode("utf-8"),
        content_hash=_hash(content),
    )


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _read_manifest(root: Path) -> dict[str, Any]:
    return json.loads(_manifest_path(root).read_text(encoding="utf-8"))


def _write_manifest(root: Path, raw: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _manifest_path(root).write_text(
        raw if isinstance(raw, str) else json.dumps(raw), encoding="utf-8"
    )


def _entry(key: str, version: int, content: str) -> dict[str, Any]:
    return {
        "key": key,
        "version": version,
        "sha256": _hash(content),
        "writtenAt": "2026-08-14T19:00:00Z",
    }


def _place_managed(root: Path, key: str, content: str, version: int = 1) -> Path:
    """Pre-create a file AND its manifest entry — i.e. an SDK-managed path."""
    target = root / key / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _write_manifest(
        root,
        {
            "manifestVersion": 1,
            "entries": {f"{key}/SKILL.md": _entry(key, version, content)},
        },
    )
    return target


def _actions_by_key(report: Any) -> dict[str, Any]:
    return {a.key: a for a in report.actions}


def _error_messages(report: Any) -> list[str]:
    """All ``error`` action messages, regardless of which key they hang off.

    Run-level (manifest) errors have no well-defined ``key`` yet, so assertions
    about them scan every error action rather than looking one up by key.
    """
    return [a.error or "" for a in report.actions if a.action == "error"]


class TestBasicWrites:
    """Basic writes and the returned report."""

    async def test_new_skill_is_written_verbatim(self, root: Path) -> None:
        report = await write_skills([_skill("pdf-extraction", 2)], root)

        target = root / "pdf-extraction" / "SKILL.md"
        assert target.read_text(encoding="utf-8") == SKILL_BODY
        assert report.ok is True
        action = _actions_by_key(report)["pdf-extraction"]
        assert action.action == "written"
        assert action.version == 2
        assert action.path is not None
        assert Path(action.path).resolve() == target.resolve()
        assert action.error is None

    async def test_skill_inputs_need_no_store(self, root: Path) -> None:
        report = await write_skills([_skill("a")], root)
        assert report.ok is True
        assert (root / "a" / "SKILL.md").exists()

    async def test_reference_inputs_resolve_through_store(
        self, root: Path, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", version=3))
        report = await write_skills([SkillReference(key="a", version=3)], root)
        assert report.ok is True
        assert (root / "a" / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY

    async def test_string_inputs_resolve_latest(
        self, root: Path, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", version=9))
        report = await write_skills(["a"], root)
        assert _actions_by_key(report)["a"].version == 9

    async def test_star_writes_everything_in_the_store(
        self, root: Path, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        for k in ("a", "b", "c"):
            store.put(make_raw_skill(key=k))
        report = await write_skills("*", root)
        assert report.ok is True
        assert len([a for a in report.actions if a.action == "written"]) == 3
        for k in ("a", "b", "c"):
            assert (root / k / "SKILL.md").exists()

    async def test_one_action_per_requested_skill(self, root: Path) -> None:
        report = await write_skills([_skill("a"), _skill("b")], root)
        assert sorted(a.key for a in report.actions) == ["a", "b"]

    async def test_empty_request_on_empty_root_is_ok(self, root: Path) -> None:
        report = await write_skills([], root)
        assert report.ok is True
        assert report.actions == []


class TestManifest:
    """Manifest format and forward compatibility."""

    async def test_manifest_format_is_exact(self, root: Path) -> None:
        await write_skills([_skill("pdf-extraction", 2)], root)

        manifest = _read_manifest(root)
        assert manifest["manifestVersion"] == 1
        entry = manifest["entries"]["pdf-extraction/SKILL.md"]
        assert entry["key"] == "pdf-extraction"
        assert entry["version"] == 2
        assert entry["sha256"] == _hash(SKILL_BODY)
        assert isinstance(entry["writtenAt"], str)

    async def test_entry_paths_are_forward_slash_relative(self, root: Path) -> None:
        await write_skills([_skill("a")], root)
        keys = list(_read_manifest(root)["entries"].keys())
        assert keys == ["a/SKILL.md"]
        assert "\\" not in keys[0]
        assert not keys[0].startswith("/")

    async def test_unknown_fields_are_preserved_on_rewrite(self, root: Path) -> None:
        entry = _entry("a", 1, SKILL_BODY)
        entry["futureEntryField"] = "keep-me"
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "futureTopLevelField": {"keep": True},
                "entries": {"a/SKILL.md": entry},
            },
        )
        (root / "a").mkdir()
        (root / "a" / "SKILL.md").write_text(SKILL_BODY, encoding="utf-8")

        await write_skills([_skill("a", 2, SKILL_BODY + "more\n")], root)

        manifest = _read_manifest(root)
        assert manifest["futureTopLevelField"] == {"keep": True}
        assert manifest["entries"]["a/SKILL.md"]["futureEntryField"] == "keep-me"


class TestReconcileSemantics:
    """The reconcile state table."""

    async def test_unchanged_managed_file_is_skipped_current(self, root: Path) -> None:
        target = _place_managed(root, "a", SKILL_BODY)
        before = target.stat().st_mtime_ns

        report = await write_skills([_skill("a")], root)

        assert _actions_by_key(report)["a"].action == "skipped_current"
        assert target.read_text(encoding="utf-8") == SKILL_BODY
        assert target.stat().st_mtime_ns == before

    async def test_new_version_updates(self, root: Path) -> None:
        _place_managed(root, "a", SKILL_BODY, version=1)
        new_content = SKILL_BODY + "second version\n"

        report = await write_skills([_skill("a", 2, new_content)], root)

        action = _actions_by_key(report)["a"]
        assert action.action == "updated"
        assert action.version == 2
        assert (root / "a" / "SKILL.md").read_text(encoding="utf-8") == new_content
        assert _read_manifest(root)["entries"]["a/SKILL.md"]["version"] == 2

    async def test_local_tampering_is_overwritten(self, root: Path) -> None:
        target = _place_managed(root, "a", SKILL_BODY)
        target.write_text("locally tampered\n", encoding="utf-8")

        report = await write_skills([_skill("a")], root)

        assert _actions_by_key(report)["a"].action == "updated"
        assert target.read_text(encoding="utf-8") == SKILL_BODY

    async def test_prune_removes_formerly_managed_skill(self, root: Path) -> None:
        _place_managed(root, "gone", SKILL_BODY)

        report = await write_skills([], root)

        assert _actions_by_key(report)["gone"].action == "removed"
        assert not (root / "gone" / "SKILL.md").exists()
        assert not (root / "gone").exists()
        assert _read_manifest(root)["entries"] == {}

    async def test_prune_false_keeps_the_file(self, root: Path) -> None:
        target = _place_managed(root, "gone", SKILL_BODY)

        report = await write_skills([], root, prune=False)

        assert target.exists()
        assert [a for a in report.actions if a.action == "removed"] == []
        assert "gone/SKILL.md" in _read_manifest(root)["entries"]

    async def test_prune_does_not_touch_unmanaged_files(self, root: Path) -> None:
        _place_managed(root, "gone", SKILL_BODY)
        bystander = root / "user-notes.md"
        bystander.write_text("mine\n", encoding="utf-8")
        user_dir_file = root / "user-skill" / "SKILL.md"
        user_dir_file.parent.mkdir()
        user_dir_file.write_text("hand written\n", encoding="utf-8")

        await write_skills([], root)

        assert bystander.read_text(encoding="utf-8") == "mine\n"
        assert user_dir_file.read_text(encoding="utf-8") == "hand written\n"

    async def test_prune_refusal_for_unownable_path_reports_the_version(
        self, root: Path
    ) -> None:
        """A prune refusal carries the manifest's version.

        A manifest entry whose path is not one this SDK could have written is
        refused rather than removed. The entry is in hand at that point, so the
        error action must carry its version — otherwise a prune *failure* is
        strictly less informative than a prune *success*, which does report it.
        """
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {
                    # Right key, wrong filename — not a path this SDK could own.
                    "orphan/NOTES.md": _entry("orphan", 7, SKILL_BODY),
                },
            },
        )

        report = await write_skills([], root)

        action = _actions_by_key(report)["orphan"]
        assert action.action == "error"
        assert action.version == 7

    async def test_prune_refusal_for_symlinked_target_reports_the_version(
        self, root: Path
    ) -> None:
        """Same contract on the symlink refusal path (prune side)."""
        if not hasattr(os, "symlink"):
            pytest.skip("platform has no symlink support")
        (root / "a").mkdir()
        outside_file = root.parent / "victim.md"
        outside_file.write_text("victim content\n", encoding="utf-8")
        (root / "a" / "SKILL.md").symlink_to(outside_file)
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {"a/SKILL.md": _entry("a", 4, "victim content\n")},
            },
        )

        report = await write_skills([], root)

        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.version == 4

    async def test_unresolvable_request_still_reports_no_version(
        self, root: Path
    ) -> None:
        """The other half of the contract: do not invent a version.

        A reference that could not be retrieved has neither a manifest entry
        nor a ``Skill``, so there is no version to report and ``version`` stays
        ``None``. Without this, "always populate version" would be satisfied by
        fabricating one.
        """
        report = await write_skills([SkillReference(key="ghost", version=3)], root)

        action = _actions_by_key(report)["ghost"]
        assert action.action == "error"
        assert action.version is None

    async def test_prune_keeps_directory_when_not_empty(self, root: Path) -> None:
        _place_managed(root, "a", SKILL_BODY)
        extra = root / "a" / "user-file.txt"
        extra.write_text("keep\n", encoding="utf-8")

        report = await write_skills([], root)

        assert _actions_by_key(report)["a"].action == "removed"
        assert not (root / "a" / "SKILL.md").exists()
        assert extra.exists()


class TestRootHandling:
    """Root resolution."""

    async def test_absent_leaf_root_is_created(self, tmp_path: Path) -> None:
        target_root = tmp_path / "skills"
        report = await write_skills([_skill("a")], target_root)
        assert report.ok is True
        assert (target_root / "a" / "SKILL.md").exists()

    async def test_missing_ancestors_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            await write_skills([_skill("a")], tmp_path / "a" / "b" / "c")

    async def test_root_that_is_a_file_raises(self, tmp_path: Path) -> None:
        file_root = tmp_path / "not-a-dir"
        file_root.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            await write_skills([_skill("a")], file_root)

    async def test_accepts_string_root(self, root: Path) -> None:
        report = await write_skills([_skill("a")], str(root))
        assert report.ok is True


class TestSkillsArgumentErrors:
    """A bare string that is not ``"*"`` raises.

    A ``ValueError``, not a ``TypeError``: a string *is* an accepted argument
    type here, since ``"*"`` means "everything the store holds", so this is an
    acceptable type carrying an invalid value. The accessors' equivalent guard
    is a ``TypeError`` because a string is never a valid argument there.
    """

    async def test_bare_non_star_string_raises_value_error(self, root: Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            await write_skills("pdf-extraction", root)

        # Naming the accepted forms is the actionable half of the message.
        assert '"*"' in str(excinfo.value)

    async def test_star_is_accepted(self, root: Path) -> None:
        """Positive control — otherwise the guard above could reject every string."""
        store = InMemorySkillStore()
        store.put(
            {
                "key": "a",
                "version": 1,
                "content": SKILL_BODY,
                "contentHash": _hash(SKILL_BODY),
            }
        )
        skills_module._set_store(store)

        report = await write_skills("*", root)

        assert report.ok is True
        assert (root / "a" / "SKILL.md").exists()

    async def test_bare_string_writes_nothing(self, root: Path) -> None:
        """The raise precedes any filesystem work.

        Asserting only the raise would also pass for an implementation that
        created one directory per character before failing.
        """
        with pytest.raises(ValueError):
            await write_skills("abc", root)

        assert list(root.iterdir()) == []


class TestResilience:
    """Unavailable retrieval and timeout."""

    async def test_keep_is_the_default_and_does_not_raise(self, root: Path) -> None:
        existing = _place_managed(root, "a", SKILL_BODY)

        report = await write_skills([SkillReference(key="a", version=1)], root)

        assert report.ok is False
        assert _actions_by_key(report)["a"].action == "error"
        assert existing.read_text(encoding="utf-8") == SKILL_BODY

    async def test_raise_mode_propagates(self, root: Path) -> None:
        with pytest.raises(RuntimeError, match=r"(?i)(unavailable|skill store)"):
            await write_skills(
                [SkillReference(key="a", version=1)], root, on_unavailable="raise"
            )

    async def test_store_error_is_reported_not_raised(
        self, root: Path, exploding_store: Any
    ) -> None:
        report = await write_skills([SkillReference(key="a", version=1)], root)

        assert report.ok is False
        assert _actions_by_key(report)["a"].action == "error"

    async def test_exhausted_timeout_behaves_as_unavailable(
        self, root: Path, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a"))
        report = await write_skills(
            [SkillReference(key="a", version=1)], root, timeout=0
        )
        assert report.ok is False
        assert not (root / "a" / "SKILL.md").exists()

    async def test_exhausted_timeout_raises_in_raise_mode(
        self, root: Path, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a"))
        with pytest.raises(RuntimeError, match=r"(?i)(unavailable|timeout|timed out)"):
            await write_skills(
                [SkillReference(key="a", version=1)],
                root,
                timeout=0,
                on_unavailable="raise",
            )

    async def test_exhausted_timeout_stops_pruning(
        self, root: Path, store: InMemorySkillStore
    ) -> None:
        """The deadline bounds pruning too, not just retrieval and the writes.

        A run whose writes all land just inside the deadline would otherwise go
        on to stat, unlink and rmdir every stale manifest entry unbounded — the
        opposite of what a small ``timeout`` asks for.
        """
        existing = _place_managed(root, "stale", SKILL_BODY)

        report = await write_skills([], root, timeout=0)

        assert report.ok is False
        assert existing.exists(), "prune ran past the exhausted deadline"
        assert any("timeout was exhausted" in m for m in _error_messages(report))
        # The entry survives, so the next reconcile picks it up.
        assert "stale/SKILL.md" in _read_manifest(root)["entries"]

    async def test_a_verification_failure_never_prunes_the_good_copy(
        self, root: Path
    ) -> None:
        """A store may key ``all_objects`` differently from the object's own key.

        The on-disk copy lives under the object's own key, so a failure recorded
        under the *store's* dict key would drop the real key out of the
        requested set and let prune delete the last known-good copy.
        """

        class AliasKeyedStore:
            """Keys objects by an internal id, not by the skill's own key."""

            def __init__(self, raw: dict[str, Any]) -> None:
                self._raw = raw

            def get_object(self, kind: str, key: str) -> dict[str, Any] | None:
                return None

            def all_objects(self, kind: str) -> dict[str, dict[str, Any]]:
                return {"internal-uuid-1": self._raw}

        existing = _place_managed(root, "pdf-extraction", SKILL_BODY)
        tampered = {
            "key": "pdf-extraction",
            "version": 1,
            "content": "tampered\n",
            "contentHash": _hash(SKILL_BODY),  # does not match the content
        }
        skills_module._set_store(AliasKeyedStore(tampered))

        report = await write_skills("*", root)

        assert report.ok is False
        assert existing.read_text(encoding="utf-8") == SKILL_BODY
        assert [a.action for a in report.actions] == ["error"]
        assert _actions_by_key(report)["pdf-extraction"].action == "error"
        assert "pdf-extraction/SKILL.md" in _read_manifest(root)["entries"]

    async def test_unavailable_run_does_not_corrupt_manifest(self, root: Path) -> None:
        _place_managed(root, "a", SKILL_BODY)
        before = _read_manifest(root)

        await write_skills([SkillReference(key="b", version=1)], root)

        assert (
            _read_manifest(root)["entries"]["a/SKILL.md"]
            == (before["entries"]["a/SKILL.md"])
        )


class TestVerifyThenWrite:
    """Hash re-verified immediately before writing."""

    async def test_hash_mismatch_aborts_the_write(
        self, root: Path, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        bad = Skill(
            key="a",
            version=1,
            content=SKILL_BODY.encode("utf-8"),
            content_hash="0" * 64,
        )

        report = await write_skills([bad], root)

        assert report.ok is False
        assert _actions_by_key(report)["a"].action == "error"
        assert not (root / "a" / "SKILL.md").exists()
        assert len(recording_emitter.signals(INTEGRITY_SIGNAL)) == 1

    async def test_oversize_skill_aborts_the_write(self, root: Path) -> None:
        oversize = "x" * (64 * 1024 + 1)
        report = await write_skills([_skill("a", 1, oversize)], root)
        assert report.ok is False
        assert not (root / "a" / "SKILL.md").exists()

    async def test_mismatch_does_not_disturb_existing_managed_file(
        self, root: Path
    ) -> None:
        target = _place_managed(root, "a", SKILL_BODY)
        bad = Skill(key="a", version=2, content=b"new content\n", content_hash="f" * 64)

        await write_skills([bad], root)

        assert target.read_text(encoding="utf-8") == SKILL_BODY


class TestAtomicityAndPermissions:
    """Atomic writes, no partial files, 0644."""

    async def test_written_file_is_0644_and_not_executable(self, root: Path) -> None:
        await write_skills([_skill("a")], root)
        mode = stat.S_IMODE((root / "a" / "SKILL.md").stat().st_mode)
        assert mode == 0o644
        assert not mode & stat.S_IXUSR
        assert not mode & stat.S_IXGRP
        assert not mode & stat.S_IXOTH

    async def test_write_goes_through_a_single_atomic_rename(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control for the interception hook.

        Without this, the ``spy.calls == []`` assertions in the failure tests
        below and in the traversal matrix could pass in a suite where the hook
        is never reachable at all.
        """
        spy = _ReplaceSpy().install(monkeypatch)

        report = await write_skills([_skill("a")], root)

        assert report.ok is True
        # The temp file is created in the *same* directory
        # as the target, so the rename is atomic rather than cross-device.
        _assert_atomic_rename_of(spy, root / "a")
        assert (root / "a" / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY

    async def test_rename_failure_leaves_prior_content_intact(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = _place_managed(root, "a", SKILL_BODY)
        spy = _ReplaceSpy(fail=True).install(monkeypatch)

        report = await write_skills([_skill("a", 2, "brand new content\n")], root)

        # The injected failure — not an unrelated rejection, and not an
        # implementation that attempted nothing — is what produced the error.
        _assert_atomic_rename_of(spy, target.parent)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert _INJECTED in (action.error or "")

        assert target.read_text(encoding="utf-8") == SKILL_BODY
        # No temp artifact survives the failed run.
        assert sorted(p.name for p in target.parent.iterdir()) == ["SKILL.md"]

    async def test_no_partial_file_at_target_after_failure(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _ReplaceSpy(fail=True).install(monkeypatch)

        report = await write_skills([_skill("a")], root)

        _assert_atomic_rename_of(spy, root / "a")

        assert report.ok is False
        assert _INJECTED in (_actions_by_key(report)["a"].error or "")
        assert not (root / "a" / "SKILL.md").exists()
        # Neither a partial target nor a leaked temp file.
        skill_dir = root / "a"
        leftovers = (
            sorted(p.name for p in skill_dir.iterdir()) if skill_dir.exists() else []
        )
        assert leftovers == []

    async def test_manifest_is_valid_json_after_a_run_with_errors(
        self, root: Path
    ) -> None:
        report = await write_skills([_skill("a"), _skill("../evil")], root)
        assert report.ok is False
        assert isinstance(_read_manifest(root), dict)


# ---------------------------------------------------------------------------
# Security abuse matrix
# ---------------------------------------------------------------------------

HOSTILE_KEYS = [
    "../evil",
    "..",
    ".",
    "",
    "/etc/cron.d/x",
    "..\\evil",
    "c:evil",
    "skill:ads",
    "sk\0ill",
    "-skill",
    "Evil",
    "a/b",
    "x" * 257,
    "a/../../b",
    "./a",
    " leading-space",
    "trailing-space ",
]


class TestPathTraversal:
    """Nothing is ever written outside the root."""

    @pytest.mark.parametrize("hostile_key", HOSTILE_KEYS)
    async def test_hostile_key_is_rejected(
        self, tmp_path: Path, hostile_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        outside_before = sorted(p.name for p in tmp_path.iterdir())
        spy = _ReplaceSpy().install(monkeypatch)

        report = await write_skills([_skill(hostile_key)], root)

        assert report.ok is False
        assert [a.action for a in report.actions if a.key == hostile_key] == ["error"]

        # The SDK's key validation — not the operating system — must be what
        # stopped this. An overlong key exceeds NAME_MAX, a null byte raises in
        # the path API, and an absolute path outside the root usually fails on
        # permissions, so "an error was reported" is not evidence of a defense
        # (and the absolute-path verdict would flip on a privileged runner).
        # Assert instead that no write was ever attempted.
        assert spy.calls == []

        # Nothing created outside the root, and no skill directory inside it.
        assert sorted(p.name for p in tmp_path.iterdir()) == outside_before
        assert [p.name for p in root.iterdir() if p.name != MANIFEST_NAME] == []
        assert list(root.rglob("SKILL.md")) == []

    async def test_interception_hook_fires_for_a_valid_key(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control for the ``spy.calls == []`` assertion above."""
        spy = _ReplaceSpy().install(monkeypatch)

        report = await write_skills([_skill("ok-key")], root)

        assert report.ok is True
        assert [Path(call.dst).name for call in spy.calls] == ["SKILL.md"]

    async def test_long_but_filesystem_legal_key_is_written(self, root: Path) -> None:
        """The ≤ 256 length bound cannot be exercised through ``write_skills``.

        A key becomes a single directory name and NAME_MAX is 255 bytes on Linux
        and macOS, so the longest key the data model permits cannot exist on
        disk at all. Assert the accepting side at the largest writable length;
        the bound itself is covered by the pure layers (config validation and
        accessor revalidation).
        """
        key = "k" * 255
        report = await write_skills([_skill(key)], root)

        assert report.ok is True
        assert (root / key / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY

    async def test_key_at_the_data_model_bound_is_reported_not_raised(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 256-character key is valid to every pure layer but fits no filesystem.

        Config validation and the accessors must both accept exactly 256
        characters, yet NAME_MAX is 255
        on Linux and macOS, so this key reaches ``write_skills`` legitimately and
        cannot become a directory. Every outcome must be visible in
        the report, so it must surface as an ``error`` action rather than an
        ``OSError`` escaping the call — which would also skip the manifest rewrite
        and orphan any file already written in the same run.
        """
        spy = _ReplaceSpy().install(monkeypatch)
        long_key = "a" * 256

        report = await write_skills([_skill("good"), _skill(long_key)], root)

        by_key = _actions_by_key(report)
        assert by_key[long_key].action == "error"
        assert by_key["good"].action == "written"
        # The bare-filename ``dst`` of a ``dir_fd``-relative rename carries no
        # directory, so "the path does not contain the hostile key" is no longer
        # a meaningful check. Assert the stronger thing instead: the only rename
        # that happened was into the valid skill's own directory. Through the
        # shared helper, so the check holds on the path fallback too — reading
        # ``dst_dir_id`` directly would compare ``None`` there and fail a run
        # that had in fact renamed correctly.
        _assert_atomic_rename_of(spy, root / "good")
        # The valid skill is fully reconciled: written AND recorded, not orphaned.
        assert (root / "good" / "SKILL.md").exists()
        assert "good/SKILL.md" in _read_manifest(root)["entries"]

    async def test_valid_keys_still_write_alongside_rejected_ones(
        self, root: Path
    ) -> None:
        report = await write_skills([_skill("good"), _skill("../evil")], root)
        by_key = _actions_by_key(report)
        assert by_key["good"].action == "written"
        assert by_key["../evil"].action == "error"
        assert (root / "good" / "SKILL.md").exists()

    async def test_traversal_key_does_not_create_parent_files(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        await write_skills([_skill("../../escaped")], root)
        assert not (tmp_path / "escaped").exists()
        assert not (tmp_path.parent / "escaped").exists()


@pytest.mark.skipif(
    not hasattr(os, "symlink"), reason="platform has no symlink support"
)
class TestSymlinkAttacks:
    """Never write through a symlink."""

    async def test_symlinked_root_raises(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_root = tmp_path / "link"
        link_root.symlink_to(real_dir, target_is_directory=True)

        with pytest.raises(ValueError):
            await write_skills([_skill("a")], link_root)

        assert list(real_dir.iterdir()) == []

    async def test_symlinked_skill_directory_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "a").symlink_to(outside, target_is_directory=True)

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        assert _actions_by_key(report)["a"].action == "error"
        assert list(outside.iterdir()) == []

    async def test_symlinked_target_file_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        outside_file = tmp_path / "victim.md"
        outside_file.write_text("victim content\n", encoding="utf-8")
        (root / "a").mkdir()
        (root / "a" / "SKILL.md").symlink_to(outside_file)
        # Manifest lists the path so clobber protection is not what saves us.
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {"a/SKILL.md": _entry("a", 1, "victim content\n")},
            },
        )

        report = await write_skills([_skill("a", 2, "attacker payload\n")], root)

        assert report.ok is False
        assert _actions_by_key(report)["a"].action == "error"
        assert outside_file.read_text(encoding="utf-8") == "victim content\n"

    async def test_symlinked_target_is_not_pruned(self, tmp_path: Path) -> None:
        """A manifest-listed path that is a symlink is refused, not unlinked.

        Asserting only that the victim file survives proves nothing here:
        unlinking a symlink never touches its target, so that assertion holds
        for an implementation with no symlink check at all. The observable
        contract is the refusal itself (prune path).
        """
        root = tmp_path / "skills"
        root.mkdir()
        outside_file = tmp_path / "victim.md"
        outside_file.write_text("victim content\n", encoding="utf-8")
        (root / "a").mkdir()
        link = root / "a" / "SKILL.md"
        link.symlink_to(outside_file)
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {"a/SKILL.md": _entry("a", 1, "victim content\n")},
            },
        )

        report = await write_skills([], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        assert [a for a in report.actions if a.action == "removed"] == []
        # The symlink itself is left in place and stays managed.
        assert link.is_symlink()
        assert "a/SKILL.md" in _read_manifest(root)["entries"]
        assert outside_file.read_text(encoding="utf-8") == "victim content\n"

    @_needs_dir_fd
    async def test_directory_swapped_at_the_rename_cannot_redirect_the_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The swap window is closed, not merely narrowed.

        Every check in the world is worthless if the final rename re-resolves
        ``<root>/<key>`` from its path: an attacker holding write permission on
        the managed root can replace the validated directory with a symlink in
        between and redirect the write out of the root. The rename is therefore
        performed relative to a descriptor pinned to the directory that was
        checked, so it follows the inode rather than the name.
        """
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        race = _SwapDirectoryDuring("replace", root / "a", outside).install(monkeypatch)

        report = await write_skills([_skill("a")], root)

        assert race.swapped is True, "the race never fired; the test proves nothing"
        assert list(outside.iterdir()) == []
        assert (race.moved_to / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY
        assert report.ok is True

    @_needs_dir_fd
    async def test_directory_swapped_at_the_prune_cannot_redirect_the_unlink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same window on the destructive side.

        ``unlink`` never follows a *trailing* symlink, but it does resolve the
        directory above it, so the swap turns a prune into a delete of an
        attacker-chosen outside file. The unlink is descriptor-relative for the
        same reason the rename is.
        """
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "SKILL.md"
        victim.write_text("precious\n", encoding="utf-8")
        _place_managed(root, "a", SKILL_BODY)
        race = _SwapDirectoryDuring("unlink", root / "a", outside).install(monkeypatch)

        report = await write_skills([], root)

        assert race.swapped is True, "the race never fired; the test proves nothing"
        assert victim.read_text(encoding="utf-8") == "precious\n"
        assert not (race.moved_to / "SKILL.md").exists()
        assert [a.action for a in report.actions if a.key == "a"] == ["removed"]


@pytest.mark.skipif(
    not hasattr(os, "symlink"), reason="platform has no symlink support"
)
class TestRootSwapRaces:
    """The swap one level up: the managed *root*, not ``<root>/<key>``.

    ``_resolve_root`` validates the root once and returns a path; nothing holds
    it open. Each write and each prune then opens ``<root>/<key>`` *by path*
    with ``O_NOFOLLOW | O_DIRECTORY`` and pins that. ``O_NOFOLLOW`` guards only
    the final component, so the root and every ancestor are re-resolved on
    every such open, and a root swapped for a symlink after validation
    redirects the open — and with it every descriptor-relative step behind it
    — into the attacker's directory. ``atomic_write_in`` does refuse the
    manifest rewrite afterwards, but by then the skill file is already outside
    the root.

    ``TestSymlinkAttacks`` proves the ``<root>/<key>`` swap is closed. These
    three state the same contract for the root — nothing lands outside it, no
    outside file is overwritten, no outside file is removed. The code does not
    yet meet it, so they fail, and the red run is the demonstration. The fix is
    to pin the root once at the top of ``write_skills`` and walk from it:
    ``os.mkdir(key, dir_fd=root_fd)``, ``os.open(key, ..., dir_fd=root_fd)``,
    and ``atomic_write(..., dir_fd=root_fd)`` for the manifest.
    """

    @_needs_dir_fd
    async def test_root_swapped_at_the_skill_directory_create_cannot_redirect_the_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First reconcile against a fresh root: the skill directory does not
        exist yet, so ``open_or_create_directory`` calls ``os.mkdir(<root>/a)``.
        The swap fires there; the ``mkdir`` and the ``O_NOFOLLOW`` open that
        follows both resolve through the link, and ``SKILL.md`` is written into
        ``<outside>/a/``."""
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        race = _SwapRootDuring("mkdir", root, "a", outside).install(monkeypatch)

        report = await write_skills([_skill("a")], root)

        assert race.swapped is True, "the race never fired; the test proves nothing"
        assert list(outside.iterdir()) == []
        # Either the skill landed in the real root or the run says it did not.
        assert report.ok is False or (
            (race.moved_to / "a" / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY
        )

    @_needs_dir_fd
    async def test_root_swapped_at_the_skill_directory_open_cannot_clobber_an_outside_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Update of an already-managed skill: the swap fires at the first
        ``O_NOFOLLOW`` open of ``<root>/a`` (the orphan-temp sweep's), and the
        write re-opens by path and inherits it. ``<outside>/a/SKILL.md`` — a
        file the manifest never recorded — is replaced with LaunchDarkly-served
        content."""
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside"
        (outside / "a").mkdir(parents=True)
        victim = outside / "a" / "SKILL.md"
        victim.write_text("precious\n", encoding="utf-8")
        _place_managed(root, "a", SKILL_BODY)
        race = _SwapRootDuring("open", root, "a", outside).install(monkeypatch)

        report = await write_skills([_skill("a", 2, "served update\n")], root)

        assert race.swapped is True, "the race never fired; the test proves nothing"
        assert victim.read_text(encoding="utf-8") == "precious\n"
        assert report.ok is False or (
            (race.moved_to / "a" / "SKILL.md").read_text(encoding="utf-8")
            == "served update\n"
        )

    @_needs_dir_fd
    async def test_root_swapped_at_the_prune_cannot_redirect_the_unlink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The destructive side. A prune of a formerly-managed key opens
        ``<root>/a`` by path, unlinks ``SKILL.md`` relative to that descriptor,
        then ``rmdir``s the directory — all three resolve through the swapped
        root, so an outside file and its directory are removed."""
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside"
        (outside / "a").mkdir(parents=True)
        victim = outside / "a" / "SKILL.md"
        victim.write_text("precious\n", encoding="utf-8")
        _place_managed(root, "a", SKILL_BODY)
        race = _SwapRootDuring("open", root, "a", outside).install(monkeypatch)

        report = await write_skills([], root)

        assert race.swapped is True, "the race never fired; the test proves nothing"
        assert victim.exists() and victim.read_text(encoding="utf-8") == "precious\n"
        assert report.ok is False or not (race.moved_to / "a" / "SKILL.md").exists()


class TestWithoutDirFd:
    """The full-path fallback for platforms with no ``*at()`` family.

    On Windows ``os.open`` cannot open a directory at all, so acquiring the
    descriptor must not even be attempted there — a fallback reached only after
    a descriptor open would leave every write, prune and manifest rewrite
    failing rather than falling back. These tests force the flag off so the
    fallback is exercised on POSIX too.
    """

    @pytest.fixture(autouse=True)
    def _no_dir_fd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Models Windows: no ``*at()`` family, and directories cannot be opened.

        Forcing the flag off alone would not reproduce the platform, because
        ``os.open`` on a directory succeeds on POSIX — the fallback would be
        reached either way. Making that call raise the ``PermissionError``
        Windows raises is what proves the descriptor open is never attempted.
        """
        monkeypatch.setattr(safe_fs_module, "SUPPORTS_DIR_FD", False)
        real_open = os.open

        def no_directory_open(path: Any, *args: Any, **kwargs: Any) -> int:
            if os.path.isdir(path):
                raise PermissionError(13, "Permission denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(safe_fs_module.os, "open", no_directory_open)

    async def test_write_prune_and_manifest_all_succeed(self, root: Path) -> None:
        first = await write_skills([_skill("a"), _skill("b")], root)
        assert first.ok is True, _error_messages(first)
        assert (root / "a" / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY
        assert _manifest_path(root).exists()
        assert stat.S_IMODE((root / "a" / "SKILL.md").stat().st_mode) == 0o644

        second = await write_skills([_skill("a")], root)

        assert second.ok is True, _error_messages(second)
        assert not (root / "b" / "SKILL.md").exists()
        assert "b/SKILL.md" not in _read_manifest(root)["entries"]

    async def test_a_symlinked_skill_directory_is_still_refused(
        self, root: Path, tmp_path: Path
    ) -> None:
        """The fallback keeps the ``lstat`` floor: no writing through a link."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "a").symlink_to(outside, target_is_directory=True)

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        assert list(outside.iterdir()) == []


class TestNonRegularFiles:
    """A managed path that is not a regular file is refused, never read."""

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this platform")
    async def test_a_fifo_at_the_managed_path_does_not_block(self, root: Path) -> None:
        """Reading a FIFO with no writer blocks forever.

        Same attacker capability the symlink checks defend against: swapping a
        managed ``SKILL.md`` for a FIFO would otherwise hang the whole reconcile
        — and the caller's event loop with it — well past any ``timeout``, since
        the deadline is only consulted between steps.
        """
        skill_dir = root / "a"
        skill_dir.mkdir()
        os.mkfifo(skill_dir / "SKILL.md")
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {"a/SKILL.md": _entry("a", 1, SKILL_BODY)},
            },
        )

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        assert "regular file" in action.error
        assert stat.S_ISFIFO(os.lstat(skill_dir / "SKILL.md").st_mode)


class TestClobberProtection:
    """Destructive ops only on manifest-listed paths."""

    async def test_unmanaged_file_is_never_overwritten(self, root: Path) -> None:
        target = root / "a" / "SKILL.md"
        target.parent.mkdir()
        target.write_text("user authored\n", encoding="utf-8")

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        assert target.read_text(encoding="utf-8") == "user authored\n"

    async def test_unmanaged_file_is_never_deleted(self, root: Path) -> None:
        target = root / "a" / "SKILL.md"
        target.parent.mkdir()
        target.write_text("user authored\n", encoding="utf-8")

        await write_skills([], root)

        assert target.read_text(encoding="utf-8") == "user authored\n"

    async def test_manifest_entry_with_mismatched_key_does_not_authorize(
        self, root: Path
    ) -> None:
        target = root / "a" / "SKILL.md"
        target.parent.mkdir()
        target.write_text("user authored\n", encoding="utf-8")
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {
                    "a/SKILL.md": _entry("different-key", 1, "user authored\n")
                },
            },
        )

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        assert _actions_by_key(report)["a"].action == "error"
        assert target.read_text(encoding="utf-8") == "user authored\n"


DIVERGENT_CONTENT = "existing content\n"


def _live_entries() -> dict[str, Any]:
    """A parseable entries map that really does claim ``a/SKILL.md`` as managed."""
    return {"a/SKILL.md": _entry("a", 1, DIVERGENT_CONTENT)}


# The first six variants are unparseable: ``entries`` is missing, the wrong type,
# or the whole document is garbage. That makes "performed no destructive action"
# arithmetic rather than a defense — with no entries to act on, a file at a
# managed path is protected by clobber protection and there is nothing to prune,
# so those cases pass against an implementation that simply treats a corrupt
# manifest as an empty one.
#
# The ``*_live_entries`` variants are the ones that actually test round-tripping: corrupt
# ONLY in ``manifestVersion``, with a valid entries map listing the managed path
# under a matching key. The implementation has everything it needs to overwrite
# and to prune, and must refuse anyway.
CORRUPT_MANIFESTS: list[tuple[str, Any]] = [
    ("garbage", "{not json at all"),
    ("empty", ""),
    ("wrong_types", {"manifestVersion": 1, "entries": ["a/SKILL.md"]}),
    ("entries_missing", {"manifestVersion": 1}),
    ("future_version", {"manifestVersion": 2, "entries": {}}),
    ("version_not_int", {"manifestVersion": "1", "entries": {}}),
    ("future_version_live_entries", {"manifestVersion": 2, "entries": _live_entries()}),
    (
        "version_not_int_live_entries",
        {"manifestVersion": "1", "entries": _live_entries()},
    ),
]

LIVE_ENTRY_MANIFESTS: list[tuple[str, Any]] = [
    case for case in CORRUPT_MANIFESTS if case[0].endswith("_live_entries")
]


class TestCorruptManifest:
    """Corrupt manifest fails closed, non-destructively."""

    @pytest.mark.parametrize(
        "raw",
        [case[1] for case in CORRUPT_MANIFESTS],
        ids=[case[0] for case in CORRUPT_MANIFESTS],
    )
    async def test_no_destructive_action_and_error_reported(
        self, root: Path, raw: Any
    ) -> None:
        target = root / "a" / "SKILL.md"
        target.parent.mkdir()
        target.write_text(DIVERGENT_CONTENT, encoding="utf-8")
        _write_manifest(root, raw)

        report = await write_skills([_skill("a", 2, "new content\n")], root)

        assert report.ok is False
        # The error must name the manifest. For the unparseable variants the file
        # at the managed path is also unmanaged, so a bare "some error happened"
        # assertion is satisfied by clobber protection alone and says nothing
        # about whether the manifest state was detected at all.
        errors = _error_messages(report)
        assert any("manifest" in e.lower() for e in errors), errors
        assert target.read_text(encoding="utf-8") == DIVERGENT_CONTENT

    async def test_run_level_error_carries_the_empty_key_sentinel(
        self, root: Path
    ) -> None:
        """A run-level error has no skill key to hang off.

        The empty string is public API surface: a caller grouping the report by
        key has to know the sentinel exists. Asserted here rather than in the
        parametrized cases above so it is a statement about the manifest error
        specifically, not about whichever error happens to come first.
        """
        _write_manifest(root, "{not json at all")

        report = await write_skills([_skill("a")], root)

        manifest_errors = [
            action
            for action in report.errors
            if "manifest" in (action.error or "").lower()
        ]
        assert manifest_errors, _error_messages(report)
        assert all(action.key == "" for action in manifest_errors)
        # A per-skill error in the same report still carries its real key, so the
        # sentinel is not simply "every error action has an empty key".
        assert all(
            action.key != ""
            for action in report.errors
            if action not in manifest_errors
        )

    @pytest.mark.parametrize(
        "raw",
        [case[1] for case in LIVE_ENTRY_MANIFESTS],
        ids=[case[0] for case in LIVE_ENTRY_MANIFESTS],
    )
    async def test_managed_file_is_not_pruned_when_only_the_version_is_corrupt(
        self, root: Path, raw: Any
    ) -> None:
        """The prune counterpart of the live-entries cases.

        Here the implementation can read the entries map and knows exactly which
        file it owns, so refusing to remove it is a real decision rather than an
        absence of information.
        """
        target = root / "a" / "SKILL.md"
        target.parent.mkdir()
        target.write_text(DIVERGENT_CONTENT, encoding="utf-8")
        _write_manifest(root, raw)

        report = await write_skills([], root)

        assert report.ok is False
        assert target.read_text(encoding="utf-8") == DIVERGENT_CONTENT
        assert [a for a in report.actions if a.action == "removed"] == []
        errors = _error_messages(report)
        assert any("manifest" in e.lower() for e in errors), errors

    async def test_nothing_is_pruned_under_a_corrupt_manifest(self, root: Path) -> None:
        target = root / "a" / "SKILL.md"
        target.parent.mkdir()
        target.write_text(DIVERGENT_CONTENT, encoding="utf-8")
        _write_manifest(root, "{not json at all")

        report = await write_skills([], root)

        assert report.ok is False
        assert target.exists()
        assert [a for a in report.actions if a.action == "removed"] == []
        errors = _error_messages(report)
        assert any("manifest" in e.lower() for e in errors), errors

    async def test_brand_new_paths_may_still_be_written(self, root: Path) -> None:
        _write_manifest(root, "{not json at all")

        report = await write_skills([_skill("fresh")], root)

        actions = _actions_by_key(report)
        assert actions["fresh"].action == "written"
        assert (root / "fresh" / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY

    async def test_corrupt_manifest_file_is_not_destroyed(self, root: Path) -> None:
        _write_manifest(root, "{not json at all")

        await write_skills([], root)

        assert _manifest_path(root).exists()
        assert _manifest_path(root).read_text(encoding="utf-8") == "{not json at all"


# The three literal cases the security review names for the prune path.
#
# The distinction from ``TestCorruptManifest`` above is the whole point: a
# corrupt manifest suppresses every destructive action wholesale, so those tests
# say nothing about these. Each manifest here is *well-formed* — parseable,
# a ``manifestVersion`` this release understands, a real ``entries`` map, and an
# entry whose ``key`` is a perfectly valid skill key that is genuinely absent
# from the requested set. The implementation has every input it needs to prune
# and must refuse anyway, because the recorded *path* is not one this SDK could
# have written.
HOSTILE_RECORDED_PATHS: list[str] = [
    # Absolute: the classic. A recorded path read back and unlinked as-is is a
    # delete of an attacker-chosen file with the reconcile's privileges.
    "/etc/passwd",
    # Traversing: the same attack for an implementation that rejects a leading
    # slash and then joins the rest onto the root.
    "../../../etc/passwd",
]


class _UnlinkSpy:
    """Records every ``os.unlink`` while delegating to the real one.

    Asserting only that ``/etc/passwd`` still exists proves nothing: the test
    process cannot delete it anyway, so that assertion passes against an
    implementation with no path check at all — permissions would be doing the
    work. What has teeth is that the removal is never *attempted*: the refusal
    happens above the syscall, on a path the SDK recomputes rather than trusts.
    """

    def __init__(self) -> None:
        self.targets: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _UnlinkSpy:
        real = os.unlink

        def spy(path: Any, *args: Any, **kwargs: Any) -> None:
            self.targets.append(os.fsdecode(path))
            real(path, *args, **kwargs)

        # ``safe_fs_module.os`` *is* the ``os`` module, so this covers both the
        # descriptor-relative ``os.unlink(name, dir_fd=...)`` and the
        # ``Path.unlink`` used on the no-``*at()`` floor.
        monkeypatch.setattr(safe_fs_module.os, "unlink", spy)
        return self


class TestHostileManifestPrune:
    """A well-formed manifest naming a path this SDK could not have written.

    The manifest is untrusted input. It is a plain file on the customer's disk
    that anything with write access to the managed root can edit, and ``prune``
    is the one code path in the SDK that deletes. So a recorded path never
    authorizes its own removal: it must match ``<key>/SKILL.md`` for a
    re-validated key, and the target is recomputed from the *current* managed
    root instead of being read back out of the entry.
    """

    @pytest.mark.parametrize("recorded", HOSTILE_RECORDED_PATHS)
    async def test_recorded_path_outside_the_root_is_refused(
        self, root: Path, recorded: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _UnlinkSpy().install(monkeypatch)
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {recorded: _entry("a", 1, SKILL_BODY)},
            },
        )

        report = await write_skills([], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        # The refusal is about ownership of the path, not about the file's state.
        assert "could own" in action.error
        assert [a for a in report.actions if a.action == "removed"] == []
        # Nothing was even attempted, let alone completed.
        assert spy.targets == []
        assert Path("/etc/passwd").exists()
        # Left in place rather than tidied away: dropping the entry would let a
        # single hostile edit erase the SDK's own record of what it manages.
        assert recorded in _read_manifest(root)["entries"]

    async def test_entry_under_a_since_symlinked_parent_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recorded path is the SDK's own, and is still not enough.

        Here the entry is exactly what a legitimate reconcile writes —
        ``a/SKILL.md`` under key ``a`` — so the shape check that catches the two
        cases above passes. What changed is the disk underneath it: ``<root>/a``
        is now a symlink to somewhere else. This is the case a validate-then-act
        implementation fails, because the manifest and the entry are both
        entirely legitimate; only the current state of the parent is not.
        """
        root = tmp_path / "skills"
        root.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        victim = elsewhere / "SKILL.md"
        victim.write_text("victim content\n", encoding="utf-8")

        # Managed legitimately first, so the manifest entry is one this SDK
        # really did write...
        managed = _place_managed(root, "a", SKILL_BODY)
        # ...then the parent directory is swapped for a link out of the root.
        managed.unlink()
        (root / "a").rmdir()
        (root / "a").symlink_to(elsewhere, target_is_directory=True)

        spy = _UnlinkSpy().install(monkeypatch)
        report = await write_skills([], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        assert "symlink" in action.error
        assert [a for a in report.actions if a.action == "removed"] == []
        assert spy.targets == []
        # The file the symlink pointed at is untouched, and so is the link.
        assert victim.read_text(encoding="utf-8") == "victim content\n"
        assert (root / "a").is_symlink()
        assert "a/SKILL.md" in _read_manifest(root)["entries"]


class TestWriteSkillsTelemetry:
    """Materialized / revoked signals from write_skills."""

    async def test_materialized_signal_per_action(
        self, root: Path, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        _place_managed(root, "same", SKILL_BODY)
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {
                    "same/SKILL.md": _entry("same", 1, SKILL_BODY),
                    "stale/SKILL.md": _entry("stale", 1, "old\n"),
                },
            },
        )
        (root / "stale").mkdir()
        (root / "stale" / "SKILL.md").write_text("old\n", encoding="utf-8")

        await write_skills(
            [_skill("same"), _skill("stale", 2, "fresh\n"), _skill("brand-new")],
            root,
        )

        signals = recording_emitter.signals(MATERIALIZED_SIGNAL)
        by_key = {s["skill_key"]: s for s in signals}
        assert len(signals) == 3
        assert by_key["same"]["reconcile_action"] == "skipped_current"
        assert by_key["stale"]["reconcile_action"] == "updated"
        assert by_key["brand-new"]["reconcile_action"] == "written"

    async def test_materialized_signal_properties(
        self, root: Path, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)

        await write_skills([_skill("a")], root)

        props = recording_emitter.signals(MATERIALIZED_SIGNAL)[0]
        assert props["skill_key"] == "a"
        assert props["content_bytes"] == len(SKILL_BODY.encode("utf-8"))
        assert props["content_hash"] == _hash(SKILL_BODY)
        assert props["reconcile_action"] == "written"
        assert props["language"] == "python"

    async def test_no_filesystem_paths_in_telemetry(
        self, root: Path, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)

        await write_skills([_skill("a")], root)

        for _signal, props in recording_emitter.records:
            assert "target_path" not in props
            for value in props.values():
                assert str(root) not in str(value)

    async def test_no_skill_body_in_telemetry(
        self, root: Path, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)

        await write_skills([_skill("a")], root)

        for _signal, props in recording_emitter.records:
            for value in props.values():
                assert "Do the thing." not in str(value)

    async def test_revoked_signal_on_prune(
        self, root: Path, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        _place_managed(root, "gone", SKILL_BODY, version=4)

        await write_skills([], root)

        revoked = recording_emitter.signals(REVOKED_SIGNAL)
        assert len(revoked) == 1
        assert revoked[0]["skill_key"] == "gone"
        assert revoked[0]["version"] == 4
        assert revoked[0]["removed_from_disk"] is True
        assert revoked[0]["language"] == "python"

    async def test_revoked_signal_redacts_an_untrusted_manifest_version(
        self, root: Path, recording_emitter: Any
    ) -> None:
        """The manifest is untrusted, so its version is shape-checked first.

        Anything with write access to the managed root can plant an arbitrary
        string here; echoing it verbatim would publish attacker-controlled
        content — a skill body, or PII — as a signal property.
        """
        skills_module._set_emitter_for_testing(recording_emitter)
        target = root / "gone" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text(SKILL_BODY, encoding="utf-8")
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {
                    "gone/SKILL.md": {
                        "key": "gone",
                        "version": "Do the thing. " * 8,
                        "sha256": _hash(SKILL_BODY),
                    }
                },
            },
        )

        await write_skills([], root)

        revoked = recording_emitter.signals(REVOKED_SIGNAL)
        assert len(revoked) == 1
        assert "version" not in revoked[0]
        assert revoked[0]["skill_key"] == "gone"
        for value in revoked[0].values():
            assert "Do the thing." not in str(value)

    async def test_no_revoked_signal_when_prune_disabled(
        self, root: Path, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        _place_managed(root, "gone", SKILL_BODY)

        await write_skills([], root, prune=False)

        assert recording_emitter.signals(REVOKED_SIGNAL) == []

    async def test_write_skills_records_no_signal_outside_the_approved_set(
        self, root: Path, recording_emitter: Any
    ) -> None:
        """Allowlist sweep over a run that exercises all four actions.

        The accessor-side half of this sweep is
        ``test_accessors_record_no_signal_outside_the_approved_set`` in
        test_skills.py. Asserted over recorded strings, so no module-level
        signal-name constant is required of the implementation.
        """
        skills_module._set_emitter_for_testing(recording_emitter)
        for key, content in (("same", SKILL_BODY), ("stale", "old\n"), ("gone", "g\n")):
            (root / key).mkdir()
            (root / key / "SKILL.md").write_text(content, encoding="utf-8")
        _write_manifest(
            root,
            {
                "manifestVersion": 1,
                "entries": {
                    "same/SKILL.md": _entry("same", 1, SKILL_BODY),
                    "stale/SKILL.md": _entry("stale", 1, "old\n"),
                    "gone/SKILL.md": _entry("gone", 1, "g\n"),
                },
            },
        )

        report = await write_skills(
            [_skill("same"), _skill("stale", 2, "fresh\n"), _skill("brand-new")],
            root,
        )

        # Positive control: the subset assertion is vacuous unless the run
        # really did produce all four actions and record for them.
        assert {a.action for a in report.actions} == {
            "skipped_current",
            "updated",
            "written",
            "removed",
        }
        recorded = {signal for signal, _props in recording_emitter.records}
        assert recorded <= APPROVED_SIGNALS, (
            f"unapproved signal(s): {sorted(recorded - APPROVED_SIGNALS)}"
        )
        assert not recorded & REMOVED_SIGNALS
        assert recorded == {MATERIALIZED_SIGNAL, REVOKED_SIGNAL}

    async def test_no_ld_track_calls_from_write_skills(
        self, root: Path, mock_ld_client: Any
    ) -> None:
        await init_client(client=mock_ld_client)

        await write_skills([_skill("a"), _skill("../evil")], root)

        mock_ld_client.track.assert_not_called()

    async def test_throwing_emitter_never_breaks_the_reconcile(
        self, root: Path, throwing_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(throwing_emitter)

        report = await write_skills([_skill("a")], root)

        assert report.ok is True
        assert (root / "a" / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY

    async def test_integrity_signal_property_keys_match_across_layers(
        self, root: Path, recording_emitter: Any
    ) -> None:
        """The same defect, caught at either layer, records
        the same property keys.

        Verification runs twice by design: once at the accessor boundary and
        again immediately before a write. The signal contract marks ``expected_hash``
        optional, so an implementation that populates it on one path and omits
        it on the other passes every other assertion here while making the
        signal's shape depend on which internal code path noticed. Oversize
        content is the case reachable from both layers with the expected hash in
        hand throughout.
        """
        skills_module._set_emitter_for_testing(recording_emitter)
        oversize = "x" * (64 * 1024 + 1)
        content_hash = _hash(oversize)

        # Layer 1 — the accessor boundary.
        store = InMemorySkillStore()
        store.put(
            {
                "key": "big",
                "version": 1,
                "content": oversize,
                "contentHash": content_hash,
            }
        )
        skills_module._set_store(store)
        assert await get_skill("big") is None

        # Layer 2 — verify-then-write, on a directly constructed Skill.
        report = await write_skills(
            [
                Skill(
                    key="big",
                    version=1,
                    content=oversize.encode("utf-8"),
                    content_hash=content_hash,
                )
            ],
            root,
        )
        assert report.ok is False

        failures = recording_emitter.signals(INTEGRITY_SIGNAL)
        assert len(failures) == 2, failures
        accessor_keys, write_keys = (set(props) for props in failures)
        assert accessor_keys == write_keys, (
            f"accessor-only keys: {sorted(accessor_keys - write_keys)}; "
            f"write-only keys: {sorted(write_keys - accessor_keys)}"
        )
        assert "expected_hash" in accessor_keys


# ---------------------------------------------------------------------------
# Self-healing partial reconciles
# ---------------------------------------------------------------------------


def _place_unmanaged(root: Path, key: str, content: str) -> Path:
    """A file at a managed path with **no** manifest entry.

    Exactly the state a reconcile killed between the content writes and the
    final manifest rewrite leaves behind — and, indistinguishably on disk, the
    state a customer authoring their own file there creates. Which is why the
    bytes are the only thing that may decide between them.
    """
    target = root / key / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


class TestCrashMidReconcileRecovery:
    """A crash between the writes and the manifest rewrite must not wedge a skill."""

    async def test_byte_identical_unmanaged_file_is_adopted(self, root: Path) -> None:
        """The whole point: the second reconcile repairs the first one's crash.

        Without adoption every later reconcile takes the clobber-refusal branch
        forever, because the file is at a managed path with no manifest entry.
        """
        target = _place_unmanaged(root, "a", SKILL_BODY)

        first = await write_skills([_skill("a")], root)

        assert first.ok is True, _error_messages(first)
        action = _actions_by_key(first)["a"]
        assert action.action == "skipped_current"
        assert action.version == 1
        assert action.path == str(target)
        # Adopted, not rewritten, and now recorded.
        assert target.read_text(encoding="utf-8") == SKILL_BODY
        entry = _read_manifest(root)["entries"]["a/SKILL.md"]
        assert entry["key"] == "a"
        assert entry["version"] == 1
        assert entry["sha256"] == _hash(SKILL_BODY)

        # And the run after it is an ordinary no-op, through the managed path.
        second = await write_skills([_skill("a")], root)
        assert second.ok is True, _error_messages(second)
        assert _actions_by_key(second)["a"].action == "skipped_current"

    async def test_adoption_writes_nothing(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adoption is a manifest edit, not a write. Nothing touches the bytes."""
        _place_unmanaged(root, "a", SKILL_BODY)
        spy = _ReplaceSpy().install(monkeypatch)

        report = await write_skills([_skill("a")], root)

        assert report.ok is True, _error_messages(report)
        assert spy.calls == []

    async def test_adoption_reports_skipped_current_not_a_new_action_kind(
        self, root: Path, recording_emitter: Any
    ) -> None:
        """``skipped_current`` is reused deliberately — no ``adopted`` kind exists."""
        skills_module._set_emitter_for_testing(recording_emitter)
        _place_unmanaged(root, "a", SKILL_BODY)

        report = await write_skills([_skill("a")], root)

        assert {a.action for a in report.actions} == {"skipped_current"}
        props = recording_emitter.signals(MATERIALIZED_SIGNAL)[0]
        assert props["reconcile_action"] == "skipped_current"
        assert props["skill_key"] == "a"

    async def test_an_adopted_file_is_prunable_afterwards(self, root: Path) -> None:
        """The documented caveat, pinned.

        Adoption records a manifest entry, so a later reconcile may prune the
        file. That is correct rather than a weakening: only content byte-identical
        to what LaunchDarkly resolved is ever adopted, so the prune removes
        content LaunchDarkly delivered — exactly what would have happened had the
        crash never occurred.
        """
        target = _place_unmanaged(root, "a", SKILL_BODY)
        await write_skills([_skill("a")], root)

        report = await write_skills([], root)

        assert report.ok is True, _error_messages(report)
        assert _actions_by_key(report)["a"].action == "removed"
        assert not target.exists()

    async def test_differing_unmanaged_content_is_still_refused(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The clobber guarantee, restated against the adoption rule.

        Adoption compares bytes, so anything that is not byte-identical to the
        resolved content falls through to the same refusal as before.
        """
        target = _place_unmanaged(root, "a", "user authored\n")
        spy = _ReplaceSpy().install(monkeypatch)

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        assert "did not write" in action.error
        assert target.read_text(encoding="utf-8") == "user authored\n"
        assert spy.calls == []

    async def test_a_longer_file_sharing_the_content_prefix_is_not_adopted(
        self, root: Path
    ) -> None:
        """The read is bounded at ``len(content) + 1``, and that one byte matters.

        A bound of exactly ``len(content)`` would make every file that merely
        *begins* with the resolved content hash as current, adopting — and later
        pruning — a customer file with the skill body at its head.
        """
        longer = SKILL_BODY + "and my own notes below\n"
        target = _place_unmanaged(root, "a", longer)

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        assert _actions_by_key(report)["a"].action == "error"
        assert target.read_text(encoding="utf-8") == longer

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this platform")
    async def test_an_unmanaged_fifo_is_refused_and_never_read(
        self, root: Path
    ) -> None:
        """Adoption widened the read to foreign files, so this refusal is load-bearing.

        Opening a FIFO with no writer blocks forever; the descriptor-pinned read
        opens ``O_NONBLOCK`` and rejects anything that is not a regular file, so
        this returns rather than hanging the reconcile and the event loop with it.
        """
        skill_dir = root / "a"
        skill_dir.mkdir()
        os.mkfifo(skill_dir / "SKILL.md")

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        assert "regular file" in action.error
        assert stat.S_ISFIFO(os.lstat(skill_dir / "SKILL.md").st_mode)

    async def test_a_read_failure_on_an_unmanaged_file_refuses(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed read proves nothing, so it must never become an overwrite.

        The adoption comparison is what would otherwise authorize the write, and
        a file whose bytes could not be read has not been shown to be ours.
        """
        target = _place_unmanaged(root, "a", SKILL_BODY)
        spy = _ReplaceSpy().install(monkeypatch)
        real_open = os.open

        def refuse_the_target(path: Any, *args: Any, **kwargs: Any) -> int:
            if isinstance(path, (str, os.PathLike)) and os.fspath(path) == str(target):
                raise PermissionError(13, "Permission denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(skills_fs_module.os, "open", refuse_the_target)

        report = await write_skills([_skill("a")], root)

        assert report.ok is False
        action = _actions_by_key(report)["a"]
        assert action.action == "error"
        assert action.error is not None
        # Distinguishable from the byte-mismatch refusal: this one says the
        # comparison could not be made at all.
        assert "could not be read to compare" in action.error
        assert spy.calls == []
        assert target.read_text(encoding="utf-8") == SKILL_BODY
        assert "a/SKILL.md" not in _read_manifest(root)["entries"]


# ---------------------------------------------------------------------------
# Orphaned temp files
# ---------------------------------------------------------------------------


def _temp_name(token: str = "0123456789abcdef") -> str:
    """A name ``atomic_write`` could have created for ``SKILL.md``.

    The prefix comes from ``safe_fs`` itself rather than a copy of its format
    string, so a change to the naming breaks this helper instead of silently
    making the sweep a no-op.
    """
    return f"{safe_fs_module.temp_name_prefix('SKILL.md')}{token}.tmp"


class TestOrphanedTempFiles:
    """A ``SIGKILL`` mid-write leaves a temp file nothing else records."""

    async def test_an_orphan_is_swept_on_the_next_write(self, root: Path) -> None:
        _place_managed(root, "a", SKILL_BODY)
        orphan = root / "a" / _temp_name()
        orphan.write_text("half-written body", encoding="utf-8")

        report = await write_skills([_skill("a")], root)

        assert report.ok is True, _error_messages(report)
        assert not orphan.exists()
        assert (root / "a" / "SKILL.md").read_text(encoding="utf-8") == SKILL_BODY

    async def test_an_orphan_no_longer_blocks_directory_cleanup(
        self, root: Path
    ) -> None:
        """The second-order effect: ``rmdir`` fails on a non-empty directory.

        One orphaned temp file would otherwise pin the skill's directory under
        the managed root forever, long after the skill itself was revoked.
        """
        _place_managed(root, "a", SKILL_BODY)
        orphan = root / "a" / _temp_name()
        orphan.write_text("half-written body", encoding="utf-8")

        report = await write_skills([], root)

        assert report.ok is True, _error_messages(report)
        assert _actions_by_key(report)["a"].action == "removed"
        assert not (root / "a").exists()

    @pytest.mark.parametrize(
        "innocent",
        [
            "notes.tmp",
            "SKILL.md.tmp",
            ".SKILL.md.tmp",
            ".SKILL.md..tmp",
            _temp_name("not-a-token"),
            _temp_name("0123456789abcdef") + ".bak",
            "x" + _temp_name(),
            _temp_name("0123456789abcdefff"),
        ],
    )
    async def test_a_lookalike_name_is_left_alone(
        self, root: Path, innocent: str
    ) -> None:
        """The recognizer is anchored at both ends, and the sweep deletes files.

        Anything that is not exactly the naming ``safe_fs`` produces belongs to
        the customer, whatever it resembles.
        """
        _place_managed(root, "a", SKILL_BODY)
        bystander = root / "a" / innocent
        bystander.write_text("mine\n", encoding="utf-8")

        report = await write_skills([_skill("a")], root)

        assert report.ok is True, _error_messages(report)
        assert bystander.read_text(encoding="utf-8") == "mine\n"

    @pytest.mark.skipif(
        not hasattr(os, "symlink"), reason="platform has no symlink support"
    )
    async def test_a_symlink_wearing_the_temp_name_is_not_removed(
        self, root: Path, tmp_path: Path
    ) -> None:
        """The temp naming must not become a way to have the SDK delete elsewhere.

        Only a regular file is ever swept, and the type comes off the descriptor
        rather than a followed path.
        """
        outside = tmp_path / "precious.txt"
        outside.write_text("do not delete\n", encoding="utf-8")
        _place_managed(root, "a", SKILL_BODY)
        link = root / "a" / _temp_name()
        link.symlink_to(outside)

        report = await write_skills([_skill("a")], root)

        assert report.ok is True, _error_messages(report)
        assert outside.read_text(encoding="utf-8") == "do not delete\n"
        assert link.is_symlink()


# ---------------------------------------------------------------------------
# Windows reserved device names
# ---------------------------------------------------------------------------

# Spelled out independently of the implementation's own set, so a name dropped
# from that set fails here rather than agreeing with itself.
WINDOWS_RESERVED_KEYS = (
    ["con", "prn", "aux", "nul"]
    + [f"com{digit}" for digit in range(1, 10)]
    + [f"lpt{digit}" for digit in range(1, 10)]
)


class TestWindowsReservedNames:
    """Keys Windows cannot hold as directory names, refused on every platform."""

    def test_the_set_is_exactly_twenty_two_names(self) -> None:
        assert len(WINDOWS_RESERVED_KEYS) == len(set(WINDOWS_RESERVED_KEYS)) == 22

    @pytest.mark.parametrize("reserved", WINDOWS_RESERVED_KEYS)
    async def test_a_reserved_key_is_refused_by_write_skills(
        self, root: Path, reserved: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _ReplaceSpy().install(monkeypatch)

        report = await write_skills([_skill(reserved)], root)

        assert report.ok is False
        action = _actions_by_key(report)[reserved]
        assert action.action == "error"
        assert action.error is not None
        assert "Windows reserves" in action.error
        assert reserved in action.error
        # Rejected before any filesystem call, not by the OS.
        assert spy.calls == []
        assert not (root / reserved).exists()

    @pytest.mark.parametrize("reserved", WINDOWS_RESERVED_KEYS)
    async def test_a_reserved_key_is_refused_by_the_prune_path(
        self, root: Path, reserved: str
    ) -> None:
        """``_key_rejection_reason`` gates both destructive paths, so both refuse.

        A manifest naming a reserved key is left in place rather than acted on:
        the same key check that stops the write stops the delete.
        """
        target = _place_managed(root, reserved, SKILL_BODY)

        report = await write_skills([], root)

        assert report.ok is False
        action = _actions_by_key(report)[reserved]
        assert action.action == "error"
        assert action.error is not None
        assert "left in place" in action.error
        assert target.read_text(encoding="utf-8") == SKILL_BODY

    @pytest.mark.parametrize("not_reserved", ["com0", "lpt0", "con1", "nul2", "conx"])
    async def test_neighbouring_names_are_not_reserved(
        self, root: Path, not_reserved: str
    ) -> None:
        """``com0`` and ``lpt0`` are not device names, and must still write."""
        report = await write_skills([_skill(not_reserved)], root)

        assert report.ok is True, _error_messages(report)
        assert (root / not_reserved / "SKILL.md").exists()

    @pytest.mark.parametrize("reserved", WINDOWS_RESERVED_KEYS)
    def test_a_reserved_name_is_still_a_valid_key_to_every_pure_layer(
        self, reserved: str
    ) -> None:
        """The layer choice, asserted — this is the whole point of it.

        The constraint lives in the filesystem layer and must not migrate into
        the key grammar. At the grammar level a rejection would fail the *entire*
        AI Config — model, provider, instructions, tools — for a Linux customer
        over a Windows-only constraint, and would shrink ``skill_refs``, which is
        what authorizes a prune: "this skill fails to write on Windows" would
        become "this skill gets deleted on Linux".
        """
        assert is_valid_skill_key(reserved) is True

        parsed = parse_ai_config(
            {
                "model": {"name": "claude-3"},
                "provider": {"name": "Anthropic"},
                "instructions": "You are helpful.",
                "skills": [{"key": reserved, "version": 1}],
            }
        )
        assert parsed.success is True

        refs = skill_refs(parsed.data)
        assert [ref.key for ref in refs] == [reserved]
