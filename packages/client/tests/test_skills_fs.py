"""
Tests for ``write_skills`` — filesystem materialization and manifest reconcile
semantics.

Every test writes only inside pytest's ``tmp_path``. No network, no real
LaunchDarkly client, no real skill transport.

The security abuse matrix — path traversal, symlink attacks, clobber
protection, corrupt manifests, atomicity under an injected crash, and the
materialization telemetry allowlist — is a separate module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import launchdarkly_ai_server.skills as skills_module
from launchdarkly_ai_server import (
    InMemorySkillStore,
    Skill,
    SkillReference,
    write_skills,
)

MANIFEST_NAME = ".launchdarkly-skills.json"
SKILL_BODY = "---\nname: Test Skill\n---\nDo the thing.\n"

INTEGRITY_SIGNAL = "AgentControl Skill Integrity Failure"


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
