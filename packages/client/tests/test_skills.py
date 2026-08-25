"""
Agent Skills — reference discovery: the types and the projection from a
resolved AI Config.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import pytest

from launchdarkly_ai_server import (
    Skill,
    SkillReference,
    skill_refs,
)

SKILL_BODY = "---\nname: Test Skill\n---\nDo the thing.\n"


def _hash(content: str) -> str:
    """sha256, lowercase hex, over verbatim utf-8 bytes."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _skill(
    key: str = "test-skill",
    version: int = 1,
    content: str = SKILL_BODY,
    content_hash: str | None = None,
) -> Skill:
    """Build a verified-shaped Skill directly (bypasses the accessors)."""
    return Skill(
        key=key,
        version=version,
        content=content.encode("utf-8"),
        content_hash=content_hash if content_hash is not None else _hash(content),
    )


class TestSkillTypes:
    """Immutability and optional metadata."""

    def test_skill_reference_is_immutable(self) -> None:
        ref = SkillReference(key="pdf-extraction", version=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.version = 3  # type: ignore[misc]

    def test_skill_is_immutable(self) -> None:
        skill = _skill()
        with pytest.raises(dataclasses.FrozenInstanceError):
            skill.content = b"tampered"  # type: ignore[misc]

    def test_skill_content_is_bytes(self) -> None:
        """Content is the verified verbatim bytes — opaque, never text."""
        skill = _skill()
        assert isinstance(skill.content, bytes)
        assert skill.content == SKILL_BODY.encode("utf-8")

    def test_skill_carries_optional_metadata(self) -> None:
        skill = Skill(
            key="a",
            version=1,
            content=SKILL_BODY.encode("utf-8"),
            content_hash=_hash(SKILL_BODY),
            name="A Skill",
            description="does things",
        )
        assert skill.name == "A Skill"
        assert skill.description == "does things"

    def test_skill_metadata_defaults_to_none(self) -> None:
        skill = _skill()
        assert skill.name is None
        assert skill.description is None


class TestSkillRefs:
    """Pure projection of the config's skills array."""

    def _config(self, **extra: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "instructions": "hi",
        }
        base.update(extra)
        return base

    def test_absent_skills_returns_empty_list(self) -> None:
        assert skill_refs(self._config()) == []

    def test_empty_skills_returns_empty_list(self) -> None:
        assert skill_refs(self._config(skills=[])) == []

    def test_returns_typed_references_in_order(self) -> None:
        config = self._config(
            skills=[{"key": "a", "version": 1}, {"key": "b", "version": 3}]
        )
        refs = skill_refs(config)
        assert refs == [
            SkillReference(key="a", version=1),
            SkillReference(key="b", version=3),
        ]
        assert all(isinstance(r, SkillReference) for r in refs)

    def test_dropped_entries_are_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A shortened projection is never silent.

        ``parse_ai_config`` fails the whole config closed on a malformed
        reference, so a config that reached here through it cannot contain one.
        A hand-built dict can, and feeding the shortened list to
        ``write_skills`` would prune the dropped skill's on-disk copy — so the
        drop is observable rather than silent.
        """
        config = self._config(
            skills=[
                {"key": "good", "version": 1},
                {"key": "bad", "version": 0},
                {"key": "Bad-Key", "version": 1},
                "not-an-object",
            ]
        )

        with caplog.at_level("WARNING", logger="launchdarkly_ai_server.skills"):
            refs = skill_refs(config)

        assert refs == [SkillReference(key="good", version=1)]
        assert len(caplog.records) == 3
        # The body is never echoed, and neither is the invalid key.
        assert all("skills[" in r.getMessage() for r in caplog.records)

    def test_requires_no_client_or_store(self, mock_ld_client: Any) -> None:
        """No store configured, no client initialized — still a pure projection."""
        refs = skill_refs(self._config(skills=[{"key": "a", "version": 2}]))
        assert refs == [SkillReference(key="a", version=2)]
        mock_ld_client.track.assert_not_called()
