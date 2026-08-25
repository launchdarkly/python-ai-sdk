"""
Tests for Agent Skills types, frontmatter, reference discovery, content
accessors, integrity verification, and the telemetry seam.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
import launchdarkly_ai_server.skills as skills_module
from launchdarkly_ai_server import (
    InMemorySkillStore,
    ReconcileAction,
    ReconcileReport,
    Skill,
    SkillReference,
    all_skills,
    get_client,
    get_skill,
    get_skills,
    init_client,
    shutdown,
    skill_refs,
)

SKILL_BODY = "---\nname: Test Skill\n---\nDo the thing.\n"

INTEGRITY_SIGNAL = "AgentControl Skill Integrity Failure"
MATERIALIZED_SIGNAL = "AgentControl Skill Materialized"
REVOKED_SIGNAL = "AgentControl Skill Revoked Received"

# The three signal names are an allowlist, not a floor: any
# other name reaching the emitter is a regression.
APPROVED_SIGNALS = frozenset({INTEGRITY_SIGNAL, MATERIALIZED_SIGNAL, REVOKED_SIGNAL})

# These two were considered and deliberately excluded from SDK emission —
# named explicitly rather than relying on the subset check to be read as
# covering them.
REMOVED_SIGNALS = frozenset(
    {
        "AgentControl Skill SDK Reference Returned",
        "AgentControl Skill Content Retrieved",
    }
)


pytestmark = pytest.mark.usefixtures("reset_skill_state")
"""Every test in this module runs against freshly cleared module state."""


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
        content=content,
        content_hash=content_hash if content_hash is not None else _hash(content),
    )


# ---------------------------------------------------------------------------
# Unencodable content
#
# A lone surrogate has no UTF-8 encoding, so there are no bytes LaunchDarkly
# could have hashed and the object is not authentic. ``json.loads`` is how one
# actually arrives: a payload carries the escape, and the parser hands back a
# real unpaired surrogate.
# ---------------------------------------------------------------------------

UNENCODABLE_BODIES = (
    json.loads(r'"hi \ud800 there"'),  # lone high surrogate
    # A lone *low* surrogate, which is the one range errors="surrogateescape"
    # smuggles through (as a raw 0x80 byte) while raising on everything else.
    json.loads(r'"hi \udc80 there"'),
)

NON_STRICT_HANDLERS = (
    "surrogatepass",
    "surrogateescape",
    "replace",
    "ignore",
    "backslashreplace",
    "xmlcharrefreplace",
    "namereplace",
)
"""Every ``str.encode`` error handler that is not ``strict``.

``verified_bytes`` must use none of them: each one *fabricates* bytes for input
that has no encoding, and fabricated bytes can satisfy the hash comparison.
"""


def _fabricated_hash_cases() -> list[Any]:
    """One case per (body, handler) pair the handler can actually encode.

    Each carries the sha256 of the bytes *that* handler would have produced, so
    the case is not vacuous: an implementation that reached for the handler
    would encode successfully, match the pinned hash, and return content
    LaunchDarkly never delivered. Handlers that raise on a given body are
    skipped — for that input they are as strict as ``strict``, so there is
    nothing to detect.
    """
    cases: list[Any] = []
    for index, body in enumerate(UNENCODABLE_BODIES):
        for handler in NON_STRICT_HANDLERS:
            try:
                fabricated = body.encode("utf-8", errors=handler)
            except UnicodeEncodeError:
                continue
            cases.append(
                pytest.param(
                    body,
                    hashlib.sha256(fabricated).hexdigest(),
                    id=f"body{index}-{handler}",
                )
            )
    return cases


FABRICATED_HASH_CASES = _fabricated_hash_cases()


class TestSkillTypes:
    """Immutability and ReconcileReport.ok."""

    def test_skill_reference_is_immutable(self) -> None:
        ref = SkillReference(key="pdf-extraction", version=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.version = 3  # type: ignore[misc]

    def test_skill_is_immutable(self) -> None:
        skill = _skill()
        with pytest.raises(dataclasses.FrozenInstanceError):
            skill.content = "tampered"  # type: ignore[misc]

    def test_skill_carries_optional_metadata(self) -> None:
        skill = Skill(
            key="a",
            version=1,
            content=SKILL_BODY,
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

    def test_report_ok_true_when_no_error_action(self) -> None:
        report = ReconcileReport(
            actions=[
                ReconcileAction(key="a", action="written", version=1),
                ReconcileAction(key="b", action="skipped_current", version=2),
                ReconcileAction(key="c", action="removed"),
                ReconcileAction(key="d", action="updated", version=3),
            ]
        )
        assert report.ok is True

    def test_report_ok_false_with_error_action(self) -> None:
        report = ReconcileReport(
            actions=[
                ReconcileAction(key="a", action="written", version=1),
                ReconcileAction(key="b", action="error", error="nope"),
            ]
        )
        assert report.ok is False

    def test_empty_report_is_ok(self) -> None:
        assert ReconcileReport(actions=[]).ok is True

    def test_report_errors_lists_error_actions_in_order(self) -> None:
        """The report exposes its error actions itself."""
        first = ReconcileAction(key="b", action="error", error="first")
        second = ReconcileAction(key="d", action="error", error="second")
        report = ReconcileReport(
            actions=[
                ReconcileAction(key="a", action="written", version=1),
                first,
                ReconcileAction(key="c", action="skipped_current", version=2),
                second,
                ReconcileAction(key="e", action="removed"),
            ]
        )
        assert report.errors == [first, second]

    def test_report_errors_empty_when_no_error_action(self) -> None:
        report = ReconcileReport(
            actions=[
                ReconcileAction(key="a", action="written", version=1),
                ReconcileAction(key="b", action="removed"),
            ]
        )
        assert report.errors == []

    def test_empty_report_has_no_errors(self) -> None:
        assert ReconcileReport(actions=[]).errors == []

    def test_report_ok_and_errors_always_agree(self) -> None:
        """``ok`` is true iff ``errors`` is empty, on the same objects."""
        clean = ReconcileReport(
            actions=[ReconcileAction(key="a", action="written", version=1)]
        )
        failed = ReconcileReport(
            actions=[
                ReconcileAction(key="a", action="written", version=1),
                ReconcileAction(key="b", action="error", error="nope"),
            ]
        )
        for report in (clean, failed, ReconcileReport(actions=[])):
            assert report.ok is (report.errors == [])


class TestPackageExports:
    """
    The on-the-wire / on-disk constants are public API and
    must be reachable from the package root, not only from the sub-path module.

    The literal values are spelled out here on purpose: this is the one place
    the constants themselves are asserted, so importing them to build the
    expectation would make the assertion circular. Every other filesystem test
    keeps writing the literals by hand for the same reason.
    """

    def test_constants_are_exported_from_the_package_root(self) -> None:
        import launchdarkly_ai_server as package

        assert package.SKILL_OBJECT_KIND == "skill"
        assert package.SKILL_FILENAME == "SKILL.md"
        assert package.MANIFEST_FILENAME == ".launchdarkly-skills.json"
        assert package.MANIFEST_VERSION == 1

    def test_constants_are_listed_in_dunder_all(self) -> None:
        """A name absent from ``__all__`` is not part of the public surface."""
        import launchdarkly_ai_server as package

        expected = {
            "SKILL_OBJECT_KIND",
            "SKILL_FILENAME",
            "MANIFEST_FILENAME",
            "MANIFEST_VERSION",
        }
        assert expected <= set(package.__all__)

    def test_content_cap_is_not_public_api(self) -> None:
        """The content cap stays internal to ``skills_core`` — see the
        ``MAX_SKILL_CONTENT_BYTES`` docstring there for why it is not exported."""
        import launchdarkly_ai_server as package
        from launchdarkly_ai_server import skills_core

        assert skills_core.MAX_SKILL_CONTENT_BYTES == 65536
        assert "MAX_SKILL_CONTENT_BYTES" not in package.__all__
        assert not hasattr(package, "MAX_SKILL_CONTENT_BYTES")

    def test_closed_set_types_are_exported_from_the_package_root(self) -> None:
        """The two closed-set unions are public API, not implementation detail.

        ``ReconcileActionKind`` types the ``ReconcileAction.action`` field every
        consumer of a report reads and switches on, and ``OnUnavailable`` types
        a public keyword argument of ``write_skills``. ``agents.md`` forbids
        handler packages from importing sub-path modules, so a name exported
        only from the implementation module has no supported import path.
        """
        import launchdarkly_ai_server as package

        assert hasattr(package, "ReconcileActionKind")
        assert hasattr(package, "OnUnavailable")
        assert {"ReconcileActionKind", "OnUnavailable"} <= set(package.__all__)

    def test_exported_action_union_admits_exactly_the_five_actions(self) -> None:
        """The union must match the actions a report can actually carry.

        Spelled out rather than imported from the implementation for the same
        reason as the constants above: deriving the expectation from the thing
        under test would make the assertion circular.
        """
        import typing

        import launchdarkly_ai_server as package

        assert set(typing.get_args(package.ReconcileActionKind)) == {
            "written",
            "updated",
            "skipped_current",
            "removed",
            "error",
        }
        assert set(typing.get_args(package.OnUnavailable)) == {"keep", "raise"}


class TestFrontmatter:
    """Bounded, safe, lazy YAML frontmatter parsing."""

    def test_valid_frontmatter_parses(self) -> None:
        content = "---\nname: test\nversion: 1\n---\nBody text\n"
        result = _skill(content=content).frontmatter()
        assert result == {"name": "test", "version": 1}

    def test_absent_frontmatter_returns_none(self) -> None:
        content = "# Just markdown\n\nNo frontmatter here.\n"
        assert _skill(content=content).frontmatter() is None

    def test_unterminated_block_returns_none(self) -> None:
        content = "---\nname: test\nnever closed\n"
        assert _skill(content=content).frontmatter() is None

    def test_malformed_yaml_returns_none(self) -> None:
        content = "---\nname: [unclosed\n  bad: : :\n---\nBody\n"
        assert _skill(content=content).frontmatter() is None

    def test_non_mapping_frontmatter_returns_none(self) -> None:
        content = "---\n- one\n- two\n---\nBody\n"
        assert _skill(content=content).frontmatter() is None

    def test_oversize_block_returns_none(self) -> None:
        big = "\n".join(f"key{i}: {'x' * 80}" for i in range(200))
        content = f"---\n{big}\n---\nBody\n"
        assert len(big) > 8 * 1024
        assert _skill(content=content).frontmatter() is None

    def test_deep_nesting_returns_none(self) -> None:
        block = "".join(f"{'  ' * i}k{i}:\n" for i in range(14)) + f"{'  ' * 14}v: 1\n"
        content = f"---\n{block}---\nBody\n"
        started = time.monotonic()
        result = _skill(content=content).frontmatter()
        assert result is None
        assert time.monotonic() - started < 5.0

    def test_single_alias_returns_none(self) -> None:
        """Alias resolution is *disabled*, not bounded, so one alias is
        already disqualifying. This minimal case is the actual boundary the rule
        draws; the billion-laughs bomb below is only a corollary of it."""
        content = "---\nname: test\nanchored: &a 1\naliased: *a\n---\nBody\n"
        assert _skill(content=content).frontmatter() is None

    def test_billion_laughs_does_not_hang_or_crash(self) -> None:
        """The classic bomb, which the alias rule rejects on sight.

        The threat here is *not* memory blow-up: PyYAML resolves aliases as
        shared references, so plain ``yaml.safe_load`` parses this in about a
        millisecond and returns a 7-key dict. Nor does any other bound catch it
        — it is ~300 bytes and 6 levels deep, inside both the 8 KB and depth-10
        limits. The contract asserted is the one from
        ``test_single_alias_returns_none``: an alias is present ⇒ None. The
        elapsed-time bound below only guards a parser that *does* expand.
        """
        bomb = (
            "---\n"
            "a: &a ['x','x','x','x','x','x','x','x','x']\n"
            "b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]\n"
            "c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]\n"
            "d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]\n"
            "e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d]\n"
            "f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e]\n"
            "g: [*f,*f,*f,*f,*f,*f,*f,*f,*f]\n"
            "---\nBody\n"
        )
        started = time.monotonic()
        result = _skill(content=bomb).frontmatter()
        elapsed = time.monotonic() - started
        assert result is None
        assert elapsed < 5.0

    def test_python_object_tag_is_inert(self) -> None:
        content = "---\nevil: !!python/object/apply:os.system ['echo pwned']\n---\nB\n"
        assert _skill(content=content).frontmatter() is None

    def test_custom_tag_is_inert(self) -> None:
        content = "---\nevil: !SomeType {a: 1}\n---\nBody\n"
        assert _skill(content=content).frontmatter() is None

    def test_returns_none_when_yaml_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pyyaml is dev-only; absence must not raise."""
        monkeypatch.setitem(sys.modules, "yaml", None)
        content = "---\nname: test\n---\nBody\n"
        assert _skill(content=content).frontmatter() is None

    def test_yaml_is_not_imported_at_module_scope(self) -> None:
        """The import must live inside frontmatter(); a module-level ``import
        yaml`` would bind a ``yaml`` attribute on the skills module and make
        pyyaml a de-facto runtime dependency."""
        assert not hasattr(skills_module, "yaml")


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

    def test_emits_no_telemetry(self, recording_emitter: Any) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        skill_refs(self._config(skills=[{"key": "a", "version": 1}]))
        assert recording_emitter.records == []

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


class TestInMemorySkillStore:
    """The public in-memory store implementation."""

    def test_get_object_round_trips(self, make_raw_skill: Any) -> None:
        raw = make_raw_skill(key="pdf-extraction", version=2)
        s = InMemorySkillStore({"pdf-extraction": raw})
        assert s.get_object("skill", "pdf-extraction") == raw

    def test_get_object_unknown_key_returns_none(self) -> None:
        assert InMemorySkillStore().get_object("skill", "nope") is None

    def test_put_then_get(self, make_raw_skill: Any) -> None:
        s = InMemorySkillStore()
        raw = make_raw_skill(key="a")
        s.put(raw)
        assert s.get_object("skill", "a") == raw

    def test_all_objects_returns_everything(self, make_raw_skill: Any) -> None:
        s = InMemorySkillStore()
        s.put(make_raw_skill(key="a"))
        s.put(make_raw_skill(key="b"))
        assert set(s.all_objects("skill").keys()) == {"a", "b"}

    def test_all_objects_unknown_kind_is_empty(self, make_raw_skill: Any) -> None:
        s = InMemorySkillStore()
        s.put(make_raw_skill(key="a"))
        assert s.all_objects("flag") == {}

    def test_put_notifies_skill_kind_listeners(self, make_raw_skill: Any) -> None:
        """``add_listener`` is part of the seam, so its one
        implementation carries a smoke test for the callback contract: the raw
        object, verbatim and unverified, as a single positional argument."""
        s = InMemorySkillStore()
        seen: list[dict[str, Any]] = []
        s.add_listener("skill", seen.append)
        raw = make_raw_skill(key="a")

        s.put(raw)

        assert seen == [raw]

    def test_put_does_not_notify_other_kind_listeners(
        self, make_raw_skill: Any
    ) -> None:
        s = InMemorySkillStore()
        seen: list[dict[str, Any]] = []
        s.add_listener("flag", seen.append)

        s.put(make_raw_skill(key="a"))

        assert seen == []


class TestStoreConfiguration:
    """Store wiring on the lifecycle layer."""

    async def test_configured_via_init_client_option(
        self, make_raw_skill: Any, mock_ld_client: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a"))
        await init_client(options={"skillStore": store}, client=mock_ld_client)
        skill = await get_skill("a")
        assert skill is not None
        assert skill.key == "a"

    async def test_get_skill_raises_actionably_when_no_store(self) -> None:
        with pytest.raises(RuntimeError, match="skill store"):
            await get_skill("a")

    async def test_get_skills_raises_actionably_when_no_store(self) -> None:
        with pytest.raises(RuntimeError, match="skill store"):
            await get_skills([SkillReference(key="a", version=1)])

    async def test_all_skills_raises_actionably_when_no_store(self) -> None:
        with pytest.raises(RuntimeError, match="skill store"):
            await all_skills()

    async def test_shutdown_clears_the_store(
        self, make_raw_skill: Any, mock_ld_client: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a"))
        await init_client(options={"skillStore": store}, client=mock_ld_client)
        assert await get_skill("a") is not None

        await shutdown()

        with pytest.raises(RuntimeError, match="skill store"):
            await get_skill("a")

    async def test_skill_store_is_applied_on_every_init_client_call(
        self, make_raw_skill: Any
    ) -> None:
        """``skillStore`` is the one option a second call applies.

        ``init_client`` is idempotent for the client singleton, and on a second
        call every other option is ignored. ``skillStore`` is applied anyway,
        on purpose: it is what lets a client that was lazily auto-initialized,
        or initialized without a store, be given one afterwards. Both halves are
        asserted on the same pair of calls, because each is meaningless without
        the other.
        """
        first_store = InMemorySkillStore()
        first_store.put(make_raw_skill(key="first"))
        second_store = InMemorySkillStore()
        second_store.put(make_raw_skill(key="second"))

        first_client = MagicMock()
        second_client = MagicMock()

        await init_client(options={"skillStore": first_store}, client=first_client)
        await init_client(options={"skillStore": second_store}, client=second_client)

        # Half one: the client singleton is unchanged — the second call is a
        # no-op for it, so the second client was discarded.
        assert get_client() is first_client

        # Half two: the store was nevertheless swapped.
        assert await get_skill("second") is not None
        assert await get_skill("first") is None

    async def test_init_client_without_a_store_leaves_the_configured_one(
        self, make_raw_skill: Any
    ) -> None:
        """Only a non-None ``skillStore`` replaces the configured store.

        Otherwise a bare ``init_client()`` from an unrelated code path — the
        lazy auto-init, say — would silently unconfigure skills.
        """
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a"))
        await init_client(options={"skillStore": store}, client=MagicMock())

        await init_client(client=MagicMock())

        assert await get_skill("a") is not None

    async def test_failed_init_client_leaves_no_store_configured(
        self, monkeypatch: pytest.MonkeyPatch, make_raw_skill: Any
    ) -> None:
        """A raising ``init_client`` must not leave global state behind.

        Installing the store before the SDK-key check would leave the accessors
        working against a store the application believes was never installed,
        masking a failed initialization.
        """
        monkeypatch.delenv("LD_SDK_KEY", raising=False)
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a"))

        with pytest.raises(RuntimeError, match="No LaunchDarkly SDK key"):
            await init_client(options={"skillStore": store})

        with pytest.raises(RuntimeError, match="skill store"):
            await get_skill("a")

    async def test_reset_for_testing_clears_the_store(
        self, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a"))
        skills_module._set_store(store)
        assert await get_skill("a") is not None

        lifecycle_module._reset_for_testing()

        with pytest.raises(RuntimeError, match="skill store"):
            await get_skill("a")


class TestAccessorArgumentErrors:
    """A bare string is a type error, not a reference.

    ``str`` satisfies ``Sequence[str]``, so the annotation on ``get_skills``
    admits a bare string and only this runtime guard catches it; iterating one
    would look up a skill per character. Deliberately a *different* class from
    ``write_skills``'s bare-string rejection, which is a ``ValueError`` because
    a string is an accepted argument type there.
    """

    async def test_bare_string_raises_type_error(
        self, store: InMemorySkillStore
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            await get_skills("pdf-extraction")  # type: ignore[arg-type]

        # The message has to name the fix, not merely reject the input.
        assert "[key]" in str(excinfo.value)

    async def test_bare_string_is_rejected_before_the_store_is_consulted(
        self, make_raw_skill: Any
    ) -> None:
        """The guard is an argument check, so it precedes store resolution.

        Asserting the raise alone would also pass if the string were iterated
        into single-character lookups that all missed, so pin that no lookup
        happened at all.
        """
        looked_up: list[str] = []

        class _RecordingStore:
            def get_object(self, kind: str, key: str) -> dict[str, Any] | None:
                looked_up.append(key)
                return None

            def all_objects(self, kind: str) -> dict[str, dict[str, Any]]:
                return {}

        skills_module._set_store(_RecordingStore())

        with pytest.raises(TypeError):
            await get_skills("abc")  # type: ignore[arg-type]

        assert looked_up == []


class TestGetSkill:
    """Single-skill accessor."""

    async def test_returns_verified_skill(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="pdf-extraction", version=2))
        skill = await get_skill("pdf-extraction")
        assert skill is not None
        assert skill.key == "pdf-extraction"
        assert skill.version == 2
        assert skill.content == SKILL_BODY
        assert skill.content_hash == _hash(SKILL_BODY)
        assert skill.name == "Test Skill"
        assert skill.description == "A skill used in tests."

    async def test_version_omitted_returns_newest_available(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", version=7))
        skill = await get_skill("a")
        assert skill is not None
        assert skill.version == 7

    async def test_exact_version_match_returns_skill(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", version=3))
        skill = await get_skill("a", version=3)
        assert skill is not None
        assert skill.version == 3

    async def test_other_version_returns_none(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", version=3))
        assert await get_skill("a", version=2) is None
        assert await get_skill("a", version=4) is None

    async def test_missing_key_returns_none(self, store: InMemorySkillStore) -> None:
        assert await get_skill("nope") is None

    async def test_multibyte_content_verifies(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        content = "---\nname: emoji\n---\n🚀 unicode ✅ body\n"
        store.put(make_raw_skill(key="a", content=content))
        skill = await get_skill("a")
        assert skill is not None
        assert skill.content == content


class TestGetSkills:
    """Batch accessor."""

    async def test_mixed_refs_and_strings(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", version=1))
        store.put(make_raw_skill(key="b", version=5))
        result = await get_skills([SkillReference(key="a", version=1), "b"])
        assert [s.key for s in result] == ["a", "b"]

    async def test_preserves_input_order(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        for k in ("a", "b", "c"):
            store.put(make_raw_skill(key=k))
        result = await get_skills(["c", "a", "b"])
        assert [s.key for s in result] == ["c", "a", "b"]

    async def test_missing_entries_are_omitted(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a"))
        result = await get_skills(["a", "missing"])
        assert [s.key for s in result] == ["a"]

    async def test_version_mismatch_is_omitted(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", version=2))
        result = await get_skills([SkillReference(key="a", version=1)])
        assert result == []

    async def test_empty_input_returns_empty_list(
        self, store: InMemorySkillStore
    ) -> None:
        assert await get_skills([]) == []

    async def test_integrity_failure_omitted_and_signalled(
        self,
        store: InMemorySkillStore,
        make_raw_skill: Any,
        recording_emitter: Any,
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        store.put(make_raw_skill(key="good-a"))
        store.put(make_raw_skill(key="bad", contentHash="0" * 64))
        store.put(make_raw_skill(key="good-b"))

        result = await get_skills(["good-a", "bad", "good-b"])

        assert [s.key for s in result] == ["good-a", "good-b"]
        failures = recording_emitter.signals(INTEGRITY_SIGNAL)
        assert len(failures) == 1
        assert failures[0]["skill_key"] == "bad"


class TestAllSkills:
    """All_skills accessor."""

    async def test_returns_every_verified_skill(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        for k in ("a", "b", "c"):
            store.put(make_raw_skill(key=k))
        result = await all_skills()
        assert {s.key for s in result} == {"a", "b", "c"}

    async def test_empty_store_returns_empty_list(
        self, store: InMemorySkillStore
    ) -> None:
        assert await all_skills() == []

    async def test_omits_skills_that_fail_verification(
        self, store: InMemorySkillStore, make_raw_skill: Any, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        store.put(make_raw_skill(key="good"))
        store.put(make_raw_skill(key="bad", contentHash="deadbeef"))
        result = await all_skills()
        assert {s.key for s in result} == {"good"}
        assert len(recording_emitter.signals(INTEGRITY_SIGNAL)) == 1


class TestIntegrityVerification:
    """Mandatory verification at the accessor boundary."""

    async def test_hash_mismatch_withholds_skill(
        self, store: InMemorySkillStore, make_raw_skill: Any, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        store.put(make_raw_skill(key="a", contentHash="a" * 64))
        assert await get_skill("a") is None
        assert len(recording_emitter.signals(INTEGRITY_SIGNAL)) == 1

    async def test_tampered_content_withholds_skill(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        raw = make_raw_skill(key="a")
        raw["content"] = raw["content"] + "x"  # hash now stale by one byte
        store.put(raw)
        assert await get_skill("a") is None

    async def test_oversize_content_rejected(
        self, store: InMemorySkillStore, make_raw_skill: Any, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        oversize = "x" * (64 * 1024 + 1)
        store.put(make_raw_skill(key="a", content=oversize))
        assert await get_skill("a") is None
        assert len(recording_emitter.signals(INTEGRITY_SIGNAL)) == 1

    async def test_content_at_size_cap_is_accepted(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        at_cap = "x" * (64 * 1024)
        store.put(make_raw_skill(key="a", content=at_cap))
        skill = await get_skill("a")
        assert skill is not None
        assert len(skill.content) == 64 * 1024

    async def test_key_at_length_bound_from_store_accepted(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        """The accepting side of the <= 256 bound.

        ``write_skills`` cannot reach this bound (a key is one directory name and
        NAME_MAX is 255), so config validation and this accessor-side
        revalidation are the only two layers where 256 is observable at all. The
        rejecting side is ``test_invalid_key_from_store_rejected["x" * 257]``.
        """
        key = "a" * 256
        store.put(make_raw_skill(key=key))
        skill = await get_skill(key)
        assert skill is not None
        assert skill.key == key

    @pytest.mark.parametrize(
        "bad_key",
        [
            "Evil",
            "-leading-dash",
            ".hidden",
            "has space",
            "a/b",
            "../escape",
            "",
            "x" * 257,
        ],
    )
    async def test_invalid_key_from_store_rejected(
        self, make_raw_skill: Any, bad_key: str
    ) -> None:
        """A hostile store may serve any key — the accessor revalidates."""
        raw = make_raw_skill(key="placeholder")
        raw["key"] = bad_key
        skills_module._set_store(InMemorySkillStore({bad_key: raw}))
        assert await get_skill(bad_key) is None

    @pytest.mark.parametrize("bad_version", [0, -1, 2.5, "2", None, True])
    async def test_invalid_version_from_store_rejected(
        self, make_raw_skill: Any, bad_version: Any
    ) -> None:
        raw = make_raw_skill(key="a", version=bad_version)
        skills_module._set_store(InMemorySkillStore({"a": raw}))
        assert await get_skill("a") is None

    async def test_missing_content_rejected(self, make_raw_skill: Any) -> None:
        raw = make_raw_skill(key="a")
        del raw["content"]
        skills_module._set_store(InMemorySkillStore({"a": raw}))
        assert await get_skill("a") is None

    async def test_missing_content_hash_rejected(self, make_raw_skill: Any) -> None:
        raw = make_raw_skill(key="a")
        del raw["contentHash"]
        skills_module._set_store(InMemorySkillStore({"a": raw}))
        assert await get_skill("a") is None

    async def test_uppercase_hash_rejected(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        """Hashes are lowercase hex; a non-canonical hash is not authentic."""
        store.put(make_raw_skill(key="a", contentHash=_hash(SKILL_BODY).upper()))
        assert await get_skill("a") is None

    @pytest.mark.parametrize(("body", "fabricated_hash"), FABRICATED_HASH_CASES)
    async def test_unencodable_content_withheld(
        self, recording_emitter: Any, body: str, fabricated_hash: str
    ) -> None:
        """Content with no UTF-8 encoding is withheld, and the signal recorded.

        ``str.encode`` raises on a lone surrogate, so ``verified_bytes`` has an
        exception to catch and never sees bytes for this content at all.

        The parametrization is what makes that observable. The guard must never
        pass ``errors="surrogatepass"``, or any other non-strict handler: each
        of them fabricates bytes for input that has no encoding, and fabricated
        bytes can satisfy the hash comparison. Every case here supplies the
        sha256 of the bytes one such handler would have produced, so an
        implementation that reached for one would verify this object
        successfully and hand back content LaunchDarkly never sent. An
        arbitrary wrong hash would not catch that — the mismatch check would
        reject the input before the encoder guard was reached.
        """
        with pytest.raises(UnicodeEncodeError):
            body.encode("utf-8")  # the premise: there is no encoding to hash

        skills_module._set_emitter_for_testing(recording_emitter)
        skills_module._set_store(
            InMemorySkillStore(
                {
                    "a": {
                        "key": "a",
                        "version": 1,
                        "content": body,
                        "contentHash": fabricated_hash,
                    }
                }
            )
        )

        assert await get_skill("a") is None
        assert len(recording_emitter.signals(INTEGRITY_SIGNAL)) == 1

    async def test_store_error_does_not_leak_content(
        self, exploding_store: Any
    ) -> None:
        assert await get_skill("a") is None
        assert await get_skills(["a"]) == []
        assert await all_skills() == []


class TestTelemetrySeam:
    """Internal emitter seam, no client.track, no context."""

    async def test_default_emitter_is_noop(
        self, store: InMemorySkillStore, make_raw_skill: Any
    ) -> None:
        store.put(make_raw_skill(key="a", contentHash="0" * 64))
        assert await get_skill("a") is None  # no emitter injected, no raise

    async def test_integrity_signal_properties(
        self,
        store: InMemorySkillStore,
        make_raw_skill: Any,
        recording_emitter: Any,
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        store.put(make_raw_skill(key="a", version=4, contentHash="b" * 64))

        await get_skill("a")

        props = recording_emitter.signals(INTEGRITY_SIGNAL)[0]
        assert props["skill_key"] == "a"
        assert props["version"] == 4
        assert props["expected_hash"] == "b" * 64
        assert props["observed_hash"] == _hash(SKILL_BODY)
        assert props["language"] == "python"

    async def test_skill_body_never_appears_in_signals(
        self, store: InMemorySkillStore, make_raw_skill: Any, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        store.put(make_raw_skill(key="a", contentHash="c" * 64))

        await get_skill("a")

        for _signal, props in recording_emitter.records:
            for value in props.values():
                assert "Do the thing." not in str(value)

    # ``skill_key`` and ``expected_hash`` are copied off the wire, so a hostile
    # store can smuggle the body through either one and publish it in a signal
    # that is otherwise body-free. The sweep above cannot see that: it serves a
    # well-formed 64-character digest under a valid key, so neither replacement
    # branch ever runs, and it passes even against an implementation that copies
    # both fields verbatim. These two cases are what make the rule observable.
    # Both assert the body's *absence* rather than the placeholder's exact
    # spelling, which is not part of the contract.

    async def test_body_smuggled_through_content_hash_is_redacted(
        self, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        body = "UNIQUE-SECRET-BODY-VIA-HASH"
        skills_module._set_store(
            InMemorySkillStore(
                {"a": {"key": "a", "version": 1, "content": body, "contentHash": body}}
            )
        )

        assert await get_skill("a") is None

        signals = recording_emitter.signals(INTEGRITY_SIGNAL)
        assert len(signals) == 1
        for value in signals[0].values():
            assert body not in str(value)

    async def test_body_smuggled_through_the_key_is_redacted(
        self, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        # Uppercase and a path separator, so this is not a valid skill key and
        # the invalid-key branch is the one that has to redact it.
        body = "UNIQUE-SECRET-BODY-VIA-KEY/../x"
        skills_module._set_store(
            InMemorySkillStore(
                {body: {"key": body, "version": 1, "content": "x", "contentHash": "y"}}
            )
        )

        assert await get_skill(body) is None

        signals = recording_emitter.signals(INTEGRITY_SIGNAL)
        assert len(signals) == 1
        for value in signals[0].values():
            assert body not in str(value)

    async def test_no_ld_track_calls_from_accessors(
        self, make_raw_skill: Any, mock_ld_client: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a"))
        store.put(make_raw_skill(key="bad", contentHash="0" * 64))
        await init_client(options={"skillStore": store}, client=mock_ld_client)

        await get_skill("a")
        await get_skill("bad")
        await get_skills(["a"])
        await all_skills()

        mock_ld_client.track.assert_not_called()

    async def test_throwing_emitter_never_breaks_the_operation(
        self, store: InMemorySkillStore, make_raw_skill: Any, throwing_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(throwing_emitter)
        store.put(make_raw_skill(key="bad", contentHash="0" * 64))
        store.put(make_raw_skill(key="good"))

        assert await get_skill("bad") is None
        good = await get_skill("good")
        assert good is not None
        assert good.key == "good"

    async def test_accessors_record_no_signal_outside_the_approved_set(
        self, store: InMemorySkillStore, make_raw_skill: Any, recording_emitter: Any
    ) -> None:
        """The three names are an allowlist, not a floor.

        Asserted over the recorded strings, so nothing here mandates a
        particular module-level constant. The write-side half of this sweep is
        ``test_write_skills_records_no_signal_outside_the_approved_set`` in
        test_skills_fs.py, where all four reconcile actions can be staged.

        Guards the most likely regression: an implementation that also emits
        ``AgentControl Skill Content Retrieved`` from ``get_skill``, or
        ``AgentControl Skill SDK Reference Returned`` from ``skill_refs``,
        passes every other test in this class.
        """
        skills_module._set_emitter_for_testing(recording_emitter)
        store.put(make_raw_skill(key="good"))
        store.put(make_raw_skill(key="tampered", contentHash="0" * 64))

        assert await get_skill("good") is not None
        assert await get_skill("tampered") is None
        await get_skills(["good", "tampered"])
        await all_skills()
        skill_refs({"skills": [{"key": "good", "version": 1}]})

        recorded = {signal for signal, _props in recording_emitter.records}
        assert recorded <= APPROVED_SIGNALS, (
            f"unapproved signal(s): {sorted(recorded - APPROVED_SIGNALS)}"
        )
        assert not recorded & REMOVED_SIGNALS
        # Positive control: a subset assertion is satisfied vacuously by an
        # implementation that records nothing at all.
        assert INTEGRITY_SIGNAL in recorded
