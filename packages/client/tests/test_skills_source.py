"""
Tests for ``SkillSource`` and ``config_skills``, and for ``write_skills`` and
``watch_skills`` given a source.

A fixed reference list never re-reads its config, so a watcher fed one cannot
see a config move ``pdf-extraction`` from v2 to v3: v2 stays on disk reported
``error`` and v3 is never written. A source is read on every reconcile, and the
watcher re-reconciles when one of its ``config_keys`` changes in the base client.

The base client is a hand-built fake exposing ``variation``, ``track``,
``flag_tracker`` and ``is_offline``, never a ``MagicMock``: an unspecced mock
passes the watcher's ``flag_tracker`` probe and then never fires, which is the
silent failure the probe exists to catch. One test at the end drives the real
base SDK through ``TestDataV2``, because a fake encodes its author's belief about
the change payload.

Every test writes only inside pytest's ``tmp_path``. The watcher runs a real
worker thread, so tests wait on its reconcile counter rather than on sleeps,
except to assert that something does *not* happen.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import (
    MANIFEST_FILENAME,
    InMemorySkillStore,
    ReconcileReport,
    Skill,
    SkillReference,
    SkillSource,
    config_skills,
    set_skill_store,
    skills_fs,
    skills_source,
    skills_watch,
    watch_skills,
    write_skills,
)
from launchdarkly_ai_server import skills as skills_module
from launchdarkly_ai_server.sdk_usage import SDK_USAGE_EVENT, reset_sdk_usage
from launchdarkly_ai_server.skills_core import SKILL_OBJECT_KIND

CONTEXT = {"kind": "user", "key": "user-1"}
CONFIG_KEY = "support-agent"


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    lifecycle_module._reset_for_testing()
    reset_sdk_usage()
    yield
    lifecycle_module._reset_for_testing()
    reset_sdk_usage()


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeFlagChange:
    """The shape of ``ldclient.interfaces.FlagChange``: a ``key`` property."""

    def __init__(self, key: str) -> None:
        self.key = key


class FakeFlagTracker:
    def __init__(self) -> None:
        self.listeners: list[Callable[[Any], None]] = []
        self.in_listener = False
        self.firing_thread: int | None = None

    def add_listener(self, fn: Callable[[Any], None]) -> None:
        self.listeners.append(fn)

    def remove_listener(self, fn: Callable[[Any], None]) -> None:
        self.listeners.remove(fn)

    def fire(self, key: str) -> None:
        """Calls every listener synchronously, as the base SDK's delivery thread does."""
        self.firing_thread = threading.get_ident()
        self.in_listener = True
        try:
            for fn in list(self.listeners):
                fn(FakeFlagChange(key))
        finally:
            self.in_listener = False


class FakeLDClient:
    """A base client with the surface the skills code touches, and no more."""

    def __init__(self, value: Any = None, *, offline: bool = False) -> None:
        self.value = value
        self.raises: Exception | None = None
        self.offline = offline
        self.flag_tracker = FakeFlagTracker()
        self.variation_calls: list[tuple[str, Any, Any]] = []
        self.tracked: list[tuple[str, Any]] = []

    def variation(self, key: str, context: Any, default: Any) -> Any:
        self.variation_calls.append((key, context, default))
        if self.raises is not None:
            raise self.raises
        return self.value

    def track(
        self, event: str, context: Any, data: Any = None, metric: Any = None
    ) -> None:
        self.tracked.append((event, data))

    def is_offline(self) -> bool:
        return self.offline

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def usage_helpers(self) -> list[str]:
        return [
            data["helper"] for event, data in self.tracked if event == SDK_USAGE_EVENT
        ]


class VariationOnlyClient:
    """A client brought to ``init_client`` that implements only ``variation``."""

    def variation(self, key: str, context: Any, default: Any) -> Any:
        return None

    def track(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def ai_config(
    *refs: tuple[str, int], enabled: bool = True, **fields: Any
) -> dict[str, Any]:
    """A variation value as the base SDK returns it."""
    value: dict[str, Any] = {
        "_ldMeta": {"enabled": enabled, "variationKey": "v1", "version": 1},
        "model": {"name": "gpt-test"},
        "provider": {"name": "openai"},
        "instructions": "Be helpful.",
    }
    if refs:
        value["skills"] = [{"key": key, "version": version} for key, version in refs]
    value.update(fields)
    return value


class CountingSource:
    """A hand-built source that records every read and where it ran."""

    def __init__(
        self,
        answer: Any,
        config_keys: tuple[str, ...] = (CONFIG_KEY,),
        tracker: FakeFlagTracker | None = None,
    ) -> None:
        self.answer = answer
        self.reads = 0
        self.threads: list[int] = []
        self.in_listener: list[bool] = []
        self._tracker = tracker
        self.source = SkillSource(self.refs, config_keys=config_keys)

    def refs(self) -> Any:
        self.reads += 1
        self.threads.append(threading.get_ident())
        self.in_listener.append(bool(self._tracker and self._tracker.in_listener))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer() if callable(self.answer) else self.answer


def skill_file(root: Path, key: str) -> Path:
    return root / key / "SKILL.md"


def manifest_versions(root: Path) -> dict[str, int]:
    entries = json.loads((root / MANIFEST_FILENAME).read_text())["entries"]
    return {entry["key"]: entry["version"] for entry in entries.values()}


def summary(report: ReconcileReport) -> list[tuple[str, str, int | None]]:
    return sorted((a.key, a.action, a.version) for a in report.actions)


def install(client: Any) -> Any:
    lifecycle_module._set_client_for_testing(client)
    return client


# ---------------------------------------------------------------------------
# SkillSource construction
# ---------------------------------------------------------------------------


class TestSkillSource:
    def test_is_immutable(self) -> None:
        source = SkillSource(list, config_keys=["a"])
        with pytest.raises(dataclasses.FrozenInstanceError):
            source.config_keys = ("b",)  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            source.refs = list  # type: ignore[assignment]

    def test_config_keys_is_held_as_a_tuple_copy(self) -> None:
        keys = ["support-agent", "triage"]
        source = SkillSource(list, config_keys=keys)
        keys.append("later")
        assert source.config_keys == ("support-agent", "triage")
        assert isinstance(source.config_keys, tuple)

    def test_config_keys_defaults_to_empty(self) -> None:
        assert SkillSource(list).config_keys == ()

    def test_a_bare_string_is_a_type_error(self) -> None:
        """Iterated, it would be thirteen one-character keys that never change."""
        with pytest.raises(TypeError, match="single string"):
            SkillSource(list, config_keys="support-agent")

    @pytest.mark.parametrize("refs", [None, "support-agent", [SkillReference("a", 1)]])
    def test_a_non_callable_refs_is_a_type_error(self, refs: Any) -> None:
        with pytest.raises(TypeError, match="callable"):
            SkillSource(refs)

    @pytest.mark.parametrize("keys", [[1], [None], ["a", b"b"]])
    def test_a_non_string_key_is_a_type_error(self, keys: Any) -> None:
        with pytest.raises(TypeError, match="only strings"):
            SkillSource(list, config_keys=keys)

    @pytest.mark.parametrize("keys", [{"a"}, iter(["a"]), 5])
    def test_a_non_sequence_is_a_type_error(self, keys: Any) -> None:
        with pytest.raises(TypeError, match="sequence"):
            SkillSource(list, config_keys=keys)

    def test_an_empty_key_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="empty"):
            SkillSource(list, config_keys=["a", ""])


# ---------------------------------------------------------------------------
# config_skills — the classification table
# ---------------------------------------------------------------------------


class TestConfigSkillsClassification:
    """Only a disabled variation and a parsed config answer; everything else is a
    failed read, raised rather than returned as ``[]``, which would prune."""

    async def test_no_initialized_client_is_a_failed_read(self) -> None:
        source = config_skills(CONFIG_KEY, CONTEXT)
        with pytest.raises(RuntimeError, match=f"'{CONFIG_KEY}'.*no AI client"):
            await source.refs()  # type: ignore[misc]

    async def test_it_never_initializes_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unlike ``inspect_config``, which initializes lazily from ``LD_SDK_KEY``."""
        monkeypatch.setenv("LD_SDK_KEY", "sdk-not-a-real-key")
        source = config_skills(CONFIG_KEY, CONTEXT)
        with pytest.raises(RuntimeError, match="no AI client"):
            await source.refs()  # type: ignore[misc]
        assert lifecycle_module._client is None

    async def test_a_raising_variation_is_a_failed_read(self) -> None:
        client = install(FakeLDClient())
        client.raises = ConnectionError("store lock poisoned")
        source = config_skills(CONFIG_KEY, CONTEXT)
        with pytest.raises(RuntimeError, match="ConnectionError: store lock poisoned"):
            await source.refs()  # type: ignore[misc]

    async def test_the_default_is_a_failed_read_and_the_default_is_none(self) -> None:
        """Flag not found, client not ready and evaluation error all arrive as
        the default, so it must be distinguishable from every real answer."""
        client = install(FakeLDClient(None))
        source = config_skills(CONFIG_KEY, CONTEXT)
        with pytest.raises(RuntimeError, match="default"):
            await source.refs()  # type: ignore[misc]
        assert client.variation_calls == [(CONFIG_KEY, CONTEXT, None)]

    @pytest.mark.parametrize("value", ["a string", 3, True, [ai_config()]])
    async def test_a_value_that_is_not_an_object_is_a_failed_read(
        self, value: Any
    ) -> None:
        install(FakeLDClient(value))
        with pytest.raises(RuntimeError, match="not an object"):
            await config_skills(CONFIG_KEY, CONTEXT).refs()  # type: ignore[misc]

    @pytest.mark.parametrize(
        "meta",
        [
            "missing",
            None,
            "enabled",
            {},
            {"enabled": "true"},
            {"enabled": 1},
            {"enabled": None},
        ],
    )
    async def test_an_object_without_a_boolean_enabled_is_a_failed_read(
        self, meta: Any
    ) -> None:
        value = ai_config(("pdf-extraction", 2))
        if meta == "missing":
            del value["_ldMeta"]
        else:
            value["_ldMeta"] = meta
        install(FakeLDClient(value))
        with pytest.raises(RuntimeError, match="_ldMeta"):
            await config_skills(CONFIG_KEY, CONTEXT).refs()  # type: ignore[misc]

    async def test_an_enabled_config_that_fails_to_parse_is_a_failed_read(
        self,
    ) -> None:
        value = ai_config(("pdf-extraction", 2))
        del value["model"]
        install(FakeLDClient(value))
        with pytest.raises(RuntimeError, match="failed to parse"):
            await config_skills(CONFIG_KEY, CONTEXT).refs()  # type: ignore[misc]

    @pytest.mark.parametrize(
        "skills",
        [
            None,
            "pdf-extraction",
            {"key": "pdf-extraction", "version": 2},
            [{"key": "pdf-extraction", "version": 2}, {"key": "Bad Key", "version": 1}],
            [{"key": "pdf-extraction", "version": 0}],
            [{"key": "pdf-extraction"}],
        ],
    )
    async def test_a_malformed_skills_field_is_a_failed_read(self, skills: Any) -> None:
        """Never read as "no skills"."""
        install(FakeLDClient(ai_config(skills=skills)))
        with pytest.raises(RuntimeError, match="skills"):
            await config_skills(CONFIG_KEY, CONTEXT).refs()  # type: ignore[misc]

    async def test_a_disabled_variation_answers_empty(self) -> None:
        """An answer from LaunchDarkly: its skills leave disk. Whatever else the
        disabled variation carries is not read."""
        install(FakeLDClient(ai_config(("pdf-extraction", 2), enabled=False, skills=7)))
        assert await config_skills(CONFIG_KEY, CONTEXT).refs() == []  # type: ignore[misc]

    async def test_a_parsed_config_answers_its_references(self) -> None:
        install(FakeLDClient(ai_config(("pdf-extraction", 2), ("triage", 5))))
        assert await config_skills(CONFIG_KEY, CONTEXT).refs() == [  # type: ignore[misc]
            SkillReference("pdf-extraction", 2),
            SkillReference("triage", 5),
        ]

    async def test_a_parsed_config_without_skills_answers_empty(self) -> None:
        install(FakeLDClient(ai_config()))
        assert await config_skills(CONFIG_KEY, CONTEXT).refs() == []  # type: ignore[misc]

    async def test_an_async_variation_is_awaited(self) -> None:
        class AsyncClient(FakeLDClient):
            async def variation(self, key: str, context: Any, default: Any) -> Any:  # type: ignore[override]
                self.variation_calls.append((key, context, default))
                return self.value

        install(AsyncClient(ai_config(("pdf-extraction", 2))))
        assert await config_skills(CONFIG_KEY, CONTEXT).refs() == [  # type: ignore[misc]
            SkillReference("pdf-extraction", 2)
        ]

    async def test_the_source_follows_exactly_its_one_key(self) -> None:
        source = config_skills(CONFIG_KEY, CONTEXT)
        assert source.config_keys == (CONFIG_KEY,)

    async def test_each_read_evaluates_afresh(self) -> None:
        client = install(FakeLDClient(ai_config(("pdf-extraction", 2))))
        source = config_skills(CONFIG_KEY, CONTEXT)
        first = await source.refs()  # type: ignore[misc]
        client.value = ai_config(("pdf-extraction", 3))
        second = await source.refs()  # type: ignore[misc]
        assert first == [SkillReference("pdf-extraction", 2)]
        assert second == [SkillReference("pdf-extraction", 3)]
        assert len(client.variation_calls) == 2


class TestConfigSkillsTelemetry:
    async def test_usage_is_reported_on_the_call_not_on_each_read(
        self, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        client = install(FakeLDClient(ai_config(("pdf-extraction", 2))))
        source = config_skills(CONFIG_KEY, CONTEXT)
        assert client.usage_helpers() == ["client.configSkills"]

        for _ in range(3):
            await source.refs()  # type: ignore[misc]

        # Three evaluations, and nothing tracked but the one usage event: no
        # $ld:ai:* telemetry, no client.skillRefs, no skills signal.
        assert len(client.variation_calls) == 3
        assert client.tracked == [
            (SDK_USAGE_EVENT, client.tracked[0][1]),
        ]
        assert client.usage_helpers() == ["client.configSkills"]
        assert recording_emitter.records == []

    async def test_usage_is_reported_before_the_arguments_are_checked(self) -> None:
        client = install(FakeLDClient())
        with pytest.raises(TypeError, match="empty"):
            config_skills("", CONTEXT)
        assert client.usage_helpers() == ["client.configSkills"]

    async def test_write_and_watch_given_a_config_source_report_no_skill_refs(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="pdf-extraction", version=2))
        set_skill_store(store)
        client = install(FakeLDClient(ai_config(("pdf-extraction", 2))))
        source = config_skills(CONFIG_KEY, CONTEXT)

        await write_skills(source, tmp_path / "a")
        _report, watcher = await watch_skills(source, tmp_path / "b", debounce=0.01)
        try:
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles >= 1)
        finally:
            watcher.close()

        assert len(client.variation_calls) == 3
        # Each read, re-reconciles included, adds nothing: only the three calls report.
        assert sorted(client.usage_helpers()) == [
            "client.configSkills",
            "client.watchSkills",
            "client.writeSkills",
        ]


# ---------------------------------------------------------------------------
# write_skills given a source
# ---------------------------------------------------------------------------


MALFORMED_ANSWERS = [
    pytest.param(None, id="None"),
    pytest.param("*", id="star"),
    pytest.param("pdf-extraction", id="bare-key"),
    pytest.param(b"pdf-extraction", id="bytes"),
    pytest.param(3, id="int"),
    pytest.param({"pdf-extraction": 2}, id="dict"),
    pytest.param({SkillReference("a", 1)}, id="set"),
    pytest.param([1], id="int-entry"),
    pytest.param([SkillReference("a", 1), None], id="None-entry"),
    pytest.param([{"key": "a", "version": 1}], id="dict-entry"),
    pytest.param(
        [Skill(key="a", version=1, content=b"x", content_hash="0" * 64)],
        id="Skill-entry",
    ),
    pytest.param([SkillReference("Bad Key", 1)], id="invalid-reference-key"),
    pytest.param([SkillReference("a", 0)], id="invalid-version"),
    pytest.param([SkillReference("a", True)], id="bool-version"),  # type: ignore[arg-type]
    pytest.param(["Bad Key"], id="invalid-key-string"),
]


class TestWriteSkillsFromASource:
    @pytest.fixture
    def two_on_disk(self, tmp_path: Path, make_raw_skill: Any) -> Any:
        """A store holding a v1 and b v2, and a root they were written to."""

        async def make() -> tuple[InMemorySkillStore, Path]:
            store = InMemorySkillStore()
            store.put(make_raw_skill(key="a", version=1, content="a body"))
            store.put(make_raw_skill(key="b", version=2, content="b body"))
            set_skill_store(store)
            root = tmp_path / "skills"
            report = await write_skills(["a", "b"], root)
            assert report.ok
            return store, root

        return make

    async def test_a_source_reconciles_exactly_as_its_answer_would(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        for key, version in [("a", 1), ("b", 2), ("c", 1)]:
            store.put(make_raw_skill(key=key, version=version, content=f"{key} body"))
        set_skill_store(store)
        answer = [SkillReference("a", 1), "b", SkillReference("missing", 1)]

        reports = []
        for root, skills in [
            (tmp_path / "list", answer),
            (tmp_path / "source", SkillSource(lambda: answer)),
        ]:
            await write_skills(["a", "c"], root)
            reports.append(await write_skills(skills, root))

        assert summary(reports[0]) == summary(reports[1])
        # A real reconcile, not two empty ones.
        assert ("missing", "error", None) in summary(reports[1])
        assert skill_file(tmp_path / "source", "b").read_text() == "b body"

    async def test_sync_async_and_tuple_answers_are_all_accepted(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a", version=1, content="a body"))
        set_skill_store(store)

        async def async_refs() -> list[SkillReference]:
            await asyncio.sleep(0)
            return [SkillReference("a", 1)]

        for name, refs in [
            ("sync", lambda: [SkillReference("a", 1)]),
            ("async", async_refs),
            ("tuple", lambda: ("a",)),
        ]:
            report = await write_skills(SkillSource(refs), tmp_path / name)
            assert report.ok, name
            assert skill_file(tmp_path / name, "a").read_text() == "a body"

    async def test_refs_is_read_exactly_once_even_when_it_raises(
        self, tmp_path: Path
    ) -> None:
        set_skill_store(InMemorySkillStore())
        source = CountingSource(RuntimeError("config service down"))
        report = await write_skills(source.source, tmp_path / "skills")
        assert source.reads == 1
        assert [(a.key, a.action) for a in report.actions] == [("", "error")]
        assert "config service down" in (report.actions[0].error or "")
        assert report.ok is False

    @pytest.mark.parametrize("answer", MALFORMED_ANSWERS)
    async def test_a_malformed_answer_is_an_unavailable_retrieval(
        self, two_on_disk: Any, answer: Any
    ) -> None:
        _store, root = await two_on_disk()
        report = await write_skills(SkillSource(lambda: answer), root)

        assert [(a.key, a.action) for a in report.actions] == [("", "error")]
        assert "skill source" in (report.actions[0].error or "")
        assert report.ok is False
        assert skill_file(root, "a").read_text() == "a body"
        assert skill_file(root, "b").read_text() == "b body"
        assert manifest_versions(root) == {"a": 1, "b": 2}

    async def test_a_source_answering_none_after_two_writes_leaves_both(
        self, two_on_disk: Any
    ) -> None:
        """The case that matters most: ``None`` must never read as "no skills"."""
        _store, root = await two_on_disk()
        report = await write_skills(SkillSource(lambda: None), root)
        assert all(a.action != "removed" for a in report.actions)
        assert manifest_versions(root) == {"a": 1, "b": 2}

    @pytest.mark.parametrize("answer", MALFORMED_ANSWERS)
    async def test_raise_mode_names_the_skill_source(
        self, two_on_disk: Any, answer: Any
    ) -> None:
        _store, root = await two_on_disk()
        with pytest.raises(RuntimeError, match="skill source"):
            await write_skills(
                SkillSource(lambda: answer), root, on_unavailable="raise"
            )
        assert manifest_versions(root) == {"a": 1, "b": 2}

    async def test_a_raising_source_in_raise_mode_names_the_skill_source(
        self, two_on_disk: Any
    ) -> None:
        _store, root = await two_on_disk()
        source = CountingSource(KeyError("nope"))
        with pytest.raises(RuntimeError, match="skill source raised KeyError"):
            await write_skills(source.source, root, on_unavailable="raise")
        assert source.reads == 1

    async def test_one_invalid_entry_fails_the_read_rather_than_being_dropped(
        self, two_on_disk: Any, make_raw_skill: Any
    ) -> None:
        """Dropping it would shorten the requested set and prune b."""
        store, root = await two_on_disk()
        store.put(make_raw_skill(key="c", version=1, content="c body"))
        report = await write_skills(
            SkillSource(lambda: [SkillReference("c", 1), {"key": "b"}]), root
        )
        assert [(a.key, a.action) for a in report.actions] == [("", "error")]
        assert not skill_file(root, "c").exists()
        assert manifest_versions(root) == {"a": 1, "b": 2}

    async def test_a_zero_timeout_never_reads_the_source(
        self, two_on_disk: Any
    ) -> None:
        _store, root = await two_on_disk()
        source = CountingSource(["a"])
        report = await write_skills(source.source, root, timeout=0)
        assert source.reads == 0
        assert [(a.key, a.action) for a in report.actions] == [("", "error")]
        assert "timeout" in (report.actions[0].error or "")

    async def test_an_async_source_pending_at_the_deadline_is_cancelled(
        self, two_on_disk: Any
    ) -> None:
        _store, root = await two_on_disk()
        cancelled = asyncio.Event()

        async def hangs() -> list[str]:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return ["a"]

        started = time.monotonic()
        report = await write_skills(SkillSource(hangs), root, timeout=0.2)
        assert time.monotonic() - started < 5
        assert cancelled.is_set()
        assert [(a.key, a.action) for a in report.actions] == [("", "error")]
        assert "did not answer" in (report.actions[0].error or "")
        assert manifest_versions(root) == {"a": 1, "b": 2}

    async def test_a_bad_root_raises_before_the_source_is_read(
        self, tmp_path: Path
    ) -> None:
        set_skill_store(InMemorySkillStore())
        not_a_directory = tmp_path / "file"
        not_a_directory.write_text("")
        source = CountingSource(["a"])
        with pytest.raises(ValueError, match="not a directory"):
            await write_skills(source.source, not_a_directory)
        assert source.reads == 0


# ---------------------------------------------------------------------------
# watch_skills following config changes
# ---------------------------------------------------------------------------


class TestFixedFormsNeverSubscribe:
    @pytest.mark.parametrize("skills", ["*", ["a"]])
    async def test_a_fixed_list_and_star_register_no_config_listener(
        self, tmp_path: Path, skills: Any
    ) -> None:
        set_skill_store(InMemorySkillStore())
        client = install(FakeLDClient())
        _report, watcher = await watch_skills(skills, tmp_path / "s", debounce=0.01)
        watcher.close()
        assert client.flag_tracker.listeners == []

    @pytest.mark.parametrize("client", [None, VariationOnlyClient()])
    @pytest.mark.parametrize("skills", ["*", ["a"]])
    async def test_they_need_no_change_listener_api(
        self, tmp_path: Path, skills: Any, client: Any
    ) -> None:
        set_skill_store(InMemorySkillStore())
        if client is not None:
            install(client)
        _report, watcher = await watch_skills(skills, tmp_path / "s", debounce=0.01)
        watcher.close()


class TestNoChangeSignal:
    """A watcher that never hears about config changes looks exactly like one
    whose config never changed, so a missing signal raises."""

    class NoAddListener:
        def variation(self, *_args: Any) -> Any:
            return None

        def track(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        flag_tracker = object()

    @pytest.mark.parametrize(
        ("client", "cause"),
        [
            (None, "no AI client is initialized"),
            (VariationOnlyClient(), "no flag_tracker"),
            (NoAddListener(), "no flag_tracker"),
            (FakeLDClient(offline=True), "offline"),
        ],
        ids=["no-client", "variation-only", "tracker-without-add", "offline"],
    )
    async def test_it_raises_naming_both_remedies_and_detaches(
        self, tmp_path: Path, client: Any, cause: str
    ) -> None:
        store = InMemorySkillStore()
        set_skill_store(store)
        if client is not None:
            install(client)
        source = CountingSource(["a"])

        with pytest.raises(RuntimeError) as raised:
            await watch_skills(source.source, tmp_path / "s", debounce=0.01)

        message = str(raised.value)
        assert cause in message
        assert "LDClient" in message
        assert "config_keys=()" in message
        assert "watcher.notify()" in message
        assert store._listeners.get(SKILL_OBJECT_KIND, []) == []
        if isinstance(client, FakeLDClient):
            assert client.flag_tracker.listeners == []
        # Refused before the initial reconcile read anything.
        assert source.reads == 0

    async def test_an_is_offline_that_raises_is_treated_as_offline(
        self, tmp_path: Path
    ) -> None:
        class RaisingOffline(FakeLDClient):
            def is_offline(self) -> bool:
                raise RuntimeError("config not loaded")

        store = InMemorySkillStore()
        set_skill_store(store)
        client = install(RaisingOffline())
        with pytest.raises(RuntimeError, match=r"offline.*watcher\.notify"):
            await watch_skills(CountingSource(["a"]).source, tmp_path / "s")
        assert store._listeners.get(SKILL_OBJECT_KIND, []) == []
        assert client.flag_tracker.listeners == []

    async def test_empty_config_keys_probe_nothing(self, tmp_path: Path) -> None:
        set_skill_store(InMemorySkillStore())
        source = CountingSource([], config_keys=())
        _report, watcher = await watch_skills(source.source, tmp_path / "s")
        watcher.close()
        assert source.reads == 1

    async def test_notify_schedules_a_reconcile_through_the_debounce(
        self, tmp_path: Path
    ) -> None:
        """The empty-``config_keys`` form depends on it."""
        set_skill_store(InMemorySkillStore())
        source = CountingSource([], config_keys=())
        _report, watcher = await watch_skills(
            source.source, tmp_path / "s", debounce=0.3
        )
        try:
            watcher.notify()
            time.sleep(0.1)
            assert watcher.reconciles == 0  # still inside the window
            assert wait_until(lambda: watcher.reconciles == 1)
            assert source.reads == 2
        finally:
            watcher.close()


class TestFollowingConfigChanges:
    async def _watch(
        self, tmp_path: Path, source: SkillSource, debounce: float = 0.05, **kwargs: Any
    ) -> tuple[ReconcileReport, Any]:
        return await watch_skills(source, tmp_path / "s", debounce=debounce, **kwargs)

    async def test_every_trigger_reads_the_source_once(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        """A store change re-reads the source just as a config change does: the
        two arrive on independent connections, so a store change is often when
        the source's answer becomes resolvable."""
        store = InMemorySkillStore()
        set_skill_store(store)
        client = install(FakeLDClient())
        source = CountingSource(["a"])
        _report, watcher = await self._watch(tmp_path, source.source)
        try:
            assert source.reads == 1

            store.put(make_raw_skill(key="a", version=1))
            assert wait_until(lambda: watcher.reconciles == 1)
            assert source.reads == 2

            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 2)
            assert source.reads == 3

            time.sleep(0.3)
            assert (watcher.reconciles, source.reads) == (2, 3)
        finally:
            watcher.close()

    async def test_an_unrelated_key_does_not_wake_the_watcher(
        self, tmp_path: Path
    ) -> None:
        set_skill_store(InMemorySkillStore())
        client = install(FakeLDClient())
        source = CountingSource(["a"])
        # A long window keeps the wake flag observable after a followed change.
        _report, watcher = await self._watch(tmp_path, source.source, debounce=5)
        try:
            client.flag_tracker.fire("some-other-flag")
            assert not watcher._wake.is_set()
            # The positive control, before the worker can clear the flag: the
            # listener is the same bound method the tracker holds.
            watcher._stop.set()
            client.flag_tracker.listeners[0](FakeFlagChange(CONFIG_KEY))
            assert watcher._wake.is_set()
        finally:
            watcher.close()
        assert watcher.reconciles == 0

    async def test_a_followed_key_wakes_it_and_an_unrelated_one_never_does(
        self, tmp_path: Path
    ) -> None:
        set_skill_store(InMemorySkillStore())
        client = install(FakeLDClient())
        source = CountingSource(["a"])
        _report, watcher = await self._watch(tmp_path, source.source)
        try:
            for _ in range(3):
                client.flag_tracker.fire("some-other-flag")
                time.sleep(0.1)
            assert watcher.reconciles == 0
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 1)
        finally:
            watcher.close()

    async def test_the_listener_only_signals(self, tmp_path: Path) -> None:
        """The base SDK calls it on its delivery thread holding the store lock
        that every evaluation takes, so the source is read on the worker."""
        set_skill_store(InMemorySkillStore())
        client = install(FakeLDClient())
        source = CountingSource(["a"], tracker=client.flag_tracker)
        _report, watcher = await self._watch(tmp_path, source.source, debounce=0.2)
        try:
            client.flag_tracker.fire(CONFIG_KEY)
            assert source.reads == 1  # the listener returned without reading
            assert wait_until(lambda: watcher.reconciles == 1)
            assert source.reads == 2
            assert source.in_listener == [False, False]
            assert source.threads[1] != client.flag_tracker.firing_thread
        finally:
            watcher.close()

    async def test_it_registers_before_the_initial_reconcile(
        self, tmp_path: Path
    ) -> None:
        """A config change landing during the initial read is followed by one
        more reconcile, not lost."""
        set_skill_store(InMemorySkillStore())
        client = install(FakeLDClient())

        def changes_during_the_first_read() -> list[str]:
            if source.reads == 1:
                client.flag_tracker.fire(CONFIG_KEY)
            return []

        source = CountingSource(changes_during_the_first_read)
        _report, watcher = await self._watch(tmp_path, source.source)
        try:
            assert wait_until(lambda: watcher.reconciles == 1)
            assert source.reads == 2
        finally:
            watcher.close()

    async def test_the_debounce_spans_both_trigger_kinds(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        """One window merges store and config notifications alike.

        Spread out so only the debounce can merge them, as in the store-only
        coalescing test, and with the same result: each change restarts the
        window, so the whole mixed burst is one reconcile and one read.
        """
        store = InMemorySkillStore()
        set_skill_store(store)
        client = install(FakeLDClient())
        source = CountingSource(["a"])
        _report, watcher = await self._watch(tmp_path, source.source, debounce=0.6)
        try:
            for i in range(6):
                if i % 2:
                    client.flag_tracker.fire(CONFIG_KEY)
                else:
                    store.put(make_raw_skill(key=f"skill-{i}"))
                time.sleep(0.1)
            assert wait_until(lambda: watcher.reconciles >= 1)
            # Settle well past another window, so a second would have landed.
            time.sleep(1.5)
            assert watcher.reconciles == 1
            assert source.reads == 1 + 1
        finally:
            watcher.close()

    async def test_close_detaches_the_config_listener(self, tmp_path: Path) -> None:
        set_skill_store(InMemorySkillStore())
        client = install(FakeLDClient())
        source = CountingSource(["a"])
        _report, watcher = await self._watch(tmp_path, source.source)
        assert len(client.flag_tracker.listeners) == 1

        watcher.close()
        watcher.close()

        assert client.flag_tracker.listeners == []
        client.flag_tracker.fire(CONFIG_KEY)
        time.sleep(0.3)
        assert (watcher.reconciles, source.reads) == (0, 1)

    async def test_a_tracker_without_remove_listener_still_closes(
        self, tmp_path: Path
    ) -> None:
        class AddOnlyTracker:
            def __init__(self) -> None:
                self.listeners: list[Any] = []

            def add_listener(self, fn: Any) -> None:
                self.listeners.append(fn)

        client = FakeLDClient()
        client.flag_tracker = AddOnlyTracker()  # type: ignore[assignment]
        install(client)
        set_skill_store(InMemorySkillStore())
        source = CountingSource(["a"])
        _report, watcher = await self._watch(tmp_path, source.source)
        watcher.close()
        for fn in client.flag_tracker.listeners:
            fn(FakeFlagChange(CONFIG_KEY))
        time.sleep(0.3)
        assert source.reads == 1


class TestVersionBump:
    """The store holds pdf-extraction v2, the config pins v2, and the initial
    reconcile writes it. Either arrival order converges on v3, replaced in
    place: no reconcile reports ``removed``."""

    async def _start(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> tuple[InMemorySkillStore, FakeLDClient, Path, Any, list[ReconcileReport]]:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="pdf-extraction", version=2, content="v2 body"))
        set_skill_store(store)
        client = install(FakeLDClient(ai_config(("pdf-extraction", 2))))
        root = tmp_path / "s"
        reports: list[ReconcileReport] = []
        report, watcher = await watch_skills(
            config_skills(CONFIG_KEY, CONTEXT),
            root,
            debounce=0.05,
            on_reconcile=reports.append,
        )
        reports.insert(0, report)
        assert summary(report) == [("pdf-extraction", "written", 2)]
        return store, client, root, watcher, reports

    @staticmethod
    def _drop_v2(store: InMemorySkillStore) -> None:
        del store._versions["pdf-extraction"][2]

    @staticmethod
    def _end_state(root: Path) -> tuple[str, dict[str, int]]:
        return skill_file(root, "pdf-extraction").read_text(), manifest_versions(root)

    async def test_config_first(self, tmp_path: Path, make_raw_skill: Any) -> None:
        store, client, root, watcher, reports = await self._start(
            tmp_path, make_raw_skill
        )
        try:
            client.value = ai_config(("pdf-extraction", 3))
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 1)
            assert [(a.key, a.action) for a in reports[1].actions] == [
                ("pdf-extraction", "error")
            ]
            assert self._end_state(root) == ("v2 body", {"pdf-extraction": 2})

            store.put(
                make_raw_skill(key="pdf-extraction", version=3, content="v3 body")
            )
            assert wait_until(lambda: watcher.reconciles == 2)
            assert summary(reports[2]) == [("pdf-extraction", "updated", 3)]
            assert reports[2].ok
            config_first = self._end_state(root)
        finally:
            watcher.close()

        assert config_first == ("v3 body", {"pdf-extraction": 3})
        assert all(a.action != "removed" for r in reports for a in r.actions)

    async def test_store_first(self, tmp_path: Path, make_raw_skill: Any) -> None:
        store, client, root, watcher, reports = await self._start(
            tmp_path, make_raw_skill
        )
        try:
            self._drop_v2(store)
            store.put(
                make_raw_skill(key="pdf-extraction", version=3, content="v3 body")
            )
            assert wait_until(lambda: watcher.reconciles == 1)
            assert [(a.key, a.action) for a in reports[1].actions] == [
                ("pdf-extraction", "error")
            ]
            assert self._end_state(root) == ("v2 body", {"pdf-extraction": 2})

            client.value = ai_config(("pdf-extraction", 3))
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 2)
            assert summary(reports[2]) == [("pdf-extraction", "updated", 3)]
            store_first = self._end_state(root)
        finally:
            watcher.close()

        # Identical to the config-first end state.
        assert store_first == ("v3 body", {"pdf-extraction": 3})
        assert all(a.action != "removed" for r in reports for a in r.actions)


class PartlyUnreadableStore(InMemorySkillStore):
    """A store whose reads raise for the keys in ``unreadable``: an unavailable
    retrieval for those references, not an ``absent`` one."""

    def __init__(self) -> None:
        super().__init__()
        self.unreadable: set[str] = set()

    def get_object(
        self, kind: str, key: str, version: int | None = None
    ) -> dict[str, Any] | None:
        if key in self.unreadable:
            raise ConnectionError(f"store cannot read {key}")
        return super().get_object(kind, key, version)


class TestDroppingAReference:
    """A dropped reference is pruned on the next reconcile whose retrieval is
    available. An ``absent`` reference beside it is a per-key outcome and does
    not hold the prune; only an unavailable retrieval does."""

    async def _watch(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> tuple[PartlyUnreadableStore, FakeLDClient, Path, Any, list[ReconcileReport]]:
        store = PartlyUnreadableStore()
        store.put(make_raw_skill(key="pdf-extraction", version=2))
        store.put(make_raw_skill(key="triage", version=1, content="triage body"))
        set_skill_store(store)
        client = install(FakeLDClient(ai_config(("pdf-extraction", 2), ("triage", 1))))
        reports: list[ReconcileReport] = []
        root = tmp_path / "s"
        _report, watcher = await watch_skills(
            config_skills(CONFIG_KEY, CONTEXT),
            root,
            debounce=0.05,
            on_reconcile=reports.append,
        )
        assert skill_file(root, "pdf-extraction").exists()
        assert skill_file(root, "triage").exists()
        return store, client, root, watcher, reports

    async def test_a_clean_drop_prunes_it(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        _store, client, root, watcher, reports = await self._watch(
            tmp_path, make_raw_skill
        )
        try:
            client.value = ai_config(("triage", 1))
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 1)
        finally:
            watcher.close()
        assert summary(reports[0]) == [
            ("pdf-extraction", "removed", 2),
            ("triage", "skipped_current", 1),
        ]
        assert not skill_file(root, "pdf-extraction").exists()

    async def test_an_absent_reference_beside_it_does_not_hold_the_prune(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        """The absent skill keeps its file under its own key, and the run is not
        incomplete, so the dropped skill still goes in the same run."""
        store, client, root, watcher, reports = await self._watch(
            tmp_path, make_raw_skill
        )
        try:
            del store._versions["triage"]
            client.value = ai_config(("triage", 1))
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 1)
        finally:
            watcher.close()
        assert [(a.key, a.action) for a in reports[0].actions] == [
            ("triage", "error"),
            ("pdf-extraction", "removed"),
        ]
        assert not skill_file(root, "pdf-extraction").exists()
        assert skill_file(root, "triage").read_text() == "triage body"
        assert manifest_versions(root) == {"triage": 1}

    async def test_an_unavailable_retrieval_holds_the_prune_until_it_is_available(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        store, client, root, watcher, reports = await self._watch(
            tmp_path, make_raw_skill
        )
        try:
            store.unreadable.add("next-skill")
            client.value = ai_config(("triage", 1), ("next-skill", 1))
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 1)
            assert all(a.action != "removed" for a in reports[0].actions)
            assert ("next-skill", "error", None) in summary(reports[0])
            assert skill_file(root, "pdf-extraction").exists()
            assert manifest_versions(root) == {"pdf-extraction": 2, "triage": 1}

            store.unreadable.clear()
            store.put(make_raw_skill(key="next-skill", version=1))
            assert wait_until(lambda: watcher.reconciles == 2)
        finally:
            watcher.close()
        assert summary(reports[1]) == [
            ("next-skill", "written", 1),
            ("pdf-extraction", "removed", 2),
            ("triage", "skipped_current", 1),
        ]
        assert not skill_file(root, "pdf-extraction").exists()


class TestASourceThatCannotAnswer:
    async def test_answer_raise_answer_leaves_the_disk_alone_through_the_failure(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a", version=1, content="a body"))
        store.put(make_raw_skill(key="b", version=1, content="b body"))
        set_skill_store(store)
        client = install(FakeLDClient())
        source = CountingSource(["a"])
        reports: list[ReconcileReport] = []
        root = tmp_path / "s"
        _report, watcher = await watch_skills(
            source.source, root, debounce=0.05, on_reconcile=reports.append
        )
        try:
            source.answer = RuntimeError("config unavailable")
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 1)
            # A 'keep' failure reaches on_reconcile, so a caller watching ok sees it.
            assert reports[0].ok is False
            assert [(a.key, a.action) for a in reports[0].actions] == [("", "error")]
            assert skill_file(root, "a").read_text() == "a body"
            assert manifest_versions(root) == {"a": 1}

            # The next trigger is the retry.
            source.answer = ["b"]
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: watcher.reconciles == 2)
        finally:
            watcher.close()
        assert reports[1].ok
        assert summary(reports[1]) == [("a", "removed", 1), ("b", "written", 1)]

    async def test_raise_on_the_initial_reconcile_detaches_every_listener(
        self, tmp_path: Path
    ) -> None:
        store = InMemorySkillStore()
        set_skill_store(store)
        client = install(FakeLDClient())
        source = CountingSource(None)
        with pytest.raises(RuntimeError, match="skill source"):
            await watch_skills(source.source, tmp_path / "s", on_unavailable="raise")
        assert store._listeners.get(SKILL_OBJECT_KIND, []) == []
        assert client.flag_tracker.listeners == []

    async def test_keep_on_the_initial_reconcile_returns_the_run_level_error(
        self, tmp_path: Path
    ) -> None:
        set_skill_store(InMemorySkillStore())
        install(FakeLDClient())
        report, watcher = await watch_skills(
            CountingSource(None).source, tmp_path / "s"
        )
        watcher.close()
        assert [(a.key, a.action) for a in report.actions] == [("", "error")]

    async def test_raise_on_a_later_reconcile_is_logged_and_the_watch_survives(
        self, tmp_path: Path, make_raw_skill: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = InMemorySkillStore()
        store.put(make_raw_skill(key="a", version=1, content="a body"))
        set_skill_store(store)
        client = install(FakeLDClient())
        source = CountingSource(["a"])
        reports: list[ReconcileReport] = []
        _report, watcher = await watch_skills(
            source.source,
            tmp_path / "s",
            on_unavailable="raise",
            debounce=0.05,
            on_reconcile=reports.append,
        )
        try:
            with caplog.at_level("ERROR"):
                source.answer = RuntimeError("down")
                client.flag_tracker.fire(CONFIG_KEY)
                assert wait_until(lambda: source.reads == 2)
                assert wait_until(lambda: "re-reconcile raised" in caplog.text)
            source.answer = ["a"]
            client.flag_tracker.fire(CONFIG_KEY)
            assert wait_until(lambda: len(reports) == 1)
            assert reports[0].ok
        finally:
            watcher.close()


# ---------------------------------------------------------------------------
# Module layout
# ---------------------------------------------------------------------------


class TestModuleLayout:
    def test_only_the_source_and_the_watcher_import_lifecycle(self) -> None:
        import inspect

        for module in (skills_module, skills_fs):
            assert "lifecycle" not in inspect.getsource(module), module.__name__
        for module in (skills_source, skills_watch):
            assert "from .lifecycle import get_client\n" in inspect.getsource(module)

    def test_write_skills_recognizes_a_source_without_importing_config_skills(
        self,
    ) -> None:
        import inspect

        assert "skills_source" not in inspect.getsource(skills_fs)


# ---------------------------------------------------------------------------
# The real base SDK
# ---------------------------------------------------------------------------


class TestWithTheRealBaseSdk:
    """A fake encodes its author's belief about the change payload, so one
    version bump runs end to end through the base SDK's own test data source.

    It has to be ``TestDataV2`` registered as a synchronizer: v1 ``TestData``
    writes to the store without going through the update sink, so it fires no
    change events, and an initializer alone delivers no updates.
    """

    async def test_a_version_bump_reaches_disk(
        self, tmp_path: Path, make_raw_skill: Any
    ) -> None:
        import ldclient
        import ldclient.datasystem
        from ldclient import Config
        from ldclient.integrations.test_datav2 import TestDataV2

        def pin(version: int) -> Any:
            return (
                td.flag(CONFIG_KEY)
                .variations(ai_config(("pdf-extraction", version)))
                .variation_for_all(0)
            )

        td = TestDataV2.data_source()
        td.update(pin(2))
        datasystem = ldclient.datasystem.custom()
        datasystem.initializers([td.builder])
        datasystem.synchronizers(td.builder)
        ld_client = ldclient.LDClient(
            Config(
                "sdk-test-key",
                datasystem_config=datasystem.build(),
                send_events=False,
                diagnostic_opt_out=True,
            )
        )
        try:
            assert ld_client.is_initialized()
            install(ld_client)
            store = InMemorySkillStore()
            store.put(
                make_raw_skill(key="pdf-extraction", version=2, content="v2 body")
            )
            store.put(
                make_raw_skill(key="pdf-extraction", version=3, content="v3 body")
            )
            set_skill_store(store)
            reports: list[ReconcileReport] = []
            root = tmp_path / "s"

            report, watcher = await watch_skills(
                config_skills(CONFIG_KEY, CONTEXT),
                root,
                debounce=0.05,
                on_reconcile=reports.append,
            )
            try:
                assert summary(report) == [("pdf-extraction", "written", 2)]
                # Queued here; the listener fires later on the synchronizer thread.
                td.update(pin(3))
                assert wait_until(lambda: watcher.reconciles >= 1)
            finally:
                watcher.close()
        finally:
            ld_client.close()

        assert summary(reports[-1]) == [("pdf-extraction", "updated", 3)]
        assert skill_file(root, "pdf-extraction").read_text() == "v3 body"
        assert manifest_versions(root) == {"pdf-extraction": 3}
