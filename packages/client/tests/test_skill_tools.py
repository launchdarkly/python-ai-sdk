"""Skill tools (TESTING.md 3.21a): the ``tools`` a skill envelope pins, read onto
``Skill.tools`` and offered to handlers after the variation's own
``config["tools"]``.

Built against hand-made envelopes until LaunchDarkly serves them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
import launchdarkly_ai_server.skills as skills_module
import launchdarkly_ai_server.skills_fs as skills_fs_module
from launchdarkly_ai_server import (
    MANIFEST_FILENAME,
    InMemorySkillStore,
    ProviderHandler,
    Skill,
    all_skills,
    config,
    get_skill,
    get_skill_result,
    get_skills,
    graph,
    inspect_config,
    resolve_graph,
    set_skill_store,
    write_skills,
)
from launchdarkly_ai_server.skills_core import (
    INTEGRITY_FAILURE_EVENT,
    _config_with_skill_tools,
)

pytestmark = pytest.mark.usefixtures("reset_skill_state")

CONTEXT = {"kind": "user", "key": "u1"}
BODY = "---\nname: PDF\n---\nExtract.\n"
LOGGER = "launchdarkly_ai_server.skills_core"
SKILL = "pdf-extraction"


def _tool(name: str, description: str = "d") -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
        "customParameters": {"x": 1},
    }


def _envelope(
    key: str = SKILL, version: int = 3, tools: Any = None, *, omit: bool = False
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "key": key,
        "version": version,
        "content": BODY,
        "contentHash": hashlib.sha256(BODY.encode()).hexdigest(),
        "name": "PDF",
        "description": "Extracts PDFs.",
    }
    if not omit:
        raw["tools"] = tools if tools is not None else []
    return raw


def _store(*envelopes: dict[str, Any]) -> InMemorySkillStore:
    return InMemorySkillStore({f"{e['key']}:{e['version']}": e for e in envelopes})


def _cfg(tools: dict[str, Any] | None = None, skills: Any = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model": {"name": "m"},
        "provider": {"name": "TestProvider"},
        "instructions": "x",
    }
    if tools is not None:
        cfg["tools"] = tools
    if skills is not None:
        cfg["skills"] = skills
    return cfg


def _variation(**kwargs: Any) -> dict[str, Any]:
    return {
        **_cfg(**kwargs),
        "_ldMeta": {
            "enabled": True,
            "variationKey": "v1",
            "version": 1,
            "mode": "messages",
        },
    }


def _ref(key: str, version: int = 3) -> dict[str, Any]:
    return {"key": key, "version": version}


def _names(tools: Any) -> list[str]:
    return [t["name"] for t in tools]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == LOGGER
    ]


@pytest.fixture
def mock_ld_client() -> Any:
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    client.variation = AsyncMock(return_value=None)
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


def _capturing_handler(seen: list[tuple[Any, dict[str, Any]]]) -> ProviderHandler:
    async def fn(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        seen.append((config, dict(tool_handlers or {})))
        return {"output": "ok", "usage": {}}

    async def stream_fn(
        config: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        seen.append((config, dict(tool_handlers or {})))
        yield {"type": "done", "output": "ok", "usage": {}}

    return ProviderHandler(
        fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
    )


async def _run_config(
    path: str, tool_handlers: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    seen: list[tuple[Any, dict[str, Any]]] = []
    instance = config(
        key="doc-agent", handler=_capturing_handler(seen), tool_handlers=tool_handlers
    )
    if path == "invoke":
        await instance.invoke("q", CONTEXT)
    else:
        async for _ in instance.stream("q", CONTEXT):
            pass
    assert len(seen) == 1
    return seen[0]


# ---------------------------------------------------------------------------
# 3.19 / 3.21a — Reading tools onto Skill
# ---------------------------------------------------------------------------


class TestReadingToolsOntoSkill:
    async def test_skill_tools_is_an_immutable_ordered_list_and_empty_when_none(
        self,
    ) -> None:
        set_skill_store(_store(_envelope(omit=True)))
        skill = await get_skill(SKILL)
        assert skill is not None
        assert skill.tools == ()

        set_skill_store(_store(_envelope(tools=[_tool("t")])))
        skill = await get_skill(SKILL)
        assert skill is not None
        assert isinstance(skill.tools, tuple)
        with pytest.raises(TypeError):
            skill.tools[0] = _tool("other")  # type: ignore[index]
        with pytest.raises(dataclasses.FrozenInstanceError):
            skill.tools = ()  # type: ignore[misc]

    async def test_order_is_the_envelopes_order(self) -> None:
        set_skill_store(_store(_envelope(tools=[_tool("b"), _tool("a"), _tool("c")])))
        skill = await get_skill(SKILL)
        assert skill is not None
        assert _names(skill.tools) == ["b", "a", "c"]

    async def test_every_field_of_a_kept_entry_arrives_unchanged(self) -> None:
        set_skill_store(_store(_envelope(tools=[_tool("t")])))
        skill = await get_skill(SKILL)
        assert skill is not None
        assert skill.tools == (_tool("t"),)

    async def test_a_returned_list_shares_no_mutable_state_with_the_store(
        self, mock_ld_client: Any
    ) -> None:
        set_skill_store(_store(_envelope(tools=[_tool("t")])))
        first = await get_skill(SKILL)
        assert first is not None
        first.tools[0]["parameters"]["changed"] = True
        first.tools[0]["description"] = "changed"
        second = await get_skill(SKILL)
        assert second is not None
        assert second.tools == (_tool("t"),)

        mock_ld_client.variation = AsyncMock(
            return_value=_variation(skills=[_ref(SKILL)])
        )
        cfg, _ = await _run_config("invoke")
        assert cfg["tools"] == {"t": _tool("t")}

    async def test_every_accessor_carries_it(self) -> None:
        set_skill_store(_store(_envelope(tools=[_tool("t")])))
        expected = (_tool("t"),)
        one = await get_skill(SKILL)
        assert one is not None and one.tools == expected
        assert [s.tools for s in await get_skills([SKILL])] == [expected]
        assert [s.tools for s in await all_skills()] == [expected]
        outcome = await get_skill_result(SKILL)
        assert outcome.reason == "ok"
        assert outcome.skill is not None and outcome.skill.tools == expected

    def test_a_hand_built_skill_has_no_tools_unless_given_some(self) -> None:
        assert Skill(key="a", version=1, content=b"", content_hash="h").tools == ()
        given = Skill(
            key="a", version=1, content=b"", content_hash="h", tools=(_tool("t"),)
        )
        assert given.tools == (_tool("t"),)
        hash(given)


# ---------------------------------------------------------------------------
# 3.21a — Bad tool data drops only that tool
# ---------------------------------------------------------------------------


class TestBadToolDataDropsOnlyThatTool:
    @pytest.mark.parametrize(
        "tools",
        [{"t": _tool("t")}, "t", 3, None],
        ids=["object", "string", "number", "null"],
    )
    async def test_a_tools_that_is_not_an_array_drops_every_tool(
        self, tools: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        raw = _envelope()
        raw["tools"] = tools
        set_skill_store(_store(raw))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            outcome = await get_skill_result(SKILL)
        assert outcome.reason == "ok"
        assert outcome.skill is not None and outcome.skill.tools == ()
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert f"'{SKILL}' version 3" in warnings[0]

    @pytest.mark.parametrize(
        ("bad", "label"),
        [
            pytest.param(
                {"type": "function", "name": "mid", "parameters": "x"},
                "'mid'",
                id="parameters-string",
            ),
            pytest.param({"type": "function", "name": "mid"}, "'mid'", id="no-params"),
            pytest.param(
                {"type": "function", "parameters": {}}, "tools[1]", id="no-name"
            ),
            pytest.param(
                {"type": "function", "name": 7, "parameters": {}},
                "tools[1]",
                id="name-not-string",
            ),
            pytest.param("mid", "tools[1]", id="not-an-object"),
        ],
    )
    async def test_an_entry_with_no_string_name_or_no_object_parameters_is_dropped(
        self, bad: Any, label: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(_store(_envelope(tools=[_tool("first"), bad, _tool("last")])))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            skill = await get_skill(SKILL)
        assert skill is not None
        assert _names(skill.tools) == ["first", "last"]
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert f"'{SKILL}' version 3" in warnings[0] and label in warnings[0]

    @pytest.mark.parametrize("kind", ["custom", None], ids=["custom", "missing"])
    async def test_an_entry_whose_type_is_not_function_is_dropped(
        self, kind: Any
    ) -> None:
        bad = _tool("mid")
        if kind is None:
            del bad["type"]
        else:
            bad["type"] = kind
        set_skill_store(_store(_envelope(tools=[_tool("first"), bad])))
        skill = await get_skill(SKILL)
        assert skill is not None
        assert _names(skill.tools) == ["first"]

    async def test_a_second_entry_with_the_same_name_in_one_skill_is_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(
            _store(_envelope(tools=[_tool("t", "first"), _tool("t", "second")]))
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            skill = await get_skill(SKILL)
        assert skill is not None
        assert skill.tools == (_tool("t", "first"),)
        assert len(_warnings(caplog)) == 1

    async def test_each_drop_warning_never_carries_parameters_description_or_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bad = {
            "type": "function",
            "name": "line\nbreak SECRET",
            "description": "SECRET-DESC",
            "parameters": "SECRET-PARAMS",
        }
        set_skill_store(_store(_envelope(tools=[bad])))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            await get_skill(SKILL)
        text = "\n".join(_warnings(caplog))
        assert "tools[0]" in text
        for secret in ("SECRET", "Extract."):
            assert secret not in text

    async def test_no_integrity_signal_no_record(
        self, caplog: pytest.LogCaptureFixture, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        raw = _envelope(tools=[_tool("a"), {"name": 1}, _tool("a"), _tool("x")])
        raw["tools"][3]["type"] = "other"
        bad_container = _envelope("other-skill")
        bad_container["tools"] = "nope"
        set_skill_store(_store(raw, bad_container))
        with caplog.at_level(logging.DEBUG):
            await get_skill(SKILL)
            await get_skill("other-skill")
        assert recording_emitter.records == []
        assert INTEGRITY_FAILURE_EVENT not in caplog.text

    async def test_a_drop_warns_once_per_process(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(_store(_envelope(tools=[{"name": "bad"}])))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            await get_skill(SKILL)
            await get_skill(SKILL)
        assert len(_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# 3.21a — No integrity hash for tools in v1 (known limit)
# ---------------------------------------------------------------------------


class TestNoIntegrityHashForTools:
    async def test_a_changed_tool_still_verifies(self) -> None:
        raw = _envelope(tools=[_tool("t")])
        store = _store(raw)
        set_skill_store(store)
        raw["tools"][0]["description"] = "tampered"
        outcome = await get_skill_result(SKILL)
        assert outcome.reason == "ok"
        assert outcome.skill is not None
        assert outcome.skill.tools[0]["description"] == "tampered"

    async def test_skill_content_hash_is_computed_over_content_only(self) -> None:
        set_skill_store(
            _store(
                _envelope("a", tools=[_tool("x")]), _envelope("b", tools=[_tool("y")])
            )
        )
        a, b = await get_skills(["a", "b"])
        assert a.content_hash == b.content_hash


# ---------------------------------------------------------------------------
# 3.21a / 3.22 — Tools stay in memory only
# ---------------------------------------------------------------------------


class TestToolsStayInMemoryOnly:
    async def test_write_skills_writes_no_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            skills_fs_module, "_utc_timestamp", lambda: "2026-11-18T00:00:00Z"
        )
        with_tools, without = tmp_path / "with", tmp_path / "without"
        with_tools.mkdir()
        without.mkdir()
        set_skill_store(_store(_envelope(tools=[_tool("get-weather")])))
        report = await write_skills([SKILL], with_tools)
        assert not report.errors
        set_skill_store(_store(_envelope(omit=True)))
        await write_skills([SKILL], without)

        assert sorted(p.name for p in (with_tools / SKILL).iterdir()) == ["SKILL.md"]
        assert (with_tools / SKILL / "SKILL.md").read_text() == BODY
        raw_manifest = (with_tools / MANIFEST_FILENAME).read_text()
        assert "get-weather" not in raw_manifest
        assert raw_manifest == (without / MANIFEST_FILENAME).read_text()

    async def test_tools_do_not_change_a_reconcile(self, tmp_path: Path) -> None:
        set_skill_store(_store(_envelope(omit=True)))
        await write_skills([SKILL], tmp_path)
        set_skill_store(_store(_envelope(tools=[_tool("t")])))
        report = await write_skills([SKILL], tmp_path)
        assert [a.action for a in report.actions] == ["skipped_current"]


# ---------------------------------------------------------------------------
# 3.21a / 1.3 / 3.7 — Offering skill tools to the provider
# ---------------------------------------------------------------------------


class TestOfferingSkillTools:
    @pytest.mark.parametrize("path", ["invoke", "stream"])
    async def test_skill_tools_come_after_the_variations_tools(
        self, mock_ld_client: Any, path: str
    ) -> None:
        set_skill_store(_store(_envelope("s", tools=[_tool("c")])))
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(
                tools={"a": _tool("a"), "b": _tool("b")}, skills=[_ref("s")]
            )
        )
        cfg, _ = await _run_config(path)
        assert list(cfg["tools"]) == ["a", "b", "c"]
        assert cfg["tools"]["c"] == _tool("c")

    def test_skills_are_taken_in_the_variations_skills_order(self) -> None:
        set_skill_store(
            _store(
                _envelope("s1", tools=[_tool("one-a"), _tool("one-b")]),
                _envelope("s2", tools=[_tool("two")]),
            )
        )
        merged = _config_with_skill_tools("k", _cfg(skills=[_ref("s2"), _ref("s1")]))
        assert list(merged["tools"]) == ["two", "one-a", "one-b"]

    def test_each_skill_is_read_at_the_version_the_variation_pins(self) -> None:
        set_skill_store(
            _store(
                _envelope("s1", version=3, tools=[_tool("v3")]),
                _envelope("s1", version=4, tools=[_tool("v4")]),
            )
        )
        merged = _config_with_skill_tools("k", _cfg(skills=[_ref("s1", 3)]))
        assert list(merged["tools"]) == ["v3"]

    def test_an_exact_duplicate_is_offered_once_where_it_first_appears(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(
            _store(
                _envelope("a", tools=[_tool("a1"), _tool("get-weather")]),
                _envelope("b", tools=[_tool("get-weather"), _tool("b1")]),
            )
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = _config_with_skill_tools("k", _cfg(skills=[_ref("a"), _ref("b")]))
        assert list(merged["tools"]) == ["a1", "get-weather", "b1"]
        assert _warnings(caplog) == []

    def test_a_variation_and_a_skill_that_share_a_tool_key_the_variation_wins(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        own = _tool("get-weather", description="variation")
        set_skill_store(
            _store(_envelope("s", tools=[_tool("get-weather", description="skill")]))
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = _config_with_skill_tools(
                "doc-agent", _cfg(tools={"get-weather": own}, skills=[_ref("s")])
            )
        assert merged["tools"] == {"get-weather": own}
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        for part in ("'doc-agent'", "'s' version 3", "'get-weather'"):
            assert part in warnings[0]

    def test_an_exact_duplicate_compares_every_field_including_custom_parameters(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        other = _tool("t")
        other["customParameters"] = {"x": 2}
        set_skill_store(
            _store(_envelope("a", tools=[_tool("t")]), _envelope("b", tools=[other]))
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = _config_with_skill_tools("k", _cfg(skills=[_ref("a"), _ref("b")]))
        assert merged["tools"] == {"t": _tool("t")}
        assert len(_warnings(caplog)) == 1

    def test_two_skills_with_the_same_tool_key_and_different_definitions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(
            _store(
                _envelope("a", tools=[_tool("t", description="first")]),
                _envelope("b", tools=[_tool("t", description="second")]),
            )
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = _config_with_skill_tools("k", _cfg(skills=[_ref("a"), _ref("b")]))
        assert merged["tools"] == {"t": _tool("t", description="first")}
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        for part in ("'a' version 3", "'b' version 3", "'t'"):
            assert part in warnings[0]

    async def test_each_conflict_warns_once_per_process(
        self, mock_ld_client: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(
            _store(
                _envelope("a", tools=[_tool("t", description="first")]),
                _envelope("b", tools=[_tool("t", description="second"), _tool("v")]),
            )
        )
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(
                tools={"v": _tool("v")}, skills=[_ref("a"), _ref("b")]
            )
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            await _run_config("invoke")
            await _run_config("invoke")
        assert len(_warnings(caplog)) == 2

    @pytest.mark.parametrize("path", ["invoke", "stream"])
    async def test_the_allowlist_is_the_variations_tools_plus_its_skills_tools(
        self, mock_ld_client: Any, path: str
    ) -> None:
        """Also: a skill tool needs a registered handler, and a tool pinned by a
        skill the store holds but this variation does not attach is not offered."""
        set_skill_store(
            _store(
                _envelope("s", tools=[_tool("get-weather"), _tool("unregistered")]),
                _envelope("not-attached", tools=[_tool("other-skill-tool")]),
            )
        )
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(tools={"safe": _tool("safe")}, skills=[_ref("s")])
        )
        cfg, handlers = await _run_config(
            path,
            {
                "safe": MagicMock(),
                "get-weather": MagicMock(),
                "privileged": MagicMock(),
                "other-skill-tool": MagicMock(),
            },
        )
        assert set(handlers) == {"safe", "get-weather"}
        assert "other-skill-tool" not in cfg["tools"]

    async def test_skill_tools_are_attached_tools(self, mock_ld_client: Any) -> None:
        """TESTING.md 3.7: no config.tools, one skill pinning get-weather."""
        set_skill_store(_store(_envelope("s", tools=[_tool("get-weather")])))
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(skills=[_ref("s")])
        )
        _, handlers = await _run_config(
            "invoke", {"safe": MagicMock(), "get-weather": MagicMock()}
        )
        assert set(handlers) == {"get-weather"}

    @pytest.mark.parametrize(
        "reason", ["absent", "integrity_failure", "wrong_version", "store_unavailable"]
    )
    def test_a_skill_that_cannot_be_read_adds_no_tools_and_the_run_goes_on(
        self,
        reason: str,
        caplog: pytest.LogCaptureFixture,
        recording_emitter: Any,
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        good = _envelope("good", tools=[_tool("g")])
        broken = _envelope("broken", tools=[_tool("b")])
        if reason == "integrity_failure":
            broken["contentHash"] = "0" * 64
        store: Any = _store(good, broken) if reason != "absent" else _store(good)
        if reason == "wrong_version":
            real_get = store.get_object

            def answer_with_v3(kind: str, key: str, version: Any = None) -> Any:
                return real_get(kind, key, 3 if key == "broken" else version)

            store.get_object = answer_with_v3
        if reason == "store_unavailable":
            real = store.get_object

            def get_object(kind: str, key: str, version: Any = None) -> Any:
                if key == "broken":
                    raise OSError("down")
                return real(kind, key, version)

            store.get_object = get_object
        set_skill_store(store)
        version = 9 if reason == "wrong_version" else 3
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = _config_with_skill_tools(
                "k",
                _cfg(
                    tools={"own": _tool("own")},
                    skills=[_ref("broken", version), _ref("good")],
                ),
            )
        assert list(merged["tools"]) == ["own", "g"]
        unavailable = [w for w in _warnings(caplog) if "unavailable" in w]
        assert len(unavailable) == 1
        assert "'broken'" in unavailable[0] and reason in unavailable[0]
        first_signals = list(recording_emitter.records)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _config_with_skill_tools("k", _cfg(skills=[_ref("broken", version)]))
        assert len([w for w in _warnings(caplog) if "unavailable" in w]) == 1
        integrity_signals = [r for r in first_signals if "Integrity" in r[0]]
        assert len(integrity_signals) == (1 if reason == "integrity_failure" else 0)

    async def test_no_skill_store_means_no_skill_tools_and_no_error(
        self, mock_ld_client: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(tools={"own": _tool("own")}, skills=[_ref("s")])
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            cfg, _ = await _run_config("invoke")
        assert list(cfg["tools"]) == ["own"]
        assert _warnings(caplog) == []

    async def test_a_variation_with_no_skills_reads_no_store(
        self, mock_ld_client: Any
    ) -> None:
        store = MagicMock()
        set_skill_store(store)
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(tools={"own": _tool("own")})
        )
        cfg, _ = await _run_config("invoke")
        assert cfg["tools"] == {"own": _tool("own")}
        store.get_object.assert_not_called()
        store.all_objects.assert_not_called()

    async def test_reading_skills_for_their_tools_reports_no_usage(
        self, mock_ld_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        helpers: list[str] = []
        monkeypatch.setattr(
            skills_module, "report_usage", lambda helper, *a: helpers.append(helper)
        )
        set_skill_store(_store(_envelope("s", tools=[_tool("t")])))
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(skills=[_ref("s")])
        )
        cfg, _ = await _run_config("invoke")
        assert list(cfg["tools"]) == ["t"]
        assert helpers == []

    @pytest.mark.parametrize(
        "skills", [None, "a", [{"key": "a"}]], ids=["null", "string", "no-version"]
    )
    def test_a_malformed_skills_field_does_not_fail_the_run(self, skills: Any) -> None:
        set_skill_store(_store(_envelope("a", tools=[_tool("a1")])))
        cfg = _cfg()
        cfg["skills"] = skills
        assert _config_with_skill_tools("k", cfg) is cfg

    async def test_inspect_config_returns_only_the_variations_own_tools(
        self, mock_ld_client: Any
    ) -> None:
        set_skill_store(_store(_envelope("s", tools=[_tool("t")])))
        mock_ld_client.variation = AsyncMock(
            return_value=_variation(tools={"own": _tool("own")}, skills=[_ref("s")])
        )
        info = await inspect_config("doc-agent", CONTEXT)
        assert list(info["config"]["tools"]) == ["own"]

    async def test_every_graph_node_runs_with_its_skill_tools(
        self, mock_ld_client: Any
    ) -> None:
        """``GraphNode.config`` stays the variation's own, since native adapters
        merge for themselves (2.2); ``graph()`` runs each node with its skill tools."""
        set_skill_store(_store(_envelope("s", tools=[_tool("t")])))

        async def variation(key: str, ctx: Any, default: Any) -> Any:
            if key == "graph-key":
                return {"root": "root", "edges": {}}
            return _variation(skills=[_ref("s")])

        mock_ld_client.variation = AsyncMock(side_effect=variation)
        definition = await resolve_graph("graph-key", context=CONTEXT)
        assert definition.root is not None
        assert "tools" not in definition.root.config

        seen: list[tuple[Any, dict[str, Any]]] = []
        await graph(
            "graph-key",
            handlers=[_capturing_handler(seen)],
            tool_handlers={"t": MagicMock(), "other": MagicMock()},
        ).invoke("hi", CONTEXT)
        assert set(seen[0][1]) == {"t"}
        assert list(seen[0][0]["tools"]) == ["t"]

    def test_shadow_variations_get_no_skill_tools(self) -> None:
        """A shadow runs with no tools (README) and no tool handlers, so the
        skill tools are not merged in and the store is not read."""
        from launchdarkly_ai_server.experimental.shadow_requests import (
            _shadow_variation,
        )

        store = _store(_envelope("s", tools=[_tool("t")]))
        store.get_object = MagicMock(wraps=store.get_object)  # type: ignore[method-assign]
        set_skill_store(store)
        parsed = _shadow_variation(
            "k", _variation(tools={"own": _tool("own")}, skills=[_ref("s")])
        )
        assert parsed is not None
        assert "tools" not in parsed[0]
        store.get_object.assert_not_called()

    def test_an_exact_duplicate_compares_the_five_tool_fields_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing field equals only a missing one; a field outside the five
        does not make two definitions differ."""
        no_description = _tool("t")
        del no_description["description"]
        null_description = {**_tool("t"), "description": None}
        extra = {**_tool("u"), "extra": 1}
        set_skill_store(
            _store(
                _envelope("a", tools=[no_description, _tool("u")]),
                _envelope("b", tools=[null_description, extra]),
            )
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = _config_with_skill_tools("k", _cfg(skills=[_ref("a"), _ref("b")]))
        assert merged["tools"] == {"t": no_description, "u": _tool("u")}
        warnings = _warnings(caplog)
        assert len(warnings) == 1 and "'t'" in warnings[0]


async def test_a_routed_graph_node_keeps_its_own_skill_and_handoff_tools(
    mock_ld_client: Any,
) -> None:
    set_skill_store(_store(_envelope("s", tools=[_tool("get-weather")])))

    async def variation(key: str, ctx: Any, default: Any) -> Any:
        if key == "graph-key":
            return {"root": "root", "edges": {"root": [{"key": "a"}, {"key": "b"}]}}
        if key == "root":
            return _variation(tools={"own": _tool("own")}, skills=[_ref("s")])
        return _variation(tools={"leaf": _tool("leaf")})

    mock_ld_client.variation = AsyncMock(side_effect=variation)
    seen: list[tuple[Any, dict[str, Any]]] = []
    await graph(
        "graph-key",
        handlers=[_capturing_handler(seen)],
        tool_handlers={
            "own": MagicMock(),
            "get-weather": MagicMock(),
            "leaf": MagicMock(),
        },
    ).invoke("hi", CONTEXT)
    root_cfg, root_handlers = seen[0]
    names = list(root_cfg["tools"])
    assert names[:2] == ["own", "get-weather"]
    assert any(n.startswith("__handoff_") for n in names[2:])
    assert {"own", "get-weather"} <= set(root_handlers)
    for cfg, handlers in seen[1:]:
        assert "get-weather" not in cfg["tools"] and "get-weather" not in handlers


def test_an_unreadable_skill_before_the_store_initializes_still_warns_after(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store()
    ready = False
    store.is_initialized = lambda: ready  # type: ignore[attr-defined]
    set_skill_store(store)
    cfg = _cfg(skills=[_ref("s")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _config_with_skill_tools("k", cfg)
        _config_with_skill_tools("k", cfg)
        ready = True
        _config_with_skill_tools("k", cfg)
        _config_with_skill_tools("k", cfg)
    # One before initialization and one after, however many runs each side.
    assert len([w for w in _warnings(caplog) if "unavailable" in w]) == 2


async def test_a_257_character_tool_name_is_not_quoted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    long_name = "a" * 257
    set_skill_store(_store(_envelope(tools=[{"name": long_name}])))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await get_skill(SKILL)
    text = "\n".join(_warnings(caplog))
    assert "tools[0]" in text and long_name not in text
