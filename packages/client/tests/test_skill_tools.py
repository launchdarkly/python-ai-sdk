"""Skill tools: the ``tools`` a skill envelope pins, read onto ``Skill.tools`` and
offered to handlers after the variation's own ``config["tools"]``.

Built against hand-made envelopes until LaunchDarkly serves them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
import launchdarkly_ai_server.skills as skills_module
from launchdarkly_ai_server import (
    MANIFEST_FILENAME,
    InMemorySkillStore,
    ProviderHandler,
    Skill,
    config,
    get_skill,
    graph,
    resolve_graph,
    set_skill_store,
    write_skills,
)
from launchdarkly_ai_server.skills_core import (
    config_with_skill_tools,
    verify_raw_skill,
)

pytestmark = pytest.mark.usefixtures("reset_skill_state")

CONTEXT = {"kind": "user", "key": "u1"}
BODY = "---\nname: PDF\n---\nExtract.\n"
LOGGER = "launchdarkly_ai_server.skills_core"


def _tool(name: str, description: str = "d") -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
        "customParameters": {"x": 1},
    }


def _envelope(
    key: str = "pdf-extraction", version: int = 3, tools: Any = None, **extra: Any
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "key": key,
        "version": version,
        "content": BODY,
        "contentHash": hashlib.sha256(BODY.encode()).hexdigest(),
        "name": "PDF",
        "description": "Extracts PDFs.",
        **extra,
    }
    if tools is not None:
        raw["tools"] = tools
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


# ---------------------------------------------------------------------------
# Reading tools from the envelope
# ---------------------------------------------------------------------------


class TestSkillTools:
    def test_tools_are_read_in_author_order(self) -> None:
        tools = [_tool("get-weather"), _tool("get-time")]
        skill = verify_raw_skill(_envelope(tools=tools))
        assert skill is not None
        assert skill.tools == tuple(tools)
        assert isinstance(skill.tools, tuple)

    def test_no_tools_field_means_an_empty_tuple(self) -> None:
        skill = verify_raw_skill(_envelope())
        assert skill is not None
        assert skill.tools == ()

    def test_a_constructed_skill_defaults_to_no_tools(self) -> None:
        skill = Skill(key="a", version=1, content=b"", content_hash="h")
        assert skill.tools == ()

    def test_a_skill_with_tools_is_still_hashable(self) -> None:
        skill = verify_raw_skill(_envelope(tools=[_tool("t")]))
        assert skill is not None
        hash(skill)

    def test_tools_are_copied_from_the_store(self) -> None:
        tools = [_tool("t")]
        skill = verify_raw_skill(_envelope(tools=tools))
        assert skill is not None
        skill.tools[0]["parameters"]["changed"] = True
        assert "changed" not in tools[0]["parameters"]

    async def test_get_skill_returns_the_tools(self) -> None:
        set_skill_store(_store(_envelope(tools=[_tool("t")])))
        skill = await get_skill("pdf-extraction", version=3)
        assert skill is not None
        assert [t["name"] for t in skill.tools] == ["t"]

    @pytest.mark.parametrize(
        "tools",
        [{"t": _tool("t")}, "t", 3, None],
        ids=["object", "string", "int", "null"],
    )
    def test_tools_that_are_not_an_array_drop_every_tool(
        self, tools: Any, caplog: pytest.LogCaptureFixture, recording_emitter: Any
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        raw = _envelope()
        raw["tools"] = tools
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            skill = verify_raw_skill(raw)
        assert skill is not None
        assert skill.tools == ()
        assert "pdf-extraction" in caplog.text
        assert recording_emitter.records == []

    @pytest.mark.parametrize(
        ("bad", "label"),
        [
            pytest.param(
                {"type": "function", "parameters": {}}, "index 1", id="no-name"
            ),
            pytest.param(
                {"name": 7, "parameters": {}}, "index 1", id="name-not-string"
            ),
            pytest.param({"name": "bad-tool"}, "'bad-tool'", id="no-parameters"),
            pytest.param(
                {"name": "bad-tool", "parameters": []},
                "'bad-tool'",
                id="parameters-list",
            ),
            pytest.param("bad-tool", "index 1", id="not-an-object"),
        ],
    )
    def test_a_malformed_entry_drops_only_that_tool(
        self,
        bad: Any,
        label: str,
        caplog: pytest.LogCaptureFixture,
        recording_emitter: Any,
    ) -> None:
        skills_module._set_emitter_for_testing(recording_emitter)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            skill = verify_raw_skill(
                _envelope(tools=[_tool("first"), bad, _tool("last")])
            )
        assert skill is not None
        assert [t["name"] for t in skill.tools] == ["first", "last"]
        warnings = [r.getMessage() for r in caplog.records]
        assert len(warnings) == 1
        assert "pdf-extraction" in warnings[0] and label in warnings[0]
        assert recording_emitter.records == []

    def test_an_unsafe_tool_name_is_not_quoted_in_the_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            verify_raw_skill(_envelope(tools=[{"name": "line\nbreak secret"}]))
        assert "secret" not in caplog.text
        assert "index 0" in caplog.text

    def test_tools_do_not_affect_integrity(self) -> None:
        """contentHash covers only the markdown: tools can change and it still verifies."""
        assert verify_raw_skill(_envelope(tools=[_tool("a")])) is not None
        assert verify_raw_skill(_envelope(tools=[_tool("b")])) is not None


# ---------------------------------------------------------------------------
# Merging into a config
# ---------------------------------------------------------------------------


class TestConfigWithSkillTools:
    def test_skill_tools_follow_the_variation_tools_in_skills_order(self) -> None:
        set_skill_store(
            _store(
                _envelope("a", tools=[_tool("a1"), _tool("a2")]),
                _envelope("b", tools=[_tool("b1")]),
            )
        )
        cfg = _cfg(tools={"own": _tool("own")}, skills=[_ref("b"), _ref("a")])
        merged = config_with_skill_tools(cfg)
        assert list(merged["tools"]) == ["own", "b1", "a1", "a2"]
        assert merged["tools"]["a1"] == _tool("a1")
        assert "b1" not in cfg["tools"]

    def test_a_config_without_tools_gets_the_skill_tools(self) -> None:
        set_skill_store(_store(_envelope("a", tools=[_tool("a1")])))
        merged = config_with_skill_tools(_cfg(skills=[_ref("a")]))
        assert list(merged["tools"]) == ["a1"]

    def test_the_pinned_version_supplies_the_tools(self) -> None:
        set_skill_store(
            _store(
                _envelope("a", version=1, tools=[_tool("old")]),
                _envelope("a", version=2, tools=[_tool("new")]),
            )
        )
        merged = config_with_skill_tools(_cfg(skills=[_ref("a", 1)]))
        assert list(merged["tools"]) == ["old"]

    @pytest.mark.parametrize(
        "cfg",
        [
            pytest.param(_cfg(tools={"own": _tool("own")}), id="no-skills"),
            pytest.param(_cfg(skills=[]), id="empty-skills"),
        ],
    )
    def test_a_config_without_skills_is_returned_as_is(self, cfg: Any) -> None:
        set_skill_store(_store(_envelope("a", tools=[_tool("a1")])))
        assert config_with_skill_tools(cfg) is cfg

    def test_skills_without_tools_leave_the_config_as_is(self) -> None:
        set_skill_store(_store(_envelope("a")))
        cfg = _cfg(skills=[_ref("a")])
        assert config_with_skill_tools(cfg) is cfg

    def test_an_exact_duplicate_across_skills_is_offered_once_without_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(
            _store(
                _envelope("a", tools=[_tool("shared"), _tool("a1")]),
                _envelope("b", tools=[_tool("shared"), _tool("b1")]),
            )
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = config_with_skill_tools(_cfg(skills=[_ref("a"), _ref("b")]))
        assert list(merged["tools"]) == ["shared", "a1", "b1"]
        assert caplog.records == []

    def test_the_variation_definition_wins_a_shared_key_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        own = _tool("shared", description="variation")
        set_skill_store(
            _store(_envelope("a", tools=[_tool("shared", description="skill")]))
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = config_with_skill_tools(
                _cfg(tools={"shared": own}, skills=[_ref("a")])
            )
        assert merged["tools"] == {"shared": own}
        assert "'shared'" in caplog.text and "'a'" in caplog.text

    def test_the_variation_wins_even_over_an_identical_skill_tool(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(_store(_envelope("a", tools=[_tool("shared")])))
        cfg = _cfg(tools={"shared": _tool("shared")}, skills=[_ref("a")])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = config_with_skill_tools(cfg)
        assert list(merged["tools"]) == ["shared"]
        assert len(caplog.records) == 1

    def test_the_first_skill_wins_a_conflicting_key_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(
            _store(
                _envelope("a", tools=[_tool("shared", description="first")]),
                _envelope("b", tools=[_tool("shared", description="second")]),
            )
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = config_with_skill_tools(_cfg(skills=[_ref("a"), _ref("b")]))
        assert merged["tools"]["shared"]["description"] == "first"
        assert "'b'" in caplog.text

    def test_an_unavailable_skill_adds_nothing_and_the_rest_still_do(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        set_skill_store(_store(_envelope("b", tools=[_tool("b1")])))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = config_with_skill_tools(_cfg(skills=[_ref("a"), _ref("b")]))
        assert list(merged["tools"]) == ["b1"]
        assert "'a'" in caplog.text and "absent" in caplog.text

    def test_a_skill_that_fails_integrity_adds_no_tools(self) -> None:
        raw = _envelope("a", tools=[_tool("a1")])
        raw["contentHash"] = "0" * 64
        set_skill_store(_store(raw))
        cfg = _cfg(skills=[_ref("a")])
        assert config_with_skill_tools(cfg) is cfg

    def test_no_store_leaves_the_config_as_is_and_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = _cfg(skills=[_ref("a")])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert config_with_skill_tools(cfg) is cfg
            assert config_with_skill_tools(cfg) is cfg
        assert len(caplog.records) == 1
        assert "skill store" in caplog.text

    @pytest.mark.parametrize(
        "skills", [None, "a", [{"key": "a"}]], ids=["null", "string", "no-version"]
    )
    def test_a_malformed_skills_field_leaves_the_config_as_is(
        self, skills: Any
    ) -> None:
        set_skill_store(_store(_envelope("a", tools=[_tool("a1")])))
        cfg = _cfg()
        cfg["skills"] = skills
        assert config_with_skill_tools(cfg) is cfg


# ---------------------------------------------------------------------------
# What handlers receive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["invoke", "stream"])
async def test_config_offers_skill_tools_and_allows_their_handlers(
    mock_ld_client: Any, path: str
) -> None:
    """The allowlist is the variation's tools plus its skills' tools, and still
    never a tool neither one names."""
    set_skill_store(_store(_envelope("a", tools=[_tool("a1")])))
    mock_ld_client.variation = AsyncMock(
        return_value=_variation(tools={"own": _tool("own")}, skills=[_ref("a")])
    )
    seen: list[tuple[Any, dict[str, Any]]] = []
    instance = config(
        key="k",
        handler=_capturing_handler(seen),
        tool_handlers={"own": MagicMock(), "a1": MagicMock(), "other": MagicMock()},
    )
    if path == "invoke":
        await instance.invoke("q", CONTEXT)
    else:
        async for _ in instance.stream("q", CONTEXT):
            pass
    cfg, tool_handlers = seen[0]
    assert list(cfg["tools"]) == ["own", "a1"]
    assert set(tool_handlers) == {"own", "a1"}


async def test_graph_nodes_offer_their_skill_tools(mock_ld_client: Any) -> None:
    set_skill_store(_store(_envelope("a", tools=[_tool("a1")])))

    async def variation(key: str, ctx: Any, default: Any) -> Any:
        if key == "graph-key":
            return {"root": "root", "edges": {}}
        return _variation(skills=[_ref("a")])

    mock_ld_client.variation = AsyncMock(side_effect=variation)
    definition = await resolve_graph("graph-key", context=CONTEXT)
    assert definition.root is not None
    assert list(definition.root.config["tools"]) == ["a1"]

    seen: list[tuple[Any, dict[str, Any]]] = []
    await graph(
        "graph-key",
        handlers=[_capturing_handler(seen)],
        tool_handlers={"a1": MagicMock(), "other": MagicMock()},
    ).invoke("hi", CONTEXT)
    assert set(seen[0][1]) == {"a1"}


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


async def test_write_skills_writes_no_tools(tmp_path: Path) -> None:
    set_skill_store(_store(_envelope(tools=[_tool("get-weather")])))
    report = await write_skills(["pdf-extraction"], tmp_path)
    assert not report.errors
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert files == [MANIFEST_FILENAME, "pdf-extraction", "pdf-extraction/SKILL.md"]
    assert (tmp_path / "pdf-extraction" / "SKILL.md").read_text() == BODY
    manifest = (tmp_path / MANIFEST_FILENAME).read_text()
    assert "get-weather" not in manifest and "tools" not in json.loads(manifest)


def test_shadow_variations_offer_their_skill_tools() -> None:
    from launchdarkly_ai_server.experimental.shadow_requests import _shadow_variation

    set_skill_store(_store(_envelope("a", tools=[_tool("a1")])))
    parsed = _shadow_variation("k", _variation(skills=[_ref("a")]))
    assert parsed is not None
    assert list(parsed[0]["tools"]) == ["a1"]
