"""
Tests for TESTING.md §2.x Claude Agents Skills Integration (A.18).

``query()`` is replaced with a fake async generator that records the options it was given and can
look at the filesystem while the run is in progress. The skill store is a real
``InMemorySkillStore``, the filesystem is a per-test ``tmp_path`` passed as ``skills_temp_dir``,
and spans go to a real ``TracerProvider``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage, StreamEvent
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import launchdarkly_ai_claude_agents.handler as handler_mod
import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_claude_agents import ClaudeRead, create_claude_agents_handler
from launchdarkly_ai_claude_agents.experimental.skills import (
    create_claude_agents_skills_handler,
)
from launchdarkly_ai_server import NATIVE_TOOL_KEY, config, init_client, shutdown
from launchdarkly_ai_server.experimental.skills import (
    InMemorySkillStore,
    set_skill_store,
)

# ---------------------------------------------------------------------------
# Spans: share whichever SDK provider is installed, or install one
# ---------------------------------------------------------------------------

_exporter = InMemorySpanExporter()
_provider = trace.get_tracer_provider()
if not isinstance(_provider, TracerProvider):
    _provider = TracerProvider()
    trace.set_tracer_provider(_provider)
_provider.add_span_processor(SimpleSpanProcessor(_exporter))


@pytest.fixture(autouse=True)
def _reset() -> Any:
    """Clears the exporter and the configured skill store around each test."""
    _exporter.clear()
    lifecycle_module._reset_for_testing()
    yield
    lifecycle_module._reset_for_testing()


_PATHS = ["invoke", "stream"]

SKILL_BODY = "---\nname: {key}\n---\nVersion {version} of {key}.\n"

BASE_CONFIG: dict[str, Any] = {
    "model": {"name": "claude-opus-4-5"},
    "provider": {"name": "Anthropic"},
    "instructions": "You are helpful.",
}


def _config(*refs: tuple[str, int], **extra: Any) -> dict[str, Any]:
    return {
        **BASE_CONFIG,
        "skills": [{"key": key, "version": version} for key, version in refs],
        **extra,
    }


def _body(key: str, version: int) -> str:
    return SKILL_BODY.format(key=key, version=version)


def _raw(key: str, version: int, *, content_hash: str | None = None) -> dict[str, Any]:
    body = _body(key, version)
    return {
        "key": key,
        "version": version,
        "content": body,
        "contentHash": content_hash or hashlib.sha256(body.encode()).hexdigest(),
    }


def _store(*objects: dict[str, Any]) -> InMemorySkillStore:
    store = InMemorySkillStore()
    for raw in objects:
        store.put(raw)
    set_skill_store(store)
    return store


def _result(subtype: str = "success") -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=subtype != "success",
        num_turns=1,
        session_id="sess-1",
        usage={"input_tokens": 3, "output_tokens": 2},
        result="done",
    )


def _text_event(text: str) -> StreamEvent:
    return StreamEvent(
        uuid="u1",
        session_id="sess-1",
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    )


During = Callable[[dict[str, Any]], Awaitable[None]]


class _FakeQuery:
    """Stands in for ``query()``: records each call's options, runs *during* while the run is in
    progress, then replays *messages*."""

    def __init__(
        self, messages: list[Any] | None = None, during: During | None = None
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = messages if messages is not None else [_result()]
        self.during = during

    @property
    def options(self) -> Any:
        assert len(self.calls) == 1
        return self.calls[0]["options"]

    async def __call__(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append(kwargs)
        if self.during is not None:
            await self.during(kwargs)
        for message in self.messages:
            yield message


async def _run(
    handler: Any,
    path: str,
    cfg: dict[str, Any],
    tool_handlers: dict[str, Any] | None = None,
    user_input: str = "q",
) -> Any:
    if path == "invoke":
        return await handler(cfg, user_input, tool_handlers or {})
    events = []
    async for event in await handler.stream(cfg, user_input, tool_handlers or {}):
        events.append(event)
    return events


def _plugin_root(options: Any) -> Path:
    return Path(options.plugins[0]["path"])


def _run_dir(options: Any) -> Path:
    return _plugin_root(options).parent


def _read_stub() -> MagicMock:
    stub = MagicMock()
    setattr(stub, NATIVE_TOOL_KEY, ClaudeRead)
    return stub


# ---------------------------------------------------------------------------
# Routing, options and the no-skills path
# ---------------------------------------------------------------------------


class TestRoutingAndOptions:
    def test_routes_exactly_as_the_core_handler_does(self) -> None:
        assert create_claude_agents_skills_handler().provides_for == (
            create_claude_agents_handler().provides_for
        )
        assert tuple(create_claude_agents_skills_handler().provides_for) == (
            "Anthropic",
            "agent",
        )

    def test_is_absent_from_the_package_root(self) -> None:
        """A fresh interpreter, so this module's own import of the entry point doesn't count."""
        probe = (
            "import sys, launchdarkly_ai_claude_agents as p;"
            "assert 'launchdarkly_ai_claude_agents.experimental' not in sys.modules;"
            "assert not hasattr(p, 'create_claude_agents_skills_handler');"
            "assert 'create_claude_agents_skills_handler' not in p.__all__"
        )
        subprocess.run([sys.executable, "-c", probe], check=True)

    @pytest.mark.parametrize("path", _PATHS)
    async def test_forwards_the_host_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1))
        fake = _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(
            cwd="/srv/app", setting_sources=["project"], skills_temp_dir=tmp_path
        )
        await _run(handler, path, _config(("a", 1)))
        assert fake.options.cwd == "/srv/app"
        assert fake.options.setting_sources == ["project"]

    @pytest.mark.parametrize("path", _PATHS)
    async def test_sends_an_explicit_empty_setting_sources_with_skills(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        """``claude-agent-sdk`` sends ``--setting-sources=user,project`` when ``skills`` is set and
        ``setting_sources`` is ``None``, so the handler must never leave it ``None``."""
        _store(_raw("a", 1))
        fake = _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 1)))
        assert fake.options.skills == ["launchdarkly:a"]
        assert fake.options.setting_sources == []

    @pytest.mark.parametrize("path", _PATHS)
    @pytest.mark.parametrize("skills", [None, []])
    async def test_a_config_with_no_skills_runs_as_the_core_handler_would(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        path: str,
        skills: list[Any] | None,
    ) -> None:
        """No store is configured: a variation with no skills needs none."""
        cfg = dict(BASE_CONFIG) if skills is None else {**BASE_CONFIG, "skills": skills}
        core, experimental = _FakeQuery(), _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", core)
        await _run(create_claude_agents_handler(cwd="/srv/app"), path, cfg)
        monkeypatch.setattr(handler_mod, "query", experimental)
        await _run(
            create_claude_agents_skills_handler(
                cwd="/srv/app", skills_temp_dir=tmp_path
            ),
            path,
            cfg,
        )
        assert experimental.options == core.options
        assert experimental.options.plugins == []
        assert experimental.options.skills is None
        assert "Skill" not in experimental.options.tools
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Materialization: one private directory per invoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS)
class TestMaterialization:
    async def test_writes_each_skill_into_a_private_plugin_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1), _raw("b", 1))
        seen: dict[str, Any] = {}

        async def during(kwargs: dict[str, Any]) -> None:
            plugin_root = _plugin_root(kwargs["options"])
            seen["run_dir"] = plugin_root.parent
            seen["a"] = (plugin_root / "skills" / "a" / "SKILL.md").read_text()
            seen["b"] = (plugin_root / "skills" / "b" / "SKILL.md").read_text()

        fake = _FakeQuery(during=during)
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 1), ("b", 1)))

        run_dir = seen["run_dir"]
        assert run_dir.parent == Path(os.path.realpath(tmp_path))
        assert run_dir.name.startswith("ld-skills-")
        assert str(run_dir) == os.path.realpath(run_dir)
        assert _plugin_root(fake.options).name == "launchdarkly"
        assert seen["a"] == _body("a", 1)
        assert seen["b"] == _body("b", 1)

    async def test_honors_a_pinned_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1), _raw("a", 2), _raw("a", 3))
        seen: list[str] = []

        async def during(kwargs: dict[str, Any]) -> None:
            skill = _plugin_root(kwargs["options"]) / "skills" / "a" / "SKILL.md"
            seen.append(skill.read_text())

        monkeypatch.setattr(handler_mod, "query", _FakeQuery(during=during))
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 2)))
        assert seen == [_body("a", 2)]

    async def test_removes_the_directory_after_a_successful_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1))
        fake = _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 1)))
        assert not _run_dir(fake.options).exists()
        assert list(tmp_path.iterdir()) == []

    async def test_removes_the_directory_when_query_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1))

        async def during(_kwargs: dict[str, Any]) -> None:
            raise RuntimeError("the CLI went away")

        monkeypatch.setattr(handler_mod, "query", _FakeQuery(during=during))
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        with pytest.raises(RuntimeError, match="the CLI went away"):
            await _run(handler, path, _config(("a", 1)))
        assert list(tmp_path.iterdir()) == []

    async def test_removes_the_directory_after_a_non_success_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1))
        monkeypatch.setattr(
            handler_mod, "query", _FakeQuery([_result("error_max_turns")])
        )
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        with pytest.raises(RuntimeError, match="error_max_turns"):
            await _run(handler, path, _config(("a", 1)))
        assert list(tmp_path.iterdir()) == []

    async def test_a_new_invoke_does_not_write_a_detached_skill(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        """Detaching a skill is a config change: the next invoke's ``skill_refs`` no longer names
        it, so it is not written."""
        _store(_raw("a", 1), _raw("b", 1))
        held: list[list[str]] = []

        async def during(kwargs: dict[str, Any]) -> None:
            skills_dir = _plugin_root(kwargs["options"]) / "skills"
            held.append(sorted(p.name for p in skills_dir.iterdir() if p.is_dir()))

        monkeypatch.setattr(handler_mod, "query", _FakeQuery(during=during))
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 1), ("b", 1)))
        await _run(handler, path, _config(("a", 1)))
        assert held == [["a", "b"], ["a"]]


async def test_removes_the_directory_when_a_stream_consumer_stops_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _store(_raw("a", 1))
    fake = _FakeQuery([_text_event("hel"), _text_event("lo"), _result()])
    monkeypatch.setattr(handler_mod, "query", fake)
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    gen = await handler.stream(_config(("a", 1)), "q", {})
    async for _event in gen:
        assert _run_dir(fake.options).exists()
        break
    await gen.aclose()
    assert list(tmp_path.iterdir()) == []


async def test_concurrent_invokes_do_not_share_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _store(_raw("a", 1))
    started = {"first": asyncio.Event(), "second": asyncio.Event()}
    release = {"first": asyncio.Event(), "second": asyncio.Event()}
    dirs: dict[str, Path] = {}

    async def during(kwargs: dict[str, Any]) -> None:
        name = kwargs["prompt"]
        dirs[name] = _run_dir(kwargs["options"])
        started[name].set()
        await release[name].wait()

    monkeypatch.setattr(handler_mod, "query", _FakeQuery(during=during))
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    cfg = _config(("a", 1))
    first = asyncio.create_task(handler(cfg, "first", {}))
    second = asyncio.create_task(handler(cfg, "second", {}))
    await started["first"].wait()
    await started["second"].wait()

    assert dirs["first"] != dirs["second"]
    assert dirs["first"].exists() and dirs["second"].exists()

    release["first"].set()
    await first
    assert not dirs["first"].exists()
    assert (dirs["second"] / "launchdarkly" / "skills" / "a" / "SKILL.md").exists()

    release["second"].set()
    await second
    assert list(tmp_path.iterdir()) == []


async def test_a_failed_removal_is_logged_and_does_not_replace_the_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import launchdarkly_ai_claude_agents.experimental.skills as skills_mod

    _store(_raw("a", 1))
    monkeypatch.setattr(handler_mod, "query", _FakeQuery())

    def refuse(_path: str) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(skills_mod.shutil, "rmtree", refuse)
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    result = await handler(_config(("a", 1)), "q", {})
    assert result["output"] == "done"
    assert any("could not remove" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


class _NotYetInitialized(InMemorySkillStore):
    def is_initialized(self) -> bool:
        return False


def _no_store() -> None:
    pass


def _uninitialized_store() -> None:
    store = _NotYetInitialized()
    store.put(_raw("a", 1))
    set_skill_store(store)


def _withheld_by_verification() -> None:
    _store(_raw("a", 1, content_hash="0" * 64))


def _pin_not_held() -> None:
    _store(_raw("a", 1), _raw("a", 3))


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize(
    "arrange",
    [_no_store, _uninitialized_store, _withheld_by_verification, _pin_not_held],
    ids=["no store", "not initialized", "withheld", "absent pin"],
)
async def test_a_reconcile_that_is_not_ok_fails_before_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    arrange: Callable[[], None],
) -> None:
    arrange()
    fake = _FakeQuery()
    monkeypatch.setattr(handler_mod, "query", fake)
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not started") as excinfo:
        await _run(handler, path, _config(("a", 2)))
    assert fake.calls == []
    # Each failed key, with its action's error text.
    assert "a: skill " in str(excinfo.value)
    # The directory is removed on this path too.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", _PATHS)
async def test_one_failed_skill_stops_the_run_and_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    _store(_raw("good", 1), _raw("bad", 1, content_hash="0" * 64))
    fake = _FakeQuery()
    monkeypatch.setattr(handler_mod, "query", fake)
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        await _run(handler, path, _config(("good", 1), ("bad", 1)))
    message = str(excinfo.value)
    # The run does not start with the skill that could be written.
    assert fake.calls == []
    assert "bad: skill 'bad' failed integrity verification" in message
    assert "good: " not in message
    # No skill content, and no path.
    assert _body("good", 1) not in message
    assert _body("bad", 1) not in message
    assert str(tmp_path) not in message
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# What query() receives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS)
class TestQueryOptions:
    async def test_one_local_plugin_with_no_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1))
        manifests: list[bool] = []

        async def during(kwargs: dict[str, Any]) -> None:
            root = _plugin_root(kwargs["options"])
            manifests.append((root / ".claude-plugin").exists())

        fake = _FakeQuery(during=during)
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 1)))
        root = _plugin_root(fake.options)
        assert fake.options.plugins == [{"type": "local", "path": str(root)}]
        assert root.name == "launchdarkly"
        assert manifests == [False]

    async def test_enables_only_the_variation_skills_by_qualified_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("b", 1), _raw("a", 1), _raw("unattached", 1))
        fake = _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("b", 1), ("a", 1)))
        assert fake.options.skills == ["launchdarkly:b", "launchdarkly:a"]

    async def test_skill_joins_the_builtin_set_once_last(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1))
        fake = _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 1)))
        assert fake.options.tools == ["Skill"]
        assert "Skill" not in fake.options.allowed_tools

        with_read = _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", with_read)
        await _run(handler, path, _config(("a", 1)), {"read": _read_stub()})
        assert with_read.options.tools == ["Read", "Skill"]
        assert "Skill" not in with_read.options.allowed_tools

    async def test_no_config_parameter_reaches_the_handler_owned_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        _store(_raw("a", 1))
        cfg = _config(
            ("a", 1),
            model={
                "name": "claude-opus-4-5",
                "parameters": {
                    "plugins": [{"type": "local", "path": "/from-config"}],
                    "skills": "all",
                    "tools": ["Bash"],
                    "setting_sources": ["user"],
                },
            },
        )
        fake = _FakeQuery()
        monkeypatch.setattr(handler_mod, "query", fake)
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, cfg)
        assert fake.options.plugins == [
            {"type": "local", "path": str(_plugin_root(fake.options))}
        ]
        assert fake.options.skills == ["launchdarkly:a"]
        assert fake.options.tools == ["Skill"]
        assert fake.options.setting_sources == []

    async def test_a_skill_call_gets_an_execute_tool_span(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
    ) -> None:
        """The hooks are installed whenever ``Skill`` is in ``tools``, even with no other tool."""
        _store(_raw("a", 1))

        async def during(kwargs: dict[str, Any]) -> None:
            hooks = kwargs["options"].hooks
            event = {
                "tool_name": "Skill",
                "tool_use_id": "tu-skill",
                "tool_input": {"skill": "launchdarkly:a"},
                "session_id": "sess-1",
            }
            await hooks["PreToolUse"][0].hooks[0](
                {**event, "hook_event_name": "PreToolUse"}, "tu-skill", None
            )
            await hooks["PostToolUse"][0].hooks[0](
                {**event, "hook_event_name": "PostToolUse", "tool_response": "ok"},
                "tu-skill",
                None,
            )

        monkeypatch.setattr(handler_mod, "query", _FakeQuery(during=during))
        handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
        await _run(handler, path, _config(("a", 1)))

        finished = list(_exporter.get_finished_spans())
        root = next(s for s in finished if s.name == "invoke_agent")
        skill = next(s for s in finished if s.name == "execute_tool Skill")
        assert skill.attributes["gen_ai.tool.name"] == "Skill"
        assert skill.parent is not None
        assert skill.parent.span_id == root.context.span_id


# ---------------------------------------------------------------------------
# Through config(): the normal tracking path
# ---------------------------------------------------------------------------

CONTEXT = {"kind": "user", "key": "user-1"}


@pytest.fixture
async def ld_client() -> Any:
    await shutdown()
    client = MagicMock()
    client.track = MagicMock()
    client.flush = MagicMock()
    client.close = MagicMock()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=client)
    yield client
    await shutdown()


def _variation(client: MagicMock, *refs: tuple[str, int]) -> None:
    async def variation(*_args: Any) -> dict[str, Any]:
        return {
            **_config(*refs),
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "agent",
            },
        }

    client.variation = variation


def _tracked(client: MagicMock) -> list[str]:
    return [entry.args[0] for entry in client.track.call_args_list]


async def test_a_fail_closed_throw_is_tracked_as_a_generation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ld_client: MagicMock
) -> None:
    _variation(ld_client, ("a", 1))
    fake = _FakeQuery()
    monkeypatch.setattr(handler_mod, "query", fake)
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not started"):
        await config(key="triage", handler=handler).invoke("q", CONTEXT)
    assert fake.calls == []
    assert "$ld:ai:generation:error" in _tracked(ld_client)


async def test_a_skill_call_reports_no_tool_call_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ld_client: MagicMock
) -> None:
    """``Skill`` is not an AI Config tool, so its call gets a span and no ``$ld:ai:tool_call``."""
    _variation(ld_client, ("a", 1))
    _store(_raw("a", 1))

    async def during(kwargs: dict[str, Any]) -> None:
        hooks = kwargs["options"].hooks
        await hooks["PreToolUse"][0].hooks[0](
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Skill",
                "tool_use_id": "tu-skill",
                "tool_input": {"skill": "launchdarkly:a"},
            },
            "tu-skill",
            None,
        )

    monkeypatch.setattr(handler_mod, "query", _FakeQuery(during=during))
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    await config(key="triage", handler=handler).invoke("q", CONTEXT)
    tracked = _tracked(ld_client)
    assert "$ld:ai:generation:success" in tracked
    assert "$ld:ai:tool_call" not in tracked


# ---------------------------------------------------------------------------
# Not covered: the core handler never materializes skills
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS)
async def test_the_core_handler_ignores_a_variation_skills(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    _store(_raw("a", 1))
    fake = _FakeQuery()
    monkeypatch.setattr(handler_mod, "query", fake)
    await _run(create_claude_agents_handler(), path, _config(("a", 1)))
    assert fake.options.plugins == []
    assert fake.options.skills is None
    assert fake.options.tools == []
