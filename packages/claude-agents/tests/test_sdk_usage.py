"""Tests for SDK helper usage events — claude-agents."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage

import launchdarkly_ai_claude_agents.handler as handler_mod
import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_claude_agents import (
    claude_agents,
    claude_graph,
    create_claude_agents_handler,
    create_claude_agents_skills_handler,
    to_claude_agents,
)
from launchdarkly_ai_claude_agents._version import PACKAGE_NAME
from launchdarkly_ai_claude_agents._version import __version__ as PACKAGE_VERSION
from launchdarkly_ai_server import (
    SDK_INFO_CONTEXT,
    InMemorySkillStore,
    __version__,
    init_client,
    set_skill_store,
    shutdown,
)

CONTEXT = {"kind": "user", "key": "user-1"}


def _fake() -> MagicMock:
    client = MagicMock()
    client.track = MagicMock()
    client.variation = AsyncMock(return_value=None)
    client.flush = AsyncMock()
    client.close = AsyncMock()
    return client


async def _settle(value: Any) -> None:
    if hasattr(value, "__await__"):
        try:
            await value
        except Exception:
            return


@pytest.fixture
async def client() -> Any:
    await shutdown()
    installed = _fake()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=installed)
    installed.track.reset_mock()
    yield installed
    await shutdown()


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        ("claude-agents.claudeAgents", lambda: claude_agents("k", "q", CONTEXT)),
        ("claude-agents.createClaudeAgentsHandler", create_claude_agents_handler),
        ("claude-agents.claudeGraph", lambda: claude_graph("k")),
        ("claude-agents.toClaudeAgents", lambda: to_claude_agents(None)),
        # Its own ID, and not also the core factory's.
        (
            "claude-agents.createClaudeAgentsSkillsHandler",
            create_claude_agents_skills_handler,
        ),
    ],
)
async def test_helper_sends_one_usage_event(
    client: MagicMock, helper: str, call: Any
) -> None:
    try:
        await _settle(call())
    except Exception:
        pass
    calls = [
        entry
        for entry in client.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]
    assert [entry.args[2].get("helper") for entry in calls] == [helper]
    assert calls[0].args[1] == SDK_INFO_CONTEXT
    assert calls[0].args[2] == {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": helper,
        "helperPackageName": PACKAGE_NAME,
        "helperPackageVersion": PACKAGE_VERSION,
    }
    assert calls[0].args[3] == 1


def _helpers(client: MagicMock) -> list[str]:
    return [
        entry.args[2].get("helper")
        for entry in client.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]


async def _drain(stream: Any) -> None:
    try:
        async for _ in stream:
            pass
    except Exception:
        return


async def test_graph_wrapper_invoke_and_stream_report_only_the_wrapper(
    client: MagicMock,
) -> None:
    """The returned graph's methods are not separate ``client.graph.*`` calls."""
    instance = claude_graph("k")
    await _settle(instance.invoke("q", CONTEXT))
    await _drain(instance.stream("q", CONTEXT))
    assert _helpers(client) == ["claude-agents.claudeGraph"]


async def test_native_adapter_invoke_reports_only_the_adapter(
    client: MagicMock,
) -> None:
    async def _definition() -> Any:
        return SimpleNamespace(enabled=False, key="k")

    runner = to_claude_agents(_definition(), {"context": CONTEXT})
    with pytest.raises(ValueError, match="disabled"):
        await runner.invoke("q")
    assert _helpers(client) == ["claude-agents.toClaudeAgents"]


_SKILL_BODY = "---\nname: a\n---\nSkill a.\n"
_SKILLS_CONFIG: dict[str, Any] = {
    "model": {"name": "claude-opus-4-5"},
    "provider": {"name": "Anthropic"},
    "instructions": "You are helpful.",
    "skills": [{"key": "a", "version": 1}],
}


async def _fake_query(**_kwargs: Any) -> Any:
    yield ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="sess-1",
        usage={"input_tokens": 3, "output_tokens": 2},
        result="done",
    )


@pytest.mark.parametrize("path", ["invoke", "stream"])
async def test_skills_run_reports_only_the_factory(
    client: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    """A run with skills reads the variation's references and writes the skills on the
    customer's behalf. They never called ``skill_refs`` or ``write_skills``, so neither
    ``client.skillRefs`` nor ``client.writeSkills`` is sent for them."""
    store = InMemorySkillStore()
    store.put(
        {
            "key": "a",
            "version": 1,
            "content": _SKILL_BODY,
            "contentHash": hashlib.sha256(_SKILL_BODY.encode()).hexdigest(),
        }
    )
    set_skill_store(store)
    monkeypatch.setattr(handler_mod, "query", _fake_query)
    handler = create_claude_agents_skills_handler(skills_temp_dir=tmp_path)
    if path == "invoke":
        await handler(_SKILLS_CONFIG, "q", {})
    else:
        await _drain(await handler.stream(_SKILLS_CONFIG, "q", {}))
    assert _helpers(client) == ["claude-agents.createClaudeAgentsSkillsHandler"]


async def test_held_helper_flushes_with_its_package() -> None:
    """A helper called before init keeps its package fields until the flush."""
    await shutdown()
    lifecycle_module._reset_for_testing()
    try:
        create_claude_agents_handler()
    except Exception:
        pass  # The factory may need credentials; it reports first.
    installed = _fake()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=installed)
    calls = [
        entry.args[2]
        for entry in installed.track.call_args_list
        if entry.args[0] == "$ld:ai:sdk:usage"
    ]
    assert calls == [
        {
            "aiSdkName": "launchdarkly-ai-server",
            "aiSdkVersion": __version__,
            "aiSdkLanguage": "python",
            "helper": "claude-agents.createClaudeAgentsHandler",
            "helperPackageName": PACKAGE_NAME,
            "helperPackageVersion": PACKAGE_VERSION,
        }
    ]
    await shutdown()
