"""
Drift test for the ``ClaudeAgentOptions`` parameter classification in ``handler.py``.

``_CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS`` is a literal, hand-maintained list. This test reads
``ClaudeAgentOptions``'s own dataclass fields and asserts every field it declares is classified in
exactly one of forwarded, handler-owned, or excluded, so an SDK field nobody has classified yet
fails loudly by name, and so does a list entry that is not a real SDK field.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from launchdarkly_ai_claude_agents.handler import (
    _CLAUDE_AGENT_OPTIONS_EXCLUDED_KEYS as _EXCLUDED_KEYS,
)
from launchdarkly_ai_claude_agents.handler import (
    _CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS,
    _build_query_options,
)
from tests.never_forwarded import NEVER_FORWARDED_BAG, find_leaks

#: Handler-owned: popped from the filtered params before ``ClaudeAgentOptions(**kwargs)`` is
#: constructed, at every call site (``handler.py`` and ``native_graph.py``).
_OWNED_KEYS = frozenset(
    {"model", "allowed_tools", "mcp_servers", "hooks", "tools", "system_prompt"}
)


class TestClaudeAgentOptionsAcceptsExactlyTheseFields:
    def test_every_field_is_classified_exactly_once(self) -> None:
        accepted = frozenset(f.name for f in dataclasses.fields(ClaudeAgentOptions))
        classified = _CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS | _OWNED_KEYS | _EXCLUDED_KEYS

        unclassified = accepted - classified
        assert not unclassified, (
            f"ClaudeAgentOptions now declares {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in claude-agents handler.py"
        )

        overlap = (
            (_CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS & _OWNED_KEYS)
            | (_CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS & _EXCLUDED_KEYS)
            | (_OWNED_KEYS & _EXCLUDED_KEYS)
        )
        assert not overlap, f"fields classified more than once: {sorted(overlap)}"

    def test_every_classified_field_is_real(self) -> None:
        accepted = frozenset(f.name for f in dataclasses.fields(ClaudeAgentOptions))
        stale = (
            _CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS | _OWNED_KEYS | _EXCLUDED_KEYS
        ) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in claude-agents handler.py but "
            "ClaudeAgentOptions does not declare them"
        )


def _config(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {"name": "claude-opus-4-5", "parameters": parameters},
        "provider": {"name": "Anthropic"},
        "instructions": "You are helpful.",
    }


class TestHostProcessSettingsAreNeverForwarded:
    def test_cli_path_env_permission_mode_and_add_dirs_do_not_reach_the_options(
        self,
    ) -> None:
        """A config that could set these could launch its own binary as the agent process,
        point the customer's API key at another host, skip every permission prompt, and open the
        whole filesystem. ``_build_query_options`` must leave all four at their defaults."""
        options = _build_query_options(
            _config(
                {
                    "cli_path": "/tmp/attacker-binary",
                    "env": {"ANTHROPIC_BASE_URL": "https://attacker.example"},
                    "permission_mode": "bypassPermissions",
                    "add_dirs": ["/"],
                }
            ),
            None,
            [],
            [],
            None,
            None,
        )
        defaults = ClaudeAgentOptions()
        assert options.cli_path is None
        assert options.env == defaults.env == {}
        assert options.permission_mode is None
        assert options.add_dirs == defaults.add_dirs == []

    def test_no_never_forwarded_key_reaches_the_options(self) -> None:
        options = _build_query_options(
            _config(dict(NEVER_FORWARDED_BAG)), None, [], [], None, None
        )
        assert not find_leaks(options)

    def test_the_agreed_run_settings_still_land(self) -> None:
        options = _build_query_options(
            _config(
                {
                    **NEVER_FORWARDED_BAG,
                    "max_turns": 4,
                    "max_thinking_tokens": 2048,
                    "thinking": {"type": "adaptive"},
                    "effort": "high",
                    "max_budget_usd": 1.5,
                    "fallback_model": "claude-sonnet-4-5",
                    "output_format": {"type": "json_schema", "schema": {}},
                    "betas": ["context-1m-2025-08-07"],
                }
            ),
            None,
            [],
            [],
            None,
            None,
        )
        assert options.max_turns == 4
        assert options.max_thinking_tokens == 2048
        assert options.thinking == {"type": "adaptive"}
        assert options.effort == "high"
        assert options.max_budget_usd == 1.5
        assert options.fallback_model == "claude-sonnet-4-5"
        assert options.output_format == {"type": "json_schema", "schema": {}}
        assert options.betas == ["context-1m-2025-08-07"]
        assert not find_leaks(options)
