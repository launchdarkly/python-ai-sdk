"""
Drift test for the ``ClaudeAgentOptions`` parameter classification in ``handler.py``.

``_CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS`` is a literal, hand-maintained list. This test reads
``ClaudeAgentOptions``'s own dataclass fields and asserts every field it declares is classified in
exactly one of forwarded, handler-owned, or excluded, so an SDK field nobody has classified yet
fails loudly by name, and so does a list entry that is not a real SDK field.
"""

from __future__ import annotations

import dataclasses

from claude_agent_sdk import ClaudeAgentOptions

from launchdarkly_ai_claude_agents.handler import (
    _CLAUDE_AGENT_OPTIONS_EXCLUDED_KEYS as _EXCLUDED_KEYS,
)
from launchdarkly_ai_claude_agents.handler import _CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS

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
