"""Cross-handler invariant: no handler forwards a never-forwarded key from ``model.parameters``.

Each handler forwards only its own literal allowlist of model and run settings. This checks every
one of those lists, in all six packages, against the shared list of keys that must never be
forwarded (``never_forwarded.py``). Each package's own tests check its real call sites with the
same keys.
"""

from __future__ import annotations

import importlib

import pytest

from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from tests.never_forwarded import (
    NEVER_FORWARDED_BAG,
    NEVER_FORWARDED_KEYS,
    find_leaks,
    leaked,
)

#: Handler module to the forwarded lists it defines. Listed rather than discovered so that a list
#: renamed or added without being checked here fails below by name.
FORWARDED_LISTS: dict[str, set[str]] = {
    "launchdarkly_ai_claude_agents.handler": {"_CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS"},
    "launchdarkly_ai_claude_messages.handler": {"_MESSAGES_FORWARDED_KEYS"},
    "launchdarkly_ai_openai_agents.handler": {"_MODEL_SETTINGS_FORWARDED_KEYS"},
    "launchdarkly_ai_openai_messages.handler": {"_RESPONSES_FORWARDED_KEYS"},
    "launchdarkly_ai_langchain_agents.handler": {
        "_CHAT_OPENAI_FORWARDED_KEYS",
        "_CHAT_ANTHROPIC_FORWARDED_KEYS",
        "_CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS",
    },
    "launchdarkly_ai_langchain_messages.handler": {
        "_CHAT_OPENAI_FORWARDED_KEYS",
        "_CHAT_ANTHROPIC_FORWARDED_KEYS",
        "_CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS",
    },
}

_CASES = [
    (module, name)
    for module, names in FORWARDED_LISTS.items()
    for name in sorted(names)
]


@pytest.mark.parametrize("module", sorted(FORWARDED_LISTS))
def test_every_forwarded_list_is_checked(module: str) -> None:
    mod = importlib.import_module(module)
    defined = {name for name in vars(mod) if name.endswith("_FORWARDED_KEYS")}
    assert defined == FORWARDED_LISTS[module], (
        f"{module} defines forwarded lists {sorted(defined)}; "
        f"this test checks {sorted(FORWARDED_LISTS[module])}"
    )


@pytest.mark.parametrize(("module", "name"), _CASES)
def test_no_forwarded_list_holds_a_never_forwarded_key(module: str, name: str) -> None:
    forwarded: frozenset[str] = getattr(importlib.import_module(module), name)
    assert not forwarded & NEVER_FORWARDED_KEYS, (
        f"{module}.{name} forwards {sorted(forwarded & NEVER_FORWARDED_KEYS)}"
    )
    assert select_forwarded_parameters(NEVER_FORWARDED_BAG, forwarded) == {}


def test_find_leaks_finds_a_marker_at_any_depth() -> None:
    class _Holder:
        def __init__(self) -> None:
            self.options = {"nested": [("x", leaked("cli_path"))]}

    assert find_leaks({"a": _Holder(), "b": leaked("env")}) == {"cli_path", "env"}
    assert find_leaks({"a": "fine", "b": [1, 2]}) == set()
