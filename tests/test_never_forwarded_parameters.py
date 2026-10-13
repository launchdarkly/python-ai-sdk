"""Cross-handler invariant: no handler forwards a never-forwarded key from ``model.parameters``.

Each handler forwards only its own literal allowlist of model and run settings. This checks every
one of those lists against the shared list of keys that must never be forwarded
(``never_forwarded.py``). Each package's own tests check its real call sites with the same keys.

Every handler module in ``packages/*/src/launchdarkly_ai_*`` must be listed in ``FORWARDED_LISTS``,
so a new handler package fails here until it declares an allowlist.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from tests.never_forwarded import (
    NEVER_FORWARDED_BAG,
    NEVER_FORWARDED_KEYS,
    find_leaks,
    leaked,
)

#: Handler module to the forwarded lists it defines. The lists are named rather than discovered so
#: that a list renamed or added without being checked here fails below by name; the modules are
#: checked against ``discover_handler_modules`` so a new handler module must be added here.
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

_PACKAGES_DIR = Path(__file__).resolve().parent.parent / "packages"


def _module_name(path: Path, src: Path) -> str:
    """``src/launchdarkly_ai_x/handler.py`` to ``launchdarkly_ai_x.handler``."""
    return ".".join(path.relative_to(src).with_suffix("").parts).removesuffix(
        ".__init__"
    )


def _provides_handler(tree: ast.Module) -> bool:
    """A function returning ``ProviderHandler``, or a ``create_handler``/``ProviderHandler`` call."""
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.returns is not None
            and ast.unparse(node.returns).split(".")[-1] == "ProviderHandler"
        ):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name in {"create_handler", "ProviderHandler"}:
                return True
    return False


def _defines_provider_handler(trees: dict[Path, ast.Module]) -> bool:
    return any(
        isinstance(node, ast.ClassDef) and node.name == "ProviderHandler"
        for tree in trees.values()
        for node in ast.walk(tree)
    )


def discover_handler_modules(packages_dir: Path = _PACKAGES_DIR) -> set[str]:
    """Modules under ``packages/*/src/launchdarkly_ai_*`` that build a ``ProviderHandler``.

    Read from source rather than imported, so a package whose provider SDK isn't installed is
    still found. The package that defines ``ProviderHandler`` itself is the client core, not a
    handler, and is skipped along with packages that build no handler (the meta-package).
    """
    found: set[str] = set()
    for package in sorted(packages_dir.glob("*/src/launchdarkly_ai_*")):
        trees = {
            path: ast.parse(path.read_text()) for path in sorted(package.rglob("*.py"))
        }
        if _defines_provider_handler(trees):
            continue
        found |= {
            _module_name(path, package.parent)
            for path, tree in trees.items()
            if _provides_handler(tree)
        }
    return found


def test_every_handler_module_has_forwarded_lists() -> None:
    discovered = discover_handler_modules()
    assert discovered, f"found no handler modules under {_PACKAGES_DIR}"
    assert discovered <= FORWARDED_LISTS.keys(), (
        f"handler modules with no entry in FORWARDED_LISTS: {sorted(discovered - FORWARDED_LISTS.keys())}"
    )
    assert FORWARDED_LISTS.keys() <= discovered, (
        f"FORWARDED_LISTS names modules that build no handler: {sorted(FORWARDED_LISTS.keys() - discovered)}"
    )


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
