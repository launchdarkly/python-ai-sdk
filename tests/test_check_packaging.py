"""Tests the import scan in scripts/check_packaging.py (TESTING.md §5.3).

The install half needs built wheels and PyPI, so CI runs it as ``make check-packaging``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts import check_packaging


def _scan(tmp_path: Path, source: str) -> list[str]:
    (tmp_path / "mod.py").write_text(textwrap.dedent(source))
    return check_packaging.source_imports(tmp_path)


def test_reads_module_level_and_lazy_imports(tmp_path: Path) -> None:
    source = """
        import pkg_a
        from pkg_b.sub import thing

        def factory():
            import importlib
            return importlib.import_module("pkg_c.models")
    """
    assert _scan(tmp_path, source) == ["pkg_a", "pkg_b", "pkg_c"]


def test_an_import_guarded_by_import_error_is_optional(tmp_path: Path) -> None:
    source = """
        import importlib

        try:
            import optional_a
        except ImportError:
            optional_a = None

        def load():
            try:
                return importlib.import_module("optional_b")
            except (ModuleNotFoundError, RuntimeError):
                raise ImportError("pip install optional-b")
    """
    assert _scan(tmp_path, source) == []


def test_the_handlers_of_a_guard_are_not_guarded(tmp_path: Path) -> None:
    source = """
        try:
            import optional_a
        except ImportError:
            import required_fallback
    """
    assert _scan(tmp_path, source) == ["required_fallback"]


def test_type_checking_imports_are_skipped(tmp_path: Path) -> None:
    source = """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import typing_only
        else:
            import runtime_only
    """
    assert _scan(tmp_path, source) == ["runtime_only"]


def test_stdlib_and_relative_imports_are_skipped(tmp_path: Path) -> None:
    source = """
        from __future__ import annotations
        import json
        from collections.abc import Callable
        from . import sibling
        from .sub import other
    """
    assert _scan(tmp_path, source) == []


def test_every_declared_extra_is_checked() -> None:
    planned = {run.label for run in check_packaging.runs(check_packaging.packages())}
    assert "launchdarkly-ai-langchain-messages[anthropic]" in planned
    assert "launchdarkly-ai-python[otel]" in planned
    assert "launchdarkly-ai-server[ld]" in planned


def test_core_is_checked_without_extras() -> None:
    planned = check_packaging.runs(check_packaging.packages())
    base = {run.package.name: run.extras for run in planned if run.base}
    assert base["launchdarkly-ai-server"] == ()
    assert base["launchdarkly-ai-python"] == ()


def test_an_undeclared_dynamic_import_fails(tmp_path: Path) -> None:
    source = """
        import importlib

        def factory():
            return importlib.import_module("langchain_anthropic")
    """
    package = check_packaging.Package(
        name="launchdarkly-ai-example",
        public=[],
        imports=_scan(tmp_path, source),
        dependencies=["langchain-core>=1.6"],
    )
    # Installed, because another dependency pulled it in, but not declared.
    providers = {"langchain_anthropic": ["langchain-anthropic"]}
    assert check_packaging.undeclared_imports(providers, package.declared()) == [
        "langchain_anthropic (from langchain-anthropic, not declared)"
    ]


def test_declared_and_missing_imports() -> None:
    declared = {"launchdarkly-ai-example", "typing-extensions"}
    providers: dict[str, list[str] | None] = {
        "launchdarkly_ai_example": ["launchdarkly_ai_example"],
        "typing_extensions": ["typing_extensions"],
        "absent": None,
    }
    assert check_packaging.undeclared_imports(providers, declared) == [
        "absent (not installed)"
    ]


def test_an_extra_without_expected_imports_fails() -> None:
    package = check_packaging.Package(
        name="launchdarkly-ai-example",
        public=[],
        imports=[],
        optional={"new": ["some-dist>=1"]},
    )
    with pytest.raises(SystemExit, match=r"launchdarkly-ai-example\[new\]"):
        check_packaging.runs([package])


def test_overrides_restate_extras_requested_on_an_internal_dependency() -> None:
    package = check_packaging.Package(
        name="launchdarkly-ai-python",
        public=[],
        imports=[],
        dependencies=["launchdarkly-ai-server>=0.4.1"],
        optional={
            "ld": ["launchdarkly-ai-server[ld]"],
            "otel": ["launchdarkly-ai-server[otel]"],
        },
    )
    assert package.override_extras("launchdarkly-ai-server", ("otel",)) == "[otel]"
    assert package.override_extras("launchdarkly-ai-server", ()) == ""
