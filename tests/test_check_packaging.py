"""Tests the import scan in scripts/check_packaging.py (TESTING.md §5.3).

The install half needs built wheels and PyPI, so CI runs it as ``make check-packaging``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

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
