"""Install each built wheel into an empty virtualenv and check it is usable there.

Usage: uv run python scripts/check_packaging.py <dist-dir>

For each package, in a fresh virtualenv with only what its install command pulls in:

1. every public module imports, and every name in its ``__all__`` resolves;
2. every module its source imports unconditionally is installed. That includes lazy
   imports (``importlib.import_module("openai")`` inside a factory), which step 1 never
   reaches. Imports under ``except ImportError`` or ``TYPE_CHECKING`` are optional.

Catches what the workspace test run cannot, because the workspace installs every
package's dependencies at once: an import the package does not declare, a declared range
that will not install, an extra that does not deliver what it promises.

Internal dependencies (``launchdarkly-ai-*``) resolve to the wheels in <dist-dir>, not
PyPI: between releases a handler may require a core version only this commit builds.
Third-party dependencies resolve from PyPI against the declared ranges.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The extras the README's install command uses, and what each must make importable.
EXTRAS = {"launchdarkly-ai-server": "[ld]", "launchdarkly-ai-python": "[ld]"}
EXTRA_IMPORTS = {
    "launchdarkly-ai-server": ["ldclient"],
    "launchdarkly-ai-python": ["ldclient"],
}
# A uv override replaces every requirement on its name, extras included, so restate
# the extras a package's own extra requests on an internal dependency.
OVERRIDE_EXTRAS = {("launchdarkly-ai-python", "launchdarkly-ai-server"): "[ld]"}

_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

PUBLIC_CHECK = """
import importlib, sys
for name in sys.argv[1:]:
    module = importlib.import_module(name)
    for attr in getattr(module, "__all__", []):
        getattr(module, attr)
"""

INSTALLED_CHECK = """
import importlib.util, sys
missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("imported but not installed: " + ", ".join(missing))
"""


@dataclass(frozen=True)
class Package:
    name: str
    public: list[str]
    """Modules whose ``__all__`` is the public contract."""
    imports: list[str]
    """Top-level modules the source imports unconditionally."""


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in _IMPORT_ERRORS for n in names)


class _Imports(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found: set[str] = set()
        self._optional = 0

    def visit_Try(self, node: ast.Try) -> None:
        guarded = any(_catches_import_error(h) for h in node.handlers)
        self._optional += guarded
        for stmt in node.body:
            self.visit(stmt)
        self._optional -= guarded
        for part in (*node.handlers, *node.orelse, *node.finalbody):
            self.visit(part)

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        ):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def _add(self, module: str) -> None:
        if not self._optional:
            self.found.add(module.split(".")[0])

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self._add(node.module)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
        if (
            name == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            if isinstance(node.args[0].value, str):
                self._add(node.args[0].value)
        self.generic_visit(node)


def source_imports(src: Path) -> list[str]:
    visitor = _Imports()
    for path in src.rglob("*.py"):
        visitor.visit(ast.parse(path.read_text(), str(path)))
    return sorted(visitor.found - set(sys.stdlib_module_names) - {"__future__"})


def packages() -> list[Package]:
    found = []
    for pyproject in sorted(ROOT.glob("packages/*/pyproject.toml")):
        config = tomllib.loads(pyproject.read_text())
        top_level = [
            Path(p).name
            for p in config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        ]
        reported = [p.stem for p in (pyproject.parent / "api-report").glob("*.txt")]
        found.append(
            Package(
                name=config["project"]["name"],
                public=sorted({*top_level, *reported}),
                imports=source_imports(pyproject.parent / "src"),
            )
        )
    return found


def wheel_for(dist: Path, name: str) -> Path:
    matches = list(dist.glob(f"{name.replace('-', '_')}-*.whl"))
    if len(matches) != 1:
        raise SystemExit(f"{name}: expected one wheel in {dist}, found {len(matches)}")
    return matches[0]


def check(package: Package, dist: Path, internal: list[str], tmp: Path) -> bool:
    name = package.name
    venv = tmp / name
    python = venv / "bin" / "python"
    requirement = f"{name}{EXTRAS.get(name, '')} @ {wheel_for(dist, name).as_uri()}"
    # Overriding the package being installed would drop its own extras.
    overrides = tmp / f"{name}.overrides.txt"
    overrides.write_text(
        "".join(
            f"{dep}{OVERRIDE_EXTRAS.get((name, dep), '')} @ {wheel_for(dist, dep).as_uri()}\n"
            for dep in internal
            if dep != name
        )
    )
    steps = [
        ["uv", "venv", "--quiet", "--python", "3.12", str(venv)],
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            "--override",
            str(overrides),
            requirement,
        ],
        # -I: import from the installed wheel, never the source tree.
        [str(python), "-I", "-c", PUBLIC_CHECK, *package.public],
        [
            str(python),
            "-I",
            "-c",
            INSTALLED_CHECK,
            *package.imports,
            *EXTRA_IMPORTS.get(name, []),
        ],
    ]
    for step in steps:
        if subprocess.run(step).returncode != 0:
            print(f"FAIL {name}", file=sys.stderr)
            return False
    print(f"ok   {name}: {', '.join(package.public)}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    dist = Path(sys.argv[1]).resolve()
    found = packages()
    with tempfile.TemporaryDirectory() as tmp_dir:
        internal = [p.name for p in found]
        failed = [p.name for p in found if not check(p, dist, internal, Path(tmp_dir))]
    if failed:
        print(f"{len(failed)} package(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
