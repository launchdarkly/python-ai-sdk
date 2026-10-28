"""Install each built wheel into an empty virtualenv and check it is usable there.

Usage: uv run python scripts/check_packaging.py <dist-dir>

For each package, in a fresh virtualenv with the package alone, no extras:

1. every public module imports, and every name in its ``__all__`` resolves;
2. every module its source imports unconditionally comes from a distribution the
   package declares. Installed is not enough: a module another dependency happens to
   pull in still fails. That includes lazy imports (``importlib.import_module("openai")``
   inside a factory), which step 1 never reaches. Imports under ``except ImportError``
   or ``TYPE_CHECKING`` are optional.

Installing with no extras is what proves the optional dependencies stay optional: core
loads without ``launchdarkly-server-sdk`` and the OpenTelemetry SDK.

Then each declared extra gets a venv of its own, installed as ``<package>[<extra>]``,
and the modules it promises in ``EXTRA_IMPORTS`` must import there. An extra missing
from ``EXTRA_IMPORTS`` fails the check, so a new extra cannot go unverified.

Catches what the workspace test run cannot, because the workspace installs every
package's dependencies at once: an import the package does not declare, a declared range
that will not install, an extra that does not deliver what it promises.

Internal dependencies (``launchdarkly-ai-*``) resolve to the wheels in <dist-dir>, not
PyPI: between releases a handler may require a core version only this commit builds.
Third-party dependencies resolve from PyPI against the declared ranges.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What each declared extra must make importable.
_OTEL = ["opentelemetry.sdk", "opentelemetry.exporter.otlp.proto.http"]
EXTRA_IMPORTS = {
    ("launchdarkly-ai-server", "ld"): ["ldclient"],
    ("launchdarkly-ai-server", "otel"): _OTEL,
    ("launchdarkly-ai-python", "ld"): ["ldclient"],
    ("launchdarkly-ai-python", "otel"): _OTEL,
    ("launchdarkly-ai-langchain-agents", "anthropic"): ["langchain_anthropic"],
    ("launchdarkly-ai-langchain-agents", "aws"): ["langchain_aws"],
    ("launchdarkly-ai-langchain-messages", "openai"): ["langchain_openai"],
    ("launchdarkly-ai-langchain-messages", "anthropic"): ["langchain_anthropic"],
    ("launchdarkly-ai-langchain-messages", "aws"): ["langchain_aws"],
}

_REQUIREMENT = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")

_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

PUBLIC_CHECK = """
import importlib, sys
for name in sys.argv[1:]:
    module = importlib.import_module(name)
    for attr in getattr(module, "__all__", []):
        getattr(module, attr)
"""

# Third-party modules an extra installs: their ``__all__`` is not our contract, and may
# name lazy attributes that need further extras (``langchain_aws.ChatAnthropicBedrock``).
IMPORT_CHECK = """
import importlib, sys
for name in sys.argv[1:]:
    importlib.import_module(name)
"""

# Prints which installed distributions provide each module, so the caller can compare
# them with what the package declares. ``null`` means the module is not installed.
PROVIDERS = """
import importlib.metadata, importlib.util, json, sys
dists = importlib.metadata.packages_distributions()
print(json.dumps({
    name: dists.get(name, []) if importlib.util.find_spec(name) else None
    for name in sys.argv[1:]
}))
"""


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def undeclared_imports(
    providers: dict[str, list[str] | None], declared: set[str]
) -> list[str]:
    """Each import that no declared distribution provides, with the reason."""
    problems = []
    for module, dists in sorted(providers.items()):
        if dists is None:
            problems.append(f"{module} (not installed)")
        elif not {_normalize(d) for d in dists} & declared:
            found = ", ".join(sorted(dists)) or "no distribution"
            problems.append(f"{module} (from {found}, not declared)")
    return problems


@dataclass(frozen=True)
class Package:
    name: str
    public: list[str]
    """Modules whose ``__all__`` is the public contract."""
    imports: list[str]
    """Top-level modules the source imports unconditionally."""
    dependencies: list[str] = field(default_factory=list)
    optional: dict[str, list[str]] = field(default_factory=dict)
    """``[project.optional-dependencies]``: extra name to its requirements."""

    def override_extras(self, dep: str, extras: tuple[str, ...]) -> str:
        """The extras this install requests on ``dep``, as ``[a,b]`` or ``""``.

        A uv override replaces every requirement on its name, extras included, so the
        override for an internal dependency must restate them.
        """
        requested: set[str] = set()
        for req in [*self.dependencies, *(r for e in extras for r in self.optional[e])]:
            m = _REQUIREMENT.match(req)
            if m and m.group(1) == dep and m.group(2):
                requested.update(x.strip() for x in m.group(2).split(","))
        return f"[{','.join(sorted(requested))}]" if requested else ""

    def declared(self) -> set[str]:
        """The package itself and its required dependencies, normalized."""
        names = {_normalize(self.name)}
        for req in self.dependencies:
            m = _REQUIREMENT.match(req)
            if m:
                names.add(_normalize(m.group(1)))
        return names


@dataclass(frozen=True)
class Run:
    package: Package
    extras: tuple[str, ...]
    """Extras to install: none for the base run, one declared extra otherwise."""
    base: bool

    @property
    def label(self) -> str:
        return self.package.name + (f"[{','.join(self.extras)}]" if self.extras else "")


def runs(found: list[Package]) -> list[Run]:
    """The base install of every package, then one run per declared extra."""
    unmapped = [
        f"{p.name}[{e}]"
        for p in found
        for e in p.optional
        if (p.name, e) not in EXTRA_IMPORTS
    ]
    if unmapped:
        raise SystemExit(f"no EXTRA_IMPORTS entry for: {', '.join(unmapped)}")
    base = [Run(p, (), base=True) for p in found]
    return base + [Run(p, (e,), base=False) for p in found for e in p.optional]


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
        project = config["project"]
        found.append(
            Package(
                name=project["name"],
                public=sorted({*top_level, *reported}),
                imports=source_imports(pyproject.parent / "src"),
                dependencies=project.get("dependencies", []),
                optional=project.get("optional-dependencies", {}),
            )
        )
    return found


def wheel_for(dist: Path, name: str) -> Path:
    matches = list(dist.glob(f"{name.replace('-', '_')}-*.whl"))
    if len(matches) != 1:
        raise SystemExit(f"{name}: expected one wheel in {dist}, found {len(matches)}")
    return matches[0]


def check(run: Run, dist: Path, internal: list[str], tmp: Path) -> tuple[bool, str]:
    """Returns whether the run passed, and its output, buffered so parallel logs stay whole."""
    package = run.package
    name = package.name
    slug = run.label.replace("[", "-").replace("]", "").replace(",", "-")
    venv = tmp / slug
    python = venv / "bin" / "python"
    extras = f"[{','.join(run.extras)}]" if run.extras else ""
    requirement = f"{name}{extras} @ {wheel_for(dist, name).as_uri()}"
    # Overriding the package being installed would drop its own extras.
    overrides = tmp / f"{slug}.overrides.txt"
    overrides.write_text(
        "".join(
            f"{dep}{package.override_extras(dep, run.extras)} @ {wheel_for(dist, dep).as_uri()}\n"
            for dep in internal
            if dep != name
        )
    )
    extra_imports = [m for e in run.extras for m in EXTRA_IMPORTS[(name, e)]]
    install = [
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
    ]
    if not run.base:
        # The base run already checked the package itself; this one checks the extra.
        steps = [*install, [str(python), "-I", "-c", IMPORT_CHECK, *extra_imports]]
        return _run_steps(
            run.label, steps, f"ok   {run.label}: {', '.join(extra_imports)}"
        )
    steps = [
        *install,
        # -I: import from the installed wheel, never the source tree.
        [str(python), "-I", "-c", PUBLIC_CHECK, *package.public],
    ]
    ok, output = _run_steps(run.label, steps, "")
    if not ok:
        return ok, output
    result = subprocess.run(
        [str(python), "-I", "-c", PROVIDERS, *package.imports],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, output + result.stderr + f"FAIL {run.label}\n"
    problems = undeclared_imports(json.loads(result.stdout), package.declared())
    if problems:
        return False, (
            output
            + f"imported but not declared: {'; '.join(problems)}\n"
            + f"FAIL {run.label}\n"
        )
    return True, output + f"ok   {run.label}: {', '.join(package.public)}\n"


def _run_steps(label: str, steps: list[list[str]], ok: str) -> tuple[bool, str]:
    output = []
    for step in steps:
        result = subprocess.run(step, capture_output=True, text=True)
        output += [result.stdout, result.stderr]
        if result.returncode != 0:
            return False, "".join(output) + f"FAIL {label}\n"
    return True, "".join(output) + ok + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    dist = Path(sys.argv[1]).resolve()
    found = packages()
    internal = [p.name for p in found]
    planned = runs(found)
    failed = []
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool,
    ):
        results = pool.map(lambda r: check(r, dist, internal, Path(tmp_dir)), planned)
        for run, (ok, output) in zip(planned, results, strict=True):
            print(output, end="", flush=True)
            if not ok:
                failed.append(run.label)
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
