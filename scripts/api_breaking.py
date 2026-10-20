"""Fail on an undeclared breaking change to the public API.

    uv run python -m scripts.api_breaking pr --base <ref> --title <PR title> [--labels a,b]
    uv run python -m scripts.api_breaking release --base <ref> [--labels a,b]

``pr`` compares each public module with ``--base``. A breaking change passes only if
the PR title declares it with ``!`` (``feat!: ...``, ``fix(client)!: ...``), which makes
release-please cut a major version (a minor one before 1.0.0), or if an owner applied
the ``api-compatible`` label because they judge the flagged change compatible.

``release`` runs on the release PR. For each package whose version changes, it
compares the package's public modules with its last release tag, and fails on a
breaking change unless the new version is a major one (a minor one before 1.0.0), or
an owner applied the ``api-compatible`` label to the release PR.

Only names in ``__all__`` count. Experimental modules are skipped: an experimental
feature may change in a minor release.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import griffe

from scripts.api_report import ROOT, public_modules, resolve

BREAKING_TITLE = re.compile(r"^[a-z]+(\([^)]*\))?!:")
COMPATIBLE_LABEL = "api-compatible"


def stable_modules(root: Path) -> dict[str, Path]:
    return {m: p for m, p in public_modules(root).items() if ".experimental." not in m}


def load(root: Path, ref: str | None, tops: set[str]) -> griffe.ModulesCollection:
    """Load the top-level packages ``tops`` from the working tree, or from ``ref``."""
    loader = griffe.GriffeLoader(
        search_paths=sorted(str(p) for p in root.glob("packages/*/src"))
    )
    relative = sorted(str(p.relative_to(root)) for p in root.glob("packages/*/src"))
    for top in sorted(tops):
        if ref is None:
            loader.load(top)
            continue
        try:
            griffe.load_git(
                top,
                ref=ref,
                repo=root,
                search_paths=relative,
                modules_collection=loader.modules_collection,
            )
        except (ImportError, griffe.GitError):  # the package did not exist yet
            pass
    loader.resolve_aliases(external=False, implicit=False)
    return loader.modules_collection


def compare(
    old: griffe.ModulesCollection, new: griffe.ModulesCollection, modules: list[str]
) -> list[str]:
    """Explain every breaking change to the ``__all__`` names of ``modules``.

    griffe compares the two modules' public members: with ``__all__`` defined, only
    its names, following re-exports to where they are defined. A name dropped from
    ``__all__`` while its object still exists is breaking too, and griffe does not
    see it, so it is checked here.
    """
    style = (
        griffe.ExplanationStyle.GITHUB
        if os.environ.get("GITHUB_ACTIONS")
        else griffe.ExplanationStyle.MARKDOWN
    )
    found: list[str] = []
    for module in modules:
        try:
            old_module = old.get_member(module)
        except KeyError:  # a new module breaks nothing
            continue
        try:
            new_module = new.get_member(module)
        except KeyError:
            found.append(f"{module}: the module was removed")
            continue
        dropped = sorted(
            set(map(str, old_module.exports or []))
            - set(map(str, new_module.exports or []))
        )
        found += [f"{module}.{name}: removed from __all__" for name in dropped]
        found += [
            b.explain(style)
            for b in griffe.find_breaking_changes(old_module, new_module)
            if not (
                b.kind is griffe.BreakageKind.OBJECT_REMOVED and b.obj.name in dropped
            )
            and ".experimental." not in f"{b.obj.path}."
        ]
        shared = sorted(
            set(map(str, old_module.exports or []))
            & set(map(str, new_module.exports or []))
        )
        for name in shared:
            found += returns_changed(
                f"{module}.{name}",
                resolve(old, old_module, name),
                resolve(new, new_module, name),
            )
    return found


def returns_changed(path: str, old: Any, new: Any) -> list[str]:
    """griffe does not flag a changed return type, which callers may not be able to handle."""
    pairs = [(path, old, new)]
    if old.is_class and new.is_class:
        pairs += [
            (f"{path}.{name}", member, new.members[name])
            for name, member in old.members.items()
            if not name.startswith("_") and name in new.members
        ]
    return [
        f"{where}: return type changed from {o.returns} to {n.returns}"
        for where, o, n in pairs
        if not o.is_alias
        and not n.is_alias
        and o.is_function
        and n.is_function
        and str(o.returns) != str(n.returns)
    ]


def tops_of(modules: list[str]) -> set[str]:
    return {m.split(".")[0] for m in modules}


def check_pr(root: Path, base: str, title: str, labels: list[str]) -> int:
    modules = sorted(stable_modules(root))
    found = compare(
        load(root, base, tops_of(modules)), load(root, None, tops_of(modules)), modules
    )
    if not found:
        print("No breaking changes to the public API.")
        return 0
    print("\n".join(found))
    if BREAKING_TITLE.match(title):
        print(
            "\nDeclared breaking with `!` in the PR title: release-please will cut a major version "
            "(a minor one before 1.0.0)."
        )
        return 0
    if COMPATIBLE_LABEL in labels:
        print(
            f"\nAn owner judged these changes compatible (`{COMPATIBLE_LABEL}` label)."
        )
        return 0
    print(
        f"\n{len(found)} breaking change(s) to the public API. Declare a breaking change with `!` "
        "in the PR title (`feat!: ...`, `fix(scope)!: ...`). If an owner judges the change "
        f"compatible, they apply the `{COMPATIBLE_LABEL}` label instead."
    )
    return 1


def version(text: str) -> tuple[int, int, int]:
    major, minor, patch = (int(n) for n in re.findall(r"\d+", text)[:3])
    return major, minor, patch


def allows_breaking(old: str, new: str) -> bool:
    """True if ``old`` → ``new`` is a major bump, or a minor one before 1.0.0."""
    (o_major, o_minor, _), (n_major, n_minor, _) = version(old), version(new)
    return n_major > o_major or (o_major == n_major == 0 and n_minor > o_minor)


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def check_release(root: Path, base: str, labels: list[str] | None = None) -> int:
    config = json.loads((root / "release-please-config.json").read_text())["packages"]
    new_versions = json.loads((root / ".release-please-manifest.json").read_text())
    old_versions = json.loads(
        git(root, "show", f"{base}:.release-please-manifest.json").stdout
    )
    by_package: dict[str, list[str]] = {}
    for module, package_dir in stable_modules(root).items():
        by_package.setdefault(str(package_dir.relative_to(root)), []).append(module)
    failed = False
    for package, new in sorted(new_versions.items()):
        old = old_versions.get(package)
        modules = sorted(by_package.get(package, []))
        if old is None or old == new or not modules:
            continue
        tag = f"{config[package]['component']}-{old}"
        if git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}").returncode:
            print(
                f"{package}: no release tag {tag}, so there is nothing to compare with"
            )
            continue
        found = compare(
            load(root, tag, tops_of(modules)),
            load(root, None, tops_of(modules)),
            modules,
        )
        if not found or allows_breaking(old, new):
            print(f"{package}: {old} -> {new} matches its changes since {tag}")
            continue
        failed = True
        print("\n".join(found))
        print(
            f"{package}: {old} -> {new}, but the changes above since {tag} are breaking. "
            "It needs a major version (a minor one before 1.0.0): a merged PR that made one "
            "of them was not declared with `!` in its title."
        )
    if failed and COMPATIBLE_LABEL in (labels or []):
        print(
            f"\nAn owner judged these changes compatible (`{COMPATIBLE_LABEL}` label)."
        )
        return 0
    if failed:
        print(
            f"\nIf an owner judges these changes compatible, they apply the `{COMPATIBLE_LABEL}` "
            "label to this release PR."
        )
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("pr")
    pr.add_argument("--base", required=True)
    pr.add_argument("--title", required=True)
    pr.add_argument("--labels", default="")
    release = sub.add_parser("release")
    release.add_argument("--base", required=True)
    release.add_argument("--labels", default="")
    args = parser.parse_args(argv)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if args.command == "pr":
        return check_pr(ROOT, args.base, args.title, labels)
    return check_release(ROOT, args.base, labels)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
