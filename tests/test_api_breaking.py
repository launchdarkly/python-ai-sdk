"""Breaking changes to the public API must be declared, and releases must match them."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts import api_breaking

V1 = """
from demo_sdk.run import run

def make(value: str) -> str:
    return value

__all__ = ["make", "run"]
"""
RUN = "def run(key: str) -> str:\n    return key\n"
BETA = '__all__ = ["try_it"]\n\ndef try_it() -> None: ...\n'


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-package repository whose first commit is released as ``demo-sdk-1.2.0``."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    # Keep a developer's global git config (signing, hooks) out of these repositories,
    # including the git that griffe runs to read an old ref.
    gitconfig = tmp_path.parent / f"{tmp_path.name}.gitconfig"
    gitconfig.write_text("[user]\n\tname = test\n\temail = test@example.com\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    src = tmp_path / "packages/demo/src/demo_sdk"
    write(src / "__init__.py", V1)
    write(src / "run.py", RUN)
    write(src / "experimental/__init__.py", "")
    write(src / "experimental/beta.py", BETA)
    write(
        tmp_path / "release-please-config.json",
        json.dumps({"packages": {"packages/demo": {"component": "demo-sdk"}}}),
    )
    write(
        tmp_path / ".release-please-manifest.json",
        json.dumps({"packages/demo": "1.2.0"}),
    )
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "v1")
    git(tmp_path, "tag", "demo-sdk-1.2.0")
    return tmp_path


def change(repo: Path, relative: str, text: str) -> None:
    write(repo / "packages/demo/src/demo_sdk" / relative, text)
    git(repo, "commit", "-q", "-am", "change")


def breakages(repo: Path) -> list[str]:
    modules = sorted(api_breaking.stable_modules(repo))
    tops = api_breaking.tops_of(modules)
    return api_breaking.compare(
        api_breaking.load(repo, "demo-sdk-1.2.0", tops),
        api_breaking.load(repo, None, tops),
        modules,
    )


def test_no_change_breaks_nothing(repo: Path) -> None:
    assert breakages(repo) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def run(key: str, strict: bool) -> str: ...\n", "added as required"),
        ("def run(name: str) -> str: ...\n", "removed"),
        ("def run(key: str) -> int: ...\n", "return type changed from str to int"),
    ],
)
def test_a_breaking_signature_change_is_found(
    repo: Path, source: str, expected: str
) -> None:
    change(repo, "run.py", source)
    found = breakages(repo)
    assert len(found) >= 1
    assert any(expected.lower() in line.lower() for line in found), found


def test_an_added_optional_parameter_or_name_breaks_nothing(repo: Path) -> None:
    change(repo, "run.py", "def run(key: str, strict: bool = False) -> str: ...\n")
    change(
        repo,
        "__init__.py",
        V1.replace('"make", "run"', '"extra", "make", "run"') + "extra = 1\n",
    )
    assert breakages(repo) == []


def test_a_name_removed_from_all_is_breaking(repo: Path) -> None:
    change(repo, "__init__.py", V1.replace('"make", "run"', '"run"'))
    assert breakages(repo) == ["demo_sdk.make: removed from __all__"]


def test_an_experimental_module_may_break(repo: Path) -> None:
    change(
        repo,
        "experimental/beta.py",
        '__all__ = ["try_it"]\n\ndef try_it(x: int) -> None: ...\n',
    )
    assert breakages(repo) == []


@pytest.mark.parametrize(
    ("title", "labels", "code"),
    [
        ("feat: change run", [], 1),
        ("feat!: change run", [], 0),
        ("fix(client)!: change run", [], 0),
        ("feat: !important", [], 1),
        ("feat: change run", ["api-compatible"], 0),
    ],
)
def test_a_breaking_pr_must_declare_it(
    repo: Path, title: str, labels: list[str], code: int
) -> None:
    change(repo, "run.py", "def run(key: str, strict: bool) -> str: ...\n")
    assert api_breaking.check_pr(repo, "demo-sdk-1.2.0", title, labels) == code


def test_a_pr_without_breaking_changes_passes(repo: Path) -> None:
    assert api_breaking.check_pr(repo, "demo-sdk-1.2.0", "feat: nothing", []) == 0


def release(
    repo: Path,
    released: str,
    proposed: str,
    breaking: bool,
    labels: list[str] | None = None,
) -> int:
    """Release ``released`` as tagged, then propose ``proposed`` the way a release PR does."""
    git(repo, "tag", "-d", "demo-sdk-1.2.0")
    write(
        repo / ".release-please-manifest.json", json.dumps({"packages/demo": released})
    )
    git(repo, "commit", "-q", "--allow-empty", "-am", "released")
    git(repo, "tag", f"demo-sdk-{released}")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if breaking:
        change(repo, "run.py", "def run(key: str, strict: bool) -> str: ...\n")
    write(
        repo / ".release-please-manifest.json", json.dumps({"packages/demo": proposed})
    )
    git(repo, "commit", "-q", "-am", "release PR")
    return api_breaking.check_release(repo, base, labels)


@pytest.mark.parametrize(
    ("released", "proposed", "breaking", "code"),
    [
        ("1.2.0", "1.3.0", True, 1),
        ("1.2.0", "1.2.1", True, 1),
        ("1.2.0", "2.0.0", True, 0),
        ("1.2.0", "1.3.0", False, 0),
        ("0.3.0", "0.4.0", True, 0),
        ("0.3.0", "0.3.1", True, 1),
        ("0.9.0", "1.0.0", True, 0),
    ],
)
def test_a_release_version_matches_its_changes(
    repo: Path, released: str, proposed: str, breaking: bool, code: int
) -> None:
    assert release(repo, released, proposed, breaking) == code


def test_a_package_with_no_release_tag_is_skipped(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = "HEAD"
    git(repo, "tag", "-d", "demo-sdk-1.2.0")
    write(
        repo / ".release-please-manifest.json", json.dumps({"packages/demo": "1.3.0"})
    )
    git(repo, "commit", "-q", "-am", "release PR")
    assert api_breaking.check_release(repo, f"{base}~1") == 0
    assert "no release tag demo-sdk-1.2.0" in capsys.readouterr().out


def test_an_owner_can_judge_a_release_compatible(repo: Path) -> None:
    # The label passes an undeclared breaking change on the release PR, as on any other PR.
    assert release(repo, "1.2.0", "1.2.1", True, ["api-compatible"]) == 0


def test_another_label_does_not_pass_a_breaking_release(repo: Path) -> None:
    assert release(repo, "1.2.0", "1.2.1", True, ["docs"]) == 1
