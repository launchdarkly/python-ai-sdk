"""
Agent Skills for Claude Agents.

``create_claude_agents_skills_handler`` builds a Claude Agents handler that gives each run the
skills attached to its variation. Exported from the package root.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Sequence

from claude_agent_sdk import SettingSource

from launchdarkly_ai_server import (
    AiConfigRep,
    ProviderHandler,
    ReconcileReport,
    SkillReference,
    report_usage,
    skill_refs,
    write_skills,
)

from ._version import PACKAGE_NAME, __version__
from .handler import QueryPreparation, _create_claude_agents_handler

__all__ = ["create_claude_agents_skills_handler"]

logger = logging.getLogger(__name__)

PLUGIN_NAME = "launchdarkly"
"""The local plugin's directory name, and so its name: the CLI names a plugin with no manifest
after its directory, and lists its skills as ``launchdarkly:<key>``."""

_RUN_DIR_PREFIX = "ld-skills-"
"""Prefix of each per-invoke directory under *skills_temp_dir*."""


def _materialization_error(
    report: ReconcileReport, refs: Sequence[SkillReference]
) -> RuntimeError:
    """Why a run did not start, naming each skill that could not be written.

    Built from the report's action errors, which carry no skill content. A run-level error
    carries an empty key, so the requested keys are named as well.
    """
    failures = [
        f"{action.key}: {action.error}" if action.key else f"{action.error}"
        for action in report.errors
    ]
    requested = ", ".join(ref.key for ref in refs)
    return RuntimeError(
        "Claude agent run not started: the skills its variation references could not be "
        f"written (requested: {requested}). {'; '.join(failures)}"
    )


def _remove_run_directory(path: str) -> None:
    """Removes one run's directory. A failure is logged, never raised over the run's own outcome.

    Synchronous on purpose: with no ``await`` in it, a cancellation arriving during teardown
    cannot interrupt it and leave the directory behind.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(
            "[LaunchDarkly] could not remove the Claude agent skills directory %s: %s",
            path,
            exc,
        )


async def _prepare_skills(
    config: AiConfigRep, parent: str | os.PathLike[str] | None
) -> QueryPreparation | None:
    """Writes the variation's skills into a directory private to this run, and returns the
    options that hand it to ``query()`` as a local plugin. ``None`` when the variation has no
    skills, so that run is exactly the core handler's.

    Fails closed: if any skill cannot be written, the run does not start. The variation names
    that skill, so running without it would run a different agent from the one configured.
    """
    refs = skill_refs(config)
    if not refs:
        return None

    run_dir = os.path.realpath(tempfile.mkdtemp(prefix=_RUN_DIR_PREFIX, dir=parent))
    try:
        plugin_root = os.path.join(run_dir, PLUGIN_NAME)
        # ``write_skills`` creates only the missing leaf of its root.
        os.mkdir(plugin_root)
        # A fresh directory: there is nothing to prune, and nothing else writes here.
        report = await write_skills(
            refs, os.path.join(plugin_root, "skills"), prune=False
        )
        if not report.ok:
            raise _materialization_error(report, refs)
    except BaseException:
        _remove_run_directory(run_dir)
        raise

    async def _cleanup() -> None:
        _remove_run_directory(run_dir)

    return QueryPreparation(
        options={
            "plugins": [{"type": "local", "path": plugin_root}],
            # Qualified: a bare name would also match a host plugin's skill of the same name.
            "skills": [f"{PLUGIN_NAME}:{ref.key}" for ref in refs],
        },
        extra_builtins=["Skill"],
        cleanup=_cleanup,
    )


def create_claude_agents_skills_handler(
    *,
    capture_content: bool | None = None,
    cwd: str | os.PathLike[str] | None = None,
    setting_sources: list[SettingSource] | None = None,
    skills_temp_dir: str | os.PathLike[str] | None = None,
) -> ProviderHandler:
    """A Claude Agents handler that gives each run the skills attached to its variation.

    Takes the options of ``create_claude_agents_handler``, which behave the same way here, plus
    *skills_temp_dir*: the parent of the directory each run writes its skills into, by default
    ``tempfile.gettempdir()``.

    On every invoke and stream, the variation's ``skills`` are written with ``write_skills``,
    through the store configured with ``set_skill_store``, into a new directory under
    *skills_temp_dir*. That directory is passed to ``query()`` as a local plugin, only those
    skills are enabled, the ``Skill`` tool is turned on, and the directory is removed when the
    run ends. A variation with no skills runs exactly as with ``create_claude_agents_handler``.

    ``setting_sources`` is always sent, ``[]`` unless given. ``claude-agent-sdk`` would
    otherwise load the ``user`` and ``project`` settings sources whenever ``skills`` is set.

    If any of the variation's skills cannot be written, because no store is configured, the
    store has not received its first payload, a skill fails verification, or a pinned version
    is absent, the run raises before the model is called.

    .. code-block:: python

        import os

        from launchdarkly_ai_server import FDv2SkillStore, config, set_skill_store
        from launchdarkly_ai_claude_agents import create_claude_agents_skills_handler

        store = FDv2SkillStore(os.environ["LD_SDK_KEY"]).start()
        store.wait_for_skills(timeout=10)
        set_skill_store(store)

        triage = config(key="triage-agent", handler=create_claude_agents_skills_handler())
    """
    report_usage(
        "claude-agents.createClaudeAgentsSkillsHandler", PACKAGE_NAME, __version__
    )

    async def _prepare(config: AiConfigRep) -> QueryPreparation | None:
        return await _prepare_skills(config, skills_temp_dir)

    return _create_claude_agents_handler(
        capture_content=capture_content,
        cwd=cwd,
        setting_sources=setting_sources,
        _prepare=_prepare,
    )
