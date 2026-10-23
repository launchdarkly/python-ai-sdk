"""
Agent Skills (experimental).

Configure a store with ``set_skill_store``, then read verified skill content with
``get_skill``, ``get_skills`` or ``all_skills``, or write it to disk with
``write_skills``. To keep the skills an AI Config pins on disk as the config
changes, use ``watch_skills(config_skills(key, context), root)``. These names
may change in a minor release, so review the changelog when you upgrade.
"""

from ..skills import (
    InMemorySkillStore,
    all_skills,
    get_skill,
    get_skill_result,
    get_skills,
    set_skill_store,
    skill_refs,
)
from ..skills_core import SkillStore
from ..skills_fdv2 import FDv2SkillStore, StoreDiagnostics
from ..skills_fs import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    SKILL_FILENAME,
    OnUnavailable,
    write_skills,
)
from ..skills_source import config_skills
from ..skills_watch import SkillWatcher, watch_skills
from ..types import (
    ReconcileAction,
    ReconcileActionKind,
    ReconcileReport,
    Skill,
    SkillOutcome,
    SkillOutcomeReason,
    SkillReference,
    SkillSource,
)

__all__ = [  # noqa: RUF022
    # configuration
    "set_skill_store",
    "SkillStore",
    "InMemorySkillStore",
    # LaunchDarkly delivery store and on-change re-reconcile
    "FDv2SkillStore",
    "StoreDiagnostics",
    "watch_skills",
    "SkillWatcher",
    # a requested set read on every reconcile, following AI Config changes
    "SkillSource",
    "config_skills",
    # accessors
    "skill_refs",
    "get_skill",
    "get_skill_result",
    "get_skills",
    "all_skills",
    "write_skills",
    # value types
    "Skill",
    "SkillReference",
    "SkillOutcome",
    "ReconcileAction",
    "ReconcileReport",
    # literal types for typed consumers
    "ReconcileActionKind",
    "OnUnavailable",
    "SkillOutcomeReason",
    # on-disk filenames and manifest version
    "SKILL_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
]
