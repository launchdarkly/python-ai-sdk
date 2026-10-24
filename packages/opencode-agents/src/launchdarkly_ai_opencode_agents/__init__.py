"""LaunchDarkly AI SDK - integration for OpenCode agents.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .builtins import (
    OpenCodeApplyPatch,
    OpenCodeBash,
    OpenCodeEdit,
    OpenCodeGlob,
    OpenCodeGrep,
    OpenCodeList,
    OpenCodeRead,
    OpenCodeSkill,
    OpenCodeTask,
    OpenCodeTodoWrite,
    OpenCodeWebFetch,
    OpenCodeWebSearch,
    OpenCodeWrite,
)
from .handler import OpenCodeSessionError, create_opencode_handler, opencode_agents

__all__ = [
    "OpenCodeApplyPatch",
    "OpenCodeBash",
    "OpenCodeEdit",
    "OpenCodeGlob",
    "OpenCodeGrep",
    "OpenCodeList",
    "OpenCodeRead",
    "OpenCodeSessionError",
    "OpenCodeSkill",
    "OpenCodeTask",
    "OpenCodeTodoWrite",
    "OpenCodeWebFetch",
    "OpenCodeWebSearch",
    "OpenCodeWrite",
    "create_opencode_handler",
    "opencode_agents",
]

register_ai_sdk_package(PACKAGE_NAME, __version__)
