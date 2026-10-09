"""LaunchDarkly AI SDK - integration for Claude agents.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from . import native_graph  # noqa: F401
from ._version import PACKAGE_NAME, __version__
from .builtins import (
    ClaudeBash,
    ClaudeEdit,
    ClaudeGlob,
    ClaudeGrep,
    ClaudeNotebookEdit,
    ClaudeRead,
    ClaudeTodoWrite,
    ClaudeWebFetch,
    ClaudeWebSearch,
    ClaudeWrite,
)
from .graph import claude_graph
from .handler import (
    build_prompt,
    build_tool_mcp,
    claude_agents,
    create_claude_agents_handler,
    partition_tools,
)
from .native_graph import to_claude_agents

__all__ = [
    "ClaudeBash",
    "ClaudeEdit",
    "ClaudeGlob",
    "ClaudeGrep",
    "ClaudeNotebookEdit",
    "ClaudeRead",
    "ClaudeTodoWrite",
    "ClaudeWebFetch",
    "ClaudeWebSearch",
    "ClaudeWrite",
    "build_prompt",
    "build_tool_mcp",
    "claude_agents",
    "claude_graph",
    "create_claude_agents_handler",
    "partition_tools",
    "to_claude_agents",
]

register_ai_sdk_package(PACKAGE_NAME, __version__)
