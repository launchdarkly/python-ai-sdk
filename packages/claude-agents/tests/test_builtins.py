"""
Tests for §4 Claude Agents built-ins (builtins.py).
Reference: TESTING.md §4
"""

from launchdarkly_ai_claude_agents.builtins import (
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
from launchdarkly_ai_server import NativeTool

ALL_BUILTINS = [
    ("ClaudeBash", ClaudeBash, "Bash"),
    ("ClaudeRead", ClaudeRead, "Read"),
    ("ClaudeEdit", ClaudeEdit, "Edit"),
    ("ClaudeWrite", ClaudeWrite, "Write"),
    ("ClaudeGlob", ClaudeGlob, "Glob"),
    ("ClaudeGrep", ClaudeGrep, "Grep"),
    ("ClaudeWebFetch", ClaudeWebFetch, "WebFetch"),
    ("ClaudeWebSearch", ClaudeWebSearch, "WebSearch"),
    ("ClaudeTodoWrite", ClaudeTodoWrite, "TodoWrite"),
    ("ClaudeNotebookEdit", ClaudeNotebookEdit, "NotebookEdit"),
]


class TestClaudeBuiltins:
    def test_each_export_is_native_tool_instance(self) -> None:
        for name, sentinel, _ in ALL_BUILTINS:
            assert isinstance(sentinel, NativeTool), f"{name} is not a NativeTool"

    def test_each_has_correct_tool_name(self) -> None:
        for name, sentinel, expected_name in ALL_BUILTINS:
            assert sentinel.tool_name == expected_name, (
                f"{name}.tool_name expected '{expected_name}', got '{sentinel.tool_name}'"
            )

    def test_all_id_symbols_are_unique(self) -> None:
        ids = [id(sentinel.id) for _, sentinel, _ in ALL_BUILTINS]
        assert len(ids) == len(set(ids)), "Builtin id sentinels are not unique"
