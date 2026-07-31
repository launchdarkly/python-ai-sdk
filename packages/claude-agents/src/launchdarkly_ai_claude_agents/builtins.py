"""
Claude built-in tool sentinels.

Place any of these as a value in your ``tool_handlers`` map to enable
the corresponding native Claude Code capability.

Example::

    from launchdarkly_ai_claude_agents import ClaudeWebSearch, ClaudeBash
    from launchdarkly_ai_server import graph

    result = await graph("my-flag", registry=registry).invoke(
        user_input,
        context,
        tool_handlers={
            "web-search": ClaudeWebSearch,
            "run-bash":   ClaudeBash,
        },
    )
"""

from __future__ import annotations

from launchdarkly_ai_server import NativeTool

ClaudeBash = NativeTool("Bash")
ClaudeRead = NativeTool("Read")
ClaudeEdit = NativeTool("Edit")
ClaudeWrite = NativeTool("Write")
ClaudeGlob = NativeTool("Glob")
ClaudeGrep = NativeTool("Grep")
ClaudeWebFetch = NativeTool("WebFetch")
ClaudeWebSearch = NativeTool("WebSearch")
ClaudeTodoWrite = NativeTool("TodoWrite")
ClaudeNotebookEdit = NativeTool("NotebookEdit")
