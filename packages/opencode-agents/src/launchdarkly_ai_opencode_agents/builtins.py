"""OpenCode built-in tool sentinels.

Place any of these as a value in your ``tool_handlers`` map to allow the
corresponding OpenCode built-in capability for one call. Every other
built-in, including one not named here, is denied for that call: an AI
Config variation with no tools gets a model with no tools at all.

Ids are the tool files under ``packages/opencode/src/tool/`` at tag
``v1.18.4``. That tag has no ``list`` tool, so this package does not export
one.

``OpenCodeWrite`` and ``OpenCodeApplyPatch`` are the same grant as
``OpenCodeEdit``. OpenCode's ``Permission.disabled`` remaps ``write`` and
``apply_patch`` onto the single ``edit`` permission before it matches a
rule, so an allow for any one of the three allows all three. The session
ruleset sends ``edit``.

Example::

    from launchdarkly_ai_opencode_agents import OpenCodeBash, OpenCodeRead
    from launchdarkly_ai_server import config

    response = await config(
        key="my-agent",
        handler=create_opencode_handler(cwd=..., config=..., env=...),
        tool_handlers={"run-bash": OpenCodeBash, "read-file": OpenCodeRead},
    ).invoke(user_input, context)
"""

from __future__ import annotations

from launchdarkly_ai_server import NativeTool

OpenCodeBash = NativeTool("bash")
OpenCodeRead = NativeTool("read")
OpenCodeEdit = NativeTool("edit")
OpenCodeWrite = NativeTool("write")
OpenCodeGlob = NativeTool("glob")
OpenCodeGrep = NativeTool("grep")
OpenCodeTask = NativeTool("task")
OpenCodeSkill = NativeTool("skill")
OpenCodeTodoWrite = NativeTool("todowrite")
OpenCodeWebFetch = NativeTool("webfetch")
OpenCodeWebSearch = NativeTool("websearch")
OpenCodeApplyPatch = NativeTool("apply_patch")
