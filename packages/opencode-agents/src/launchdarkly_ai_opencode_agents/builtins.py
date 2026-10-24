"""OpenCode built-in tool sentinels.

Place any of these as a value in your ``tool_handlers`` map to allow the
corresponding OpenCode built-in capability for one call. Every other
built-in, including one not named here, is denied for that call: an AI
Config variation with no tools gets a model with no tools at all.

Ids verified against the OpenCode source at tag ``v1.18.4``
(``packages/opencode/src/tool/*.ts``). ``OpenCodeList`` is the one
exception: it mirrors Vega's own ``_PINNABLE_BUILTIN_TOOLS`` pin
(``ai/vega-agentcore/vega_agentcore/opencode_config.py``) even though that
tag ships no ``list`` tool file. Passing a name OpenCode does not currently
serve is harmless — it simply never matches a tool the model is offered.

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
OpenCodeList = NativeTool("list")
OpenCodeTask = NativeTool("task")
OpenCodeSkill = NativeTool("skill")
OpenCodeTodoWrite = NativeTool("todowrite")
OpenCodeWebFetch = NativeTool("webfetch")
OpenCodeWebSearch = NativeTool("websearch")
OpenCodeApplyPatch = NativeTool("apply_patch")
