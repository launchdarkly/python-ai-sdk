"""
to_claude_agents — converts a GraphDefinition into nested claude-agent-sdk
query() calls, mirroring the TypeScript toClaudeAgents implementation.
"""

from __future__ import annotations

import re
import time
import types
import uuid
from typing import Any

from launchdarkly_ai_server import (
    NATIVE_TOOL_KEY,
    GraphDefinition,
    GraphNode,
    NativeTool,
    get_client,
    make_track_data,
    to_ld_context,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

from launchdarkly_ai_claude_agents.handler import (
    _build_hooks,
    build_prompt,
    build_tool_mcp,
    partition_tools,
)

TOOL_MCP_NAME = "tool-mcp"
SUBAGENT_MCP_NAME = "subagents"
MCP_TOOL_PREFIX = f"mcp__{TOOL_MCP_NAME}__"
SUBAGENT_TOOL_PREFIX = f"mcp__{SUBAGENT_MCP_NAME}__"


def _sanitize_name(key: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", key, flags=re.IGNORECASE)


def _wrap_native_tools(
    tool_handlers: dict[str, Any],
    ld_context: Any,
    track_data: dict[str, Any],
) -> dict[str, Any]:
    wrapped: dict[str, Any] = {}
    for name, fn in tool_handlers.items():
        if isinstance(fn, NativeTool):

            def _stub(_name: str = name) -> None:
                if ld_context:
                    get_client().track(
                        "$ld:ai:tool_call",
                        ld_context,
                        {**track_data, "toolName": _name},
                        1,
                    )

            setattr(_stub, NATIVE_TOOL_KEY, fn)
            wrapped[name] = _stub
        else:
            wrapped[name] = fn
    return wrapped


async def _reverse_traverse(
    def_obj: GraphDefinition,
    fn: Any,
) -> None:
    """Post-order (leaves first) traversal of the graph definition."""
    edges_from = def_obj.edges_from
    visited: set[str] = set()

    async def _visit(node_key: str) -> None:
        if node_key in visited:
            return
        visited.add(node_key)
        for edge in edges_from(node_key):
            await _visit(edge.target_key)
        node = def_obj.get_node(node_key)
        if node:
            await fn(node)

    root = def_obj.root
    if root:
        await _visit(root.key)


async def _run_query(
    node: GraphNode,
    input_text: str,
    variables: dict[str, Any],
    tool_handlers: dict[str, Any],
    ld_context: Any,
    graph_key: str,
    run_id: str,
    child_subagent_tools: list[Any],
) -> dict[str, Any]:
    import importlib

    sdk = importlib.import_module("claude_agent_sdk")
    ClaudeAgentOptions = sdk.ClaudeAgentOptions
    ResultMessage = sdk.ResultMessage
    query_fn = sdk.query
    create_server = sdk.create_sdk_mcp_server

    track_data = make_track_data(node, graph_key, run_id)
    wrapped = _wrap_native_tools(tool_handlers, ld_context, track_data)

    prompt, system_prompt = build_prompt(node.config, input_text, variables)
    native_tool_map, user_config_tools, native_tool_names = partition_tools(
        node.config.get("tools"), wrapped
    )

    tool_mcp = (
        await build_tool_mcp(user_config_tools, wrapped) if user_config_tools else None
    )
    child_mcp = (
        create_server(
            name=SUBAGENT_MCP_NAME, version="1.0.0", tools=child_subagent_tools
        )
        if child_subagent_tools
        else None
    )

    mcp_allowed = [MCP_TOOL_PREFIX + n for n in user_config_tools]
    # Subagent tools (SUBAGENT_TOOL_PREFIX + child name) are intentionally NOT
    # pre-approved in allowed_tools below; Claude decides whether to call them
    # and the SDK handles permission in non-interactive mode.

    mcp_servers: dict[str, Any] = {}
    if tool_mcp:
        mcp_servers[TOOL_MCP_NAME] = tool_mcp
    if child_mcp:
        mcp_servers[SUBAGENT_MCP_NAME] = child_mcp

    hooks = _build_hooks(native_tool_map)

    options = ClaudeAgentOptions(
        # Explicitly set the available built-in tools (empty list disables all).
        # When no native tools are needed, disable built-in tools so Claude
        # cannot call WebSearch/Bash/etc. and get stuck waiting for permission
        # approval (mirrors TypeScript's `tools: nativeToolNames.length > 0 ? nativeToolNames : []`).
        tools=native_tool_names if native_tool_names else [],
        # Only auto-approve user-defined MCP tools and native tools; do NOT
        # auto-approve subagent tools (child_allowed). This matches TypeScript's
        # `allowedTools: allAllowedTools.length > 0 ? allAllowedTools : undefined`
        # behaviour where child subagent tools are not pre-approved — Claude decides
        # whether to call them and the SDK handles permission in non-interactive mode.
        allowed_tools=(mcp_allowed + native_tool_names)
        if (mcp_allowed or native_tool_names)
        else [],
        mcp_servers=mcp_servers,
        hooks=hooks or {},
        **({"system_prompt": system_prompt} if system_prompt else {}),
    )

    output = ""
    raw_usage: dict[str, Any] = {}

    # Hold an explicit reference so we can call aclose() in the finally block.
    # Bare `return` inside `async for` abandons the generator — Python's asyncio
    # finalizer later tries to aclose() it and may raise RuntimeError if the
    # generator is suspended inside a real await in the SDK (AIC-2950).
    gen = query_fn(prompt=prompt, options=options)
    try:
        async for message in gen:
            if isinstance(message, ResultMessage):
                output = message.result or ""
                raw_usage = message.usage or {}
                break
    finally:
        await gen.aclose()

    input_tokens = int(raw_usage.get("input_tokens", 0))
    output_tokens = int(raw_usage.get("output_tokens", 0))
    return {
        "output": output,
        "usage": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
    }


def to_claude_agents(
    def_promise: Any,
    opts: dict[str, Any] | None = None,
) -> Any:
    """
    Converts a resolved ``GraphDefinition`` into a nested query() multi-agent
    execution where each child node becomes a sub-agent tool.

    Returns an object with a ``.invoke(input, variables)`` coroutine.
    """
    _opts = opts or {}

    async def invoke(
        input_text: str = "",
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import importlib

        sdk = importlib.import_module("claude_agent_sdk")
        tool_fn = sdk.tool

        vs = variables or {}
        def_obj: GraphDefinition = await def_promise

        if not def_obj.enabled:
            raise ValueError(f'Agent graph "{def_obj.key}" is disabled')
        root = def_obj.root
        if not root:
            raise ValueError(f'Graph "{def_obj.key}" has no root node')

        raw_ld_context = _opts.get("context")
        ld_context = (
            to_ld_context(get_client(), raw_ld_context)
            if raw_ld_context is not None
            else None
        )
        raw_handlers: dict[str, Any] = _opts.get("tool_handlers") or {}

        tracer_name = "@launchdarkly/ai-claude-agents"
        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("ld.ai.graph")
            span.set_attribute("ld.ai.graph.key", def_obj.key)
        else:
            span = None

        start_time = time.monotonic()
        run_id = str(uuid.uuid4())
        path: list[str] = []
        total_usage = {"input": 0, "output": 0, "total": 0}
        subagent_tool_ctx: dict[str, Any] = {}

        try:

            async def _build_node(node: GraphNode) -> None:
                if node.key == root.key:
                    return

                node_key = _sanitize_name(node.key)
                child_tools = [
                    subagent_tool_ctx[e.target_key]
                    for e in node.edges
                    if e.target_key in subagent_tool_ctx
                ]

                async def _subagent_execute(
                    args: Any, _node: GraphNode = node
                ) -> dict[str, Any]:
                    sub_input = (
                        args.get("input", "") if isinstance(args, dict) else str(args)
                    )
                    if ld_context:
                        td = make_track_data(_node, def_obj.key, run_id)
                        get_client().track(
                            "$ld:ai:graph:handoff_success", ld_context, td, 1
                        )

                    path.append(_node.key)
                    node_start = time.monotonic()
                    result = await _run_query(
                        _node,
                        sub_input,
                        vs,
                        raw_handlers,
                        ld_context,
                        def_obj.key,
                        run_id,
                        child_tools,
                    )
                    total_usage["input"] += result["usage"]["input"]
                    total_usage["output"] += result["usage"]["output"]
                    total_usage["total"] += result["usage"]["total"]

                    if ld_context:
                        td = make_track_data(_node, def_obj.key, run_id)
                        dur = int((time.monotonic() - node_start) * 1000)
                        client = get_client()
                        client.track("$ld:ai:duration:total", ld_context, td, dur)
                        client.track("$ld:ai:generation:success", ld_context, td, 1)
                        u = result["usage"]
                        if u["total"] > 0:
                            client.track(
                                "$ld:ai:tokens:total", ld_context, td, u["total"]
                            )
                        if u["input"] > 0:
                            client.track(
                                "$ld:ai:tokens:input", ld_context, td, u["input"]
                            )
                        if u["output"] > 0:
                            client.track(
                                "$ld:ai:tokens:output", ld_context, td, u["output"]
                            )

                    return {"content": [{"type": "text", "text": result["output"]}]}

                instructions_desc = (node.config.get("instructions") or node.key)[:120]
                subagent_mcp_tool = tool_fn(
                    node_key,
                    instructions_desc,
                    {"input": str},
                )(_subagent_execute)
                subagent_tool_ctx[node.key] = subagent_mcp_tool

            await _reverse_traverse(def_obj, _build_node)

            root_child_tools = [
                subagent_tool_ctx[e.target_key]
                for e in root.edges
                if e.target_key in subagent_tool_ctx
            ]

            path.append(root.key)
            root_start = time.monotonic()

            try:
                result = await _run_query(
                    root,
                    input_text,
                    vs,
                    raw_handlers,
                    ld_context,
                    def_obj.key,
                    run_id,
                    root_child_tools,
                )
            except Exception as exc:
                if span:
                    span.record_exception(exc)
                    span.set_status(SpanStatusCode.ERROR, str(exc))
                    span.end()
                if ld_context:
                    td = make_track_data(root, def_obj.key, run_id)
                    get_client().track(
                        "$ld:ai:graph:invocation_failure", ld_context, td, 1
                    )
                raise

            final_output = result["output"]
            root_usage = result["usage"]
            total_usage["input"] += root_usage["input"]
            total_usage["output"] += root_usage["output"]
            total_usage["total"] += root_usage["total"]

            if ld_context:
                td = make_track_data(root, def_obj.key, run_id)
                client = get_client()
                dur = int((time.monotonic() - root_start) * 1000)
                client.track("$ld:ai:duration:total", ld_context, td, dur)
                client.track("$ld:ai:generation:success", ld_context, td, 1)
                if root_usage["total"] > 0:
                    client.track(
                        "$ld:ai:tokens:total", ld_context, td, root_usage["total"]
                    )
                if root_usage["input"] > 0:
                    client.track(
                        "$ld:ai:tokens:input", ld_context, td, root_usage["input"]
                    )
                if root_usage["output"] > 0:
                    client.track(
                        "$ld:ai:tokens:output", ld_context, td, root_usage["output"]
                    )

            graph_dur = int((time.monotonic() - start_time) * 1000)

            if span:
                span.set_attribute("ld.ai.graph.path", "->".join(path))
                span.set_attribute("gen_ai.usage.input_tokens", total_usage["input"])
                span.set_attribute("gen_ai.usage.output_tokens", total_usage["output"])
                span.set_attribute("gen_ai.usage.total_tokens", total_usage["total"])
                span.set_status(SpanStatusCode.OK)
                span.end()

            if ld_context:
                root_td = make_track_data(root, def_obj.key, run_id)
                client = get_client()
                client.track(
                    "$ld:ai:graph:duration:total", ld_context, root_td, graph_dur
                )
                client.track(
                    "$ld:ai:graph:total_tokens",
                    ld_context,
                    root_td,
                    total_usage["total"],
                )
                client.track("$ld:ai:graph:path", ld_context, root_td, len(path))
                client.track("$ld:ai:graph:invocation_success", ld_context, root_td, 1)

            return {"response": final_output, "usage": total_usage}

        except Exception as exc:
            if span:
                span.record_exception(exc)
                span.set_status(SpanStatusCode.ERROR, str(exc))
                span.end()
            raise

    return types.SimpleNamespace(invoke=invoke)
