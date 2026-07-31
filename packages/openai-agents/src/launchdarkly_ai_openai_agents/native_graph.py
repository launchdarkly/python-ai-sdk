"""
toOpenAIAgents — converts a GraphDefinition into an OpenAI Agents SDK agent
tree and runs it via Runner.run, mirroring the TypeScript toOpenAIAgents.
"""

from __future__ import annotations

import re
import time
import types
import uuid
from typing import Any

from launchdarkly_ai_server import (
    GraphDefinition,
    GraphNode,
    NativeTool,
    get_client,
    make_track_data,
    parse_template,
    to_ld_context,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def _sanitize_name(key: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", key, flags=re.IGNORECASE)[:64]


def _build_instructions(node: GraphNode, variables: dict[str, Any]) -> str | None:
    config = node.config
    if config.get("instructions"):
        return parse_template(config["instructions"], variables)
    if config.get("messages"):
        sys_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        if sys_msgs:
            return parse_template("\n".join(m["content"] for m in sys_msgs), variables)
    return None


def _build_node_tools(
    node: GraphNode,
    tool_handlers: dict[str, Any],
) -> list[Any]:
    import importlib

    agents_mod = importlib.import_module("agents")
    tool_fn = agents_mod.tool

    if not node.config.get("tools"):
        return []

    result = []
    for name, tool_cfg in node.config["tools"].items():

        async def _execute(args: Any, _name: str = name) -> str:
            handler = tool_handlers.get(_name)
            if not handler or isinstance(handler, NativeTool):
                return ""
            res = await handler(args)
            return str(res)

        t = tool_fn(
            name=name,
            description=tool_cfg.get("description", ""),
            params_json_schema=tool_cfg.get("parameters") or {},
        )(_execute)
        result.append(t)
    return result


def to_openai_agents(
    def_promise: Any,
    opts: dict[str, Any] | None = None,
) -> Any:
    """
    Converts a resolved ``GraphDefinition`` into an OpenAI Agents SDK agent tree
    and returns a caller that runs the graph natively via ``Runner.run``.

    Example::

        from launchdarkly_ai_server import resolve_graph
        from launchdarkly_ai_openai_agents import to_openai_agents

        result = await to_openai_agents(
            resolve_graph("support-graph", context=ctx),
            {"context": ctx},
        ).invoke("I was double charged")
    """
    _opts = opts or {}

    async def invoke(
        input_text: str = "",
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import importlib

        agents_mod = importlib.import_module("agents")
        Agent = agents_mod.Agent
        Runner = agents_mod.Runner
        handoff_fn = agents_mod.handoff
        RunHooks = agents_mod.RunHooks

        vs = variables or {}
        def_obj: GraphDefinition = await def_promise

        if not def_obj.enabled:
            raise ValueError(f'Agent graph "{def_obj.key}" is disabled')
        root = def_obj.root
        if not root:
            raise ValueError(f'Graph "{def_obj.key}" has no root node')

        tool_handlers: dict[str, Any] = _opts.get("tool_handlers") or {}
        raw_ld_context = _opts.get("context")
        ld_context = (
            to_ld_context(get_client(), raw_ld_context)
            if raw_ld_context is not None
            else None
        )

        tracer_name = "@launchdarkly/ai-openai-agents"
        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("ld.ai.graph")
            span.set_attribute("ld.ai.graph.key", def_obj.key)
        else:
            span = None

        start_time = time.monotonic()
        run_id = str(uuid.uuid4())
        path: list[str] = []
        agent_name_to_key: dict[str, str] = {}
        agent_ctx: dict[str, Any] = {}
        edges_from = def_obj.edges_from

        # Post-order traversal (leaves first) using the edges_from function
        visited: set[str] = set()

        async def _visit(node_key: str) -> None:
            if node_key in visited:
                return
            visited.add(node_key)
            for edge in edges_from(node_key):
                await _visit(edge.target_key)

            node = def_obj.get_node(node_key)
            if node is None:
                return

            child_handoffs = []
            for edge in edges_from(node_key):
                child_agent = agent_ctx.get(edge.target_key)
                if child_agent is None:
                    raise ValueError(
                        f'Child agent "{edge.target_key}" not built before parent "{node_key}"'
                    )
                child_handoffs.append(handoff_fn(child_agent))

            instructions = _build_instructions(node, vs)
            tools = _build_node_tools(node, tool_handlers)
            agent_name = _sanitize_name(node.key)
            agent_name_to_key[agent_name] = node.key

            agent = Agent(
                name=agent_name,
                model=node.config.get("model", {}).get("name", "gpt-4o"),
                **({"instructions": instructions} if instructions else {}),
                **({"tools": tools} if tools else {}),
                **({"handoffs": child_handoffs} if child_handoffs else {}),
            )
            agent_ctx[node.key] = agent

        await _visit(root.key)

        root_agent = agent_ctx.get(root.key)
        if root_agent is None:
            raise ValueError(f'Root agent "{root.key}" was not built')

        # Lifecycle hooks for LD tracking
        class _LDHooks(RunHooks):  # type: ignore[misc, valid-type]
            async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
                node_key = agent_name_to_key.get(agent.name)
                if node_key and ld_context:
                    node = def_obj.get_node(node_key)
                    if node:
                        td = make_track_data(node, def_obj.key, run_id)
                        get_client().track(
                            "$ld:ai:generation:success", ld_context, td, 1
                        )

            async def on_handoff(
                self, context: Any, from_agent: Any, to_agent: Any
            ) -> None:
                from_key = agent_name_to_key.get(from_agent.name)
                if from_key and ld_context:
                    from_node = def_obj.get_node(from_key)
                    if from_node:
                        td = make_track_data(from_node, def_obj.key, run_id)
                        get_client().track(
                            "$ld:ai:graph:handoff_success", ld_context, td, 1
                        )
                to_key = agent_name_to_key.get(to_agent.name)
                if to_key and to_key not in path:
                    path.append(to_key)

            async def on_agent_start(self, context: Any, agent: Any) -> None:
                node_key = agent_name_to_key.get(agent.name)
                if node_key and node_key not in path:
                    path.append(node_key)

        hooks = _LDHooks()

        try:
            result = await Runner.run(root_agent, input_text, hooks=hooks)
            if span:
                span.set_status(SpanStatusCode.OK)
        except Exception as exc:
            if span:
                span.record_exception(exc)
                span.set_status(SpanStatusCode.ERROR, str(exc))
                span.end()
            if ld_context:
                td = make_track_data(root, def_obj.key, run_id)
                get_client().track("$ld:ai:graph:invocation_failure", ld_context, td, 1)
            raise

        final_output = str(result.final_output or "")
        # Sum usage across all raw_responses
        input_tokens = sum(
            getattr(r.usage, "input_tokens", 0)
            for r in result.raw_responses
            if hasattr(r, "usage")
        )
        output_tokens = sum(
            getattr(r.usage, "output_tokens", 0)
            for r in result.raw_responses
            if hasattr(r, "usage")
        )
        total_tokens = sum(
            getattr(r.usage, "total_tokens", 0)
            for r in result.raw_responses
            if hasattr(r, "usage")
        )

        total_usage = {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        }
        duration = int((time.monotonic() - start_time) * 1000)

        if span:
            span.set_attribute("ld.ai.graph.path", "->".join(path))
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.end()

        if ld_context:
            root_td = make_track_data(root, def_obj.key, run_id)
            client = get_client()
            client.track("$ld:ai:graph:duration:total", ld_context, root_td, duration)
            client.track("$ld:ai:graph:total_tokens", ld_context, root_td, total_tokens)
            client.track("$ld:ai:graph:path", ld_context, root_td, len(path))
            client.track("$ld:ai:graph:invocation_success", ld_context, root_td, 1)

        return {"response": final_output, "usage": total_usage}

    return types.SimpleNamespace(invoke=invoke)
