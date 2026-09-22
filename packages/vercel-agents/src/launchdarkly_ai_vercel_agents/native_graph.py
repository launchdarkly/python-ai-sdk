from __future__ import annotations

import time
import uuid
from types import SimpleNamespace
from typing import Any

import ai
from ai.types.tools import ToolSpec
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from launchdarkly_ai_server import (
    GraphDefinition,
    GraphNode,
    get_client,
    make_track_data,
    to_ld_context,
)

from .handler import (
    ModelSource,
    build_agent_tools,
    build_messages,
    build_request_params,
    output_of,
    resolve_model,
    usage_of,
)


def _handoff_tool(
    source: str,
    target: str,
    description: str | None,
    selected: dict[str, str | None],
) -> Any:
    async def transfer() -> str:
        selected[source] = target
        return target

    model_tool = ai.Tool(
        kind="function",
        name=f"transfer_to_{target}",
        spec=ToolSpec(
            description=description, params={"type": "object", "properties": {}}
        ),
    )
    agent_tool = ai.AgentTool(model_tool, transfer)
    try:
        object.__setattr__(agent_tool, "name", f"transfer_to_{target}")
        object.__setattr__(agent_tool, "execute", transfer)
    except (AttributeError, TypeError):
        pass
    return agent_tool


def _track(
    name: str,
    context: Any,
    node: GraphNode,
    definition: GraphDefinition,
    run_id: str,
    value: int | float,
) -> None:
    client = get_client()
    client.track(
        name,
        to_ld_context(client, context),
        make_track_data(node, definition.key, run_id),
        value,
    )


def to_vercel_agents(
    definition_promise: Any,
    opts: dict[str, Any] | None = None,
) -> Any:
    options = opts or {}

    async def invoke(
        input_text: str = "",
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        definition: GraphDefinition = await definition_promise
        if not definition.enabled:
            raise ValueError(f'Agent graph "{definition.key}" is disabled')
        if definition.root is None:
            raise ValueError(f'Graph "{definition.key}" has no root node')

        vs = variables or {}
        tool_handlers = options.get("tool_handlers") or {}
        model_source: ModelSource | None = options.get(
            "model", options.get("model_factory")
        )
        selected: dict[str, str | None] = {}
        agents: dict[str, Any] = {}
        models: dict[str, Any] = {}
        reachable: list[GraphNode] = []
        visited: set[str] = set()

        async def build(node: GraphNode) -> None:
            if node.key in visited:
                return
            visited.add(node.key)
            reachable.append(node)
            selected[node.key] = None
            model = await resolve_model(model_source, node.config, ai)
            models[node.key] = model
            tools = build_agent_tools(node.config.get("tools"), tool_handlers, ai)
            for edge in definition.edges_from(node.key):
                tools.append(
                    _handoff_tool(
                        node.key,
                        edge.target_key,
                        (edge.handoff or {}).get("description"),
                        selected,
                    )
                )
            agents[node.key] = ai.Agent(tools=tools)
            for edge in definition.edges_from(node.key):
                child = definition.get_node(edge.target_key)
                if child is not None:
                    await build(child)

        await build(definition.root)

        span = trace.get_tracer("@launchdarkly/ai-vercel-agents").start_span(
            "ld.ai.graph"
        )
        span.set_attribute("ld.ai.graph.key", definition.key)
        context = options.get("context")
        run_id = str(uuid.uuid4())
        start = time.monotonic()
        path: list[str] = []
        input_tokens = 0
        output_tokens = 0
        current = definition.root
        current_input = input_text
        final_text = ""
        try:
            while current is not None and current.key not in path:
                path.append(current.key)
                selected[current.key] = None
                node_history = history if current.key == definition.root.key else None
                messages = build_messages(
                    current.config,
                    current_input,
                    vs,
                    node_history,
                    runtime=ai,
                )
                async with agents[current.key].run(
                    model=models[current.key],
                    messages=messages,
                    params=build_request_params(current.config, ai),
                ) as stream:
                    async for _ in stream:
                        pass
                usage = usage_of(stream)
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                final_text = output_of(stream)
                if context is not None:
                    _track(
                        "$ld:ai:generation:success",
                        context,
                        current,
                        definition,
                        run_id,
                        1,
                    )
                target = selected[current.key]
                if target is None:
                    break
                if context is not None:
                    _track(
                        "$ld:ai:graph:handoff_success",
                        context,
                        current,
                        definition,
                        run_id,
                        1,
                    )
                next_node = definition.get_node(target)
                if next_node is None:
                    break
                current = next_node
                current_input = final_text

            total = input_tokens + output_tokens
            span.set_attribute("ld.ai.graph.path", "->".join(path))
            span.set_status(StatusCode.OK)
            if context is not None:
                _track(
                    "$ld:ai:graph:invocation_success",
                    context,
                    definition.root,
                    definition,
                    run_id,
                    1,
                )
                _track(
                    "$ld:ai:graph:duration:total",
                    context,
                    definition.root,
                    definition,
                    run_id,
                    (time.monotonic() - start) * 1000,
                )
                _track(
                    "$ld:ai:graph:total_tokens",
                    context,
                    definition.root,
                    definition,
                    run_id,
                    total,
                )
            return {
                "response": final_text,
                "usage": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total,
                },
            }
        except BaseException as exc:
            if isinstance(exc, Exception):
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                if context is not None:
                    _track(
                        "$ld:ai:graph:invocation_failure",
                        context,
                        definition.root,
                        definition,
                        run_id,
                        1,
                    )
            else:
                span.set_attribute("launchdarkly.run.cancelled", True)
            raise
        finally:
            span.end()

    return SimpleNamespace(invoke=invoke)
