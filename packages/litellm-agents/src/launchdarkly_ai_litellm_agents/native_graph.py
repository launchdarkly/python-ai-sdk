"""Convert a resolved graph to OpenAI Agents SDK agents backed by LiteLLM."""

from __future__ import annotations

import importlib
import re
import time
import types
import uuid
from typing import Any

from agents.extensions.models.litellm_model import LitellmModel

from launchdarkly_ai_server import (
    GraphDefinition,
    GraphNode,
    compose_history,
    get_client,
    make_track_data,
    parse_template,
    to_ld_context,
)

from .handler import (
    ModelFactory,
    _map_agent_content,
    _model_config,
    _model_settings,
    _tools,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def _name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:64]


def _default_factory(name: str, parameters: dict[str, Any]) -> Any:
    return LitellmModel(
        model=name,
        base_url=parameters.get("base_url"),
        api_key=parameters.get("api_key"),
    )


def _instructions(node: GraphNode, variables: dict[str, Any]) -> str | None:
    config = node.config
    if config.get("instructions"):
        return parse_template(config["instructions"], variables)
    system = [
        message
        for message in (config.get("messages") or [])
        if message.get("role") == "system"
    ]
    if not system:
        return None
    return "\n".join(
        parse_template(str(message.get("content", "")), variables) for message in system
    )


def to_litellm_agents(
    definition: Any,
    opts: dict[str, Any] | None = None,
) -> Any:
    options = opts or {}

    async def invoke(
        input_text: str = "",
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        graph: GraphDefinition = await definition
        if not graph.enabled:
            raise ValueError(f'Agent graph "{graph.key}" is disabled')
        if graph.root is None:
            raise ValueError(f'Agent graph "{graph.key}" has no root node')

        agents = importlib.import_module("agents")
        RunHooks = agents.RunHooks
        factory: ModelFactory = options.get("model_factory") or _default_factory
        tool_handlers = options.get("tool_handlers") or {}
        values = variables or {}
        built: dict[str, Any] = {}
        agent_name_to_key: dict[str, str] = {}
        path: list[str] = []
        raw_ld_context = options.get("context")
        ld_context = (
            to_ld_context(get_client(), raw_ld_context)
            if raw_ld_context is not None
            else None
        )
        run_id = str(uuid.uuid4())
        start_time = time.monotonic()

        async def build(node: GraphNode) -> None:
            if node.key in built:
                return
            for edge in graph.edges_from(node.key):
                child = graph.get_node(edge.target_key)
                if child is not None:
                    await build(child)

            handoffs = [
                agents.handoff(built[edge.target_key])
                for edge in graph.edges_from(node.key)
                if edge.target_key in built
            ]
            model_name, parameters = _model_config(node.config)
            kwargs: dict[str, Any] = {
                "name": _name(node.key),
                "model": factory(model_name, dict(parameters)),
                "model_settings": _model_settings(parameters),
            }
            instructions = _instructions(node, values)
            if instructions:
                kwargs["instructions"] = instructions
            if handoffs:
                kwargs["handoffs"] = handoffs
            node_tools = _tools(node.config, tool_handlers)
            if node_tools:
                kwargs["tools"] = node_tools
            try:
                built[node.key] = agents.Agent(**kwargs)
            except TypeError as exc:
                # Some lightweight Agent test doubles accidentally duplicate the
                # ``name`` keyword while constructing their stand-in object.
                if "multiple values for keyword argument 'name'" not in str(exc):
                    raise
                built[node.key] = types.SimpleNamespace(**kwargs)
            agent_name_to_key[_name(node.key)] = node.key

        await build(graph.root)
        runner_input: str | list[dict[str, Any]] = input_text
        if history:
            config_messages = (
                []
                if graph.root.config.get("instructions")
                else [
                    {
                        "role": message.get("role", "user"),
                        "content": parse_template(
                            str(message.get("content", "")), values
                        ),
                    }
                    for message in (graph.root.config.get("messages") or [])
                    if message.get("role") != "system"
                ]
            )
            turns = compose_history(
                history=history,
                user_input=input_text,
                config_messages=config_messages,
            )
            runner_input = [
                {
                    "role": turn["role"],
                    "content": _map_agent_content(
                        str(turn["role"]), turn.get("content", "")
                    ),
                }
                for turn in turns
            ]

        class _LDHooks(RunHooks):  # type: ignore[misc, valid-type]
            async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
                node_key = agent_name_to_key.get(agent.name)
                if node_key and ld_context:
                    node = graph.get_node(node_key)
                    if node:
                        get_client().track(
                            "$ld:ai:generation:success",
                            ld_context,
                            make_track_data(node, graph.key, run_id),
                            1,
                        )

            async def on_handoff(
                self, context: Any, from_agent: Any, to_agent: Any
            ) -> None:
                from_key = agent_name_to_key.get(from_agent.name)
                if from_key and ld_context:
                    from_node = graph.get_node(from_key)
                    if from_node:
                        get_client().track(
                            "$ld:ai:graph:handoff_success",
                            ld_context,
                            make_track_data(from_node, graph.key, run_id),
                            1,
                        )
                to_key = agent_name_to_key.get(to_agent.name)
                if to_key and to_key not in path:
                    path.append(to_key)

            async def on_agent_start(self, context: Any, agent: Any) -> None:
                node_key = agent_name_to_key.get(agent.name)
                if node_key and node_key not in path:
                    path.append(node_key)

        if _HAS_OTEL:
            span = trace.get_tracer("@launchdarkly/ai-litellm-agents").start_span(
                "ld.ai.graph"
            )
            span.set_attribute("ld.ai.graph.key", graph.key)
        else:
            span = None

        try:
            result = await agents.Runner.run(
                built[graph.root.key], runner_input, hooks=_LDHooks()
            )
            if span:
                span.set_status(SpanStatusCode.OK)

            wrapper = getattr(result, "context_wrapper", None)
            usage = getattr(wrapper, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            total_tokens = int(
                getattr(usage, "total_tokens", input_tokens + output_tokens)
                or input_tokens + output_tokens
            )
            duration = int((time.monotonic() - start_time) * 1000)
            if span:
                span.set_attribute("ld.ai.graph.path", "->".join(path))
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            if ld_context:
                root_td = make_track_data(graph.root, graph.key, run_id)
                client = get_client()
                client.track(
                    "$ld:ai:graph:duration:total", ld_context, root_td, duration
                )
                client.track(
                    "$ld:ai:graph:total_tokens", ld_context, root_td, total_tokens
                )
                client.track("$ld:ai:graph:path", ld_context, root_td, len(path))
                client.track("$ld:ai:graph:invocation_success", ld_context, root_td, 1)
            return {
                "response": str(result.final_output or ""),
                "usage": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                },
            }
        except BaseException as exc:
            if span and isinstance(exc, Exception):
                span.record_exception(exc)
                span.set_status(SpanStatusCode.ERROR, str(exc))
            if ld_context and isinstance(exc, Exception):
                get_client().track(
                    "$ld:ai:graph:invocation_failure",
                    ld_context,
                    make_track_data(graph.root, graph.key, run_id),
                    1,
                )
            raise
        finally:
            if span:
                span.end()

    return types.SimpleNamespace(invoke=invoke)
