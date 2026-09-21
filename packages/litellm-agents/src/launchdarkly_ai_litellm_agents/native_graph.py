"""Convert a resolved graph to OpenAI Agents SDK agents backed by LiteLLM."""

from __future__ import annotations

import importlib
import re
import types
from typing import Any

from agents.extensions.models.litellm_model import LitellmModel

from launchdarkly_ai_server import (
    GraphDefinition,
    GraphNode,
    compose_history,
    parse_template,
)

from .handler import (
    ModelFactory,
    _map_agent_content,
    _model_config,
    _model_settings,
    _tools,
)


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
        factory: ModelFactory = options.get("model_factory") or _default_factory
        tool_handlers = options.get("tool_handlers") or {}
        values = variables or {}
        built: dict[str, Any] = {}

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
        result = await agents.Runner.run(built[graph.root.key], runner_input)
        wrapper = getattr(result, "context_wrapper", None)
        usage = getattr(wrapper, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return {
            "response": str(result.final_output or ""),
            "usage": {
                "input": input_tokens,
                "output": output_tokens,
                "total": int(
                    getattr(usage, "total_tokens", input_tokens + output_tokens)
                    or input_tokens + output_tokens
                ),
            },
        }

    return types.SimpleNamespace(invoke=invoke)
