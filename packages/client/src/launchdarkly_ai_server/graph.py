from __future__ import annotations

import inspect as _inspect
import json
import logging
import re as _re
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

from .conversation import bind_conversation_id, bind_span_context
from .registry import resolve_handlers, resolve_tools
from .types import (
    AiConfigRep,
    GraphDefinition,
    GraphEdge,
    GraphNode,
    GraphStreamEvent,
    JudgeResult,
    LDContext,
    NativeTool,
    ProviderGraphResponse,
    ProviderHandler,
    TrackData,
    UsageDict,
    VariationMeta,
)
from .utils import end_span_once, model_stamps_from_meta, select_handler, to_ld_context

logger = logging.getLogger(__name__)

MAX_TRAVERSAL_DEPTH = 100
MAX_GRAPH_CACHE_SIZE = 512


def _sanitize_name(key: str) -> str:
    # Match TS sanitizeName: hyphens become underscores (tool names must be [a-zA-Z0-9_]).
    return _re.sub(r"[^a-zA-Z0-9_]", "_", key)[:64]


def _disabled_definition(key: str) -> GraphDefinition:
    async def _traverse_noop(fn: Any, ctx: dict[str, Any] | None = None) -> None:
        return None

    async def _run_node_disabled(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(f'Agent graph "{key}" is disabled')

    async def _route_disabled(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(f'Agent graph "{key}" is disabled')

    return GraphDefinition(
        key=key,
        enabled=False,
        root=None,
        get_node=lambda k: None,
        get_child_nodes=lambda k: [],
        get_parent_nodes=lambda k: [],
        terminal_nodes=lambda: [],
        is_terminal=lambda k: True,
        edges_from=lambda k: [],
        run_node=_run_node_disabled,
        route=_route_disabled,
        traverse=_traverse_noop,
        reverse_traverse=_traverse_noop,
    )


def _disabled_stream_route(key: str) -> Callable[..., Any]:
    """The ``stream_route`` a disabled graph gets. Kept beside the definition it
    pairs with, but off ``GraphDefinition`` — see ``_build_graph``'s return."""

    async def _stream_route_disabled(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[GraphStreamEvent, None]:
        raise ValueError(f'Agent graph "{key}" is disabled')
        yield  # pragma: no cover

    return _stream_route_disabled


async def _fetch_graph_variation(
    key: str,
    context: LDContext,
) -> dict[str, Any]:
    from .lifecycle import get_client, init_client

    await init_client()

    client = get_client()
    variation: Any = client.variation(key, to_ld_context(client, context), {})
    if hasattr(variation, "__await__"):
        variation = await variation

    meta = (variation.get("_ldMeta") or {}) if isinstance(variation, dict) else {}
    topology = variation if isinstance(variation, dict) else None

    if not topology or not topology.get("root"):
        return {"enabled": False, "meta": meta}

    return {"enabled": True, "topology": topology, "meta": meta}


async def _build_graph(
    key: str,
    context: LDContext,
    options: dict[str, Any],
) -> tuple[GraphDefinition, TrackData, Callable[..., Any]]:
    from .judges import run_judges
    from .lifecycle import extract_variation, get_client
    from .tracking import execute_and_track

    result = await _fetch_graph_variation(key, context)
    # Convert once so all inner track() calls use an ldclient.Context object.
    ld_ctx = to_ld_context(get_client(), context)
    enabled: bool = result["enabled"]
    topology: dict[str, Any] | None = result.get("topology")
    meta = result.get("meta", {})

    graph_track_data: TrackData = {
        "runId": str(uuid.uuid4()),
        "configKey": key,
        "variationKey": meta.get("variationKey", "") if isinstance(meta, dict) else "",
        "version": meta.get("version", 1) if isinstance(meta, dict) else 1,
        "modelName": "",
        "providerName": "",
        **model_stamps_from_meta(meta),
        "graphKey": key,
    }

    if not enabled or not topology:
        return (
            _disabled_definition(key),
            graph_track_data,
            _disabled_stream_route(key),
        )

    raw_edges_map: dict[str, list[dict[str, Any]]] = topology.get("edges") or {}
    edges: list[GraphEdge] = []
    for source_key, outgoing in raw_edges_map.items():
        for raw_edge in outgoing:
            edges.append(
                GraphEdge(
                    key=f"{source_key}-{raw_edge['key']}",
                    source_key=source_key,
                    target_key=raw_edge["key"],
                    handoff=raw_edge.get("handoff"),
                )
            )

    all_keys: set[str] = {topology["root"]}
    for edge in edges:
        all_keys.add(edge.source_key)
        all_keys.add(edge.target_key)

    def edges_from(node_key: str) -> list[GraphEdge]:
        return [e for e in edges if e.source_key == node_key]

    nodes: dict[str, GraphNode] = {}
    try:
        for node_key in all_keys:
            variation = await extract_variation(node_key, context)
            node_config: AiConfigRep = variation["config"]
            node_meta: VariationMeta = variation["meta"]
            node_edges = edges_from(node_key)
            nodes[node_key] = GraphNode(
                key=node_key,
                config=node_config,
                meta=node_meta,
                edges=node_edges,
                is_terminal=len(node_edges) == 0,
            )
    except Exception as exc:
        logger.error("Graph node variation failed: %s", exc)
        return (
            _disabled_definition(key),
            graph_track_data,
            _disabled_stream_route(key),
        )

    root_node = nodes.get(topology["root"])

    def get_node(node_key: str) -> GraphNode | None:
        return nodes.get(node_key)

    def get_child_nodes(node_key: str) -> list[GraphNode]:
        return [
            nodes[e.target_key] for e in edges_from(node_key) if e.target_key in nodes
        ]

    def get_parent_nodes(node_key: str) -> list[GraphNode]:
        return [
            nodes[e.source_key]
            for e in edges
            if e.target_key == node_key and e.source_key in nodes
        ]

    def terminal_nodes() -> list[GraphNode]:
        return [n for n in nodes.values() if len(edges_from(n.key)) == 0]

    def is_terminal(node_key: str) -> bool:
        return len(edges_from(node_key)) == 0

    # ── run_node ──────────────────────────────────────────────────────────────
    # Runs a single node: calls the handler, runs per-node judges, and tracks
    # handoff events. Mirrors the TS GraphDefinition.runNode method.

    async def run_node(
        node: GraphNode,
        input: str = "",
        opts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = opts or {}
        handlers: list[ProviderHandler] = options.get("handlers") or []
        if not handlers:
            raise ValueError(
                "run_node is not available when no handlers were provided — use a "
                "framework-native runner (to_openai_agents, to_lang_graph, to_claude_agents) instead."
            )
        handler = select_handler(node.config, node.meta, handlers, strict=False)
        tool_handlers = opts.get("tool_handlers") or options.get("tool_handlers")
        from_node: GraphNode | None = opts.get("from")

        try:
            result = await execute_and_track(
                config_key=node.key,
                config=node.config,
                meta=node.meta,
                user_context=context,
                handler=handler,
                user_input=input,
                tool_handlers=tool_handlers,
                variables=opts.get("variables"),
                graph_key=key,
                history=opts.get("history"),
            )
            response = (
                result["response"]
                if isinstance(result["response"], str)
                else str(result["response"])
            )

            judge_results = await run_judges(
                config=node.config,
                user_context=context,
                handler=handler,
                handlers=handlers,
                user_input=input,
                llm_response=response,
                base_track_data=result["track_data"],
                tool_handlers=tool_handlers,
                graph_key=key,
            )

            if from_node:
                get_client().track(
                    "$ld:ai:graph:handoff_success",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": from_node.key,
                        "targetKey": node.key,
                    },
                    1,
                )

            return {
                "response": response,
                "usage": result["usage"],
                "judge_results": judge_results,
            }
        except Exception:
            if from_node:
                get_client().track(
                    "$ld:ai:graph:handoff_failure",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": from_node.key,
                        "targetKey": node.key,
                    },
                    1,
                )
            raise

    # ── build_handoff_routing ─────────────────────────────────────────────────
    # Shared by route and stream_route so multi-edge descriptions stay identical.

    def build_handoff_routing(
        node: GraphNode,
        out_edges: list[GraphEdge],
    ) -> dict[str, Any]:
        chosen: list[str] = []
        handoff_tools: dict[str, Any] = {}
        handoff_handlers: dict[str, Any] = {}

        for edge in out_edges:
            target_key = edge.target_key
            tool_name = f"__handoff_{_sanitize_name(target_key)}"
            target_node = nodes.get(target_key)
            # The prefix is unconditional: without it, a description sourced from the target's
            # own instructions reads as a tool that does the target's work, and the model calls
            # it instead of the node's real tools.
            detail = (edge.handoff or {}).get("description") or (
                target_node.config.get("instructions", "")[:120] if target_node else ""
            )
            description = f"Transfer control to {target_key}."
            if detail:
                description = f"{description} {detail}"
            handoff_tools[tool_name] = {
                "name": tool_name,
                "type": "function",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }

            def _make_handoff_fn(t: str) -> Callable[..., str]:
                def _fn(*a: Any, **kw: Any) -> str:
                    if not chosen:
                        chosen.append(t)
                    # Selecting an edge does not end the turn; execution continues until the
                    # model produces its final text. A "transferring now" reply reads as though
                    # control has already left, and the model stops short of its own work.
                    return (
                        f"Handoff to {t} recorded. "
                        "Finish your own work and provide your final response."
                    )

                return _fn

            handoff_handlers[tool_name] = _make_handoff_fn(target_key)

        routed_config: AiConfigRep = {
            **node.config,
            "instructions": (
                (node.config.get("instructions") or "")
                + (
                    "\n\nComplete your task using your available tools first. "
                    "Only once you have your final answer, call exactly one transfer "
                    "tool to route to the next agent."
                )
            ),
            "tools": {
                **(node.config.get("tools") or {}),
                **handoff_tools,
            },
        }

        return {
            "routed_config": routed_config,
            "handoff_handlers": handoff_handlers,
            "chosen": lambda: chosen[0] if chosen else None,
        }

    # ── route ─────────────────────────────────────────────────────────────────
    # For nodes with zero/one outgoing edge, delegates to run_node and returns
    # the sole successor as `next`. For multi-edge nodes, injects synthetic
    # handoff tools so the model picks the next agent. Mirrors TS route().

    async def route(
        node: GraphNode,
        input: str = "",
        opts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = opts or {}
        handlers: list[ProviderHandler] = options.get("handlers") or []
        if not handlers:
            raise ValueError(
                "route is not available when no handlers were provided — use a "
                "framework-native runner (to_openai_agents, to_lang_graph, to_claude_agents) instead."
            )

        out_edges = edges_from(node.key)

        # Zero/one outgoing edge: run node directly; report sole child as next.
        if len(out_edges) <= 1:
            res = await run_node(node, input, opts)
            next_node = nodes.get(out_edges[0].target_key) if out_edges else None
            return {**res, "next": next_node}

        handler = select_handler(node.config, node.meta, handlers, strict=False)
        tool_handlers = opts.get("tool_handlers") or options.get("tool_handlers")

        routing = build_handoff_routing(node, out_edges)
        routed_config = routing["routed_config"]
        handoff_handlers = routing["handoff_handlers"]
        chosen = routing["chosen"]

        merged_tool_handlers = {**(tool_handlers or {}), **handoff_handlers}

        try:
            result = await execute_and_track(
                config_key=node.key,
                config=routed_config,
                meta=node.meta,
                user_context=context,
                handler=handler,
                user_input=input,
                tool_handlers=merged_tool_handlers,
                variables=opts.get("variables"),
                graph_key=key,
                history=opts.get("history"),
            )
            response = (
                result["response"]
                if isinstance(result["response"], str)
                else str(result["response"])
            )

            # Judge against the node's original config, not the routing-augmented one.
            judge_results = await run_judges(
                config=node.config,
                user_context=context,
                handler=handler,
                handlers=handlers,
                user_input=input,
                llm_response=response,
                base_track_data=result["track_data"],
                tool_handlers=tool_handlers,
                graph_key=key,
            )

            chosen_key = chosen()
            next_node = nodes.get(chosen_key) if chosen_key else None

            if next_node:
                get_client().track(
                    "$ld:ai:graph:handoff_success",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": node.key,
                        "targetKey": next_node.key,
                    },
                    1,
                )

            return {
                "response": response,
                "usage": result["usage"],
                "judge_results": judge_results,
                "next": next_node,
            }
        except Exception:
            chosen_key = chosen()
            if chosen_key:
                get_client().track(
                    "$ld:ai:graph:handoff_failure",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": node.key,
                        "targetKey": chosen_key,
                    },
                    1,
                )
            raise

    # ── stream_node / stream_route ────────────────────────────────────────────
    # Streaming counterparts to run_node / route. Python async generators cannot
    # return a value (unlike JS yield*), so callers pass an ``outcome`` dict that
    # is populated with the ProviderResponse / RouteResult fields when done.

    async def stream_node(
        node: GraphNode,
        input: str = "",
        opts: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
    ) -> AsyncGenerator[GraphStreamEvent, None]:
        from .tracking import execute_and_stream

        opts = opts or {}
        handlers: list[ProviderHandler] = options.get("handlers") or []
        if not handlers:
            raise ValueError(
                "stream_node is not available when no handlers were provided — use a "
                "framework-native runner (to_openai_agents, to_lang_graph, to_claude_agents) instead."
            )
        handler = select_handler(node.config, node.meta, handlers, strict=False)
        tool_handlers = opts.get("tool_handlers") or options.get("tool_handlers")
        from_node: GraphNode | None = opts.get("from")

        yield {"type": "node_start", "nodeKey": node.key}

        try:
            response = ""
            usage: dict[str, Any] = {"input": 0, "output": 0, "total": 0}
            track_data: TrackData = {
                "runId": str(uuid.uuid4()),
                "configKey": node.key,
                "variationKey": (
                    node.meta.get("variationKey", "")
                    if isinstance(node.meta, dict)
                    else ""
                ),
                "version": (
                    node.meta.get("version", 1) if isinstance(node.meta, dict) else 1
                ),
                "modelName": (node.config.get("model") or {}).get("name", ""),
                "providerName": (node.config.get("provider") or {}).get("name", ""),
                "graphKey": key,
            }

            async for event in execute_and_stream(
                config_key=node.key,
                config=node.config,
                meta=node.meta,
                user_context=context,
                handler=handler,
                user_input=input,
                tool_handlers=tool_handlers,
                variables=opts.get("variables"),
                graph_key=key,
                history=opts.get("history"),
            ):
                if event.get("type") == "chunk":
                    yield {
                        "type": "chunk",
                        "text": event["text"],
                        "nodeKey": node.key,
                    }
                else:
                    response = event.get("response", "")
                    usage = event.get("usage") or usage
                    track_data = event.get("track_data") or track_data

            judge_results = await run_judges(
                config=node.config,
                user_context=context,
                handler=handler,
                handlers=handlers,
                user_input=input,
                llm_response=response,
                base_track_data=track_data,
                tool_handlers=tool_handlers,
                graph_key=key,
            )

            if from_node:
                get_client().track(
                    "$ld:ai:graph:handoff_success",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": from_node.key,
                        "targetKey": node.key,
                    },
                    1,
                )

            yield {
                "type": "node_done",
                "nodeKey": node.key,
                "response": response,
                "usage": usage,
            }
            if outcome is not None:
                outcome.clear()
                outcome.update(
                    {
                        "response": response,
                        "usage": usage,
                        "judge_results": judge_results,
                        "track_data": track_data,
                    }
                )
        except Exception:
            if from_node:
                get_client().track(
                    "$ld:ai:graph:handoff_failure",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": from_node.key,
                        "targetKey": node.key,
                    },
                    1,
                )
            raise

    async def stream_route(
        node: GraphNode,
        input: str = "",
        opts: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
    ) -> AsyncGenerator[GraphStreamEvent, None]:
        from .tracking import execute_and_stream

        opts = opts or {}
        handlers: list[ProviderHandler] = options.get("handlers") or []
        if not handlers:
            raise ValueError(
                "stream_route is not available when no handlers were provided — use a "
                "framework-native runner (to_openai_agents, to_lang_graph, to_claude_agents) instead."
            )

        out_edges = edges_from(node.key)

        if len(out_edges) <= 1:
            node_outcome: dict[str, Any] = {}
            async for event in stream_node(node, input, opts, node_outcome):
                yield event
            next_node = nodes.get(out_edges[0].target_key) if out_edges else None
            if outcome is not None:
                outcome.clear()
                outcome.update({**node_outcome, "next": next_node})
            return

        handler = select_handler(node.config, node.meta, handlers, strict=False)
        tool_handlers = opts.get("tool_handlers") or options.get("tool_handlers")

        routing = build_handoff_routing(node, out_edges)
        routed_config = routing["routed_config"]
        handoff_handlers = routing["handoff_handlers"]
        chosen = routing["chosen"]

        yield {"type": "node_start", "nodeKey": node.key}

        try:
            response = ""
            usage: dict[str, Any] = {"input": 0, "output": 0, "total": 0}
            track_data: TrackData = {
                "runId": str(uuid.uuid4()),
                "configKey": node.key,
                "variationKey": (
                    node.meta.get("variationKey", "")
                    if isinstance(node.meta, dict)
                    else ""
                ),
                "version": (
                    node.meta.get("version", 1) if isinstance(node.meta, dict) else 1
                ),
                "modelName": (node.config.get("model") or {}).get("name", ""),
                "providerName": (node.config.get("provider") or {}).get("name", ""),
                "graphKey": key,
            }

            merged_tool_handlers = {**(tool_handlers or {}), **handoff_handlers}

            async for event in execute_and_stream(
                config_key=node.key,
                config=routed_config,
                meta=node.meta,
                user_context=context,
                handler=handler,
                user_input=input,
                tool_handlers=merged_tool_handlers,
                variables=opts.get("variables"),
                graph_key=key,
                history=opts.get("history"),
            ):
                if event.get("type") == "chunk":
                    yield {
                        "type": "chunk",
                        "text": event["text"],
                        "nodeKey": node.key,
                    }
                else:
                    response = event.get("response", "")
                    usage = event.get("usage") or usage
                    track_data = event.get("track_data") or track_data

            # Judge against the node's original config, not the routing-augmented one.
            judge_results = await run_judges(
                config=node.config,
                user_context=context,
                handler=handler,
                handlers=handlers,
                user_input=input,
                llm_response=response,
                base_track_data=track_data,
                tool_handlers=tool_handlers,
                graph_key=key,
            )

            chosen_key = chosen()
            next_node = nodes.get(chosen_key) if chosen_key else None

            if next_node:
                get_client().track(
                    "$ld:ai:graph:handoff_success",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": node.key,
                        "targetKey": next_node.key,
                    },
                    1,
                )

            yield {
                "type": "node_done",
                "nodeKey": node.key,
                "response": response,
                "usage": usage,
            }
            if outcome is not None:
                outcome.clear()
                outcome.update(
                    {
                        "response": response,
                        "usage": usage,
                        "judge_results": judge_results,
                        "track_data": track_data,
                        "next": next_node,
                    }
                )
        except Exception:
            chosen_key = chosen()
            if chosen_key:
                get_client().track(
                    "$ld:ai:graph:handoff_failure",
                    ld_ctx,
                    {
                        **graph_track_data,
                        "sourceKey": node.key,
                        "targetKey": chosen_key,
                    },
                    1,
                )
            raise

    # ── traverse / reverse_traverse ───────────────────────────────────────────
    # Async BFS visitors that mirror the TS GraphDefinition.traverse /
    # reverseTraverse. The visitor fn receives (node, ctx) and its return value
    # is stored in ctx[node_key]. Both handle sync and async visitors.

    async def _call_visitor(fn: Any, node: GraphNode, ctx: dict[str, Any]) -> Any:
        result = fn(node, ctx)
        if _inspect.isawaitable(result):
            return await result
        return result

    async def traverse(fn: Any, ctx: dict[str, Any] | None = None) -> Any:
        """Visit nodes root-first (BFS, shallow before deep)."""
        if root_node is None:
            return None
        ctx = ctx if ctx is not None else {}
        depths: dict[str, int] = {root_node.key: 0}
        seen: set[str] = {root_node.key}
        frontier: list[str] = [root_node.key]
        iterations = 0
        while frontier and iterations < MAX_TRAVERSAL_DEPTH:
            iterations += 1
            next_frontier: list[str] = []
            for node_key in frontier:
                depth = depths.get(node_key, 0)
                for child in get_child_nodes(node_key):
                    child_depth = depth + 1
                    if child.key not in depths or child_depth > depths[child.key]:
                        depths[child.key] = child_depth
                    if child.key not in seen:
                        seen.add(child.key)
                        next_frontier.append(child.key)
            frontier = next_frontier
        ordered = sorted(depths.keys(), key=lambda k: depths[k])
        for node_key in ordered:
            node = nodes.get(node_key)
            if node:
                ctx[node_key] = await _call_visitor(fn, node, ctx)
        return ctx.get(root_node.key)

    async def reverse_traverse(fn: Any, ctx: dict[str, Any] | None = None) -> Any:
        """Visit nodes leaf-first (terminal nodes first, root last)."""
        if root_node is None:
            return None
        ctx = ctx if ctx is not None else {}
        terminals = terminal_nodes()
        if not terminals:
            return None
        visited: set[str] = set()
        frontier: list[str] = [n.key for n in terminals]
        iterations = 0
        while frontier and iterations < MAX_TRAVERSAL_DEPTH:
            iterations += 1
            next_frontier: list[str] = []
            for node_key in frontier:
                if node_key in visited:
                    continue
                visited.add(node_key)
                if node_key == root_node.key:
                    continue
                node = nodes.get(node_key)
                if node:
                    ctx[node_key] = await _call_visitor(fn, node, ctx)
                for parent in get_parent_nodes(node_key):
                    if parent.key not in visited:
                        next_frontier.append(parent.key)
            frontier = next_frontier
        ctx[root_node.key] = await _call_visitor(fn, root_node, ctx)
        return ctx.get(root_node.key)

    graph_def = GraphDefinition(
        key=key,
        enabled=True,
        root=root_node,
        get_node=get_node,
        get_child_nodes=get_child_nodes,
        get_parent_nodes=get_parent_nodes,
        terminal_nodes=terminal_nodes,
        is_terminal=is_terminal,
        edges_from=edges_from,
        run_node=run_node,
        route=route,
        traverse=traverse,
        reverse_traverse=reverse_traverse,
    )

    return graph_def, graph_track_data, stream_route


async def resolve_graph(
    key: str,
    *,
    context: LDContext,
    handlers: list[ProviderHandler] | None = None,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
    registry: Any = None,
) -> GraphDefinition:
    """
    Resolves a graph flag into a topology definition without executing it.
    Callers should check ``definition['enabled']`` before traversing.

    ``context`` is a keyword-only argument, mirroring the TypeScript
    ``resolveGraph(key, { context, handlers, toolHandlers, registry })`` shape.
    """
    resolved_handlers = resolve_handlers(registry, handlers)
    resolved_tools = resolve_tools(registry, tool_handlers)
    options = {
        "handlers": resolved_handlers,
        "tool_handlers": resolved_tools,
        "registry": registry,
    }
    graph_def, _, _ = await _build_graph(key, context, options)
    return graph_def


class GraphInstance:
    """Return type of ``graph()``."""

    def __init__(
        self,
        key: str,
        options: dict[str, Any],
    ) -> None:
        self._key = key
        self._options = options
        self._cache: dict[
            str, tuple[GraphDefinition, TrackData, Callable[..., Any]]
        ] = {}

    async def invoke(
        self,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ProviderGraphResponse:
        from opentelemetry import trace

        from .judges import run_judges
        from .lifecycle import get_client

        ld_ctx = to_ld_context(get_client(), context)

        resolved_handlers = resolve_handlers(
            self._options.get("registry"), self._options.get("handlers")
        )
        resolved_tools = resolve_tools(
            self._options.get("registry"), self._options.get("tool_handlers")
        )
        resolved_options = {
            **self._options,
            "handlers": resolved_handlers,
            "tool_handlers": resolved_tools,
        }

        if not resolved_handlers:
            raise ValueError(
                "graph().invoke() requires handlers to be provided. Pass handlers in options, or "
                "use resolve_graph() with a framework-native runner."
            )

        try:
            cache_key: str | None = json.dumps(context, sort_keys=True)
        except (TypeError, ValueError):
            cache_key = None
        if cache_key is not None and cache_key in self._cache:
            built = self._cache[cache_key]
        else:
            built = await _build_graph(self._key, context, resolved_options)
            if cache_key is not None:
                if len(self._cache) >= MAX_GRAPH_CACHE_SIZE:
                    # Evict an arbitrary entry to keep the cache bounded.
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = built
        graph_def, graph_track_data, _ = built

        if not graph_def.enabled:
            raise ValueError(f'Agent graph "{self._key}" is disabled')

        tracer = trace.get_tracer("@launchdarkly/ai-server")
        with tracer.start_as_current_span("ld.ai.graph") as span:
            span.set_attribute("ld.ai.graph.key", self._key)

            start_time = time.monotonic()
            path: list[str] = []
            total_usage = {"input": 0, "output": 0, "total": 0}
            resolved_input = user_input or ""

            try:
                current: GraphNode | None = graph_def.root
                previous_node: GraphNode | None = None
                current_input = resolved_input
                last: dict[str, Any] | None = None
                visited: set[str] = set()
                steps = 0

                while current and steps < MAX_TRAVERSAL_DEPTH:
                    steps += 1
                    opts: dict[str, Any] = {"variables": variables}
                    if previous_node:
                        opts["from"] = previous_node
                    # History seeds the entry point only. After the root hop, nodes
                    # stay oriented through the string threading built below, so
                    # history is not re-sent to downstream handlers.
                    elif history:
                        opts["history"] = history

                    res = await graph_def.route(current, current_input, opts)
                    path.append(current.key)
                    total_usage["input"] += (
                        res["usage"].get("input", 0)
                        if isinstance(res["usage"], dict)
                        else 0
                    )
                    total_usage["output"] += (
                        res["usage"].get("output", 0)
                        if isinstance(res["usage"], dict)
                        else 0
                    )
                    total_usage["total"] += (
                        res["usage"].get("total", 0)
                        if isinstance(res["usage"], dict)
                        else 0
                    )
                    last = res

                    next_node = res.get("next")
                    if not next_node or next_node.key in visited:
                        break
                    visited.add(current.key)
                    previous_node = current
                    current = next_node
                    current_input = "\n\n".join(
                        [
                            f"[Original request]\n{resolved_input}",
                            f"[Previous agent response]\n{res['response']}",
                        ]
                    )

                final_response = (last or {}).get("response", "")

                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                client = get_client()
                client.track(
                    "$ld:ai:graph:duration:total", ld_ctx, graph_track_data, elapsed_ms
                )
                if total_usage["total"] > 0:
                    client.track(
                        "$ld:ai:graph:total_tokens",
                        ld_ctx,
                        graph_track_data,
                        total_usage["total"],
                    )
                client.track(
                    "$ld:ai:graph:path",
                    ld_ctx,
                    {**graph_track_data, "path": path},
                    len(path),
                )
                client.track(
                    "$ld:ai:graph:invocation_success", ld_ctx, graph_track_data, 1
                )

                # Optional graph-level judge run against the final response.
                judge_results: dict[str, JudgeResult] | None = None
                graph_judge: str | None = resolved_options.get("graph_judge")
                root_node = graph_def.root
                if graph_judge and root_node and resolved_handlers:
                    judge_handler = select_handler(
                        root_node.config,
                        root_node.meta,
                        resolved_handlers,
                        strict=False,
                    )
                    judge_results = await run_judges(
                        config={
                            "judgeConfiguration": {
                                "judges": [{"key": graph_judge, "samplingRate": 1}]
                            }
                        },
                        user_context=context,
                        handler=judge_handler,
                        handlers=resolved_handlers,
                        user_input=resolved_input,
                        llm_response=final_response,
                        base_track_data=graph_track_data,
                        tool_handlers=resolved_tools,
                        graph_key=self._key,
                    )

                return ProviderGraphResponse(
                    response=final_response,
                    # Named rather than splatted, so a new UsageDict member cannot silently arrive
                    # here from a dict that has no business filling it. Graph totals carry no cache
                    # breakdown: they are a sum across nodes, and the per-node detail is on the node's
                    # own spans.
                    usage=UsageDict(
                        input=total_usage["input"],
                        output=total_usage["output"],
                        total=total_usage["total"],
                    ),
                    judge_results=judge_results,
                )

            except Exception:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                client = get_client()
                client.track(
                    "$ld:ai:graph:duration:total", ld_ctx, graph_track_data, elapsed_ms
                )
                client.track(
                    "$ld:ai:graph:invocation_failure", ld_ctx, graph_track_data, 1
                )
                raise

    def stream(
        self,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[GraphStreamEvent, None]:
        """Stream graph traversal events.

        Deliberately not an ``async def`` with ``yield``: a generator body does not run until the
        first ``__anext__``, by which point a ``conversation_id`` / caller span scope wrapped around
        this call may have already exited. Binding the conversation id and capturing the OTel parent
        here — at call time — matches ``config().stream()`` and the TypeScript graph stream.
        """
        from opentelemetry import context as otel_context

        caller_context = otel_context.get_current()
        return bind_conversation_id(
            self._stream_events(user_input, context, variables, history, caller_context)
        )

    async def _stream_events(
        self,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
        caller_context: Any,
    ) -> AsyncGenerator[GraphStreamEvent, None]:
        from opentelemetry import context as otel_context
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode, set_span_in_context

        from .judges import run_judges
        from .lifecycle import get_client

        resolved_input = user_input or ""
        ld_ctx = to_ld_context(get_client(), context)

        resolved_handlers = resolve_handlers(
            self._options.get("registry"), self._options.get("handlers")
        )
        resolved_tools = resolve_tools(
            self._options.get("registry"), self._options.get("tool_handlers")
        )
        resolved_options = {
            **self._options,
            "handlers": resolved_handlers,
            "tool_handlers": resolved_tools,
        }

        if not resolved_handlers:
            raise ValueError(
                "graph().stream() requires handlers to be provided. Pass handlers in options, or "
                "use resolve_graph() with a framework-native runner."
            )

        try:
            cache_key: str | None = json.dumps(context, sort_keys=True)
        except (TypeError, ValueError):
            cache_key = None
        if cache_key is not None and cache_key in self._cache:
            built = self._cache[cache_key]
        else:
            built = await _build_graph(self._key, context, resolved_options)
            if cache_key is not None:
                if len(self._cache) >= MAX_GRAPH_CACHE_SIZE:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = built
        graph_def, graph_track_data, stream_route = built

        if not graph_def.enabled:
            raise ValueError(f'Agent graph "{self._key}" is disabled')

        tracer = trace.get_tracer("@launchdarkly/ai-server")
        span = tracer.start_span("ld.ai.graph", context=caller_context)
        span.set_attribute("ld.ai.graph.key", self._key)
        span_context = set_span_in_context(span, caller_context)
        ended: set[int] = set()
        start_time = time.monotonic()

        path: list[str] = []
        total_usage = {"input": 0, "output": 0, "total": 0}

        try:
            current: GraphNode | None = graph_def.root
            previous_node: GraphNode | None = None
            current_input = resolved_input
            last: dict[str, Any] | None = None
            visited: set[str] = set()
            steps = 0

            while current and steps < MAX_TRAVERSAL_DEPTH:
                steps += 1
                route_opts: dict[str, Any] = {"variables": variables}
                if previous_node:
                    route_opts["from"] = previous_node
                # History seeds the entry point only. After the root hop, nodes
                # stay oriented through the string threading built below, so
                # history is not re-sent to downstream handlers.
                elif history:
                    route_opts["history"] = history

                outcome: dict[str, Any] = {}
                async for event in bind_span_context(
                    stream_route(current, current_input, route_opts, outcome),
                    span_context,
                ):
                    yield event

                path.append(current.key)
                usage = outcome.get("usage") or {}
                total_usage["input"] += (
                    usage.get("input", 0) if isinstance(usage, dict) else 0
                )
                total_usage["output"] += (
                    usage.get("output", 0) if isinstance(usage, dict) else 0
                )
                total_usage["total"] += (
                    usage.get("total", 0) if isinstance(usage, dict) else 0
                )
                last = outcome

                next_node = outcome.get("next")
                if not next_node or next_node.key in visited:
                    break

                yield {
                    "type": "handoff",
                    "sourceKey": current.key,
                    "targetKey": next_node.key,
                }

                visited.add(current.key)
                previous_node = current
                current = next_node
                current_input = "\n\n".join(
                    [
                        f"[Original request]\n{resolved_input}",
                        f"[Previous agent response]\n{outcome.get('response', '')}",
                    ]
                )

            final_response = (last or {}).get("response", "")

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            client = get_client()
            client.track(
                "$ld:ai:graph:duration:total", ld_ctx, graph_track_data, elapsed_ms
            )
            if total_usage["total"] > 0:
                client.track(
                    "$ld:ai:graph:total_tokens",
                    ld_ctx,
                    graph_track_data,
                    total_usage["total"],
                )
            client.track(
                "$ld:ai:graph:path",
                ld_ctx,
                {**graph_track_data, "path": path},
                len(path),
            )
            client.track("$ld:ai:graph:invocation_success", ld_ctx, graph_track_data, 1)

            judge_results: dict[str, JudgeResult] | None = None
            graph_judge: str | None = resolved_options.get("graph_judge")
            root_node = graph_def.root
            if graph_judge and root_node and resolved_handlers:
                # Re-enter the graph span explicitly. bind_span_context only covers the
                # delegated per-node generator; this call runs in the generator body.
                token = otel_context.attach(span_context)
                try:
                    judge_handler = select_handler(
                        root_node.config,
                        root_node.meta,
                        resolved_handlers,
                        strict=False,
                    )
                    results = await run_judges(
                        config={
                            "judgeConfiguration": {
                                "judges": [{"key": graph_judge, "samplingRate": 1}]
                            }
                        },
                        user_context=context,
                        handler=judge_handler,
                        handlers=resolved_handlers,
                        user_input=resolved_input,
                        llm_response=final_response,
                        base_track_data=graph_track_data,
                        tool_handlers=resolved_tools,
                        graph_key=self._key,
                    )
                finally:
                    otel_context.detach(token)
                if results:
                    judge_results = results

            span.set_status(Status(StatusCode.OK))
            end_span_once(span, ended)

            done_event: GraphStreamEvent = {
                "type": "done",
                "response": final_response,
                "usage": total_usage,
            }
            if judge_results:
                done_event["judgeResults"] = judge_results
            yield done_event

        except Exception as err:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            client = get_client()
            client.track(
                "$ld:ai:graph:duration:total", ld_ctx, graph_track_data, elapsed_ms
            )
            client.track("$ld:ai:graph:invocation_failure", ld_ctx, graph_track_data, 1)
            span.record_exception(err)
            span.set_status(Status(StatusCode.ERROR, str(err)))
            end_span_once(span, ended)
            raise
        finally:
            end_span_once(span, ended, abandoned=True)


def graph(
    key: str,
    *,
    handlers: list[ProviderHandler] | None = None,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
    registry: Any = None,
    graph_judge: str | None = None,
) -> GraphInstance:
    """
    Creates a ``GraphInstance`` bound to *key*. Calling ``.invoke()`` fetches the
    graph topology from LD and executes it using model-driven routing.
    For framework-native runners use ``resolve_graph()`` instead.
    """
    options = {
        "handlers": handlers,
        "tool_handlers": tool_handlers,
        "registry": registry,
        "graph_judge": graph_judge,
    }
    return GraphInstance(key=key, options=options)
