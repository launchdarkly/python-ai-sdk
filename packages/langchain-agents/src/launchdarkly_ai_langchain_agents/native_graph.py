"""
toLangGraph — converts a GraphDefinition into a compiled LangGraph StateGraph,
mirroring the TypeScript toLangGraph implementation.
"""

from __future__ import annotations

import re
import time
import types
import uuid
from typing import Annotated, Any, TypedDict

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

try:
    from langgraph.graph.message import add_messages as add_messages

    _HAS_LANGGRAPH = True
except ImportError:
    # langgraph is an optional peer dependency; the symbol is only needed when
    # to_lang_graph() is actually called, at which point the dynamic import
    # inside invoke() will raise ImportError with a clear message.
    add_messages = None  # type: ignore[assignment]
    _HAS_LANGGRAPH = False


def _sanitize_name(key: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", key, flags=re.IGNORECASE)


def _build_system_prompt(node: GraphNode, variables: dict[str, Any]) -> str | None:
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

    lc_tools = importlib.import_module("langchain_core.tools")
    tool_fn = lc_tools.tool

    if not node.config.get("tools"):
        return []

    result = []
    for name, tool_cfg in node.config["tools"].items():
        schema = tool_cfg.get("parameters") or {}

        async def _handler(_name: str = name, **kwargs: Any) -> str:
            fn = tool_handlers.get(_name)
            if not fn or isinstance(fn, NativeTool):
                return ""
            res = await fn(kwargs)
            return str(res)

        t = tool_fn(
            name,
            _handler,
            description=tool_cfg.get("description", ""),
            args_schema=schema,
        )
        result.append(t)
    return result


def _extract_usage(msg: Any) -> dict[str, int]:
    meta = getattr(msg, "usage_metadata", None)
    if not meta:
        return {"input": 0, "output": 0, "total": 0}
    input_t = (
        meta.get("input_tokens", 0)
        if isinstance(meta, dict)
        else getattr(meta, "input_tokens", 0)
    )
    output_t = (
        meta.get("output_tokens", 0)
        if isinstance(meta, dict)
        else getattr(meta, "output_tokens", 0)
    )
    return {"input": input_t, "output": output_t, "total": input_t + output_t}


def to_lang_graph(
    def_promise: Any,
    opts: dict[str, Any] | None = None,
) -> Any:
    """
    Converts a resolved ``GraphDefinition`` into a compiled LangGraph
    ``StateGraph`` and returns a caller that runs it via ``compiled.invoke``.

    Example::

        from launchdarkly_ai_server import resolve_graph
        from launchdarkly_ai_langchain_agents import to_lang_graph

        result = await to_lang_graph(
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

        langgraph_mod = importlib.import_module("langgraph.graph")
        langgraph_prebuilt = importlib.import_module("langgraph.prebuilt")
        lc_msgs = importlib.import_module("langchain_core.messages")

        StateGraph = langgraph_mod.StateGraph
        START = langgraph_mod.START
        END = langgraph_mod.END
        ToolNode = langgraph_prebuilt.ToolNode
        tools_condition = langgraph_prebuilt.tools_condition

        HumanMessage = lc_msgs.HumanMessage
        SystemMessage = lc_msgs.SystemMessage

        vs = variables or {}
        def_obj: GraphDefinition = await def_promise

        if not def_obj.enabled:
            raise ValueError(f'Agent graph "{def_obj.key}" is disabled')
        root = def_obj.root
        if not root:
            raise ValueError(f'Graph "{def_obj.key}" has no root node')

        tool_handlers: dict[str, Any] = _opts.get("tool_handlers") or {}
        model_factory = _opts.get("model_factory")
        raw_ld_context = _opts.get("context")
        ld_context = (
            to_ld_context(get_client(), raw_ld_context)
            if raw_ld_context is not None
            else None
        )

        tracer_name = "@launchdarkly/ai-langchain-agents"
        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("ld.ai.graph")
            span.set_attribute("ld.ai.graph.key", def_obj.key)
        else:
            span = None

        start_time = time.monotonic()
        run_id = str(uuid.uuid4())
        path: list[str] = []
        total_usage = {"input": 0, "output": 0, "total": 0}
        edges_from = def_obj.edges_from

        # WorkflowState must reference add_messages from module-level scope.
        # With `from __future__ import annotations`, LangGraph resolves annotations
        # via get_type_hints() in the *module* global namespace — a local variable
        # would cause NameError at StateGraph(WorkflowState) time (AIC-2948).
        class WorkflowState(TypedDict):
            messages: Annotated[list[Any], add_messages]

        builder = StateGraph(WorkflowState)

        async def _traverse_node(node: GraphNode) -> None:
            node_key = _sanitize_name(node.key)
            outgoing = edges_from(node.key)
            is_terminal = node.is_terminal
            is_multi_child = len(outgoing) > 1

            # Get chat model
            if model_factory:
                chat_model = model_factory(node)
            else:
                lc_openai = importlib.import_module("langchain_openai")
                chat_model = lc_openai.ChatOpenAI(
                    model=node.config.get("model", {}).get("name", "gpt-4o")
                )

            regular_tools = _build_node_tools(node, tool_handlers)

            # Handoff tools (return Command to route)
            lc_tools = importlib.import_module("langchain_core.tools")
            tool_fn = lc_tools.tool
            langgraph_types = importlib.import_module("langgraph.types")
            Command = langgraph_types.Command

            handoff_tools = []
            for edge in outgoing:
                target_key = _sanitize_name(edge.target_key)

                async def _handoff_exec(
                    _target: str = target_key,
                    _node: GraphNode = node,
                ) -> Any:
                    if ld_context:
                        td = make_track_data(_node, def_obj.key, run_id)
                        get_client().track(
                            "$ld:ai:graph:handoff_success", ld_context, td, 1
                        )
                    return Command(goto=_target)

                ht = tool_fn(
                    f"transfer_to_{_sanitize_name(edge.target_key)}",
                    _handoff_exec,
                    description=f"Transfer control to the {edge.target_key} agent",
                    args_schema={},
                )
                handoff_tools.append(ht)

            all_tools = regular_tools + handoff_tools

            async def _node_fn(
                state: WorkflowState, _node: GraphNode = node
            ) -> dict[str, Any]:
                path.append(_node.key)
                node_start = time.monotonic()

                system_prompt = _build_system_prompt(_node, vs)
                conv_messages: list[Any] = state.get("messages", [])
                full_messages = (
                    [SystemMessage(system_prompt), *conv_messages]
                    if system_prompt
                    else list(conv_messages)
                )

                bound = (
                    chat_model.bind_tools(
                        all_tools,
                        **({"parallel_tool_calls": False} if is_multi_child else {}),
                    )
                    if all_tools
                    else chat_model
                )

                result_msg = await bound.ainvoke(full_messages)
                usage = _extract_usage(result_msg)
                total_usage["input"] += usage["input"]
                total_usage["output"] += usage["output"]
                total_usage["total"] += usage["total"]

                if ld_context:
                    td = make_track_data(_node, def_obj.key, run_id)
                    dur = int((time.monotonic() - node_start) * 1000)
                    client = get_client()
                    client.track("$ld:ai:duration:total", ld_context, td, dur)
                    client.track("$ld:ai:generation:success", ld_context, td, 1)
                    if usage["total"] > 0:
                        client.track(
                            "$ld:ai:tokens:total", ld_context, td, usage["total"]
                        )
                    if usage["input"] > 0:
                        client.track(
                            "$ld:ai:tokens:input", ld_context, td, usage["input"]
                        )
                    if usage["output"] > 0:
                        client.track(
                            "$ld:ai:tokens:output", ld_context, td, usage["output"]
                        )

                return {"messages": [result_msg]}

            builder.add_node(node_key, _node_fn)

            if all_tools:
                builder.add_node(f"{node_key}_tools", ToolNode(all_tools))

            # Edge wiring
            if node.key == root.key:
                builder.add_edge(START, node_key)

            if is_terminal:
                if all_tools:
                    builder.add_conditional_edges(
                        node_key,
                        tools_condition,
                        {"tools": f"{node_key}_tools", "__end__": END},
                    )
                    builder.add_edge(f"{node_key}_tools", node_key)
                else:
                    builder.add_edge(node_key, END)
            elif is_multi_child:
                if all_tools:
                    builder.add_conditional_edges(
                        node_key,
                        tools_condition,
                        {"tools": f"{node_key}_tools", "__end__": END},
                    )
                    builder.add_edge(f"{node_key}_tools", node_key)
                else:
                    builder.add_edge(node_key, END)
            else:
                child_key = _sanitize_name(outgoing[0].target_key)
                if all_tools:
                    builder.add_conditional_edges(
                        node_key,
                        tools_condition,
                        {"tools": f"{node_key}_tools", "__end__": child_key},
                    )
                    builder.add_edge(f"{node_key}_tools", node_key)
                else:
                    builder.add_edge(node_key, child_key)

        # Pre-order traversal (root first, to mirror TS traverse)
        pre_visited: set[str] = set()

        async def _pre_visit(node_key: str) -> None:
            if node_key in pre_visited:
                return
            pre_visited.add(node_key)
            node = def_obj.get_node(node_key)
            if node:
                await _traverse_node(node)
            for edge in edges_from(node_key):
                await _pre_visit(edge.target_key)

        await _pre_visit(root.key)

        compiled = builder.compile()

        try:
            result = await compiled.ainvoke({"messages": [HumanMessage(input_text)]})
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

        duration = int((time.monotonic() - start_time) * 1000)

        # Extract final output from last AI message
        result_messages = result.get("messages", []) if isinstance(result, dict) else []
        last_msg = result_messages[-1] if result_messages else None

        def _content_str(msg: Any) -> str:
            if msg is None:
                return ""
            c = msg.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else ""
                    for part in c
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return ""

        final_output = _content_str(last_msg)

        if span:
            span.set_attribute("ld.ai.graph.path", "->".join(path))
            span.set_attribute("gen_ai.usage.input_tokens", total_usage["input"])
            span.set_attribute("gen_ai.usage.output_tokens", total_usage["output"])
            span.set_attribute("gen_ai.usage.total_tokens", total_usage["total"])
            span.end()

        if ld_context:
            root_td = make_track_data(root, def_obj.key, run_id)
            client = get_client()
            client.track("$ld:ai:graph:duration:total", ld_context, root_td, duration)
            client.track(
                "$ld:ai:graph:total_tokens", ld_context, root_td, total_usage["total"]
            )
            client.track("$ld:ai:graph:path", ld_context, root_td, len(path))
            client.track("$ld:ai:graph:invocation_success", ld_context, root_td, 1)

        return {"response": final_output, "usage": total_usage}

    return types.SimpleNamespace(invoke=invoke)
