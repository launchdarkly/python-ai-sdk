"""
toLangGraph — converts a GraphDefinition into a compiled LangGraph StateGraph,
mirroring the TypeScript toLangGraph implementation.
"""

from __future__ import annotations

import inspect
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any, TypedDict

from langchain_core.tools import InjectedToolCallId
from pydantic import BaseModel

from launchdarkly_ai_server import (
    GraphDefinition,
    GraphNode,
    LDContext,
    NativeGraphRunner,
    NativeTool,
    compose_history,
    get_client,
    make_track_data,
    parse_template,
    report_usage,
    set_ld_span_attributes,
    to_ld_context,
    tool_result_string,
)
from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from launchdarkly_ai_server.utils import make_graph_track_data, model_parameters

from ._version import PACKAGE_NAME, __version__
from .handler import _CHAT_OPENAI_FORWARDED_KEYS, _CHAT_OPENAI_MAPPING_KEYS
from .messages import to_lang_chain_messages

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


_HANDOFF_IGNORED = "Ignored: only the first transfer in a turn routes."


class _HandoffArgs(BaseModel):
    """Hand-off tool schema: the model passes no arguments, and LangChain injects
    the call id (a dict schema would skip the injection)."""

    tool_call_id: Annotated[str, InjectedToolCallId]


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
    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool],
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
            # Handlers may be sync or async. Same rule as handler._build_agent_tools
            # and tracking.wrap_tool_handlers — awaiting a plain return raises.
            result = fn(kwargs)
            if inspect.isawaitable(result):
                result = await result
            return tool_result_string(result)

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


class LangChainNativeGraphOptions(TypedDict, total=False):
    """Options for :func:`to_lang_graph`."""

    tool_handlers: Mapping[str, Callable[..., Any] | NativeTool]
    model_factory: Callable[[GraphNode], Any]
    """Builds the chat model for a node. Defaults to ``ChatOpenAI``."""
    context: LDContext
    """LaunchDarkly context used for tracking events. Required for LD telemetry."""


def to_lang_graph(
    def_promise: Awaitable[GraphDefinition],
    opts: LangChainNativeGraphOptions | None = None,
) -> NativeGraphRunner:
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
    report_usage("langchain-agents.toLangGraph", PACKAGE_NAME, __version__)
    _opts = opts or {}

    async def invoke(
        input_text: str = "",
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
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
        ToolMessage = lc_msgs.ToolMessage

        vs = variables or {}
        def_obj: GraphDefinition = await def_promise

        if not def_obj.enabled:
            raise ValueError(f'Agent graph "{def_obj.key}" is disabled')
        root = def_obj.root
        if not root:
            raise ValueError(f'Graph "{def_obj.key}" has no root node')

        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] = (
            _opts.get("tool_handlers") or {}
        )
        model_factory = _opts.get("model_factory")
        raw_ld_context = _opts.get("context")
        ld_context = (
            to_ld_context(get_client(), raw_ld_context)
            if raw_ld_context is not None
            else None
        )

        start_time = time.monotonic()
        run_id = str(uuid.uuid4())
        tracer_name = "@launchdarkly/ai-langchain-agents"
        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("launchdarkly.graph")
        else:
            span = None

        # One try covers setup and the run, so a setup error (a graph that
        # fails to compile) is recorded on the span and tracked as an invocation
        # failure, like a run error. The span ends exactly once, in the finally.
        try:
            if span:
                set_ld_span_attributes(
                    span,
                    {
                        "__ld": make_graph_track_data(def_obj.key, run_id),
                        "ldContext": raw_ld_context,
                    },
                )
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
                    model_cfg = node.config.get("model") or {}
                    kwargs = model_parameters(node.config)
                    if kwargs:
                        kwargs = select_forwarded_parameters(
                            kwargs,
                            _CHAT_OPENAI_FORWARDED_KEYS,
                            mapping_keys=_CHAT_OPENAI_MAPPING_KEYS,
                        )
                    kwargs["model"] = model_cfg.get("name") or "gpt-4o"
                    chat_model = lc_openai.ChatOpenAI(**kwargs)

                regular_tools = _build_node_tools(node, tool_handlers)

                # Handoff tools (return Command to route)
                lc_tools = importlib.import_module("langchain_core.tools")
                tool_fn = lc_tools.tool
                langgraph_types = importlib.import_module("langgraph.types")
                Command = langgraph_types.Command

                # The node's last model call sets "first" (its first hand-off call
                # id); the tools step sets "handed_off" when that call routes.
                routing: dict[str, Any] = {"first": None, "handed_off": False}

                handoff_tools = []
                for edge in outgoing:
                    target_key = _sanitize_name(edge.target_key)

                    async def _handoff_exec(
                        tool_call_id: Annotated[str, InjectedToolCallId],
                        _target: str = target_key,
                        _target_raw: str = edge.target_key,
                        _node: GraphNode = node,
                    ) -> Any:
                        # Only the first hand-off in a model message routes, as in graph().
                        if tool_call_id != routing["first"]:
                            return _HANDOFF_IGNORED
                        routing["handed_off"] = True
                        if ld_context:
                            td = make_track_data(_node, def_obj.key, run_id)
                            get_client().track(
                                "$ld:ai:graph:handoff_success", ld_context, td, 1
                            )
                        # The ToolMessage answers the call, so the next model call
                        # never sees an unanswered tool call.
                        return Command(
                            goto=_target,
                            update={
                                "messages": [
                                    ToolMessage(
                                        content=f"Handed off to {_target_raw}.",
                                        tool_call_id=tool_call_id,
                                        name=f"transfer_to_{_target}",
                                    )
                                ]
                            },
                        )

                    ht = tool_fn(
                        f"transfer_to_{_sanitize_name(edge.target_key)}",
                        _handoff_exec,
                        description=f"Transfer control to the {edge.target_key} agent",
                        args_schema=_HandoffArgs,
                    )
                    handoff_tools.append(ht)

                all_tools = regular_tools + handoff_tools

                async def _node_fn(
                    state: WorkflowState, _node: GraphNode = node
                ) -> dict[str, Any]:
                    if _node.key not in path:
                        index = len(path)
                        path.append(_node.key)
                        if ld_context:
                            node_td = make_track_data(_node, def_obj.key, run_id)
                            get_client().track(
                                "$ld:ai:graph:node",
                                ld_context,
                                {**node_td, "nodeKey": _node.key, "index": index},
                                1,
                            )
                    node_start = time.monotonic()

                    system_prompt = _build_system_prompt(_node, vs)
                    conv_messages: list[Any] = state.get("messages", [])
                    full_messages = (
                        [SystemMessage(system_prompt), *conv_messages]
                        if system_prompt
                        else list(conv_messages)
                    )

                    # No parallel_tool_calls: Bedrock Converse rejects it. The
                    # hand-off tools keep only the first hand-off instead.
                    bound = (
                        chat_model.bind_tools(all_tools) if all_tools else chat_model
                    )

                    result_msg = await bound.ainvoke(full_messages)
                    routing["first"] = next(
                        (
                            tc.get("id")
                            for tc in getattr(result_msg, "tool_calls", None) or []
                            if tc.get("name", "").startswith("transfer_to_")
                        ),
                        None,
                    )
                    routing["handed_off"] = False
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
                else:
                    # A non-terminal node always has hand-off tools. Ending the turn
                    # without one ends the run (multi-child) or moves to the child.
                    builder.add_conditional_edges(
                        node_key,
                        tools_condition,
                        {
                            "tools": f"{node_key}_tools",
                            "__end__": END
                            if is_multi_child
                            else _sanitize_name(outgoing[0].target_key),
                        },
                    )
                    # No static edge back to the node: after a hand-off the Command
                    # routes to the target and the node must not run again.
                    builder.add_conditional_edges(
                        f"{node_key}_tools",
                        lambda _s, _r=routing, _k=node_key: (
                            END if _r["handed_off"] else _k
                        ),
                        [node_key, END],
                    )

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

            # History is a root-only concern: it seeds the initial message state the
            # entry node reads. Downstream nodes are reached through handoffs and see
            # the accumulated graph state, never the original `history` array.
            initial_messages = (
                to_lang_chain_messages(
                    compose_history(history=history, user_input=input_text)
                )
                if history
                else [HumanMessage(input_text)]
            )

            result = await compiled.ainvoke({"messages": initial_messages})

            duration = int((time.monotonic() - start_time) * 1000)

            # Extract final output from last AI message
            result_messages = (
                result.get("messages", []) if isinstance(result, dict) else []
            )
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
                span.set_attribute("launchdarkly.graph.path", "->".join(path))
                span.set_attribute("gen_ai.usage.input_tokens", total_usage["input"])
                span.set_attribute("gen_ai.usage.output_tokens", total_usage["output"])
                span.set_attribute("gen_ai.usage.total_tokens", total_usage["total"])

            if ld_context:
                graph_td = make_graph_track_data(def_obj.key, run_id)
                client = get_client()
                client.track(
                    "$ld:ai:graph:duration:total", ld_context, graph_td, duration
                )
                client.track(
                    "$ld:ai:graph:total_tokens",
                    ld_context,
                    graph_td,
                    total_usage["total"],
                )
                client.track("$ld:ai:graph:invocation_success", ld_context, graph_td, 1)

            if span:
                span.set_status(SpanStatusCode.OK)
            return {"response": final_output, "usage": total_usage}
        except Exception as exc:
            if span:
                span.record_exception(exc)
                span.set_status(SpanStatusCode.ERROR, str(exc))
            if ld_context:
                get_client().track(
                    "$ld:ai:graph:invocation_failure",
                    ld_context,
                    make_graph_track_data(def_obj.key, run_id),
                    1,
                )
            raise
        finally:
            if span:
                span.end()

    # A class, not a SimpleNamespace, so mypy checks invoke against NativeGraphRunner.
    _invoke = invoke

    class _Runner:
        invoke = staticmethod(_invoke)

    return _Runner()
