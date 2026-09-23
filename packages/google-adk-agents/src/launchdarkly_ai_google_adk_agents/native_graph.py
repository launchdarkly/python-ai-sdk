"""Native ADK workflow adapter for a LaunchDarkly agent graph."""

from __future__ import annotations

import time
import uuid
from typing import Any

from launchdarkly_ai_server import (
    get_client,
    make_track_data,
    parse_template,
    to_ld_context,
)

from .handler import (
    _session_event,
    event_text,
    history_contents,
    make_function_tool,
    native_of,
)
from .spans import usage_counts

try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.workflow import START, Workflow
except ImportError:  # pragma: no cover - unit tests patch these names
    LlmAgent = None
    InMemoryRunner = None
    FunctionTool = None
    Workflow = None
    START = "START"

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode
except ImportError:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    SpanStatusCode = None  # type: ignore[assignment,misc]

_TRACER = "@launchdarkly/ai-google-adk-agents"
_APP = "launchdarkly"


class _AdkAgents:
    def __init__(self, graph_def: Any, options: dict[str, Any]) -> None:
        self._graph = graph_def
        self._options = options

    async def invoke(
        self,
        user_input: str | None = None,
        context: Any = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        graph_def = self._graph
        if not getattr(graph_def, "enabled", True):
            raise RuntimeError(f"Graph {graph_def.key} is disabled")
        if getattr(graph_def, "root", None) is None:
            raise RuntimeError("Graph has no root")

        ld_context = None
        if context is not None:
            ld_context = to_ld_context(get_client(), context)
        span = _graph_span(graph_def.key)
        start = time.monotonic()
        run_id = str(uuid.uuid4())
        variables = variables or {}
        tool_handlers = self._options.get("tool_handlers") or {}
        handoff: dict[str, str | None] = {"target": None}
        agents = _build_agents(graph_def, variables, tool_handlers, handoff)
        root_agent = agents[graph_def.root.key]
        Workflow(name="graph", edges=[(START, root_agent)])

        path: list[str] = []
        usage = {"input": 0, "output": 0, "total": 0}
        output = ""
        try:
            current = graph_def.root
            seen: set[str] = set()
            seed = history
            while current is not None and current.key not in seen:
                seen.add(current.key)
                path.append(current.key)
                handoff["target"] = None
                output, step = await _run_agent(
                    agents[current.key],
                    user_input or "",
                    seed,
                )
                seed = None
                for key in usage:
                    usage[key] += step[key]
                target = handoff["target"]
                if target and ld_context is not None:
                    get_client().track(
                        "$ld:ai:graph:handoff_success",
                        ld_context,
                        make_track_data(current, graph_def.key, run_id),
                        1,
                    )
                current = graph_def.get_node(target) if target else None
            if span is not None:
                span.set_attribute("ld.ai.graph.path", "->".join(path))
                if SpanStatusCode is not None:
                    span.set_status(SpanStatusCode.OK)
            if ld_context is not None:
                _track_success(
                    graph_def.root,
                    graph_def.key,
                    run_id,
                    ld_context,
                    path,
                    usage,
                    start,
                )
            return {"response": output, "usage": usage}
        except Exception as exc:
            if span is not None:
                span.record_exception(exc)
                if SpanStatusCode is not None:
                    span.set_status(SpanStatusCode.ERROR, str(exc))
            if ld_context is not None:
                get_client().track(
                    "$ld:ai:graph:invocation_failure",
                    ld_context,
                    make_track_data(graph_def.root, graph_def.key, run_id),
                    1,
                )
            raise
        finally:
            if span is not None:
                span.end()


def to_adk_agents(graph_def: Any, **options: Any) -> _AdkAgents:
    """Compiles a LaunchDarkly agent graph into an ADK workflow."""
    return _AdkAgents(graph_def, options)


def _graph_span(key: str) -> Any:
    if trace is None:
        return None
    span = trace.get_tracer(_TRACER).start_span("ld.ai.graph")
    span.set_attribute("ld.ai.graph.key", key)
    return span


def _track_success(
    root: Any,
    graph_key: str,
    run_id: str,
    ld_context: Any,
    path: list[str],
    usage: dict[str, int],
    start: float,
) -> None:
    client = get_client()
    data = make_track_data(root, graph_key, run_id)
    duration = int((time.monotonic() - start) * 1000)
    client.track("$ld:ai:graph:duration:total", ld_context, data, duration)
    client.track("$ld:ai:graph:total_tokens", ld_context, data, usage["total"])
    client.track("$ld:ai:graph:path", ld_context, data, len(path))
    client.track("$ld:ai:graph:invocation_success", ld_context, data, 1)


def _build_agents(
    graph_def: Any,
    variables: dict[str, Any],
    tool_handlers: dict[str, Any],
    handoff: dict[str, str | None],
) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    _visit(graph_def, graph_def.root, agents, variables, tool_handlers, handoff)
    return agents


def _visit(
    graph_def: Any,
    node: Any,
    agents: dict[str, Any],
    variables: dict[str, Any],
    tool_handlers: dict[str, Any],
    handoff: dict[str, str | None],
) -> None:
    if node is None or node.key in agents:
        return
    targets: list[str] = []
    tools: list[Any] = []
    for edge in node.edges or []:
        target_key = getattr(edge, "target_key", None) or getattr(
            edge, "targetKey", None
        )
        if not target_key:
            continue
        targets.append(str(target_key))
        tools.append(_transfer_tool(str(target_key), handoff))
    config = node.config or {}
    tools = _node_tools(config, tool_handlers) + tools
    agents[node.key] = LlmAgent(
        name=_agent_name(node.key),
        model=str((config.get("model") or {}).get("name") or ""),
        instruction=parse_template(str(config.get("instructions") or ""), variables),
        tools=tools,
    )
    for target_key in targets:
        _visit(
            graph_def,
            graph_def.get_node(target_key),
            agents,
            variables,
            tool_handlers,
            handoff,
        )


def _node_tools(config: dict[str, Any], handlers: dict[str, Any]) -> list[Any]:
    built: list[Any] = []
    for key, spec in (config.get("tools") or {}).items():
        info = spec if isinstance(spec, dict) else {}
        name = str(info.get("name") or key)
        func = handlers.get(name) or handlers.get(key)
        if func is None or native_of(func) is not None:
            continue
        built.append(
            make_function_tool(
                FunctionTool,
                func,
                name=name,
                description=str(info.get("description") or ""),
                parameters=info.get("parameters"),
            )
        )
    return built


def _transfer_tool(target_key: str, handoff: dict[str, str | None]) -> Any:
    def _select() -> str:
        handoff["target"] = target_key
        return target_key

    return make_function_tool(FunctionTool, _select, name=f"transfer_to_{target_key}")


def _agent_name(key: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key)
    return cleaned or "agent"


async def _run_agent(
    agent: Any,
    user_input: str,
    history: list[dict[str, Any]] | None,
) -> tuple[str, dict[str, int]]:
    runner = InMemoryRunner(agent, app_name=_APP)
    session = await runner.session_service.create_session(app_name=_APP, user_id="user")
    if history:
        for content in history_contents(history):
            await runner.session_service.append_event(session, _session_event(content))
    output = ""
    usage = {"input": 0, "output": 0, "total": 0}
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=_message(user_input),
    ):
        if getattr(event, "partial", False):
            continue
        checker = getattr(event, "is_final_response", None)
        if callable(checker) and not checker():
            continue
        output = event_text(event)
        prompt, candidates, total = usage_counts(getattr(event, "usage_metadata", None))
        usage = {
            "input": usage["input"] + prompt,
            "output": usage["output"] + candidates,
            "total": usage["total"] + total,
        }
    return output, usage


def _message(text: str) -> Any:
    return type(
        "Content", (), {"role": "user", "parts": [type("Part", (), {"text": text})()]}
    )()
