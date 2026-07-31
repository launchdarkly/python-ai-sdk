"""
Claude Agents handler — uses the claude-agent-sdk `query()` + MCP tools.
Mirrors the TypeScript @launchdarkly/ai-claude-agents handler.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from launchdarkly_ai_server import (
    NATIVE_TOOL_KEY,
    AiConfigRep,
    LDContext,
    NativeTool,
    ProviderHandler,
    config,
    create_handler,
    parse_template,
    set_ld_span_attributes,
    set_openllmetry_completion,
    set_openllmetry_prompt,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

TOOL_MCP_NAME = "tool-mcp"
MCP_TOOL_PREFIX = f"mcp__{TOOL_MCP_NAME}__"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def build_tool_mcp(
    config_tools: dict[str, Any],
    handlers: dict[str, Any],
) -> Any:
    """Build an in-process SDK MCP server from LD config tools + handler functions."""
    import importlib

    sdk = importlib.import_module("claude_agent_sdk")
    tool_fn = sdk.tool
    create_server = sdk.create_sdk_mcp_server

    tool_objs = []
    for tool_name, tool_cfg in config_tools.items():

        async def _execute(args: Any, _name: str = tool_name) -> dict[str, Any]:
            handler = handlers.get(_name)
            if not handler:
                raise ValueError(f'No handler registered for tool "{_name}"')
            result = await handler(args) if _is_coroutine(handler) else handler(args)
            return {"content": [{"type": "text", "text": str(result)}]}

        mcp_tool = tool_fn(
            tool_name,
            tool_cfg.get("description", ""),
            tool_cfg.get("parameters") or {},
        )(_execute)
        tool_objs.append(mcp_tool)

    return create_server(name=TOOL_MCP_NAME, version="1.0.0", tools=tool_objs)


def partition_tools(
    config_tools: dict[str, Any] | None,
    tool_handlers: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Returns (native_tool_map, user_config_tools, native_tool_names).

    native_tool_map  : provider tool name → tracking stub (for PreToolUse hook)
    user_config_tools: LD tool definitions for user-defined tools (sent via MCP)
    native_tool_names: provider-facing names for query(options.tools=[...])
    """
    native_tool_map: dict[str, Any] = {}
    user_config_tools: dict[str, Any] = {}

    for ld_name, stub in tool_handlers.items():
        native = getattr(stub, NATIVE_TOOL_KEY, None)
        if isinstance(native, NativeTool):
            native_tool_map[native.tool_name] = stub
        elif config_tools and ld_name in config_tools:
            user_config_tools[ld_name] = config_tools[ld_name]

    return native_tool_map, user_config_tools, list(native_tool_map.keys())


def _format_history(history: list[dict[str, Any]] | None) -> str | None:
    if not history:
        return None
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "Conversation History:\n\n" + "\n".join(lines)


def build_prompt(
    config: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, str | None]:
    """Returns (prompt, system_prompt)."""
    safe_input = user_input or ""
    system_prompt: str | None = None

    if config.get("instructions"):
        system_prompt = parse_template(config["instructions"], variables)
    elif config.get("messages"):
        system_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        non_system = [m for m in config["messages"] if m.get("role") != "system"]
        system_prompt = (
            parse_template("\n".join(m["content"] for m in system_msgs), variables)
            if system_msgs
            else None
        )
        config_history = "\n".join(
            parse_template(m["content"], variables) for m in non_system
        )
        safe_input = (
            f"{config_history}\n\n{safe_input}" if config_history else safe_input
        )

    history_text = _format_history(history)
    if history_text:
        system_prompt = (
            f"{system_prompt}\n\n{history_text}" if system_prompt else history_text
        )

    return safe_input, system_prompt


def _is_coroutine(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


def _build_hooks(native_tool_map: dict[str, Any]) -> dict[str, Any] | None:
    if not native_tool_map:
        return None

    import importlib

    sdk = importlib.import_module("claude_agent_sdk")
    HookMatcher = sdk.HookMatcher

    async def _pre_tool_hook(
        input_data: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        tool_name = getattr(input_data, "tool_name", None) or (
            input_data.get("tool_name") if isinstance(input_data, dict) else None
        )
        stub = native_tool_map.get(tool_name) if tool_name is not None else None
        if stub and callable(stub):
            stub()
        return {}

    return {
        "PreToolUse": [HookMatcher(hooks=[_pre_tool_hook])],
    }


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def create_claude_agents_handler() -> ProviderHandler:
    """Creates a ``ProviderHandler`` for Anthropic's Claude via the claude-agent-sdk."""
    tracer_name = "@launchdarkly/ai-claude-agents"

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import importlib

        sdk = importlib.import_module("claude_agent_sdk")
        ClaudeAgentOptions = sdk.ClaudeAgentOptions
        ResultMessage = sdk.ResultMessage
        query_fn = sdk.query

        th = tool_handlers or {}
        vs = variables or {}

        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("claude.query")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", "anthropic")
            span.set_attribute(
                "gen_ai.request.model", config.get("model", {}).get("name", "")
            )
            set_ld_span_attributes(span, vs)
        else:
            span = None

        prompt, system_prompt = build_prompt(config, user_input, vs, history)

        # Append outputFormat instruction to system prompt
        if config.get("outputFormat"):
            schema_instr = f"Respond with valid JSON matching this schema:\n{json.dumps(config['outputFormat'])}"
            system_prompt = (
                f"{system_prompt}\n\n{schema_instr}" if system_prompt else schema_instr
            )

        if span:
            span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt})
            prompt_msgs: list[dict[str, str]] = []
            if system_prompt:
                prompt_msgs.append({"role": "system", "content": system_prompt})
            prompt_msgs.append({"role": "user", "content": prompt})
            set_openllmetry_prompt(span, prompt_msgs)

        try:
            native_tool_map, user_config_tools, native_tool_names = partition_tools(
                config.get("tools"), th
            )

            tool_mcp = (
                await build_tool_mcp(user_config_tools, th)
                if user_config_tools
                else None
            )
            mcp_allowed = [MCP_TOOL_PREFIX + n for n in user_config_tools]
            all_allowed = mcp_allowed + native_tool_names

            hooks = _build_hooks(native_tool_map)

            options = ClaudeAgentOptions(
                allowed_tools=all_allowed if all_allowed else [],
                mcp_servers={TOOL_MCP_NAME: tool_mcp} if tool_mcp else {},
                hooks=hooks or {},
                **({"system_prompt": system_prompt} if system_prompt else {}),
                **({"tools": native_tool_names} if native_tool_names else {}),
            )

            output = ""
            raw_usage: dict[str, Any] = {}
            span_ended = False

            # Hold an explicit reference so we can call aclose() in the finally
            # block below.  A bare `return` inside `async for` abandons the
            # generator — Python's asyncio finalizer later tries to aclose() it
            # and raises RuntimeError if the generator is suspended inside a real
            # await in the SDK (AIC-2950).  See Appendix A.4 in TESTING.md.
            gen = query_fn(prompt=prompt, options=options)
            try:
                async for message in gen:
                    if isinstance(message, ResultMessage):
                        output = message.result or ""
                        raw_usage = message.usage or {}
                        input_tokens = int(raw_usage.get("input_tokens", 0))
                        output_tokens = int(raw_usage.get("output_tokens", 0))
                        if span:
                            span.set_attribute(
                                "gen_ai.response.model",
                                config.get("model", {}).get("name", ""),
                            )
                            span.set_attribute(
                                "gen_ai.usage.input_tokens", input_tokens
                            )
                            span.set_attribute(
                                "gen_ai.usage.output_tokens", output_tokens
                            )
                            span.set_attribute(
                                "gen_ai.usage.total_tokens",
                                input_tokens + output_tokens,
                            )
                            span.add_event(
                                "gen_ai.content.completion",
                                {
                                    "gen_ai.completion": output
                                    if isinstance(output, str)
                                    else json.dumps(output)
                                },
                            )
                            set_openllmetry_completion(
                                span,
                                output
                                if isinstance(output, str)
                                else json.dumps(output),
                                {
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                },
                            )
                            span.set_status(SpanStatusCode.OK)
                            span.end()
                            span_ended = True
                        break
            finally:
                await gen.aclose()

            if span and not span_ended:
                span.set_status(SpanStatusCode.OK)
                span.end()
            return {"output": output, "usage": raw_usage}

        except Exception as exc:
            if span:
                span.record_exception(exc)
                span.set_status(SpanStatusCode.ERROR, str(exc))
                span.end()
            raise

    def _stream_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream_gen(
            config, user_input, tool_handlers or {}, variables or {}, history
        )

    return create_handler(("Anthropic", "agent"), _call_impl, _stream_impl)  # type: ignore[arg-type]


async def _stream_gen(
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    import importlib

    sdk = importlib.import_module("claude_agent_sdk")
    ClaudeAgentOptions = sdk.ClaudeAgentOptions
    ResultMessage = sdk.ResultMessage
    StreamEvent = sdk.StreamEvent
    query_fn = sdk.query

    tracer_name = "@launchdarkly/ai-claude-agents"
    if _HAS_OTEL:
        span = trace.get_tracer(tracer_name).start_span("claude.query.stream")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute(
            "gen_ai.request.model", config.get("model", {}).get("name", "")
        )
        set_ld_span_attributes(span, variables)
    else:
        span = None

    prompt, system_prompt = build_prompt(config, user_input, variables, history)
    if span:
        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt})
        prompt_msgs: list[dict[str, str]] = []
        if system_prompt:
            prompt_msgs.append({"role": "system", "content": system_prompt})
        prompt_msgs.append({"role": "user", "content": prompt})
        set_openllmetry_prompt(span, prompt_msgs)

    try:
        native_tool_map, user_config_tools, native_tool_names = partition_tools(
            config.get("tools"), tool_handlers
        )
        tool_mcp = (
            await build_tool_mcp(user_config_tools, tool_handlers)
            if user_config_tools
            else None
        )
        mcp_allowed = [MCP_TOOL_PREFIX + n for n in user_config_tools]
        all_allowed = mcp_allowed + native_tool_names
        hooks = _build_hooks(native_tool_map)

        options = ClaudeAgentOptions(
            allowed_tools=all_allowed if all_allowed else [],
            mcp_servers={TOOL_MCP_NAME: tool_mcp} if tool_mcp else {},
            hooks=hooks or {},
            include_partial_messages=True,
            **({"system_prompt": system_prompt} if system_prompt else {}),
            **({"tools": native_tool_names} if native_tool_names else {}),
        )

        full_output = ""

        async for message in query_fn(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
                event = message.event
                if (
                    event.get("type") == "content_block_delta"
                    and event.get("delta", {}).get("type") == "text_delta"
                ):
                    text = event["delta"].get("text", "")
                    if text:
                        yield {"type": "chunk", "text": text}
                        full_output += text
            elif isinstance(message, ResultMessage):
                raw_usage = message.usage or {}
                input_tokens = int(raw_usage.get("input_tokens", 0))
                output_tokens = int(raw_usage.get("output_tokens", 0))
                final_output = message.result or full_output
                if span:
                    span.set_attribute(
                        "gen_ai.response.model", config.get("model", {}).get("name", "")
                    )
                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                    span.set_attribute(
                        "gen_ai.usage.total_tokens", input_tokens + output_tokens
                    )
                    span.add_event(
                        "gen_ai.content.completion",
                        {
                            "gen_ai.completion": final_output
                            if isinstance(final_output, str)
                            else json.dumps(final_output)
                        },
                    )
                    set_openllmetry_completion(
                        span,
                        final_output
                        if isinstance(final_output, str)
                        else json.dumps(final_output),
                        {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    )
                    span.set_status(SpanStatusCode.OK)
                    span.end()
                yield {"type": "done", "output": final_output, "usage": raw_usage}
                return

        if span:
            span.set_status(SpanStatusCode.OK)
            span.end()
        yield {"type": "done", "output": full_output, "usage": {}}

    except Exception as exc:
        if span:
            span.record_exception(exc)
            span.set_status(SpanStatusCode.ERROR, str(exc))
            span.end()
        raise


def claude_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    variables = kwargs.pop("variables", None)
    return config(
        key=config_key, handler=create_claude_agents_handler(), **kwargs
    ).invoke(user_input, context, variables=variables)
