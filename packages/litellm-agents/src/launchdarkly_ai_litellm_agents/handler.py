from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from agents import ModelSettings
from agents.agent_output import AgentOutputSchemaBase
from agents.extensions.models.litellm_model import LitellmModel
from agents.lifecycle import RunHooksBase
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    SpanUsage,
    config,
    content_to_text,
    create_handler,
    end_span_once,
    image_block_to_url,
    parse_template,
    set_input_content_attributes,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    set_usage_span_attributes,
    text_message,
)

ModelFactory = Callable[[str, dict[str, Any]], Any]
_MODEL_SETTING_FIELDS = {
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "tool_choice",
    "parallel_tool_calls",
    "max_tokens",
    "reasoning",
    "verbosity",
    "metadata",
    "store",
    "include_usage",
    "top_logprobs",
    "extra_headers",
    "extra_query",
    "extra_body",
}


def _json_schema(output_format: dict[str, Any]) -> dict[str, Any]:
    """Coerce an AI Config outputFormat into a JSON Schema object.

    LaunchDarkly stores structured-output metadata with ``type: json_schema``. Providers require
    a real JSON Schema root type, so that sentinel is rewritten to ``object``.
    """
    schema = dict(output_format)
    if schema.get("type") in (None, "json_schema", "json"):
        schema["type"] = "object"
    return schema


class _JsonSchemaOutput(AgentOutputSchemaBase):
    """Agents SDK output type that keeps the evaluated JSON Schema without looking like a dict.

    Inheriting from ``dict`` made ``isinstance(output_type, dict)`` true, so the SDK treated the
    envelope as the schema itself and OpenAI rejected ``type: json_schema``.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def __getitem__(self, key: str) -> Any:
        if key != "schema":
            raise KeyError(key)
        return self._schema

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "output"

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return False

    def validate_json(self, json_str: str) -> Any:
        return json.loads(json_str)


def _model_config(config_value: AiConfigRep) -> tuple[str, dict[str, Any]]:
    model = config_value.get("model") or {}
    raw = model.get("parameters")
    parameters = dict(raw) if isinstance(raw, dict) else {}
    for field in (
        "model",
        "messages",
        "tools",
        "stream",
        "stream_options",
        "response_format",
        "output_format",
    ):
        parameters.pop(field, None)
    return str(model.get("name") or ""), parameters


def _default_model_factory(name: str, parameters: dict[str, Any]) -> Any:
    return LitellmModel(
        model=name,
        base_url=parameters.get("base_url"),
        api_key=parameters.get("api_key"),
    )


def _model_settings(parameters: dict[str, Any]) -> ModelSettings:
    known = {
        key: value for key, value in parameters.items() if key in _MODEL_SETTING_FIELDS
    }
    extras = {
        key: value
        for key, value in parameters.items()
        if key
        not in _MODEL_SETTING_FIELDS | {"max_turns", "base_url", "api_key", "model"}
    }
    if extras:
        known["extra_args"] = extras
    return ModelSettings(**known)


def _instructions(config_value: AiConfigRep, variables: dict[str, Any]) -> str | None:
    if config_value.get("instructions"):
        return parse_template(config_value["instructions"], variables)
    system = [
        message
        for message in (config_value.get("messages") or [])
        if message.get("role") == "system"
    ]
    if not system:
        return None
    return "\n".join(
        parse_template(str(message.get("content", "")), variables) for message in system
    )


def _map_agent_content(role: str, content: Any) -> Any:
    if role == "assistant":
        if isinstance(content, list):
            text = content_to_text(content)
        else:
            text = str(content or "")
        return [{"type": "output_text", "text": text}]
    if not isinstance(content, list):
        return content
    parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "image":
            parts.append(
                {"type": "input_image", "image_url": image_block_to_url(block)}
            )
        elif block.get("type") == "text":
            parts.append({"type": "input_text", "text": block.get("text", "")})
    return parts


def _prompt(
    config_value: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    if not config_value.get("instructions"):
        for message in config_value.get("messages") or []:
            if message.get("role") == "system":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                content = parse_template(content, variables)
            turns.append({"role": message.get("role", "user"), "content": content})
    if history:
        turns.extend(
            {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
            for message in history
            if message.get("role") != "system"
        )
    if turns:
        if user_input:
            turns.append({"role": "user", "content": user_input})
        return [
            {
                "role": turn["role"],
                "content": _map_agent_content(str(turn["role"]), turn["content"]),
            }
            for turn in turns
        ]
    return user_input or ""


async def _call_tool(handler: Any, arguments: dict[str, Any]) -> Any:
    result = handler(arguments)
    return await result if inspect.isawaitable(result) else result


def _tools(config_value: AiConfigRep, handlers: dict[str, Any]) -> list[Any]:
    import importlib

    FunctionTool = importlib.import_module("agents").FunctionTool
    tools: list[Any] = []
    for name, definition in (config_value.get("tools") or {}).items():
        handler = handlers.get(name)
        if not callable(handler):
            continue

        async def invoke(_context: Any, args: str, _handler: Any = handler) -> Any:
            try:
                arguments = json.loads(args or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("Tool arguments must be valid JSON") from exc
            return await _call_tool(_handler, arguments)

        tools.append(
            FunctionTool(
                name=name,
                description=definition.get("description", "") or "",
                params_json_schema=definition.get("parameters") or {},
                on_invoke_tool=invoke,
            )
        )
    return tools


def _usage(result: Any) -> SpanUsage:
    wrapper = getattr(result, "context_wrapper", None)
    raw = getattr(wrapper, "usage", None)
    return SpanUsage(
        input=int(getattr(raw, "input_tokens", 0) or 0),
        output=int(getattr(raw, "output_tokens", 0) or 0),
    )


def _direct_usage(raw: Any) -> SpanUsage:
    return SpanUsage(
        input=int(getattr(raw, "input_tokens", 0) or 0),
        output=int(getattr(raw, "output_tokens", 0) or 0),
    )


def _root_span(config_value: AiConfigRep, variables: dict[str, Any]) -> Any:
    model_name = _model_config(config_value)[0]
    provider = str(
        (config_value.get("provider") or {}).get("name") or "litellm"
    ).lower()
    span = trace.get_tracer("@launchdarkly/ai-litellm-agents").start_span(
        "invoke_agent"
    )
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    set_model_identity_attributes(span, provider, model_name, "litellm")
    set_ld_span_attributes(span, variables)
    return span


def _parent_context(span: Any) -> Any:
    return trace.set_span_in_context(span)


def _model_span(config_value: AiConfigRep, parent: Any) -> Any:
    model_name = _model_config(config_value)[0]
    provider = str(
        (config_value.get("provider") or {}).get("name") or "litellm"
    ).lower()
    span = trace.get_tracer("@launchdarkly/ai-litellm-agents").start_span(
        f"chat {model_name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "chat")
    set_model_identity_attributes(span, provider, model_name, "litellm")
    return span


def _tool_span(name: str, call_id: str, parent: Any) -> Any:
    span = trace.get_tracer("@launchdarkly/ai-litellm-agents").start_span(
        f"execute_tool {name}", context=parent
    )
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", name)
    span.set_attribute("gen_ai.tool.call.id", call_id)
    return span


def _finish(span: Any, model_name: str, usage: SpanUsage, ended: set[int]) -> None:
    span.set_attribute("gen_ai.response.model", model_name)
    set_usage_span_attributes(span, usage)
    span.set_status(StatusCode.OK)
    end_span_once(span, ended)


def _succeed(span: Any, ended: set[int]) -> None:
    span.set_status(StatusCode.OK)
    end_span_once(span, ended)


def _fail(span: Any, error: BaseException, ended: set[int]) -> None:
    span.record_exception(error)
    span.set_status(StatusCode.ERROR, str(error))
    end_span_once(span, ended)


class _SpanningHooks(RunHooksBase[Any, Any]):
    def __init__(
        self,
        config_value: AiConfigRep,
        parent: Any,
        capture_content: bool,
        ended: set[int],
    ) -> None:
        self.config = config_value
        self.parent = parent
        self.capture_content = capture_content
        self.ended = ended
        self.open_model: Any = None
        self.open_tools: dict[str, Any] = {}
        self.total = SpanUsage()
        self.usage_reported = False

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: str | None,
        input_items: Any,
    ) -> None:
        del context, agent
        span = _model_span(self.config, self.parent)
        self.open_model = span
        if self.capture_content:
            set_input_content_attributes(
                span,
                True,
                system_instructions=system_prompt,
                messages=[text_message("user", str(input_items))],
            )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        del context, agent
        span = self.open_model
        self.open_model = None
        if span is None:
            return
        try:
            usage = _direct_usage(getattr(response, "usage", None))
            self.total.input += usage.input
            self.total.output += usage.output
            self.usage_reported = (
                self.usage_reported or getattr(response, "usage", None) is not None
            )
            if self.capture_content:
                set_output_content_attributes(
                    span,
                    True,
                    [text_message("assistant", str(response.output or ""))],
                )
            _finish(
                span,
                _model_config(self.config)[0],
                usage,
                self.ended,
            )
        except Exception as exc:
            _fail(span, exc, self.ended)
            raise

    @staticmethod
    def _tool_key(context: Any, tool: Any) -> str:
        return str(
            getattr(context, "tool_call_id", None)
            or getattr(tool, "name", None)
            or "tool"
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        del agent
        key = self._tool_key(context, tool)
        name = str(
            getattr(context, "tool_name", None) or getattr(tool, "name", None) or "tool"
        )
        span = _tool_span(name, key, self.parent)
        self.open_tools[key] = span
        if self.capture_content:
            raw = getattr(context, "tool_arguments", None)
            if raw is not None:
                try:
                    arguments = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    arguments = raw
                set_tool_call_content_attributes(span, True, arguments=arguments)

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        del agent
        span = self.open_tools.pop(self._tool_key(context, tool), None)
        if span is None:
            return
        try:
            if self.capture_content:
                set_tool_call_content_attributes(span, True, result=result)
            _succeed(span, self.ended)
        except Exception as exc:
            _fail(span, exc, self.ended)
            raise

    def fail_open(self, error: BaseException) -> None:
        for span in self.open_tools.values():
            _fail(span, error, self.ended)
        self.open_tools.clear()
        if self.open_model is not None:
            _fail(self.open_model, error, self.ended)
            self.open_model = None

    def close_open(self, *, abandoned: bool = False, cancelled: bool = False) -> None:
        for span in self.open_tools.values():
            end_span_once(
                span,
                self.ended,
                abandoned=abandoned,
                cancelled=cancelled,
            )
        self.open_tools.clear()
        if self.open_model is not None:
            end_span_once(
                self.open_model,
                self.ended,
                abandoned=abandoned,
                cancelled=cancelled,
            )
            self.open_model = None


def _agent_and_prompt(
    config_value: AiConfigRep,
    user_input: str | None,
    handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    model_factory: ModelFactory,
) -> tuple[Any, Any, str | None, int]:
    import importlib

    Agent = importlib.import_module("agents").Agent
    name, parameters = _model_config(config_value)
    model = model_factory(name, dict(parameters))
    instructions = _instructions(config_value, variables)
    output_format = config_value.get("outputFormat")
    output_type = (
        _JsonSchemaOutput(_json_schema(output_format)) if output_format else None
    )
    agent = Agent(
        name="assistant",
        model=model,
        model_settings=_model_settings(parameters),
        **({"instructions": instructions} if instructions else {}),
        **(
            {"tools": _tools(config_value, handlers)}
            if config_value.get("tools")
            else {}
        ),
        **({"output_type": output_type} if output_type is not None else {}),
    )
    return (
        agent,
        _prompt(config_value, user_input, variables, history),
        instructions,
        int(parameters.get("max_turns", 10)),
    )


def create_litellm_agents_handler(
    *,
    model_factory: ModelFactory | None = None,
    capture_content: bool = False,
) -> ProviderHandler:
    factory = model_factory or _default_model_factory

    async def call(
        config_value: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import importlib

        values = variables or {}
        agent, prompt, instructions, max_turns = _agent_and_prompt(
            config_value,
            user_input,
            tool_handlers or {},
            values,
            history,
            factory,
        )
        span = _root_span(config_value, values)
        parent = _parent_context(span)
        model_name = _model_config(config_value)[0]
        ended: set[int] = set()
        hooks = _SpanningHooks(config_value, parent, capture_content, ended)
        try:
            if capture_content:
                set_input_content_attributes(
                    span,
                    True,
                    system_instructions=instructions,
                    messages=[text_message("user", str(prompt))],
                )
            result = await importlib.import_module("agents").Runner.run(
                agent, prompt, max_turns=max_turns, hooks=hooks
            )
            output = result.final_output
            if capture_content:
                set_output_content_attributes(
                    span, True, [text_message("assistant", str(output or ""))]
                )
            usage = _usage(result)
            if not (usage.input or usage.output) and hooks.usage_reported:
                usage = hooks.total
            _finish(span, model_name, usage, ended)
            return {
                "output": output,
                "usage": {
                    "input_tokens": usage.input,
                    "output_tokens": usage.output,
                },
            }
        except Exception as exc:
            hooks.fail_open(exc)
            if hooks.usage_reported:
                span.set_attribute("gen_ai.response.model", model_name)
                set_usage_span_attributes(span, hooks.total)
            _fail(span, exc, ended)
            raise
        finally:
            hooks.close_open(cancelled=True)
            end_span_once(span, ended, cancelled=True)

    def stream(
        config_value: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream(
            config_value,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            factory,
            capture_content=capture_content,
        )

    return create_handler(
        ("*", "agent"),
        call,
        stream,
        capture_content=capture_content,
    )


async def _stream(
    config_value: AiConfigRep,
    user_input: str | None,
    handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    model_factory: ModelFactory,
    *,
    capture_content: bool,
) -> AsyncGenerator[dict[str, Any], None]:
    import importlib

    agent, prompt, instructions, max_turns = _agent_and_prompt(
        config_value, user_input, handlers, variables, history, model_factory
    )
    span = _root_span(config_value, variables)
    parent = _parent_context(span)
    streamed: Any = None
    completed = False
    ended: set[int] = set()
    hooks = _SpanningHooks(config_value, parent, capture_content, ended)
    cancelled = False
    try:
        if capture_content:
            set_input_content_attributes(
                span,
                True,
                system_instructions=instructions,
                messages=[text_message("user", str(prompt))],
            )
        streamed = importlib.import_module("agents").Runner.run_streamed(
            agent, prompt, max_turns=max_turns, hooks=hooks
        )
        collected = ""
        async for event in streamed.stream_events():
            if getattr(event, "type", None) != "raw_response_event":
                continue
            data = getattr(event, "data", None)
            if getattr(data, "type", None) == "response.output_text.delta":
                delta = getattr(data, "delta", "")
                if isinstance(delta, str) and delta:
                    collected += delta
                    yield {"type": "chunk", "text": delta}
        output = streamed.final_output
        if output is None:
            output = collected
        usage = _usage(streamed)
        if not (usage.input or usage.output) and hooks.usage_reported:
            usage = hooks.total
        _finish(span, _model_config(config_value)[0], usage, ended)
        completed = True
        yield {
            "type": "done",
            "output": output,
            "usage": {
                "input_tokens": usage.input,
                "output_tokens": usage.output,
            },
        }
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        hooks.fail_open(exc)
        if hooks.usage_reported:
            span.set_attribute("gen_ai.response.model", _model_config(config_value)[0])
            set_usage_span_attributes(span, hooks.total)
        _fail(span, exc, ended)
        completed = True
        raise
    finally:
        if streamed is not None and not completed:
            try:
                streamed.cancel()
            except Exception:
                pass
        hooks.close_open(abandoned=not completed, cancelled=cancelled)
        if not completed:
            if hooks.usage_reported:
                span.set_attribute(
                    "gen_ai.response.model", _model_config(config_value)[0]
                )
                set_usage_span_attributes(span, hooks.total)
            end_span_once(span, ended, abandoned=True, cancelled=cancelled)


def litellm_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    variables = kwargs.pop("variables", None)
    capture_content = bool(kwargs.pop("capture_content", False))
    model_factory = kwargs.pop("model_factory", None)
    return config(
        key=config_key,
        handler=create_litellm_agents_handler(
            model_factory=model_factory, capture_content=capture_content
        ),
        **kwargs,
    ).invoke(user_input, context, variables)
