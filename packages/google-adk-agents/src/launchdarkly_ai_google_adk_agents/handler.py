"""Google ADK agent handler.

Gemini Developer API is the default transport. Vertex is an explicit opt-in and never
shares an API key with that path. A failed Gemini call is not retried on Vertex.

Non-Gemini models go through ADK's own ``LiteLlm`` adapter (``provider/model``). That
is ``google.adk.models.lite_llm``, not the LaunchDarkly LiteLLM package.
"""

from __future__ import annotations

import base64
import copy
import inspect
import os
from types import SimpleNamespace
from typing import Any

from launchdarkly_ai_server import (
    NATIVE_TOOL_KEY,
    AiConfigRep,
    NativeTool,
    config,
    create_handler,
    end_unfinished_spans,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    text_message,
)

from . import spans as spanlib

try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.models.google_llm import Gemini
    from google.adk.plugins.base_plugin import BasePlugin
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.function_tool import FunctionTool
except ImportError:  # pragma: no cover - unit tests patch these names
    LlmAgent = None
    Gemini = None
    InMemoryRunner = None
    FunctionTool = None

    class BasePlugin:  # type: ignore[no-redef]
        def __init__(self, name: str) -> None:
            self.name = name


try:
    from google.adk.models.lite_llm import LiteLlm
except Exception:  # pragma: no cover - extensions extra, or the package is absent
    LiteLlm = None

try:
    from google.adk.events.event import Event
except ImportError:  # pragma: no cover
    Event = None

try:
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai_types = None

_GOOGLE_PROVIDERS = frozenset(
    {"google", "gemini", "vertex", "google-genai", "gcp.gemini"}
)
_APP_NAME = "launchdarkly"


class LaunchDarklyTelemetryPlugin(BasePlugin):  # type: ignore[misc]
    """ADK plugin that opens ``chat`` and ``execute_tool`` spans around the runner."""

    def __init__(
        self, config: AiConfigRep, parent: Any, capture_content: bool = False
    ) -> None:
        super().__init__("launchdarkly")
        self.config = config
        self.parent = parent
        self.capture_content = capture_content
        self._tool_spans: dict[str, Any] = {}
        self._native_stubs: dict[str, Any] = {}
        self._instruction = ""
        self._user_text = ""

    def bind_turn(self, instruction: str, user_text: str, handlers: Any) -> None:
        self._instruction = instruction
        self._user_text = user_text
        self._native_stubs = native_stubs(handlers)

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> None:
        span = spanlib.start_model_span(self.config, self.parent)
        messages = [text_message("user", self._user_text)] if self._user_text else []
        set_input_content_attributes(
            span,
            self.capture_content,
            system_instructions=self._instruction or None,
            messages=messages,
        )
        output = event_text(llm_response)
        if output:
            set_output_content_attributes(
                span, self.capture_content, [text_message("assistant", output)]
            )
        prompt, candidates, _total = spanlib.usage_counts(
            getattr(llm_response, "usage_metadata", None)
        )
        spanlib.finish_model_span(
            span, self.config, spanlib.span_usage_from_counts(prompt, candidates)
        )

    async def before_tool_callback(
        self, *, tool: Any, tool_args: Any, tool_context: Any
    ) -> None:
        name = str(getattr(tool, "name", "") or "tool")
        call_id = _call_id(tool_context, name)
        self._tool_spans[call_id] = spanlib.start_tool_span(name, call_id, self.parent)
        stub = self._native_stubs.get(name)
        if callable(stub):
            stub()

    async def after_tool_callback(
        self, *, tool: Any, tool_args: Any, tool_context: Any, result: Any
    ) -> None:
        name = str(getattr(tool, "name", "") or "tool")
        call_id = _call_id(tool_context, name)
        span = self._tool_spans.pop(call_id, None)
        set_tool_call_content_attributes(
            span, self.capture_content, arguments=tool_args, result=result
        )
        spanlib.succeed_span(span)

    async def on_tool_error_callback(
        self, *, tool: Any, tool_args: Any, tool_context: Any, error: BaseException
    ) -> dict[str, str]:
        name = str(getattr(tool, "name", "") or "tool")
        call_id = _call_id(tool_context, name)
        span = self._tool_spans.pop(call_id, None) or spanlib.start_tool_span(
            name, call_id, self.parent
        )
        spanlib.fail_span(span, error)
        return {"error": str(error)}

    def abandon_open_spans(self, ended: set[int] | None = None) -> None:
        tracker = ended if isinstance(ended, set) else set()
        for span in list(self._tool_spans.values()):
            spanlib.abandon_open_spans([span], tracker)
        self._tool_spans.clear()

    def close_open_spans(self, err: Exception, ended: set[int] | None = None) -> None:
        tracker = ended if isinstance(ended, set) else set()
        for span in list(self._tool_spans.values()):
            spanlib.fail_span(span, err, tracker)
        self._tool_spans.clear()


def history_contents(history: list[dict[str, Any]] | None) -> list[Any]:
    """Prior turns as ADK contents. Image bytes are raw, not a data URL."""
    contents: list[Any] = []
    for message in history or []:
        role = "model" if message.get("role") == "assistant" else "user"
        contents.append(_content(role, _parts(message.get("content"))))
    return contents


def create_google_adk_agents_handler(
    *,
    api_key: str | None = None,
    use_vertexai: bool = False,
    project: str | None = None,
    location: str | None = None,
    model: Any = None,
    capture_content: bool = False,
) -> Any:
    """Builds a wildcard agent handler. Vertex project and location are required up front."""
    if use_vertexai:
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        if not project or not location:
            raise ValueError(
                "Vertex mode requires a Google Cloud project and location "
                "(arguments or GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION)"
            )

    async def _run(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        variables = variables or {}
        ended: set[int] = set()
        root = spanlib.start_root_span(cfg, variables)
        plugin = LaunchDarklyTelemetryPlugin(
            cfg, spanlib.parent_context_of(root), capture_content
        )
        try:
            output, usage = await _execute(
                cfg,
                user_input,
                tool_handlers,
                variables,
                history,
                plugin,
                api_key=api_key,
                use_vertexai=use_vertexai,
                project=project,
                location=location,
                model=model,
                output_schema=cfg.get("outputFormat"),
            )
            spanlib.finish_root_span(
                root,
                cfg,
                spanlib.span_usage_from_counts(usage["input"], usage["output"]),
            )
            spanlib.succeed_span(root)
            return {"output": output, "usage": usage}
        except Exception as err:
            spanlib.fail_span(root, err, ended)
            plugin.close_open_spans(err, ended)
            raise
        finally:
            plugin.abandon_open_spans(ended)
            end_unfinished_spans(root)

    async def _stream(
        cfg: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> Any:
        variables = variables or {}
        ended: set[int] = set()
        root = spanlib.start_root_span(cfg, variables)
        plugin = LaunchDarklyTelemetryPlugin(
            cfg, spanlib.parent_context_of(root), capture_content
        )
        try:
            runner, session, user_id = await _open_run(
                cfg,
                user_input,
                tool_handlers,
                variables,
                history,
                plugin,
                api_key=api_key,
                use_vertexai=use_vertexai,
                project=project,
                location=location,
                model=model,
                output_schema=None,
            )
            output = ""
            usage = _zero_usage()
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=_user_message(user_input),
            ):
                text = event_text(event)
                if getattr(event, "partial", False):
                    if text:
                        yield {"type": "chunk", "text": text}
                    continue
                if _is_final(event):
                    output = text
                    usage = _add_usage(usage, event)
            yield {"type": "done", "output": output, "usage": usage}
            spanlib.finish_root_span(
                root,
                cfg,
                spanlib.span_usage_from_counts(usage["input"], usage["output"]),
            )
            spanlib.succeed_span(root)
        except Exception as err:
            spanlib.fail_span(root, err, ended)
            plugin.close_open_spans(err, ended)
            raise
        finally:
            plugin.abandon_open_spans(ended)
            end_unfinished_spans(root)

    return create_handler(("*", "agent"), _run, _stream, capture_content)


def google_adk_agents(
    config_key: str,
    user_input: str,
    context: Any,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: one wildcard ADK handler and ``config().invoke()``."""
    variables = kwargs.pop("variables", None)
    capture_content = kwargs.pop("capture_content", False)
    api_key = kwargs.pop("api_key", None)
    use_vertexai = kwargs.pop("use_vertexai", False)
    project = kwargs.pop("project", None)
    location = kwargs.pop("location", None)
    model = kwargs.pop("model", None)
    return config(
        key=config_key,
        handler=create_google_adk_agents_handler(
            api_key=api_key,
            use_vertexai=use_vertexai,
            project=project,
            location=location,
            model=model,
            capture_content=capture_content,
        ),
        **kwargs,
    ).invoke(user_input, context, variables=variables)


async def _execute(
    cfg: AiConfigRep,
    user_input: str | None,
    tool_handlers: dict[str, Any] | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    plugin: LaunchDarklyTelemetryPlugin,
    *,
    api_key: str | None,
    use_vertexai: bool,
    project: str | None,
    location: str | None,
    model: Any,
    output_schema: Any,
) -> tuple[str, dict[str, int]]:
    runner, session, user_id = await _open_run(
        cfg,
        user_input,
        tool_handlers,
        variables,
        history,
        plugin,
        api_key=api_key,
        use_vertexai=use_vertexai,
        project=project,
        location=location,
        model=model,
        output_schema=output_schema,
    )
    output = ""
    usage = _zero_usage()
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=_user_message(user_input),
    ):
        if _is_final(event):
            output = event_text(event)
            usage = _add_usage(usage, event)
    return output, usage


async def _open_run(
    cfg: AiConfigRep,
    user_input: str | None,
    tool_handlers: dict[str, Any] | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    plugin: LaunchDarklyTelemetryPlugin,
    *,
    api_key: str | None,
    use_vertexai: bool,
    project: str | None,
    location: str | None,
    model: Any,
    output_schema: Any,
) -> tuple[Any, Any, str]:
    resolved = _resolve_model(
        cfg,
        model,
        api_key=api_key,
        use_vertexai=use_vertexai,
        project=project,
        location=location,
    )
    agent_kwargs: dict[str, Any] = {
        "name": "agent",
        "model": resolved,
        "instruction": _instruction(cfg, variables),
        "tools": _tools(cfg, tool_handlers),
    }
    if output_schema is not None:
        agent_kwargs["output_schema"] = normalize_output_schema(output_schema)
    plugin.bind_turn(_instruction(cfg, variables), user_input or "", tool_handlers)
    agent = LlmAgent(**agent_kwargs)
    runner = InMemoryRunner(agent, app_name=_APP_NAME, plugins=[plugin])
    user_id = _user_id(variables)
    session = await runner.session_service.create_session(
        app_name=_APP_NAME, user_id=user_id
    )
    for content in history_contents(history):
        await runner.session_service.append_event(session, _session_event(content))
    return runner, session, user_id


def _resolve_model(
    cfg: AiConfigRep,
    injected: Any,
    *,
    api_key: str | None,
    use_vertexai: bool,
    project: str | None,
    location: str | None,
) -> Any:
    if injected is not None:
        if callable(injected) and not isinstance(injected, type):
            return injected(cfg)
        return injected
    provider = str((cfg.get("provider") or {}).get("name") or "")
    name = str((cfg.get("model") or {}).get("name") or "")
    if "/" in name or provider.lower() not in _GOOGLE_PROVIDERS:
        model_id = name if "/" in name else f"{provider.lower()}/{name}"
        if LiteLlm is None:
            raise RuntimeError(
                "Non-Gemini models require LiteLLM (install google-adk[extensions])"
            )
        return LiteLlm(model=model_id)
    client_kwargs: dict[str, Any] = {}
    if use_vertexai:
        client_kwargs = {"vertexai": True, "project": project, "location": location}
    elif api_key:
        client_kwargs = {"api_key": api_key}
    if client_kwargs:
        return Gemini(model=name, client_kwargs=client_kwargs)
    return Gemini(model=name)


def _instruction(cfg: AiConfigRep, variables: dict[str, Any]) -> str:
    raw = cfg.get("instructions")
    if not raw:
        parts = [
            message.get("content", "")
            for message in cfg.get("messages") or []
            if message.get("role") == "system"
            and isinstance(message.get("content"), str)
        ]
        raw = "\n\n".join(parts)
    return parse_template(str(raw or ""), variables)


def normalize_output_schema(schema: Any) -> Any:
    """Copy a JSON schema and mark every property required."""
    if not isinstance(schema, dict):
        return schema
    copied = copy.deepcopy(schema)
    properties = copied.get("properties")
    if isinstance(properties, dict):
        copied.setdefault("required", list(properties))
        copied.setdefault("additionalProperties", False)
    return copied


def native_of(func: Any) -> NativeTool | None:
    if isinstance(func, NativeTool):
        return func
    native = getattr(func, NATIVE_TOOL_KEY, None)
    return native if isinstance(native, NativeTool) else None


def native_stubs(handlers: dict[str, Any] | None) -> dict[str, Any]:
    stubs: dict[str, Any] = {}
    for key, func in (handlers or {}).items():
        native = native_of(func)
        if native is None or not callable(func):
            continue
        stubs[native.tool_name] = func
        stubs[str(key)] = func
    return stubs


def _tools(cfg: AiConfigRep, handlers: dict[str, Any] | None) -> list[Any]:
    catalog = cfg.get("tools") or {}
    built: list[Any] = []
    for key, spec in catalog.items():
        info = spec if isinstance(spec, dict) else {}
        name = str(info.get("name") or key)
        func = (handlers or {}).get(name) or (handlers or {}).get(key)
        if func is None:
            continue
        native = native_of(func)
        if native is not None:
            builtin = builtin_tool(native.tool_name)
            if builtin is not None:
                built.append(builtin)
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


def adapt_handler(func: Any, name: str) -> Any:
    """ADK calls tools with keywords. Mapping-style handlers still get one dict."""
    params = [
        param
        for param in inspect.signature(func).parameters.values()
        if param.name not in {"self", "tool_context", "cls"}
    ]
    single_mapping = len(params) == 1 and params[0].name in {"args", "input"}

    def _call(*args: Any, **kwargs: Any) -> Any:
        if single_mapping and kwargs and not args:
            return func(kwargs)
        return func(*args, **kwargs)

    if inspect.iscoroutinefunction(func):
        adapted: Any = _async_adapter(_call)
    else:
        adapted = _sync_adapter(_call)
    adapted.__name__ = name
    adapted.__doc__ = inspect.getdoc(func) or ""
    return adapted


def _sync_adapter(call: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return call(*args, **kwargs)

    return wrapped


def _async_adapter(call: Any) -> Any:
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        return await call(*args, **kwargs)

    return wrapped


def make_function_tool(
    function_tool: Any,
    func: Any,
    *,
    name: str,
    description: str = "",
    parameters: Any = None,
) -> Any:
    adapted = adapt_handler(func, name)
    try:
        return function_tool(
            adapted, name=name, description=description, parameters=parameters
        )
    except TypeError:
        tool = function_tool(adapted)
        for attr, value in (
            ("name", name),
            ("description", description),
            ("parameters", parameters),
        ):
            try:
                setattr(tool, attr, value)
            except Exception:
                continue
        return tool


def builtin_tool(name: str) -> Any:
    try:
        import google.adk.tools as tools
    except ImportError:
        return None
    exported = getattr(tools, name, None)
    if exported is None or exported is FunctionTool:
        return None
    return exported


def _call_id(tool_context: Any, fallback: str) -> str:
    for attr in ("function_call_id", "functionCallId"):
        value = (
            tool_context.get(attr)
            if isinstance(tool_context, dict)
            else getattr(tool_context, attr, None)
        )
        if value:
            return str(value)
    return fallback


def _user_id(variables: dict[str, Any]) -> str:
    context = variables.get("ldContext")
    if isinstance(context, dict) and context.get("key"):
        return str(context["key"])
    return "user"


def _user_message(user_input: str | None) -> Any:
    return _content("user", [_text_part(user_input or "")])


def _session_event(content: Any) -> Any:
    author = getattr(content, "role", None) or "user"
    if Event is not None:
        try:
            return Event(author=author, content=content)
        except Exception:
            pass
    return SimpleNamespace(author=author, content=content)


def event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    if content is None and isinstance(event, dict):
        content = event.get("content")
    parts = getattr(content, "parts", None) if content is not None else None
    if parts is None and isinstance(content, dict):
        parts = content.get("parts")
    texts: list[str] = []
    for part in parts or []:
        text = (
            part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        )
        if text:
            texts.append(str(text))
    return "".join(texts)


def _is_final(event: Any) -> bool:
    if getattr(event, "partial", False):
        return False
    checker = getattr(event, "is_final_response", None)
    if callable(checker):
        return bool(checker())
    return True


def _zero_usage() -> dict[str, int]:
    return {"input": 0, "output": 0, "total": 0}


def _add_usage(usage: dict[str, int], event: Any) -> dict[str, int]:
    prompt, candidates, total = spanlib.usage_counts(
        getattr(event, "usage_metadata", None)
    )
    return {
        "input": usage["input"] + prompt,
        "output": usage["output"] + candidates,
        "total": usage["total"] + total,
    }


def _parts(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [_text_part(content)]
    if not isinstance(content, list):
        return [_text_part("")]
    parts: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(_text_part(str(block.get("text") or "")))
        elif block.get("type") == "image":
            source = block.get("source") or {}
            raw = base64.b64decode(str(source.get("data") or ""))
            parts.append(_image_part(str(source.get("media_type") or "image/png"), raw))
    return parts


def _text_part(text: str) -> Any:
    if genai_types is not None:
        return genai_types.Part(text=text)
    return SimpleNamespace(text=text, inline_data=None)


def _image_part(mime_type: str, data: bytes) -> Any:
    if genai_types is not None:
        return genai_types.Part(
            inline_data=genai_types.Blob(mime_type=mime_type, data=data)
        )
    return SimpleNamespace(
        text=None, inline_data=SimpleNamespace(mime_type=mime_type, data=data)
    )


def _content(role: str, parts: list[Any]) -> Any:
    if genai_types is not None:
        return genai_types.Content(role=role, parts=parts)
    return SimpleNamespace(role=role, parts=parts)
