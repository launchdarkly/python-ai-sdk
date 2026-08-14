from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from .judges import build_judge_tasks, run_judges
from .lifecycle import extract_variation
from .registry import resolve_handlers, resolve_tools
from .tracking import execute_and_stream, execute_and_track
from .types import (
    AiConfigRep,
    LDContext,
    NativeTool,
    ProviderHandler,
    ProviderResponse,
    StreamEvent,
    VariationMeta,
)
from .utils import (
    parse_json_with_possible_fences,
    select_handler,
    to_usage_dict,
)


def _resolve_output_format_response(
    raw_response: Any, output_format: dict[str, Any] | None
) -> Any:
    """
    Attempts to parse JSON from *raw_response* when ``output_format`` is set.
    Returns the parsed dict when successful, or the raw string when the model
    did not produce valid JSON (best-effort — agents and streaming responses
    cannot guarantee structured output).
    """
    if not output_format:
        return raw_response if raw_response is not None else ""
    if not isinstance(raw_response, str):
        return raw_response
    parsed = parse_json_with_possible_fences(raw_response)
    return parsed if parsed is not None else raw_response


class ConfigInstance:
    """Return type of ``config()``."""

    def __init__(
        self,
        key: str,
        handler: ProviderHandler | list[ProviderHandler] | None,
        tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None,
        registry: Any,  # Registry | None
        skip_judges: bool = False,
    ) -> None:
        self._key = key
        self._handler = handler
        self._tool_handlers = tool_handlers
        self._registry = registry
        self._skip_judges = skip_judges

    def _normalize_handlers(self) -> list[ProviderHandler] | None:
        if self._handler is None:
            return None
        if isinstance(self._handler, list):
            return self._handler
        return [self._handler]

    async def invoke(
        self,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse[Any]:
        resolved_handler_list = resolve_handlers(
            self._registry, self._normalize_handlers()
        )
        resolved_tools = resolve_tools(self._registry, self._tool_handlers)

        variation = await extract_variation(self._key, context)
        config: AiConfigRep = variation["config"]
        meta: VariationMeta = variation["meta"]

        if not resolved_handler_list:
            raise ValueError("No handlers provided to config()")
        handler = select_handler(config, meta, resolved_handler_list)

        result = await execute_and_track(
            config_key=self._key,
            config=config,
            meta=meta,
            user_context=context,
            handler=handler,
            user_input=user_input,
            tool_handlers=resolved_tools,
            variables=variables,
            history=history,
        )

        raw_response = result["response"]
        usage: dict[str, int] = result["usage"]
        track_data = result["track_data"]

        parsed_response = _resolve_output_format_response(
            raw_response,
            config.get("outputFormat") if isinstance(config, dict) else None,
        )

        llm_str = (
            parsed_response
            if isinstance(parsed_response, str)
            else json.dumps(parsed_response)
        )

        usage_obj = to_usage_dict(usage)

        if self._skip_judges:
            judge_tasks = await build_judge_tasks(
                config=config,
                user_context=context,
                handler=handler,
                handlers=resolved_handler_list,
                llm_response=llm_str,
                base_track_data=track_data,
            )
            return ProviderResponse(
                response=parsed_response,
                usage=usage_obj,
                judge_tasks=judge_tasks,
                track_data=track_data,
            )

        judge_results = await run_judges(
            config=config,
            user_context=context,
            handler=handler,
            handlers=resolved_handler_list,
            user_input=user_input,
            llm_response=llm_str,
            base_track_data=track_data,
            tool_handlers=resolved_tools,
        )
        return ProviderResponse(
            response=parsed_response,
            usage=usage_obj,
            judge_results=judge_results if judge_results else None,
            track_data=track_data,
        )

    async def stream(
        self,
        user_input: str | None,
        context: LDContext,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        resolved_handler_list = resolve_handlers(
            self._registry, self._normalize_handlers()
        )
        resolved_tools = resolve_tools(self._registry, self._tool_handlers)

        variation = await extract_variation(self._key, context)
        config: AiConfigRep = variation["config"]
        meta: VariationMeta = variation["meta"]

        if not resolved_handler_list:
            raise ValueError("No handlers provided to config()")
        handler = select_handler(config, meta, resolved_handler_list)

        done_event: dict[str, Any] | None = None

        async for event in execute_and_stream(
            config_key=self._key,
            config=config,
            meta=meta,
            user_context=context,
            handler=handler,
            user_input=user_input,
            tool_handlers=resolved_tools,
            variables=variables,
            history=history,
        ):
            if event.get("type") == "chunk":
                yield event
            else:
                done_event = event

        if done_event:
            track_data = done_event.get("track_data", {})
            judge_results = (
                {}
                if self._skip_judges
                else await run_judges(
                    config=config,
                    user_context=context,
                    handler=handler,
                    handlers=resolved_handler_list,
                    user_input=user_input,
                    llm_response=done_event.get("response", ""),
                    base_track_data=track_data,
                    tool_handlers=resolved_tools,
                )
            )
            yield {
                "type": "done",
                "response": done_event.get("response", ""),
                "usage": done_event.get("usage"),
                "judge_results": judge_results if judge_results else None,
            }


def config(
    *,
    key: str,
    handler: ProviderHandler | list[ProviderHandler] | None = None,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
    registry: Any = None,
    skip_judges: bool = False,
) -> ConfigInstance:
    """
    Creates a ``ConfigInstance`` bound to *key*. Accepts a single handler or a
    list of handlers. When a list is provided the correct handler is selected
    at call time based on the variation's provider and mode. Calling
    ``.invoke()`` or ``.stream()`` fetches the AI config from LD and dispatches
    to the appropriate handler.

    Pass ``skip_judges=True`` to suppress automatic judge evaluations during
    ``.invoke()`` / ``.stream()``. When set, ``invoke()`` returns
    ``judge_tasks: list[JudgeTask]`` — pre-packaged tasks ready for a background
    thread calling ``run_judge(task, handlers)``.
    """
    return ConfigInstance(
        key=key,
        handler=handler,
        tool_handlers=tool_handlers,
        registry=registry,
        skip_judges=skip_judges,
    )
