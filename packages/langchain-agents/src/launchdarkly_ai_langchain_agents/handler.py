"""
LangChain Agents handler — uses LangGraph's create_react_agent / StateGraph.
Mirrors the TypeScript @launchdarkly/ai-langchain-agents handler.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    SpanMessage,
    SpanMessagePart,
    _config,
    _create_handler,
    compose_history,
    create_run_usage,
    end_span_once,
    end_unfinished_spans,
    lang_chain_content_text,
    lang_chain_span_messages,
    lang_chain_span_usage,
    parse_template,
    report_usage,
    set_input_content_attributes,
    set_output_content_attributes,
)
from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from launchdarkly_ai_server.utils import model_parameters

from ._version import PACKAGE_NAME, __version__
from .messages import to_lang_chain_messages
from .spans import (
    build_span_callbacks,
    fail_span,
    finish_root_span,
    mark_ok,
    parent_context_of,
    start_root_span,
    succeed_span,
    to_tool_definitions,
)

#: Handler-owned, per model class. The handler always sets the model itself, and each
#: class exposes that one constructor field under two names (the field and its alias), so a
#: config value for either must never be forwarded: it would collide with the model the
#: handler resolves, or override it.
_CHAT_OPENAI_OWNED_KEYS = frozenset({"model", "model_name"})
_CHAT_ANTHROPIC_OWNED_KEYS = frozenset({"model", "model_name"})
_CHAT_BEDROCK_CONVERSE_OWNED_KEYS = frozenset({"model", "model_id"})

#: LangChain's own runtime fields, shared by every chat model class: caching, callbacks, rate
#: limiting, tracing tags and metadata, streaming mode, message format, and token counting. They
#: configure how LangChain runs the model in this process, not the model request, and several are
#: objects or callables a config cannot express. Never forwarded.
_LANGCHAIN_RUNTIME_KEYS = frozenset(
    {
        "cache",
        "callbacks",
        "custom_get_token_ids",
        "disable_streaming",
        "metadata",
        "name",
        "output_version",
        "profile",
        "rate_limiter",
        "streaming",
        "tags",
        "verbose",
    }
)

#: Every key ``ChatOpenAI`` accepts (field names plus pydantic aliases), classified by hand into
#: exactly one of: forwarded (below), handler-owned (``model``, always overwritten by the
#: resolved model name, see ``_model_constructor_kwargs``), or excluded (below).
#: ``TestChatOpenAIAcceptsExactlyTheseKeys`` in this package's tests asserts this classification
#: stays exhaustive as the SDK's own pydantic model changes.
#:
#: This is the cross-SDK list for LangChain ChatOpenAI (TESTING.md section 1.12).
_CHAT_OPENAI_FORWARDED_KEYS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "presence_penalty",
        "service_tier",
        "stop",
        "stop_sequences",
        "temperature",
        "top_logprobs",
        "top_p",
        "verbosity",
    }
)

#: Forwarded keys whose value must be an object. A config value of any other type is malformed
#: and dropped rather than passed to ``ChatOpenAI``.
_CHAT_OPENAI_MAPPING_KEYS = frozenset({"logit_bias"})

#: Accepted by ``ChatOpenAI`` but never forwarded, and why:
#: * ``api_key``, ``openai_api_key``, ``organization``, ``openai_organization``: credentials.
#: * ``base_url``, ``openai_api_base``, ``openai_proxy``: where requests go.
#: * ``client``, ``async_client``, ``root_client``, ``root_async_client``, ``http_client``,
#:   ``http_async_client``, ``http_socket_options``: raw HTTP client objects/settings.
#: * ``default_headers``, ``default_query``, ``extra_body``, ``model_kwargs``: raw request
#:   injection. ``model_kwargs`` is merged straight into the request payload, so it would carry
#:   ``extra_headers``/``extra_query`` or any other excluded key past this list.
#: * ``max_retries``, ``request_timeout``, ``timeout``, ``stream_chunk_timeout``: retries and
#:   timeouts.
#: * ``use_responses_api``, ``use_previous_response_id``, ``include``, ``reasoning``,
#:   ``truncation``, ``context_management``: API-shape switch. Each one makes ``ChatOpenAI`` call
#:   the Responses API instead of Chat Completions, which changes the API and response shape the
#:   handler gets back. ``truncation``, ``context_management`` and ``use_previous_response_id``
#:   also lean on server-side conversation state.
#: * ``reasoning_effort``: the Chat Completions spelling of ``reasoning``, left out with it so a
#:   config cannot set reasoning by one spelling and not the other.
#: * ``store``: data retention. It decides whether the request is kept on the server.
#: * ``seed``: not on the cross-SDK list.
#: * ``prompt_cache_key``: cut from the cross-SDK list, because ``ChatOpenAI`` has no such field
#:   and the spec cuts a key rather than work around it. It is not in the set below because
#:   ``ChatOpenAI`` does not accept it at all, which the drift test requires of every listed key.
#: * ``stream_usage``, ``include_response_headers``, ``disabled_params``, ``tiktoken_model_name``:
#:   runtime wiring. They change what the handler gets back and how usage is reported to it.
#: * Everything in :data:`_LANGCHAIN_RUNTIME_KEYS`.
#:
#: Named for the drift test and for review, not read at runtime: the forwarded list above already
#: leaves these out, so nothing needs to subtract them again.
_CHAT_OPENAI_EXCLUDED_KEYS = _LANGCHAIN_RUNTIME_KEYS | {
    "api_key",
    "openai_api_key",
    "organization",
    "openai_organization",
    "base_url",
    "openai_api_base",
    "openai_proxy",
    "client",
    "async_client",
    "root_client",
    "root_async_client",
    "http_client",
    "http_async_client",
    "http_socket_options",
    "default_headers",
    "default_query",
    "extra_body",
    "model_kwargs",
    "max_retries",
    "request_timeout",
    "timeout",
    "stream_chunk_timeout",
    "stream_usage",
    "include_response_headers",
    "disabled_params",
    "tiktoken_model_name",
    "use_responses_api",
    "use_previous_response_id",
    "include",
    "reasoning",
    "truncation",
    "context_management",
    "reasoning_effort",
    "store",
    "seed",
}

#: Every key ``ChatAnthropic`` accepts (field names plus pydantic aliases), classified the same way
#: as :data:`_CHAT_OPENAI_FORWARDED_KEYS`. ``TestChatAnthropicAcceptsExactlyTheseKeys`` in this
#: package's tests asserts this classification stays exhaustive.
#:
#: This is the cross-SDK list for LangChain ChatAnthropic (TESTING.md section 1.12).
#: ``max_tokens_to_sample`` is renamed to ``max_tokens`` before this list applies, and a
#: ``max_tokens`` the config also set wins (see :data:`_CHAT_ANTHROPIC_RENAMES`).
_CHAT_ANTHROPIC_FORWARDED_KEYS = frozenset(
    {
        "betas",
        "effort",
        "max_tokens",
        "max_tokens_to_sample",
        "output_config",
        "stop",
        "stop_sequences",
        "temperature",
        "thinking",
        "top_k",
        "top_p",
    }
)

#: Forwarded keys whose value must be an object. A config value of any other type is malformed
#: and dropped rather than passed to ``ChatAnthropic``.
_CHAT_ANTHROPIC_MAPPING_KEYS = frozenset({"output_config", "thinking"})

#: Alias to the name it is renamed to before forwarding. ``ChatAnthropic`` takes both spellings
#: of the one field, so passing both would leave which one wins to pydantic.
_CHAT_ANTHROPIC_RENAMES = {"max_tokens_to_sample": "max_tokens"}

#: Accepted by ``ChatAnthropic`` but never forwarded, and why:
#: * ``anthropic_api_key``, ``api_key``: credentials (field plus alias).
#: * ``anthropic_api_url``, ``base_url``, ``anthropic_proxy``: where requests go.
#: * ``inference_geo``: the region inference runs in, which decides where data is processed.
#: * ``context_management``: server-side state. It has the server clear or compact earlier
#:   context.
#: * ``default_headers``, ``model_kwargs``: raw request injection. ``model_kwargs`` is merged
#:   straight into the request, so it would carry any excluded key past this list.
#: * ``mcp_servers``: attaches remote MCP servers, which then receive the conversation.
#: * ``reuse_last_container``: reuses server-side container state from an earlier request, not
#:   a model setting.
#: * ``default_request_timeout``, ``timeout``, ``max_retries``: timeouts and retries.
#: * ``stream_usage``: how usage is reported back to the handler.
#: * Everything in :data:`_LANGCHAIN_RUNTIME_KEYS`.
#:
#: Named for the drift test and for review, not read at runtime.
_CHAT_ANTHROPIC_EXCLUDED_KEYS = _LANGCHAIN_RUNTIME_KEYS | {
    "anthropic_api_key",
    "api_key",
    "anthropic_api_url",
    "base_url",
    "anthropic_proxy",
    "inference_geo",
    "context_management",
    "default_headers",
    "model_kwargs",
    "mcp_servers",
    "reuse_last_container",
    "default_request_timeout",
    "timeout",
    "max_retries",
    "stream_usage",
}

#: Every key ``ChatBedrockConverse`` accepts (field names plus pydantic aliases), classified the
#: same way as :data:`_CHAT_OPENAI_FORWARDED_KEYS`. ``langchain-aws`` is not a dependency of this
#: package (Bedrock support is opt-in, see ``_make_default_chat_model``), so
#: ``TestChatBedrockConverseAcceptsExactlyTheseKeys`` in this package's tests skips itself when it
#: is not installed rather than asserting nothing.
#:
#: This is the cross-SDK list for LangChain ChatBedrockConverse (TESTING.md section 1.12).
_CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS = frozenset(
    {
        "max_tokens",
        "performance_config",
        "service_tier",
        "temperature",
        "top_p",
    }
)

#: Forwarded keys whose value must be an object. A config value of any other type is malformed
#: and dropped rather than passed to ``ChatBedrockConverse``.
_CHAT_BEDROCK_CONVERSE_MAPPING_KEYS = frozenset({"performance_config"})

#: Accepted by ``ChatBedrockConverse`` but never forwarded, and why:
#: * ``bedrock_api_key``, ``api_key``, ``aws_access_key_id``, ``aws_secret_access_key``,
#:   ``aws_session_token``, ``credentials_profile_name``: credentials.
#: * ``endpoint_url``, ``base_url``, ``region_name``: where requests go, and the region, which
#:   decides where data is processed.
#: * ``client``, ``bedrock_client``, ``config``: raw boto3/botocore client objects and config.
#: * ``default_headers``, ``additional_model_request_fields``: raw request injection.
#:   ``additional_model_request_fields`` is passed into the request body unfiltered.
#: * ``max_retries``, ``timeout``: retries and timeouts.
#: * ``base_model_id``, ``base_model``, ``provider``: which model the handler is talking to, the
#:   same identity as ``model_id`` (handler-owned).
#: * ``additional_model_response_field_paths``, ``raw_blocks``, ``supports_tool_choice_values``:
#:   the response shape and tool-calling behaviour the handler relies on.
#: * ``guardrails``, ``guardrail_config``, ``guard_last_turn_only``: safety configuration.
#: * ``request_metadata``: identity and attribution. It tags the request for whoever reads the
#:   invocation logs.
#: * ``system``: prompt content beyond instructions. It adds system prompt blocks on top of the
#:   config's own instructions.
#: * ``output_config``: API-shape switch. It is how ``ChatBedrockConverse`` asks for structured
#:   output, which changes the response the handler reads.
#: * ``reasoning_effort``: runtime wiring. ``ChatBedrockConverse`` turns it into model-specific
#:   ``additionalModelRequestFields``, the raw request fields the handler never sets otherwise.
#: * ``stop``, ``stop_sequences``: runtime wiring. ``ChatBedrockConverse`` merges them with the
#:   stop sequences its own structured-output prompt relies on.
#: * Everything in :data:`_LANGCHAIN_RUNTIME_KEYS`.
#:
#: Named for the drift test and for review, not read at runtime.
_CHAT_BEDROCK_CONVERSE_EXCLUDED_KEYS = _LANGCHAIN_RUNTIME_KEYS | {
    "bedrock_api_key",
    "api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "credentials_profile_name",
    "endpoint_url",
    "base_url",
    "region_name",
    "client",
    "bedrock_client",
    "config",
    "default_headers",
    "additional_model_request_fields",
    "max_retries",
    "timeout",
    "base_model_id",
    "base_model",
    "provider",
    "additional_model_response_field_paths",
    "raw_blocks",
    "supports_tool_choice_values",
    "guardrails",
    "guardrail_config",
    "guard_last_turn_only",
    "request_metadata",
    "system",
    "output_config",
    "reasoning_effort",
    "stop",
    "stop_sequences",
}


def _build_agent_tools(
    config_tools: dict[str, Any],
    tool_handlers: dict[str, Any],
) -> list[Any]:
    import importlib

    lc_tools = importlib.import_module("langchain_core.tools")
    tool_fn = lc_tools.tool

    result = []
    for name, tool_cfg in config_tools.items():
        schema = tool_cfg.get("parameters") or {}

        async def _handler(_name: str = name, **kwargs: Any) -> str:
            fn = tool_handlers.get(_name)
            if not fn:
                raise ValueError(f'No handler registered for tool "{_name}"')
            # Handlers may be sync or async. Graph ``__handoff_*`` tools stay sync so
            # routing records the selected edge on the call itself; awaiting a plain
            # return value raises. Same rule as ``tracking.wrap_tool_handlers``.
            result = fn(kwargs)
            if inspect.isawaitable(result):
                result = await result
            return str(result)

        t = tool_fn(
            name,
            _handler,
            description=tool_cfg.get("description", ""),
            args_schema=schema,
        )
        result.append(t)
    return result


def _extract_system_prompt(
    config: AiConfigRep,
    variables: dict[str, Any],
) -> str | None:
    system_prompt: str | None = None
    if config.get("instructions"):
        system_prompt = parse_template(config["instructions"], variables)
    elif config.get("messages"):
        sys_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        if sys_msgs:
            system_prompt = parse_template(
                "\n".join(m["content"] for m in sys_msgs), variables
            )

    return system_prompt


def _config_conversation_turns(
    config: AiConfigRep, variables: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "role": message.get("role"),
            "content": parse_template(message.get("content", ""), variables)
            if isinstance(message.get("content", ""), str)
            else message.get("content", ""),
        }
        for message in (config.get("messages") or [])
        if message.get("role") != "system"
    ]


def _build_initial_messages(
    config: AiConfigRep,
    user_input: str,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> list[Any]:
    if history:
        return to_lang_chain_messages(
            compose_history(
                history=history,
                user_input=user_input,
                config_messages=(
                    []
                    if config.get("instructions")
                    else _config_conversation_turns(config, variables)
                ),
            )
        )

    import importlib

    msgs_mod = importlib.import_module("langchain_core.messages")
    HumanMessage = msgs_mod.HumanMessage
    AIMessage = msgs_mod.AIMessage

    messages: list[Any] = []
    last_role: str | None = None
    if config.get("messages"):
        for msg in config["messages"]:
            if msg.get("role") == "system":
                continue
            content = parse_template(msg["content"], variables)
            if msg["role"] == "user":
                messages.append(HumanMessage(content))
            else:
                messages.append(AIMessage(content))
            last_role = msg["role"]
    if last_role != "user":
        messages.append(HumanMessage(user_input or ""))
    return messages


def _resolved_model_name(config: AiConfigRep, fallback_name: str = "") -> str:
    """Bedrock ``model.region`` is an inference-profile prefix, prepended once."""
    model = config.get("model") or {}
    name = str(model.get("name") or fallback_name)
    provider = str((config.get("provider") or {}).get("name") or "").lower()
    if provider != "bedrock":
        return name
    prefix = str(model.get("region") or "")
    if not prefix or name.startswith(f"{prefix}."):
        return name
    return f"{prefix}.{name}"


def _config_for_model_call(config: AiConfigRep) -> AiConfigRep:
    """Shallow copy with a resolved Bedrock model name. Does not mutate *config*."""
    resolved = _resolved_model_name(config)
    model = dict(config.get("model") or {})
    if model.get("name") == resolved:
        return config
    return {**config, "model": {**model, "name": resolved}}


def _model_constructor_kwargs(
    config: AiConfigRep,
    fallback_name: str,
    forwarded_keys: frozenset[str],
    *,
    mapping_keys: frozenset[str] = frozenset(),
    renames: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    params = model_parameters(config)
    for alias, name in (renames or {}).items():
        if alias in params:
            value = params.pop(alias)
            params.setdefault(name, value)
    parameters = select_forwarded_parameters(
        params, forwarded_keys, mapping_keys=mapping_keys
    )
    parameters["model"] = _resolved_model_name(config, fallback_name)
    return parameters


def _is_model_factory(llm: Any) -> bool:
    """LangChain models are callable, so ``callable`` is not enough to spot a factory."""
    return callable(llm) and not hasattr(llm, "invoke") and not hasattr(llm, "ainvoke")


def _make_default_chat_model(config: AiConfigRep) -> Any:
    """
    Instantiate the appropriate LangChain chat model based on ``config.provider.name``.
    Falls back to ``ChatOpenAI`` when the provider is not recognised.
    Requires the matching ``langchain-<provider>`` integration package to be installed.
    ``model.parameters`` are filtered to that class's forwarded list first. ``tools`` is never on
    that list, since tools are bound separately from ``config["tools"]``.
    """
    import importlib

    provider = ((config.get("provider") or {}).get("name") or "openai").lower()
    if provider == "anthropic":
        lc_anthropic = importlib.import_module("langchain_anthropic")
        return lc_anthropic.ChatAnthropic(
            **_model_constructor_kwargs(
                config,
                "claude-3-5-sonnet-20241022",
                _CHAT_ANTHROPIC_FORWARDED_KEYS,
                mapping_keys=_CHAT_ANTHROPIC_MAPPING_KEYS,
                renames=_CHAT_ANTHROPIC_RENAMES,
            )
        )
    if provider == "bedrock":
        try:
            lc_aws = importlib.import_module("langchain_aws")
        except ImportError as exc:
            raise ImportError(
                "Using Bedrock models requires langchain-aws. "
                "Install it with: pip install langchain-aws"
            ) from exc
        return lc_aws.ChatBedrockConverse(
            **_model_constructor_kwargs(
                config,
                "",
                _CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS,
                mapping_keys=_CHAT_BEDROCK_CONVERSE_MAPPING_KEYS,
            )
        )
    lc_openai = importlib.import_module("langchain_openai")
    return lc_openai.ChatOpenAI(
        **_model_constructor_kwargs(
            config,
            "gpt-4o",
            _CHAT_OPENAI_FORWARDED_KEYS,
            mapping_keys=_CHAT_OPENAI_MAPPING_KEYS,
        )
    )


async def _resolve_base_model(config: AiConfigRep, llm: Any) -> Any:
    invocation = _config_for_model_call(config)
    if llm is None:
        return _make_default_chat_model(invocation)
    if _is_model_factory(llm):
        model = llm(invocation)
        if asyncio.iscoroutine(model):
            return await model
        return model
    return llm


def _run_usage_from_messages(messages: list[Any]) -> Any:
    """Sums ``usage_metadata`` over a run's messages, the same set of numbers the callbacks see
    from the other side.

    Only ``AIMessage`` carries usage. Summing here rather than trusting the callbacks' own total
    matters when there is a real ``result``/stepped state to read: it is the same path the TypeScript
    handler takes.

    The two sides do not read the same fields, though. This one sees ``usage_metadata`` only, and the
    callbacks also fall back to ``llm_output.token_usage``. The caller reconciles them, because a
    provider that reports only in ``llm_output`` would otherwise give a successful run a root that
    says zero and chat spans that say otherwise.
    """
    run_usage = create_run_usage()
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            run_usage.add(lang_chain_span_usage(usage))
    return run_usage


def create_langchain_agents_handler(
    llm: Any = None, *, capture_content: bool = False
) -> ProviderHandler:
    """Creates a ``ProviderHandler`` for LangChain via ``create_react_agent``.

    Pass *llm* as a chat model instance, or as a function ``(config) -> model`` that is
    called after flag evaluation so ``model.parameters`` can be applied unchanged.

    Set *capture_content* to put prompts, model output, tool arguments and tool results on the
    emitted spans. It defaults to off. Conversation content is PII, so a run emits only metadata,
    meaning models, token counts, timings and tool names, until a caller asks for more.
    """
    report_usage(
        "langchain-agents.createLangChainAgentsHandler", PACKAGE_NAME, __version__
    )
    return _create_langchain_agents_handler(llm, capture_content=capture_content)


def _create_langchain_agents_handler(
    llm: Any = None, *, capture_content: bool = False
) -> ProviderHandler:
    """Non-reporting :func:`create_langchain_agents_handler`, used by this package's wrappers."""

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import importlib

        th = tool_handlers or {}
        vs = variables or {}

        span = start_root_span(config, vs)
        parent = parent_context_of(span)
        # Cleared by whichever path ends the root, so the `finally` below can tell an open root
        # from a closed one without asking the span. A mock span answers `is_recording()`
        # truthily, and the test suite is built on mock spans.
        open_root_span: Any = span

        system_prompt = _extract_system_prompt(config, vs)
        if config.get("outputFormat"):
            schema_instr = f"Respond with valid JSON matching this schema:\n{json.dumps(config['outputFormat'])}"
            system_prompt = (
                f"{system_prompt}\n\n{schema_instr}" if system_prompt else schema_instr
            )

        initial_messages = _build_initial_messages(config, user_input, vs, history)

        span_callbacks = build_span_callbacks(
            config,
            parent,
            capture_content,
            to_tool_definitions(config.get("tools") or {}),
        )

        try:
            # Inside the guard, because serialising the prompt raises on anything that is not
            # JSON-serialisable and a raise out here would leave the root open: never ended, never
            # exported, and the run gone from AI Config Monitoring with the feature_flag event on it.
            if capture_content:
                set_input_content_attributes(
                    span,
                    capture_content,
                    system_instructions=system_prompt,
                    messages=lang_chain_span_messages(initial_messages)[1],
                )
            base_model = await _resolve_base_model(config, llm)

            langgraph_prebuilt = importlib.import_module("langgraph.prebuilt")
            create_react_agent = langgraph_prebuilt.create_react_agent
            tools = _build_agent_tools(config.get("tools") or {}, th)
            agent = create_react_agent(
                base_model,
                tools,
                **({"prompt": system_prompt} if system_prompt else {}),
            )
            result = await agent.ainvoke(
                {"messages": initial_messages},
                config={"callbacks": span_callbacks.callbacks},
            )

            msgs = (
                result.get("messages", [])
                if isinstance(result, dict)
                else getattr(result, "messages", [])
            )
            run_usage = _run_usage_from_messages(msgs)
            # The two sides do not see the same fields. A message carries usage_metadata and nothing
            # else, while the callbacks read the LLMResult and fall back to llm_output.token_usage,
            # which some providers use instead. When only the callbacks saw anything, they are the
            # only record of what the run cost, and a successful root reporting zero while its own
            # chat spans report real tokens is the one outcome neither figure can be right about.
            if not run_usage.reported and span_callbacks.run_usage.reported:
                run_usage = span_callbacks.run_usage

            last_msg = msgs[-1] if msgs else None
            output = lang_chain_content_text(last_msg.content) if last_msg else ""

            # Built through the same conversion the chat span uses, not from `output`. A chat model
            # may return content as a list of blocks. Keeping that conversion here preserves
            # non-text parts in telemetry while the caller-facing output contains visible text.
            set_output_content_attributes(
                span,
                capture_content,
                lang_chain_span_messages([last_msg])[1]
                if last_msg is not None
                else [
                    SpanMessage(
                        role="assistant",
                        parts=[SpanMessagePart(type="text", content=output)],
                    )
                ],
            )
            finish_root_span(span, config, run_usage.total)
            succeed_span(span)
            open_root_span = None

            return {
                "output": output,
                "usage": {
                    "input_tokens": run_usage.total.input,
                    "output_tokens": run_usage.total.output,
                },
            }

        except Exception as exc:
            span_callbacks.close_open_spans(exc)
            # There is no `result` to sum, so the run total comes from the callbacks, which saw
            # every turn that did complete. Those tokens were billed and the root is the only span
            # a config-scoped cost query can find them on.
            if span_callbacks.run_usage.reported:
                finish_root_span(span, config, span_callbacks.run_usage.total)
            fail_span(span, exc)
            open_root_span = None
            raise
        finally:
            # Not an `except`: asyncio.CancelledError is a BaseException, so a timeout or a
            # task.cancel() never reaches the clause above. Without this the root, and any chat or
            # execute_tool span the callback handler opened but never closed, would be stranded.
            # The root is the only span carrying the feature_flag event and the launchdarkly.*
            # attributes, so the whole run would vanish from AI Config Monitoring rather than show
            # as incomplete.
            span_callbacks.cancel_open_spans()
            if open_root_span is not None and span_callbacks.run_usage.reported:
                # The turns that completed were billed, the same reason the failure path reports
                # them.
                finish_root_span(open_root_span, config, span_callbacks.run_usage.total)
            end_unfinished_spans(open_root_span)

    def _stream_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream_gen(
            llm,
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=capture_content,
        )

    return _create_handler(
        ("*", "agent"),
        _call_impl,  # type: ignore[arg-type]
        _stream_impl,  # type: ignore[arg-type]
        capture_content=capture_content,
    )


async def _stream_gen(
    llm: Any,
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    *,
    capture_content: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streams the run, emitting the same span tree as the blocking path.

    A consumer that breaks out of ``async for``, or raises inside the loop body, makes this
    generator run its ``finally`` without ever entering ``except``: ``GeneratorExit`` inherits from
    ``BaseException``, so ``except Exception`` does not see it. Without the cleanup in ``finally``
    the root span, and any ``chat``/``execute_tool`` span LangChain's end callback never fired for,
    is never ended, so it is never exported.

    ``ended`` stops the success, failure and abandonment paths from ending the same span twice.
    """
    import importlib

    span = start_root_span(config, variables)
    parent = parent_context_of(span)

    system_prompt = _extract_system_prompt(config, variables)
    initial_messages = _build_initial_messages(config, user_input, variables, history)

    span_callbacks = build_span_callbacks(
        config, parent, capture_content, to_tool_definitions(config.get("tools") or {})
    )
    ended: set[int] = set()

    # Distinguishes the two teardown reasons. A consumer that stops reading abandoned the
    # stream; a CancelledError means something cancelled the run, usually a timeout, and the
    # consumer chose nothing. The blocking path already tells these apart.
    cancelled = False
    try:
        # Inside the guard, because serialising the prompt raises on anything that is not
        # JSON-serialisable. A raise out here would leave the root open with the `finally` never
        # entered, so the run would vanish from AI Config Monitoring with its feature_flag event.
        if capture_content:
            set_input_content_attributes(
                span,
                capture_content,
                system_instructions=system_prompt,
                messages=lang_chain_span_messages(initial_messages)[1],
            )
        base_model = await _resolve_base_model(config, llm)

        langgraph_prebuilt = importlib.import_module("langgraph.prebuilt")
        create_react_agent = langgraph_prebuilt.create_react_agent
        tools = _build_agent_tools(config.get("tools") or {}, tool_handlers)
        agent = create_react_agent(
            base_model,
            tools,
            **({"prompt": system_prompt} if system_prompt else {}),
        )

        run_usage = create_run_usage()
        full_output = ""

        # agent.astream() yields state updates per graph step: { [node_name]: { messages: [...] } }
        async for step_state in agent.astream(
            {"messages": initial_messages},
            config={"callbacks": span_callbacks.callbacks},
        ):
            for step_messages in (
                step_state.values() if isinstance(step_state, dict) else []
            ):
                msgs = getattr(step_messages, "messages", None) or (
                    step_messages.get("messages", [])
                    if isinstance(step_messages, dict)
                    else []
                )
                for msg in msgs:
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        run_usage.add(lang_chain_span_usage(usage))
                    if getattr(msg, "type", None) == "ai":
                        text = lang_chain_content_text(msg.content)
                        if text:
                            yield {"type": "chunk", "text": text}
                            full_output = text

        # The same reconciliation the blocking path does, for the same reason. This walk reads
        # usage_metadata off the astream payloads, and sees nothing at all unless the graph is
        # streaming updates rather than value snapshots. The callbacks read the LLMResult and also
        # fall back to llm_output.token_usage. When only they saw anything, they are the only record
        # of what the run cost, and a successful root reporting zero while its own chat spans report
        # real tokens is the one outcome neither figure can be right about.
        if not run_usage.reported and span_callbacks.run_usage.reported:
            run_usage = span_callbacks.run_usage

        set_output_content_attributes(
            span,
            capture_content,
            [
                SpanMessage(
                    role="assistant",
                    parts=[SpanMessagePart(type="text", content=full_output)],
                )
            ],
        )
        finish_root_span(span, config, run_usage.total)
        mark_ok(span)
        end_span_once(span, ended)

        yield {
            "type": "done",
            "output": full_output,
            "usage": {
                "input_tokens": run_usage.total.input,
                "output_tokens": run_usage.total.output,
            },
        }

    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        span_callbacks.close_open_spans(exc)
        if span_callbacks.run_usage.reported:
            finish_root_span(span, config, span_callbacks.run_usage.total)
        fail_span(span, exc, ended)
        raise
    finally:
        # A no-op on the success and failure paths, because both already ended their spans through
        # `ended`. On abandonment it is the only chance to close the tree, including any chat or
        # tool span whose LangChain end callback never fired, and to report what the completed
        # turns already cost. An abandoned span is left UNSET rather than ERROR: stopping early is
        # a normal thing for a consumer to do, and LaunchDarkly's own metrics record neither a
        # success nor an error for it.
        if span is not None and id(span) not in ended:
            span_callbacks.abandon_open_spans(ended, cancelled=cancelled)
            if span_callbacks.run_usage.reported:
                finish_root_span(span, config, span_callbacks.run_usage.total)
        end_span_once(span, ended, abandoned=True, cancelled=cancelled)


def langchain_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    # Both are lifted out of kwargs: capture_content configures the handler, variables belong to
    # the invocation. Leaving either in would pass it to config(), which takes neither, so a caller
    # asking for content on spans got a TypeError instead of content.
    report_usage("langchain-agents.langchainAgents", PACKAGE_NAME, __version__)
    variables = kwargs.pop("variables", None)
    capture_content = kwargs.pop("capture_content", False)
    return _config(
        key=config_key,
        handler=_create_langchain_agents_handler(capture_content=capture_content),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
