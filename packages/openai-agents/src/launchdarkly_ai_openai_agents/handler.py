"""
OpenAI Agents handler — uses the openai-agents SDK (``agents`` package).
Mirrors the TypeScript @launchdarkly/ai-openai-agents handler.

Span construction lives in ``spans.py``. This module drives the ``agents`` SDK's own ``Runner``
and reports each per-turn and per-tool-call boundary the SDK gives us, through ``RunHooks``.

The TypeScript handler intercepts model calls by wrapping the Agents SDK's ``Model`` /
``ModelProvider`` interfaces (``SpanningModel`` / ``SpanningModelProvider``) and tool calls by
listening for the ``Agent``'s ``agent_tool_start`` / ``agent_tool_end`` events. The Python
``agents`` SDK exposes the same two boundaries more directly, as ``RunHooks`` callbacks
(``on_llm_start`` / ``on_llm_end`` for a model turn, ``on_tool_start`` / ``on_tool_end`` for a tool
call), and both fire identically on the blocking and the streaming path. Using them is simpler than
building a parallel ``Model`` wrapper and produces the same span tree, because ``Model.get_response``
in this SDK discards the same information (see ``spans.derive_finish_reason``) no matter which
interface intercepts it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    RunUsage,
    SpanUsage,
    config,
    create_handler,
    create_run_usage,
    end_span_once,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    text_message,
)

from .spans import (
    derive_finish_reason,
    fail_span,
    finish_model_span,
    finish_root_span,
    mark_ok,
    parent_context_of,
    start_model_span,
    start_root_span,
    start_tool_span,
    succeed_span,
    to_request_span_messages,
    to_response_span_messages,
    to_span_usage,
    to_tool_definitions,
    tool_arguments,
)

try:
    # The `agents` SDK validates that anything passed as `hooks=` is a `RunHooksBase` instance, so
    # `_SpanningHooks` below has to actually subclass it rather than merely duck-type it. Imported
    # once at module load, unlike the rest of this handler's dynamic `importlib.import_module`
    # calls, because a class statement needs its base at class-definition time.
    from agents.lifecycle import RunHooksBase as _RunHooksBase
except ImportError:  # pragma: no cover - `agents` is a hard dependency of this package
    _RunHooksBase = object  # type: ignore[assignment,misc]


def _build_agent_tools(
    config_tools: dict[str, Any],
    tool_handlers: dict[str, Any],
) -> list[Any]:
    # Not filtered to the tools that have a registered handler, unlike the TypeScript SDK, which
    # excludes an unregistered tool from the catalog entirely. This difference predates the span
    # work and changes what the model is offered, not what a span reports, so it is left as it is;
    # `to_tool_definitions` below records the catalog actually sent, whatever it is.
    import importlib

    agents_mod = importlib.import_module("agents")
    FunctionTool = agents_mod.FunctionTool

    result = []
    for name, tool_cfg in config_tools.items():
        # on_invoke_tool receives (ToolContext, json_args_str); decode and dispatch.
        async def _execute(_ctx: Any, args_str: str, _name: str = name) -> str:
            handler = tool_handlers.get(_name)
            if not handler or not callable(handler):
                raise ValueError(f'No handler registered for tool "{_name}"')
            try:
                args = json.loads(args_str) if args_str else {}
            except (json.JSONDecodeError, ValueError):
                args = {}
            res = await handler(args)
            return str(res)

        t = FunctionTool(
            name=name,
            description=tool_cfg.get("description", "") or "",
            params_json_schema=tool_cfg.get("parameters")
            or {"type": "object", "properties": {}},
            on_invoke_tool=_execute,
        )
        result.append(t)
    return result


def _format_history(history: list[dict[str, Any]] | None) -> str | None:
    if not history:
        return None
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "Conversation History:\n\n" + "\n".join(lines)


def _build_agent_and_prompt(
    config: AiConfigRep,
    user_input: str | None,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> tuple[Any, str, str | None]:
    import importlib

    agents_mod = importlib.import_module("agents")
    Agent = agents_mod.Agent

    safe_input = user_input or ""
    instructions: str | None = None
    prompt = safe_input

    if config.get("instructions"):
        instructions = parse_template(config["instructions"], variables)
    elif config.get("messages"):
        system_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        conv_msgs = [m for m in config["messages"] if m.get("role") != "system"]
        if system_msgs:
            instructions = parse_template(
                "\n".join(m["content"] for m in system_msgs), variables
            )
        conv_history = "\n".join(
            parse_template(m["content"], variables) for m in conv_msgs
        )
        prompt = f"{conv_history}\n\n{safe_input}" if conv_history else safe_input

    history_text = _format_history(history)
    if history_text:
        instructions = (
            f"{instructions}\n\n{history_text}" if instructions else history_text
        )

    tools = _build_agent_tools(config.get("tools") or {}, tool_handlers)

    # `outputFormat` is not wired into `Agent(output_type=...)` here, matching this handler's
    # pre-existing behaviour (and unlike the TypeScript handler, which does wire it): the Python
    # Agents SDK requires a concrete Python type for `output_type`, not a raw JSON Schema dict (see
    # `utils.build_output_type`'s docstring). Fixing that gap is a "what is sent to the model"
    # change, not a telemetry one, so it is left alone. `_call_impl` still returns the parsed
    # `final_output` object as-is when `outputFormat` is configured, matching the pre-existing
    # return-shape contract.
    agent = Agent(
        name="assistant",
        model=config.get("model", {}).get("name", "gpt-4o"),
        **({"instructions": instructions} if instructions else {}),
        **({"tools": tools} if tools else {}),
    )
    return agent, prompt, instructions


class _SpanningHooks(_RunHooksBase):
    """Opens and closes one ``chat`` span per model turn and one ``execute_tool`` span per tool
    call, all parented to the root's context.

    Subclasses ``agents.RunHooksBase``: the ``Runner`` validates that ``hooks=`` is an instance of
    it before running, so a merely duck-typed object is rejected outright.

    Owns the run's usage accumulator too, so a run that raises mid-flight still has the completed
    turns' spend somewhere the caller can read it from, via :attr:`run_usage`.
    """

    def __init__(
        self,
        config: AiConfigRep,
        parent: Any,
        capture_content: bool,
        run_usage: Any,
    ) -> None:
        self.config = config
        self.parent = parent
        self.capture_content = capture_content
        self.run_usage = run_usage
        self.open_model_span: Any = None
        self.open_tool_spans: dict[str, Any] = {}

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: str | None, input_items: Any
    ) -> None:
        span = start_model_span(self.config, self.parent)
        self.open_model_span = span
        if self.capture_content:
            set_input_content_attributes(
                span,
                self.capture_content,
                system_instructions=system_prompt,
                messages=to_request_span_messages(input_items),
                tool_definitions=to_tool_definitions(getattr(agent, "tools", []) or []),
            )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        span = self.open_model_span
        self.open_model_span = None
        output = getattr(response, "output", None) or []
        finish_reason = derive_finish_reason(output)
        usage = to_span_usage(getattr(response, "usage", None))
        # Accounting before the span, and not conditional on it. run_usage is what the caller gets
        # back as its usage bag and what LaunchDarkly's own metrics bill from, and neither depends on
        # a span existing. Returning early here dropped the tokens of every turn on an install
        # without the otel extra, which is a billing figure rather than a telemetry one.
        self.run_usage.add(usage)
        if span is None:
            return
        # Popped above, so close_open_spans can no longer reach this span: whatever happens next,
        # this method has to end it. Serialising conversation content raises on anything that is not
        # JSON-serialisable, and a raise here would otherwise leak the span with nothing tracking it.
        # The langchain-agents callback guards the identical shape for the identical reason.
        try:
            if self.capture_content:
                set_output_content_attributes(
                    span,
                    self.capture_content,
                    to_response_span_messages(output, finish_reason),
                )
            finish_model_span(span, self.config, usage, finish_reason)
        except Exception as exc:
            fail_span(span, exc)
            raise

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        call_id = getattr(context, "tool_call_id", None) or getattr(
            tool, "name", "tool"
        )
        name = getattr(context, "tool_name", None) or getattr(tool, "name", "tool")
        span = start_tool_span(str(name), str(call_id), self.parent)
        if self.capture_content:
            # `tool_arguments` is the raw JSON args string the model produced. Parsed here rather
            # than passed through, per TELEMETRY-CONTRACT.md section 12: arguments hold the object
            # the provider means, not the encoding it chose. The line this replaced cited section 7,
            # which describes what the writer does with a value it is given, not what a call site
            # should hand it. Reading it as a mandate to pass the string made this span describe the
            # same call differently from an Anthropic one.
            args = getattr(context, "tool_arguments", None)
            if args is not None:
                set_tool_call_content_attributes(
                    span, self.capture_content, arguments=tool_arguments(args)
                )
        self.open_tool_spans[str(call_id)] = span

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        call_id = str(getattr(context, "tool_call_id", None))
        span = self.open_tool_spans.pop(call_id, None)
        if span is None:
            return
        if self.capture_content:
            set_tool_call_content_attributes(span, self.capture_content, result=result)
        succeed_span(span)

    def close_open_spans(self, error: BaseException) -> None:
        """Fails every span this run has open, for the crash path.

        A tool handler's own exception, or the run raising for any other reason, propagates before
        ``on_tool_end``/``on_llm_end`` ever fire, so those spans are still open when the caller's
        ``except`` runs. Mirrors the TypeScript handler's ``attachToolSpanHooks(...).closeOpenSpans``.
        """
        for span in self.open_tool_spans.values():
            fail_span(span, error)
        self.open_tool_spans.clear()
        if self.open_model_span is not None:
            fail_span(self.open_model_span, error)
            self.open_model_span = None

    def abandon_open_spans(self, ended: set[int]) -> None:
        """Ends every span this run has open, for stream abandonment.

        Unlike :meth:`close_open_spans`, nothing failed: a consumer stopping early is normal.
        ``end_span_once`` leaves each span at ``UNSET`` and marks ``launchdarkly.stream.abandoned``,
        rather than recording an exception and setting ``ERROR``.
        """
        for span in self.open_tool_spans.values():
            end_span_once(span, ended, abandoned=True)
        self.open_tool_spans.clear()
        if self.open_model_span is not None:
            end_span_once(self.open_model_span, ended, abandoned=True)
            self.open_model_span = None


def _write_failed_run_usage(
    span: Any,
    config: AiConfigRep,
    error: BaseException,
    run_usage: RunUsage,
) -> None:
    """Writes what a failed run spent onto the root, without counting it twice.

    Two sources describe the same spend, and they overlap. The run hooks add each turn as it
    finishes, so by the time the run raises they already hold every completed turn. The exception
    also carries the SDK's own aggregate over those same turns.

    Adding the aggregate to the accumulator therefore roughly doubles the reported cost of any run
    that failed after paid turns, which MaxTurnsExceeded does by definition. The aggregate is the
    authoritative figure, so it replaces the accumulator rather than adding to it.

    When the error carries no aggregate, which is what a tool handler's own error looks like, the
    accumulator is all there is and is used instead. Nothing is written when neither has anything:
    all-zero attributes would assert the run cost nothing, which a run that died on its first call
    cannot claim.
    """
    spent = _usage_from_error(error)
    # An aggregate that reports no tokens is not an aggregate. RunContextWrapper.usage defaults to an
    # empty Usage, so the attribute is present from the moment the run starts, and an exception before
    # the first paid call carries a full set of zeros. Writing those would assert the run cost
    # nothing, which is the one claim this function's docstring says it must not make.
    if spent is not None and (spent.input or spent.output):
        finish_root_span(span, config, spent)
    elif run_usage.reported:
        finish_root_span(span, config, run_usage.total)


def _run_aggregate_usage(run_result: Any) -> SpanUsage | None:
    """The SDK's own total for a completed run, when it reported one.

    ``RunResult.context_wrapper.usage`` is the aggregate the Runner maintains itself, and it is the
    same object :func:`_usage_from_error` reads off a failure. The TypeScript handler uses it for the
    root total and the returned bag, so this does too.

    Preferred over the accumulator the run hooks fill, because the hooks are not guaranteed to fire:
    ``openai-agents`` only added the LLM lifecycle hooks in 0.2.11, and this package's floor is far
    below that. Accumulating from hooks alone handed a caller zeros for a run that really did spend,
    which is a billing figure rather than a telemetry one.

    Returns ``None`` when the aggregate is absent or reports nothing, so the caller falls back to the
    accumulator rather than asserting the run cost nothing.
    """
    context_wrapper = getattr(run_result, "context_wrapper", None)
    usage = getattr(context_wrapper, "usage", None) if context_wrapper else None
    if usage is None:
        return None
    spent = to_span_usage(usage)
    return spent if (spent.input or spent.output) else None


def _usage_from_error(error: BaseException) -> SpanUsage | None:
    """The run's spend at the point it raised, when the SDK attached one.

    ``AgentsException`` carries ``run_data.context_wrapper.usage``, the same aggregate the success
    path reads off ``result.state.usage``. The tokens a failed run already spent were really
    billed, and the root is the only span a config-scoped cost query can find them on, so dropping
    this would silently zero out a run that failed after several paid turns.

    Read structurally rather than with a provider import: a tool handler's own error propagates
    unwrapped and carries no ``run_data``, and that case has to yield ``None`` so nothing is
    written, rather than asserting the run spent nothing.
    """
    run_data = getattr(error, "run_data", None)
    if run_data is None:
        return None
    context_wrapper = getattr(run_data, "context_wrapper", None)
    usage = getattr(context_wrapper, "usage", None) if context_wrapper else None
    if usage is None:
        return None
    return to_span_usage(usage)


def create_openai_agent_handler(*, capture_content: bool = False) -> ProviderHandler:
    """Creates a ``ProviderHandler`` for OpenAI via the openai-agents SDK.

    Set *capture_content* to put prompts, model output, tool arguments and tool results on the
    emitted spans. It defaults to off. Conversation content is PII, so a run emits only metadata,
    meaning models, token counts, timings and tool names, until a caller asks for more.
    """

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import importlib

        agents_mod = importlib.import_module("agents")
        Runner = agents_mod.Runner

        th = tool_handlers or {}
        vs = variables or {}

        span = start_root_span(config, vs)
        parent = parent_context_of(span)

        agent, prompt, instructions = _build_agent_and_prompt(
            config, user_input, th, vs, history
        )
        run_usage = create_run_usage()
        hooks = _SpanningHooks(config, parent, capture_content, run_usage)
        try:
            # Inside the guard, because serialising the prompt raises on anything that is not
            # JSON-serialisable and a raise out here would leave the root open: never ended, never
            # exported, and the run gone from AI Config Monitoring with the feature_flag event on it.
            set_input_content_attributes(
                span,
                capture_content,
                system_instructions=instructions,
                messages=[text_message("user", prompt)],
            )
            result = await Runner.run(agent, prompt, hooks=hooks)
            final_output = result.final_output
            set_output_content_attributes(
                span,
                capture_content,
                [text_message("assistant", _stringify_output(final_output))],
            )
            spent = _run_aggregate_usage(result) or run_usage.total
            finish_root_span(span, config, spent)
            succeed_span(span)
            output = (
                final_output
                if config.get("outputFormat")
                else _stringify_output(final_output)
            )
            return {
                "output": output,
                "usage": {
                    "input_tokens": spent.input,
                    "output_tokens": spent.output,
                },
            }
        except Exception as exc:
            hooks.close_open_spans(exc)
            _write_failed_run_usage(span, config, exc, run_usage)
            fail_span(span, exc)
            raise

    def _stream_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream_gen(
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=capture_content,
        )

    return create_handler(("OpenAI", "agent"), _call_impl, _stream_impl)  # type: ignore[arg-type]


def _stringify_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


async def _stream_gen(
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
    the root span, and any still-open ``chat``/``execute_tool`` span, would never be ended.

    Ending the spans is not the whole story here: the ``Runner.run_streamed`` background task keeps
    the agent run going, and spending tokens, after a consumer stops reading. ``finally`` also
    cancels the streamed run itself, mirroring the TypeScript handler's ``AbortController``.
    """
    import importlib

    agents_mod = importlib.import_module("agents")
    Runner = agents_mod.Runner

    span = start_root_span(config, variables)
    parent = parent_context_of(span)

    agent, prompt, instructions = _build_agent_and_prompt(
        config, user_input, tool_handlers, variables, history
    )
    ended: set[int] = set()
    run_usage = create_run_usage()
    hooks = _SpanningHooks(config, parent, capture_content, run_usage)
    streamed: Any = None
    # Whether the vendor's stream ran to the end. Tracked separately from the span, because the
    # teardown below has to cancel the Runner on an install with no spans at all.
    stream_completed = False

    try:
        # Inside the guard, because serialising the prompt raises on anything that is not
        # JSON-serialisable. A raise out here would leave the root open with the `finally` never
        # entered, so the run would vanish from AI Config Monitoring with its feature_flag event.
        set_input_content_attributes(
            span,
            capture_content,
            system_instructions=instructions,
            messages=[text_message("user", prompt)],
        )
        streamed = Runner.run_streamed(agent, prompt, hooks=hooks)
        full_output = ""

        async for event in streamed.stream_events():
            if event.type == "raw_response_event":
                raw = getattr(event, "data", None)
                if (
                    raw is not None
                    and getattr(raw, "type", None) == "response.output_text.delta"
                ):
                    delta = getattr(raw, "delta", "")
                    if isinstance(delta, str) and delta:
                        yield {"type": "chunk", "text": delta}
                        full_output += delta

        stream_completed = True
        final_output = streamed.final_output
        output = (
            _stringify_output(final_output) if final_output is not None else full_output
        )

        set_output_content_attributes(
            span, capture_content, [text_message("assistant", output)]
        )
        # Same preference as the blocking path: the Runner's own aggregate, and the hook accumulator
        # only when it reported nothing.
        spent = _run_aggregate_usage(streamed) or run_usage.total
        finish_root_span(span, config, spent)
        mark_ok(span)
        end_span_once(span, ended)

        yield {
            "type": "done",
            "output": output,
            "usage": {
                "input_tokens": spent.input,
                "output_tokens": spent.output,
            },
        }

    except Exception as exc:
        hooks.close_open_spans(exc)
        _write_failed_run_usage(span, config, exc, run_usage)
        fail_span(span, exc, ended)
        raise
    finally:
        # A no-op on the success and failure paths: both already ended their spans through
        # `ended`. On abandonment it is the only chance to close the tree, and the only chance to
        # stop the vendor's run — breaking out of `async for` above only stops us reading; the
        # Runner's own background task keeps calling the model and spending tokens until told to
        # stop.
        # Cancelling the Runner is not telemetry, so it does not sit behind a span. Without the otel
        # extra there is no root span at all, and the guard below would have skipped the cancel
        # entirely: an early break would leave the vendor's background task calling the model and
        # spending money, which is the failure this teardown exists to prevent.
        if streamed is not None and not stream_completed:
            try:
                streamed.cancel()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        if span is not None and id(span) not in ended:
            hooks.abandon_open_spans(ended)
            if run_usage.reported:
                finish_root_span(span, config, run_usage.total)
        end_span_once(span, ended, abandoned=True)


def openai_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    # Both are lifted out of kwargs: capture_content configures the handler, variables belong to
    # the invocation. Leaving either in would pass it to config(), which takes neither, so a caller
    # asking for content on spans got a TypeError instead of content.
    variables = kwargs.pop("variables", None)
    capture_content = kwargs.pop("capture_content", False)
    return config(
        key=config_key,
        handler=create_openai_agent_handler(capture_content=capture_content),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
