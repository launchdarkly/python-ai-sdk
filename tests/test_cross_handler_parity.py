"""Cross-handler invariants: the six handlers must agree with each other.

Every handler package tests its own spans. Nothing tested that the six agree, and that is exactly
how they drifted apart: each was correct on its own terms while a single run emitted `chat` spans
that disagreed about what a finish reason or a cached token was.

These tests are the oracle for that. They live outside the packages because no package can own an
invariant about all six.

There are two kinds of check here.

The shape checks call each package's span constructors directly, with a recording tracer, so they
need no provider mocks and cannot be fooled by a handler that never reaches its own span code.

The vocabulary lock reads every span attribute literal out of the source and compares it to a
committed set. It fails whenever a key is added, removed or renamed anywhere in the SDK. That is
deliberate: an attribute is a public contract with whatever reads the traces, and changing one
should require editing this list and saying why in the commit.

See TELEMETRY-CONTRACT.md.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Package directory to the module that holds its span construction.
HANDLERS: dict[str, str] = {
    "claude-messages": "launchdarkly_ai_claude_messages.spans",
    "claude-agents": "launchdarkly_ai_claude_agents.spans",
    "openai-messages": "launchdarkly_ai_openai_messages.spans",
    "openai-agents": "launchdarkly_ai_openai_agents.spans",
    "langchain-messages": "launchdarkly_ai_langchain_messages.spans",
    "langchain-agents": "launchdarkly_ai_langchain_agents.spans",
}

#: `claude-agents` builds its `chat` span inside an inference tracker rather than in a standalone
#: function, because the Claude Agent SDK reports each inference as it streams rather than returning
#: one response per turn. The span it produces is still `chat {model}`; only the call site differs.
NO_STANDALONE_MODEL_SPAN = {"claude-agents"}

CONFIG: dict[str, Any] = {
    "model": {"name": "test-model-1"},
    "provider": {"name": "Anthropic"},
    "instructions": "Be helpful.",
}

LD_VARIABLES: dict[str, Any] = {
    "__ld": {
        "configKey": "cfg",
        "variationKey": "var",
        "runId": "run-1",
        "graphKey": "graph-1",
        "environmentId": "env-1",
    }
}


class RecordedSpan:
    def __init__(self, name: str, context: Any = None) -> None:
        self.name = name
        self.context = context
        self.attributes: dict[str, Any] = {}
        self.events: list[str] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(name)

    def set_status(self, code: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def end(self) -> None:
        pass


class RecordingTracer:
    """Stands in for the `trace` module inside a package's spans module."""

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    def get_tracer(self, name: str) -> RecordingTracer:
        return self

    def start_span(self, name: str, context: Any = None) -> RecordedSpan:
        span = RecordedSpan(name, context)
        self.spans.append(span)
        return span

    def set_span_in_context(self, span: RecordedSpan) -> Any:
        return ("context-of", span)


@pytest.fixture(params=sorted(HANDLERS), ids=sorted(HANDLERS))
def handler_spans(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Each package's spans module, with its tracer replaced by a recorder."""
    package = request.param
    module = importlib.import_module(HANDLERS[package])
    tracer = RecordingTracer()
    monkeypatch.setattr(module, "trace", tracer)
    return package, module, tracer


# ─── The surface every handler must expose ───────────────────────────────────

REQUIRED_FUNCTIONS = (
    "model_name",
    "start_root_span",
    "parent_context_of",
    "start_tool_span",
    "finish_root_span",
    "succeed_span",
    "mark_ok",
    "fail_span",
)


class TestSharedSurface:
    def test_every_package_exposes_the_same_span_functions(
        self, handler_spans: Any
    ) -> None:
        package, module, _ = handler_spans
        missing = [name for name in REQUIRED_FUNCTIONS if not hasattr(module, name)]
        assert missing == [], f"{package} is missing {missing}"

    def test_every_package_has_a_model_span_constructor(
        self, handler_spans: Any
    ) -> None:
        package, module, _ = handler_spans
        if package in NO_STANDALONE_MODEL_SPAN:
            pytest.skip(f"{package} builds its chat span inside an inference tracker")
        assert hasattr(module, "start_model_span")


# ─── Span names ──────────────────────────────────────────────────────────────


class TestSpanNames:
    def test_the_root_span_is_always_named_invoke_agent(
        self, handler_spans: Any
    ) -> None:
        _, module, tracer = handler_spans
        module.start_root_span(CONFIG, {})
        assert tracer.spans[0].name == "invoke_agent"

    def test_the_root_span_always_declares_its_operation(
        self, handler_spans: Any
    ) -> None:
        _, module, tracer = handler_spans
        module.start_root_span(CONFIG, {})
        assert tracer.spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"

    def test_the_model_span_is_always_chat_plus_the_model(
        self, handler_spans: Any
    ) -> None:
        # The semantic conventions name an inference span `{operation} {model}`. A handler that
        # emitted a bare `chat` would aggregate more neatly and tell a reader nothing.
        package, module, tracer = handler_spans
        if package in NO_STANDALONE_MODEL_SPAN:
            pytest.skip(f"{package} builds its chat span inside an inference tracker")
        module.start_model_span(CONFIG, None)
        assert tracer.spans[0].name == "chat test-model-1"
        assert tracer.spans[0].attributes["gen_ai.operation.name"] == "chat"

    def test_the_tool_span_is_always_execute_tool_plus_the_name(
        self, handler_spans: Any
    ) -> None:
        _, module, tracer = handler_spans
        module.start_tool_span("get_weather", "call-1", None)
        span = tracer.spans[0]
        assert span.name == "execute_tool get_weather"
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == "get_weather"
        assert span.attributes["gen_ai.tool.call.id"] == "call-1"


# ─── Where the LaunchDarkly identity lives ───────────────────────────────────

LD_ROOT_ATTRIBUTES = (
    "launchdarkly.operation.type",
    "launchdarkly.config.key",
    "launchdarkly.variation.key",
    "launchdarkly.run.id",
    "launchdarkly.graph.key",
)


class TestLaunchDarklyIdentity:
    def test_the_root_carries_the_full_launchdarkly_identity(
        self, handler_spans: Any
    ) -> None:
        _, module, tracer = handler_spans
        module.start_root_span(CONFIG, LD_VARIABLES)
        attrs = tracer.spans[0].attributes
        missing = [k for k in LD_ROOT_ATTRIBUTES if k not in attrs]
        assert missing == []

    def test_the_root_emits_the_feature_flag_event(self, handler_spans: Any) -> None:
        # The AI Config Monitoring traces tab finds a run through this event, not an attribute.
        _, module, tracer = handler_spans
        module.start_root_span(CONFIG, LD_VARIABLES)
        assert "feature_flag" in tracer.spans[0].events

    def test_a_tool_span_carries_no_launchdarkly_identity(
        self, handler_spans: Any
    ) -> None:
        # The root is the only span a config-scoped query finds. Duplicating the identity onto
        # children makes one run look like several.
        _, module, tracer = handler_spans
        module.start_tool_span("get_weather", "call-1", None)
        span = tracer.spans[0]
        assert [k for k in span.attributes if k.startswith("launchdarkly.")] == []
        assert "feature_flag" not in span.events

    def test_a_model_span_carries_no_launchdarkly_identity(
        self, handler_spans: Any
    ) -> None:
        package, module, tracer = handler_spans
        if package in NO_STANDALONE_MODEL_SPAN:
            pytest.skip(f"{package} builds its chat span inside an inference tracker")
        module.start_model_span(CONFIG, None)
        span = tracer.spans[0]
        assert [k for k in span.attributes if k.startswith("launchdarkly.")] == []
        assert "feature_flag" not in span.events


# ─── Model identity ──────────────────────────────────────────────────────────


class TestModelIdentity:
    def test_the_root_writes_both_provider_keys_and_the_request_model(
        self, handler_spans: Any
    ) -> None:
        # `gen_ai.system` is the pre-1.37 name and `gen_ai.provider.name` the current one. Emitting
        # only one of them either breaks old dashboards or leaves the SDK off-spec.
        _, module, tracer = handler_spans
        module.start_root_span(CONFIG, {})
        attrs = tracer.spans[0].attributes
        assert "gen_ai.system" in attrs
        assert "gen_ai.provider.name" in attrs
        assert attrs["gen_ai.request.model"] == "test-model-1"

    def test_the_langchain_handlers_keep_the_framework_on_the_legacy_key(
        self, handler_spans: Any
    ) -> None:
        # `gen_ai.provider.name` names who served the model, and its enum has no `langchain`
        # member, so the framework name stays on the older key.
        package, module, tracer = handler_spans
        if not package.startswith("langchain-"):
            pytest.skip("only the LangChain handlers split the two keys")
        module.start_root_span(CONFIG, {})
        attrs = tracer.spans[0].attributes
        assert attrs["gen_ai.system"] == "langchain"
        assert attrs["gen_ai.provider.name"] == "anthropic"

    def test_the_langchain_provider_name_is_binary_not_a_passthrough(
        self, handler_spans: Any
    ) -> None:
        # Anything that is not Anthropic is served by the OpenAI client, so the attribute follows the
        # client actually instantiated rather than whatever the config happens to name.
        package, module, _ = handler_spans
        if not package.startswith("langchain-"):
            pytest.skip("only the LangChain handlers make this choice")
        for configured, expected in (
            ("Anthropic", "anthropic"),
            ("OpenAI", "openai"),
            ("Bedrock", "openai"),
            ("Azure", "openai"),
            ("", "openai"),
        ):
            config = {**CONFIG, "provider": {"name": configured}}
            assert module.serving_provider(config) == expected, configured


# ─── The vocabulary lock ─────────────────────────────────────────────────────

#: Every span attribute key, event name and naming template the SDK emits.
#:
#: Locked on purpose. An attribute is a public contract with whatever reads the traces, so adding,
#: removing or renaming one should mean editing this list and saying why in the commit message.
#:
#: Derived from the TypeScript SDK at 5178db1 and verified to match it exactly.
EXPECTED_VOCABULARY = {
    # Operation and identity
    "gen_ai.operation.name",
    "gen_ai.system",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.response.id",
    "gen_ai.response.finish_reasons",
    "gen_ai.agent.name",
    "gen_ai.conversation.id",
    # Usage
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.total_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.completion_tokens",
    # Content, canonical
    "gen_ai.system_instructions",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.definitions",
    # Content, OpenLLMetry. These two are the prefixes handed to the indexed writer; the keys it
    # actually emits are `gen_ai.prompt.{i}.role` and friends, built by f-string and therefore
    # invisible to any static scan. TestOpenLLMetryCarrier below covers those at runtime.
    "gen_ai.prompt",
    "gen_ai.completion",
    # Tool calls
    "gen_ai.tool.name",
    "gen_ai.tool.call.id",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    # Content events
    "gen_ai.content.prompt",
    "gen_ai.content.completion",
    # LaunchDarkly
    "launchdarkly.operation.type",
    "launchdarkly.config.key",
    "launchdarkly.variation.key",
    "launchdarkly.run.id",
    "launchdarkly.graph.key",
    "launchdarkly.stream.abandoned",
    # A blocking run that was cancelled. asyncio.CancelledError is a BaseException, so it walks past
    # every `except Exception` a handler writes, and without a `finally` the run exported no span at
    # all. UNSET plus this marker, never ERROR, for the same reason as the abandoned stream above:
    # nothing failed, the caller went away. Python only. TypeScript has no cancellation that skips a
    # `catch`, so this key has no counterpart there and its absence is not drift.
    "launchdarkly.run.cancelled",
    "feature_flag",
    "feature_flag.key",
    "feature_flag.provider.name",
    "feature_flag.set.id",
    # Graph spans, unchanged from before the span work
    "ld.ai.graph",
    "ld.ai.graph.key",
    "ld.ai.graph.path",
}

#: Functions kept exported for one release that nothing calls any more.
#:
#: Their bodies are cut out before the scan below, rather than their keys being listed as expected.
#: Listing the keys does not work: `gen_ai.prompt` is written by the live content writer *and* by
#: dead `set_openllmetry_prompt`, so naming it as expected lets the dead copy satisfy the lock after
#: the live one is removed. Cutting the dead code out means only live writes can satisfy anything.
#:
#: Delete these names when the functions go. The lock will tell you if you miss one.
SUPERSEDED_FUNCTIONS = (
    "set_openllmetry_prompt",
    "set_openllmetry_completion",
)


def _without_superseded(source: str) -> str:
    """Drops the body of every superseded function, so a dead write cannot satisfy the lock."""
    for name in SUPERSEDED_FUNCTIONS:
        start = source.find(f"def {name}(")
        if start == -1:
            continue
        nxt = source.find("\ndef ", start + 1)
        end = len(source) if nxt == -1 else nxt
        source = source[:start] + source[end:]
    return source


_KEY_PATTERN = re.compile(
    r'set_attribute\(\s*f?"([^"{]+)"'
    r'|add_event\(\s*"([^"]+)"'
    r'|start_span\(\s*"(ld\.ai\.graph)"'
    r'|"(gen_ai\.[a-z_.0-9]+)"'
    r'|f"(gen_ai\.[a-z_.]+)\.\{'
    # The feature_flag event's own attributes are built as a plain dict before being handed to
    # add_event, so they never appear inside a set_attribute call.
    r'|"(feature_flag\.[a-z_.]+)"'
)


def _emitted_vocabulary() -> set[str]:
    """Every attribute key, event name and template found in the package sources."""
    found: set[str] = set()
    for path in (REPO_ROOT / "packages").glob("*/src/*/*.py"):
        for match in _KEY_PATTERN.finditer(_without_superseded(path.read_text())):
            key = next((g for g in match.groups() if g), None)
            if key and key.split(".")[0] in (
                "gen_ai",
                "launchdarkly",
                "feature_flag",
                "ld",
            ):
                found.add(key)
    return found


class TestVocabularyLock:
    def test_the_sdk_emits_no_key_this_list_does_not_know_about(self) -> None:
        unexpected = _emitted_vocabulary() - EXPECTED_VOCABULARY
        assert unexpected == set(), (
            "New span attribute keys found. If this is deliberate, add them to "
            "EXPECTED_VOCABULARY and say why in the commit message. If the two SDKs should agree, "
            "add the key to the TypeScript SDK in the same change."
        )

    def test_every_key_this_list_names_is_still_emitted(self) -> None:
        # Catches a key silently disappearing, which is how a dashboard goes blank without anything
        # failing.
        emitted = _emitted_vocabulary()
        missing = EXPECTED_VOCABULARY - emitted
        assert missing == set(), f"keys no longer emitted anywhere: {sorted(missing)}"


class TestOpenLLMetryCarrier:
    """The indexed carrier is written by f-string, so only a runtime check can see it.

    This is the one LaunchDarkly's LLM trace view reads today, so dropping it renders an empty
    transcript while every canonical attribute is still present and every static check still passes.
    The vocabulary lock above cannot cover it: `f"{prefix}.{index}.role"` has no literal key to scan
    for.
    """

    def test_the_input_side_writes_indexed_role_and_content(self) -> None:
        from launchdarkly_ai_server import set_input_content_attributes, text_message

        span = RecordedSpan("chat test-model-1")
        set_input_content_attributes(
            span,
            True,
            system_instructions="Be brief.",
            messages=[text_message("user", "hi")],
        )
        assert span.attributes["gen_ai.prompt.0.role"] == "system"
        assert span.attributes["gen_ai.prompt.0.content"] == "Be brief."
        assert span.attributes["gen_ai.prompt.1.role"] == "user"
        assert span.attributes["gen_ai.prompt.1.content"] == "hi"

    def test_the_output_side_writes_indexed_role_and_content(self) -> None:
        from launchdarkly_ai_server import set_output_content_attributes, text_message

        span = RecordedSpan("chat test-model-1")
        set_output_content_attributes(span, True, [text_message("assistant", "hello")])
        assert span.attributes["gen_ai.completion.0.role"] == "assistant"
        assert span.attributes["gen_ai.completion.0.content"] == "hello"

    def test_the_carrier_stays_behind_the_capture_gate(self) -> None:
        from launchdarkly_ai_server import set_input_content_attributes, text_message

        span = RecordedSpan("chat test-model-1")
        set_input_content_attributes(span, False, messages=[text_message("user", "hi")])
        assert span.attributes == {}


# ---------------------------------------------------------------------------
# Every span a streaming run opens must be closed by its `finally`
# ---------------------------------------------------------------------------

#: The handlers whose streaming path dispatches tools inline, in the generator itself. Each holds the
#: open `execute_tool` span in a local, so each needs that local in its `finally`.
INLINE_TOOL_LOOP_HANDLERS: dict[str, str] = {
    "claude-messages": "launchdarkly_ai_claude_messages.handler",
    "openai-messages": "launchdarkly_ai_openai_messages.handler",
    "langchain-messages": "launchdarkly_ai_langchain_messages.handler",
}

#: The handlers that dispatch tools through the vendor's own hook or callback object. The open spans
#: live in that object, so the same duty is discharged by an `abandon_open_spans`-style method.
HOOK_BASED_TOOL_HANDLERS: dict[str, str] = {
    "claude-agents": "launchdarkly_ai_claude_agents.handler",
    "openai-agents": "launchdarkly_ai_openai_agents.handler",
    "langchain-agents": "launchdarkly_ai_langchain_agents.handler",
}


def _handler_source(module_path: str) -> str:
    return inspect.getsource(importlib.import_module(module_path))


def _package_source(module_path: str) -> str:
    """The handler and its spans module together.

    Where the abandonment helper lives is a free choice: claude-agents and openai-agents keep it
    beside the hooks in handler.py, langchain-agents keeps it on the callback object in spans.py.
    The duty is the same, so the check looks in both rather than dictating the file.
    """
    spans_path = module_path.rsplit(".", 1)[0] + ".spans"
    return _handler_source(module_path) + _handler_source(spans_path)


def _finally_blocks(source: str) -> list[str]:
    """The body of every ``finally:`` in the source.

    Any of them will do, not only the last. langchain-agents cleans up in an inner ``finally`` and
    keeps an outer one for the vendor generator, so pinning this to the last block would have failed
    a handler that does the right thing in the right place.
    """
    tree = ast.parse(source)
    blocks: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            segments = [ast.get_source_segment(source, stmt) for stmt in node.finalbody]
            blocks.append("\n".join(seg for seg in segments if seg))
    return blocks


def _call_sites_only(source: str) -> str:
    """The source with every ``def`` line removed, so a definition cannot pass for a call.

    ``abandon_open_spans(`` appears in its own ``def`` line, and three of the handlers define the
    helper in the same module that has to call it. Searching the raw source therefore said yes to a
    helper nothing reached, which is the exact thing this is here to catch.
    """
    return "\n".join(
        line for line in source.split("\n") if not line.lstrip().startswith("def ")
    )


def _function_bodies(source: str, name: str) -> list[str]:
    """Every definition of ``name`` in the source, not just the first.

    langchain-agents defines ``abandon_open_spans`` twice: once on the callback handler that holds the
    spans, and once on the wrapper the handler actually calls. Checking only the first left the
    wrapper unchecked, so a wrapper that closed nothing kept this green.
    """
    bodies = [
        _function_body(source[m.start() :], name)
        for m in re.finditer(rf"^[ \t]*def {re.escape(name)}\(", source, re.M)
    ]
    assert bodies, f"{name} is not defined"
    return bodies


def _function_body(source: str, name: str) -> str:
    """The body of one function: every line indented deeper than its ``def``.

    By indentation rather than by finding the next ``def``, because these helpers are nested inside a
    factory and the last one in a factory has no sibling after it. Scanning to the next ``def`` there
    swallowed the rest of the module and read its calls as the helper's own.
    """
    match = re.search(rf"^([ \t]*)def {re.escape(name)}\(", source, re.M)
    assert match, f"{name} is not defined"
    depth = len(match.group(1))
    body: list[str] = []
    started = False
    # Skipping from the `def` line rather than from the end of the name, so a signature that wraps
    # over several lines does not end the body before it starts.
    for line in source[match.start() :].split("\n")[1:]:
        if not line.strip():
            if started:
                body.append(line)
            continue
        if len(line) - len(line.lstrip()) <= depth:
            if started:
                break
            continue
        started = True
        body.append(line)
    return "\n".join(body)


class TestStreamingTeardownClosesToolSpans:
    """A tool cancelled mid-flight must not leave its span open.

    `except Exception` does not see a `CancelledError` or a `GeneratorExit`, so the tool loop's own
    handler never runs for those, and the streaming `finally` is the only code left that can end the
    span. Four of the six handlers once held the open tool span in a local that `finally` never read,
    which exported a closed parent above a child that never arrived.

    Structural rather than behavioural on purpose: the leak is a property of which variables the
    teardown reads, and a behavioural test would need a cancellable tool per handler to say the same
    thing six times.
    """

    @pytest.mark.parametrize("name", sorted(INLINE_TOOL_LOOP_HANDLERS))
    def test_an_inline_tool_loop_tracks_its_open_span_for_the_teardown(
        self, name: str
    ) -> None:
        source = _handler_source(INLINE_TOOL_LOOP_HANDLERS[name])
        assert "start_tool_span" in source, f"{name} no longer opens tool spans"
        assert "open_tool_span" in source, (
            f"{name} opens execute_tool spans in its streaming generator but keeps no tracker for "
            "the teardown to close. A tool cancelled mid-flight will leak its span."
        )
        # The tracker has to be closed inside a `finally`, not merely mentioned somewhere after the
        # last one. Slicing from the last `finally:` to the end of the file took in every helper
        # defined below it, so a function that happened to name the tracker satisfied this while the
        # teardown closed nothing. Uses the same block reader as the hook-based checks below.
        teardowns = _finally_blocks(source)
        assert any("open_tool_span" in block for block in teardowns), (
            f"{name} tracks open_tool_span but no `finally` closes it."
        )

    @pytest.mark.parametrize("name", sorted(HOOK_BASED_TOOL_HANDLERS))
    def test_a_hook_based_handler_can_abandon_its_open_spans(self, name: str) -> None:
        source = _package_source(HOOK_BASED_TOOL_HANDLERS[name])
        # The definition, not the name: a mention in a docstring must not satisfy this.
        assert re.search(r"def abandon_open_spans\(", source), (
            f"{name} dispatches tools through a hook object, so it needs an abandon_open_spans to "
            "end the spans still open when a consumer walks away."
        )
        # And the call has to sit in a `finally`. Anywhere else is not teardown: GeneratorExit and
        # CancelledError never enter `except Exception`, and a success path does not run at all when
        # a consumer walks away, so an abandon call in either place closes nothing on the one path it
        # exists for. Definition lines are stripped first, because three of these handlers define the
        # helper in the module that has to call it and the `def` line would otherwise answer for the
        # call. Matched by prefix rather than exact name, because claude-agents binds it to a local
        # when it unpacks the hook factory's result.
        handler = _handler_source(HOOK_BASED_TOOL_HANDLERS[name])
        teardowns = [_call_sites_only(block) for block in _finally_blocks(handler)]
        assert any(re.search(r"abandon\w*\(", block) for block in teardowns), (
            f"{name} never calls its abandonment helper from a `finally`, so a consumer who walks "
            "away still leaves tool spans open."
        )

    @pytest.mark.parametrize("name", sorted(HOOK_BASED_TOOL_HANDLERS))
    def test_abandonment_is_distinct_from_failure(self, name: str) -> None:
        # Both exist for a reason: abandonment leaves UNSET, failure records the exception. Collapsing
        # them makes one handler report an error for a run another reports as a clean stop.
        source = _package_source(HOOK_BASED_TOOL_HANDLERS[name])
        assert re.search(r"def close_open_spans\(", source), (
            f"{name} lost its failure path for open spans"
        )
        # The bodies, not the names. Two helpers that both call fail_span are one helper with two
        # names, and the whole point is that abandonment does not record an exception.
        # Every definition, because a handler may wrap the one that holds the spans.
        for abandon in _function_bodies(source, "abandon_open_spans"):
            assert "fail_span" not in abandon, (
                f"{name} has an abandon_open_spans that records an exception, so an abandoned run "
                "reports an error nobody had. Abandonment leaves the span UNSET."
            )
            # A wrapper delegates rather than marking the span itself, and that is still correct.
            assert "abandoned=True" in abandon or "abandon_open_spans(" in abandon, (
                f"{name} has an abandon_open_spans that neither marks the spans abandoned nor hands "
                "off to one that does, so it closes nothing."
            )
        # At least one body has to do the recording itself. Accepting delegation alone let the
        # wrapper's hand-off keep this green while the handler it delegates to stopped failing
        # anything, which is the failure path disappearing without the lock noticing.
        close_bodies = _function_bodies(source, "close_open_spans")
        assert any("fail_span" in body for body in close_bodies), (
            f"{name} has no close_open_spans that records a failure, so a run that died mid-tool "
            "leaves its spans ended without the error that killed them."
        )
        for body in close_bodies:
            assert "fail_span" in body or "close_open_spans(" in body, (
                f"{name} has a close_open_spans that neither records the failure nor hands off to "
                "one that does, so it closes nothing."
            )
