from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

# ---------------------------------------------------------------------------
# LDContext
# ---------------------------------------------------------------------------

LDContext = dict[str, Any]
"""
Structurally compatible with all LaunchDarkly SDK context shapes.
At minimum carries ``kind`` and ``key``; additional attributes are allowed.
"""

# ---------------------------------------------------------------------------
# LDClientInterface — minimal surface needed from any LD SDK client
# ---------------------------------------------------------------------------


class LDClientInterface(Protocol):
    """
    Structural type for any LaunchDarkly client; ``ldclient.LDClient`` satisfies it.

    ``variation``, ``flush`` and ``close`` may be sync or async: the SDK awaits any
    awaitable result. ``track`` must be synchronous: the SDK calls it from sync code and
    drops the result, so an ``async def track`` would never send an event. ``context`` is
    ``Any`` because ``ldclient`` takes an ``ldclient.Context``, which the SDK builds from
    an ``LDContext`` before the call. Parameters are positional-only so implementations
    may name them freely (``ldclient`` calls the third ``variation`` argument ``default``).
    """

    def variation(self, key: str, context: Any, default_value: Any, /) -> Any: ...

    def track(
        self,
        event_name: str,
        context: Any,
        data: Any = None,
        metric_value: float | None = None,
        /,
    ) -> None: ...

    def flush(self) -> Any: ...

    def close(self) -> Any: ...


# ---------------------------------------------------------------------------
# NativeTool / NATIVE_TOOL_KEY
# ---------------------------------------------------------------------------

NATIVE_TOOL_KEY: str = "__ld_native_tool__"
"""
String key used to stash the original ``NativeTool`` instance on a tracking
stub. Handler packages read this to recover the provider tool name without a
separate data channel.
"""


class NativeTool:
    """
    Marker for a provider built-in tool. Place as a value in ``tool_handlers``
    to signal that the named tool is a native provider capability. Each instance
    carries a unique identity sentinel (``id``) so duplicate sentinels are
    detectable.
    """

    def __init__(self, tool_name: str) -> None:
        self.id: object = object()
        self.tool_name: str = tool_name


# ---------------------------------------------------------------------------
# AiConfigRep — config shape delivered by a LaunchDarkly flag variation
# ---------------------------------------------------------------------------


class ModelConfig:
    name: str
    region: str | None
    parameters: dict[str, Any] | None
    custom: dict[str, Any] | None


class Tool:
    name: str
    parameters: dict[str, Any]
    type: Literal["function"]
    custom_parameters: dict[str, Any] | None
    description: str | None


class Message:
    role: Literal["user", "assistant", "system"]
    content: str


AiConfigRep = dict[str, Any]
"""
Raw AI config dict as returned by ``parse_ai_config``. Fields include
``model``, ``provider``, at least one of ``instructions``, ``messages``, or a
non-empty ``classifiers`` list, and an optional ``skills`` array of
``{key, version}`` references. Parsing does not validate ``skills``; read it
with ``launchdarkly_ai_server.skill_refs``, which does.
"""

VariationMeta = dict[str, Any]
"""
Variation metadata: ``enabled``, ``variationKey``, ``version``, ``mode``, and
the pinned model-config identity ``modelKey`` / ``modelVersion`` (delivered in
``_ldMeta``; absent when no model config is pinned).
"""

# ---------------------------------------------------------------------------
# ProviderHandler — callable class wrapping a provider implementation
# ---------------------------------------------------------------------------

HandlerResult = dict[str, Any]
"""Return shape of a provider handler call: ``{"output": ..., "usage": ...}``."""


class HandlerStreamEvent:
    """Events a handler stream yields.

    ``chunk`` and ``done`` remain valid. A handler may instead yield the AG-UI
    text and tool subset (``TEXT_MESSAGE_*``, ``TOOL_CALL_*``) and one ``done``.
    ``config().stream()`` still yields ``chunk`` then ``done``.
    """

    pass


T = TypeVar("T")

_HandlerFn = Callable[
    [
        AiConfigRep,
        str | None,
        Mapping[str, Callable[..., Any] | NativeTool] | None,
        dict[str, Any] | None,
        list[dict[str, Any]] | None,
    ],
    Awaitable[HandlerResult],
]
_StreamFn = Callable[
    [
        AiConfigRep,
        str | None,
        Mapping[str, Callable[..., Any] | NativeTool] | None,
        dict[str, Any] | None,
        list[dict[str, Any]] | None,
    ],
    AsyncGenerator[dict[str, Any], None],
]


class ProviderHandler:
    """
    Callable handler with provider metadata.

    Wraps an async callable and exposes:
    - ``__call__`` — blocking invocation
    - ``stream``   — optional async-generator streaming (may be ``None``)
    - ``provides_for`` — ``(provider_name, mode)`` tuple or ``None``
    - ``capture_content`` — the effective content-capture decision for this handler

    ``capture_content`` is declared here so the client core can apply the handler's own content
    decision to content it writes on the handler's behalf — notably the judge's reasoning —
    without reaching into the factory's closure. ``None`` at construction reads the process
    default from ``configure`` when the value is used. An explicit ``True`` or ``False`` is an
    override. The property returns that effective bool, so judge and span paths share one gate.
    """

    provides_for: tuple[str, Literal["agent", "messages"]] | None

    def __init__(
        self,
        fn: _HandlerFn,
        provides_for: tuple[str, Literal["agent", "messages"]] | None = None,
        stream_fn: _StreamFn | None = None,
        capture_content: bool | None = None,
    ) -> None:
        self._fn = fn
        self.provides_for = provides_for
        self._stream_fn = stream_fn
        self._capture_content_override = capture_content

    @property
    def capture_content(self) -> bool:
        """Effective content capture: an explicit override, else the process default.

        Assigning ``True`` or ``False`` sets this handler's override, the same as
        passing it to the factory. It does not change the process default.
        """
        from .lifecycle import effective_capture_content

        return effective_capture_content(self._capture_content_override)

    @capture_content.setter
    def capture_content(self, value: bool) -> None:
        self._capture_content_override = value

    async def __call__(
        self,
        config: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> HandlerResult:
        from .lifecycle import note_ai_invocation

        note_ai_invocation()
        return await self._fn(config, user_input, tool_handlers, variables, history)

    async def stream(
        self,
        config: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self._stream_fn is None:
            raise NotImplementedError("This handler does not support streaming")
        from .lifecycle import note_ai_invocation

        note_ai_invocation()
        return self._stream_fn(config, user_input, tool_handlers, variables, history)

    @property
    def has_stream(self) -> bool:
        return self._stream_fn is not None


# ---------------------------------------------------------------------------
# Response / stream event types
# ---------------------------------------------------------------------------


@dataclass
class InputTokenDetails:
    """The cache breakdown behind an inclusive ``input`` figure.

    Present only when the provider reported at least one cache field. ``uncached + cache_read +
    cache_creation`` equals :attr:`UsageDict.input`.
    """

    uncached: int = 0
    cache_read: int = 0
    cache_creation: int = 0


@dataclass
class UsageDict:
    input: int = 0
    output: int = 0
    total: int = 0
    #: Cache breakdown, when the provider reported one. See :class:`InputTokenDetails`.
    input_details: InputTokenDetails | None = None


@dataclass
class JudgeResult:
    usage: UsageDict
    response: str
    score: float
    event_key: str | None = None
    """Metric name for this score. Set when the judge produced one event per label."""


@dataclass
class ProviderResponse(Generic[T]):
    response: T
    usage: UsageDict
    judge_results: dict[str, JudgeResult] | None = None
    """
    Judge evaluation results. Populated when ``skip_judges=False`` (default) and
    at least one judge ran during ``invoke()`` / ``stream()``.
    """
    judge_tasks: list[JudgeTask] | None = None
    """
    Pre-packaged judge tasks produced when ``skip_judges=True``. Each task is a
    plain serialisable dataclass that can be passed to a background thread running
    ``run_judge(task, handlers)``.

    ``None`` when ``skip_judges=False`` (judges ran inline).
    """
    track_data: TrackData | None = None
    """
    Tracking payload from this invocation. Carried inside each :class:`JudgeTask`
    so that background judge results are attributed to the originating request
    (run ID, config key, graph key, etc.).
    """


TrackData = dict[str, Any]
"""Payload attached to every LaunchDarkly tracking event."""


@dataclass
class JudgeTask:
    """
    A fully-resolved, serialisable snapshot of everything needed to execute a
    judge evaluation in a background thread, without re-fetching the variation
    from LaunchDarkly or re-running the main invocation.

    Produced by ``ConfigInstance.prepare_judge()`` on the main thread and passed
    to ``run_judge()`` in the worker via the thread's argument list or a queue.
    All fields are plain Python primitives / dicts — safe to pickle or transmit
    over any IPC channel.
    """

    config_key: str
    """The flag key used for the judge config variation."""
    judge_config: AiConfigRep
    """The already-fetched judge AI config (plain dict)."""
    judge_meta: VariationMeta
    """Variation metadata for the judge config."""
    actual_output: str
    """The LLM response to evaluate."""
    user_context: LDContext
    """The LaunchDarkly context from the originating invocation."""
    judge_provider: str | None
    """Provider name from the judge config (pre-resolved for handler selection)."""
    judge_mode: str
    """Effective mode after normalisation (e.g. ``'messages'`` or ``'agent'``)."""
    collapse_messages: bool
    """
    When ``True``, the worker must collapse ``messages`` to a single
    ``instructions`` string before calling the handler.
    """
    parent_track_data: TrackData
    """
    Track data from the parent ``invoke()`` call (run ID, config key, graph key,
    etc.). Merged into the judge tracking event so it is attributed to the
    originating request.
    """
    variables: dict[str, Any] | None = None
    """Optional extra template variables for the judge prompt."""
    evaluation_metric_key: str | None = None
    user_input: str | None = None
    """The input that produced ``actual_output``.

    Carried so this path builds the same ``message_history`` as the inline one.
    Previously absent, which showed a deferred judge a response with no
    request beside it.
    """
    trajectory: str = ""
    """The invocation's rendered tool-call trajectory.

    A plain string rather than the structured record, since every field here
    must stay picklable.
    """
    """LD metric key to track the score against."""


@dataclass
class JudgeRunResult:
    """
    Result returned by :func:`run_judge` — the judge score, reasoning text,
    token usage, and merged track data ready to pass to ``client.track()``.
    """

    score: float
    response: str
    usage: UsageDict
    track_data: TrackData
    """
    Track data with ``judgeConfigKey`` merged in. Pass this directly to
    ``get_client().track(task.evaluation_metric_key, ctx, track_data, score)``
    from the main thread after the worker finishes.
    """
    results: dict[str, JudgeResult] | None = None
    """
    Present when one judge call produced several scores. Keys are
    ``config_key.question_key``. ``score`` and ``response`` are the first
    entry. Every entry carries the call's full usage and, for a classifier,
    its ``event_key``.
    """
    metrics: list[dict[str, Any]] | None = None
    """
    One LaunchDarkly metric per classifier. When this is set, track each
    entry's ``eventKey`` and do not also track ``evaluation_metric_key``.
    """
    event_key: str | None = None
    """Metric name for ``score``. Set to the first label's event key when one call produced several."""


# Stream events
StreamChunkEvent = dict[str, Any]  # {"type": "chunk", "text": str}
StreamDoneEvent = dict[str, Any]  # {"type": "done", "response": str, "usage": ..., ...}
StreamEvent = dict[str, Any]  # StreamChunkEvent | StreamDoneEvent

# Graph stream events (camelCase public fields, matching the TypeScript SDK)
# node_start: {type, nodeKey}
# chunk: {type, text, nodeKey}
# node_done: {type, nodeKey, response, usage, judgeResults?}  (the node's own judges)
# handoff: {type, sourceKey, targetKey}
# done: {type, response, usage, judgeResults?}
GraphStreamEvent = dict[str, Any]

# Internal execute stream event that also carries track_data
ExecuteStreamDoneEvent = dict[str, Any]
ExecuteStreamEvent = dict[str, Any]

# ---------------------------------------------------------------------------
# Graph types
# ---------------------------------------------------------------------------

GraphTopology = dict[str, Any]
"""``{"root": str, "edges": {...}}`` as delivered by a graph flag variation."""


@dataclass
class GraphEdge:
    """A directed edge between two agent configs in a graph."""

    key: str
    """Stable edge identifier (``{source_key}-{target_key}``)."""
    source_key: str
    """Source node config key."""
    target_key: str
    """Target node config key."""
    handoff: dict[str, Any] | None = None
    """Optional handoff data from the graph definition."""


@dataclass
class GraphNode:
    """A node in a resolved agent graph: an evaluated agent config plus its outgoing edges."""

    key: str
    """The node's config key."""
    config: AiConfigRep
    """Evaluated agent config for this node."""
    meta: VariationMeta
    """Variation metadata for this node."""
    edges: list[GraphEdge] = field(default_factory=list)
    """Outgoing edges from this node."""
    is_terminal: bool = False
    """``True`` when the node has no outgoing edges."""


class GraphDefinition:
    """
    A resolved agent graph returned by ``resolve_graph()``. Exposes topology
    accessors and execution primitives as attributes rather than dict keys.

    Check ``.enabled`` before traversing — a disabled graph has a ``None`` root
    and its ``run_node`` / ``route`` methods raise immediately.
    """

    def __init__(
        self,
        *,
        key: str,
        enabled: bool,
        root: GraphNode | None,
        get_node: Callable[[str], GraphNode | None],
        get_child_nodes: Callable[[str], list[GraphNode]],
        get_parent_nodes: Callable[[str], list[GraphNode]],
        terminal_nodes: Callable[[], list[GraphNode]],
        is_terminal: Callable[[str], bool],
        edges_from: Callable[[str], list[GraphEdge]],
        run_node: Callable[..., Any],
        route: Callable[..., Any],
        traverse: Callable[..., Any],
        reverse_traverse: Callable[..., Any],
    ) -> None:
        self.key = key
        self.enabled = enabled
        self.root = root
        self.get_node = get_node
        self.get_child_nodes = get_child_nodes
        self.get_parent_nodes = get_parent_nodes
        self.terminal_nodes = terminal_nodes
        self.is_terminal = is_terminal
        self.edges_from = edges_from
        self.run_node = run_node
        self.route = route
        self.traverse = traverse
        self.reverse_traverse = reverse_traverse


@dataclass
class ProviderGraphResponse:
    """The value returned by ``graph().invoke()``."""

    response: str
    """The final text output (from the last node executed)."""
    usage: UsageDict
    """Aggregate token counts across all nodes."""
    judge_results: dict[str, JudgeResult] | None = None
    """Results from a graph-level judge, if configured."""


class NativeGraphRunner(Protocol):
    """
    What a native graph adapter (``to_lang_graph``, ``to_openai_agents``, …) returns.

    ``invoke`` resolves to ``{"response": str, "usage": UsageDict}``.
    """

    def invoke(
        self,
        input_text: str = "",
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> Awaitable[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Agent Skills
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillReference:
    """A version-pinned pointer to a skill, as attached to an AI Config variation."""

    key: str
    """Immutable skill key — ``^[a-z0-9][a-z0-9-]*$``, at most 256 characters."""
    version: int
    """Immutable skill version — an integer >= 1."""


@dataclass(frozen=True)
class Skill:
    """
    A single verified ``SKILL.md`` document. Immutable.

    The accessors return a ``Skill`` only after integrity verification, so
    ``content`` is exactly the bytes LaunchDarkly delivered and ``content_hash``
    is their sha256.
    """

    key: str
    version: int
    content: bytes
    """The verbatim bytes. The SDK never decodes or parses them."""
    content_hash: str
    """sha256, lowercase hex, over the verbatim bytes of ``content``."""
    name: str | None = None
    """Display name from LaunchDarkly metadata; never parsed from the content."""
    description: str | None = None
    """Description from LaunchDarkly metadata; never parsed from the content."""


SkillOutcomeReason = Literal[
    "absent", "integrity_failure", "ok", "store_unavailable", "wrong_version"
]
"""
The outcomes ``get_skill_result`` reports.

- ``ok`` — a verified skill was returned.
- ``absent`` — the store does not hold the key.
- ``integrity_failure`` — content failed verification and was withheld. Fail
  closed on this one: it can indicate tampering.
- ``store_unavailable`` — the store raised. An outage, not a deletion.
- ``wrong_version`` — the store holds a different version than the one
  requested, so nothing was returned.
"""


@dataclass(frozen=True)
class SkillOutcome:
    """
    The result of ``get_skill_result``: the skill, plus why it was or wasn't
    returned. Immutable.
    """

    skill: Skill | None
    """The verified skill; set only when ``reason == "ok"``."""
    reason: SkillOutcomeReason
    """Which outcome happened; see ``SkillOutcomeReason``."""
    detail: str | None
    """
    Human-readable message, set for every reason except ``ok``.

    Safe to log: never contains skill content or filesystem paths. Branch on
    ``reason``, not on this.
    """


ReconcileActionKind = Literal[
    "written", "updated", "skipped_current", "removed", "error"
]
"""
The per-skill outcomes ``write_skills`` reports.

- ``written`` — the file did not exist and now holds the resolved content.
- ``updated`` — a managed file held different bytes and was overwritten.
- ``skipped_current`` — the bytes on disk already are the resolved content.
- ``removed`` — the skill is no longer managed and is not on disk (whether this
  run deleted it or it was already gone).
- ``error`` — the outcome was refused or failed; see ``ReconcileAction.error``.
"""


@dataclass(frozen=True)
class ReconcileAction:
    """What ``write_skills`` did — or refused to do — for one skill."""

    key: str
    """
    The skill key, or ``""`` for a run-level failure not tied to one skill (for
    example a corrupt or unwritable manifest, or a failed retrieval). Expect
    ``""`` when grouping a report by key.
    """
    action: ReconcileActionKind
    version: int | None = None
    path: str | None = None
    """Canonical resolved path, when one was determined."""
    error: str | None = None
    """Failure detail, set only when ``action == "error"``."""


@dataclass(frozen=True)
class ReconcileReport:
    """The result of a ``write_skills`` run: one action per outcome."""

    actions: list[ReconcileAction] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` iff no action is an ``error``."""
        return not self.errors

    @property
    def errors(self) -> list[ReconcileAction]:
        """The ``error`` actions, in ``actions`` order."""
        return [a for a in self.actions if a.action == "error"]


@dataclass(frozen=True, init=False)
class SkillSource:
    """
    A requested set of skills that is read when it is needed, not fixed when
    the call is made. Immutable.

    ``write_skills`` reads it once per call; ``watch_skills`` reads it on every
    reconcile, and re-reconciles when one of the AI Configs named in
    ``config_keys`` changes. ``config_skills(key, context)`` builds the common
    case, one AI Config evaluated for one context.

    ``refs`` raising, or answering anything other than a sequence of
    ``SkillReference`` values and key strings, is a failed retrieval: nothing on
    disk changes. Raise to say "I could not tell"; never return ``[]`` for it,
    because ``[]`` prunes every skill the root manages.
    """

    refs: Callable[
        [], Sequence[SkillReference | str] | Awaitable[Sequence[SkillReference | str]]
    ]
    """Zero-argument callable, sync or async, returning the current references."""
    config_keys: tuple[str, ...]
    """The AI Config keys whose changes should make a watcher read ``refs`` again."""

    def __init__(
        self,
        refs: Callable[
            [],
            Sequence[SkillReference | str] | Awaitable[Sequence[SkillReference | str]],
        ],
        config_keys: Sequence[str] = (),
    ) -> None:
        """
        Raises:
            TypeError: If *refs* is not callable, or *config_keys* is a string
                or holds anything but non-empty strings. A bare string would be
                followed as one-character keys, none of which ever changes.
        """
        if not callable(refs):
            raise TypeError(
                f"SkillSource refs must be callable, got {type(refs).__name__}"
            )
        if isinstance(config_keys, (str, bytes)):
            raise TypeError(
                "SkillSource config_keys must be a sequence of config keys, not a "
                f"single string; got {config_keys!r}. Pass ({config_keys!r},)."
            )
        if not isinstance(config_keys, Sequence):
            raise TypeError(
                "SkillSource config_keys must be a sequence of config keys, got "
                f"{type(config_keys).__name__}"
            )
        keys = tuple(config_keys)
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(
                    "SkillSource config_keys must hold only strings, got "
                    f"{type(key).__name__}"
                )
            if not key:
                raise TypeError("SkillSource config_keys must not hold an empty key")
        object.__setattr__(self, "refs", refs)
        object.__setattr__(self, "config_keys", keys)


# ---------------------------------------------------------------------------
# Model / graph options
# ---------------------------------------------------------------------------

ModelArgs = dict[str, Any]
RoutedModelArgs = dict[str, Any]
GraphOptions = dict[str, Any]
GraphArgs = dict[str, Any]


@dataclass(frozen=True)
class SDKOptions:
    """Process-wide behavior defaults, applied by :func:`configure`.

    ``capture_content`` is the default for writing prompts, model output, tool
    arguments, tool results, and judge explanations. The built-in value is
    ``True``: a process that never calls ``configure`` sends that text. It is a
    default, not a ceiling: an explicit ``True`` or ``False`` on a handler
    factory or a convenience wrapper overrides it, and an explicit ``False``
    removes the text everywhere, including on offline evaluation spans. Further
    process-wide defaults are added as fields on this object.
    """

    capture_content: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.capture_content, bool):
            raise TypeError("SDKOptions.capture_content must be a bool")


InitClientOptions = dict[str, Any]

# ---------------------------------------------------------------------------
# Parse result helper
# ---------------------------------------------------------------------------


@dataclass
class ParseSuccess(Generic[T]):
    success: Literal[True]
    data: T


@dataclass
class ParseFailure:
    success: Literal[False]
    error: dict[str, str]


ParseResult = ParseSuccess[Any] | ParseFailure
