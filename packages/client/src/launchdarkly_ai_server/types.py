from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

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


class LDClientInterface:
    """Protocol-like base; pass any object that satisfies this surface."""

    async def variation(
        self, key: str, context: LDContext, default_value: Any
    ) -> Any: ...

    def track(
        self,
        event_name: str,
        context: LDContext,
        data: Any = None,
        metric_value: float | None = None,
    ) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


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
``model``, ``provider``, at least one of ``instructions`` / ``messages``, and an
optional ``skills`` array of ``{key, version}`` references (see ``skill_refs``).
"""

VariationMeta = dict[str, Any]
"""
Variation metadata: ``enabled``, ``variation_key``, ``version``, ``mode``.
"""

# ---------------------------------------------------------------------------
# ProviderHandler — callable class wrapping a provider implementation
# ---------------------------------------------------------------------------

HandlerResult = dict[str, Any]
"""Return shape of a provider handler call: ``{"output": ..., "usage": ...}``."""


class HandlerStreamEvent:
    """Union of chunk / done events yielded by a handler's stream method."""

    pass


T = TypeVar("T")

_HandlerFn = Callable[
    [
        AiConfigRep,
        str | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        list[dict[str, Any]] | None,
    ],
    Awaitable[HandlerResult],
]
_StreamFn = Callable[
    [
        AiConfigRep,
        str | None,
        dict[str, Any] | None,
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
    """

    provides_for: tuple[str, Literal["agent", "messages"]] | None

    def __init__(
        self,
        fn: _HandlerFn,
        provides_for: tuple[str, Literal["agent", "messages"]] | None = None,
        stream_fn: _StreamFn | None = None,
    ) -> None:
        self._fn = fn
        self.provides_for = provides_for
        self._stream_fn = stream_fn

    async def __call__(
        self,
        config: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> HandlerResult:
        return await self._fn(config, user_input, tool_handlers, variables, history)

    async def stream(
        self,
        config: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self._stream_fn is None:
            raise NotImplementedError("This handler does not support streaming")
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


# Stream events
StreamChunkEvent = dict[str, Any]  # {"type": "chunk", "text": str}
StreamDoneEvent = dict[str, Any]  # {"type": "done", "response": str, "usage": ..., ...}
StreamEvent = dict[str, Any]  # StreamChunkEvent | StreamDoneEvent

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
    A single verbatim ``SKILL.md`` document.

    Only ever constructed after integrity verification passes, so ``content``
    holds the exact byte sequence LaunchDarkly delivered and ``content_hash``
    is its sha256. Instances are immutable.
    """

    key: str
    version: int
    content: bytes
    """The verified verbatim bytes, exactly as LaunchDarkly delivered and
    hashed them. Opaque to this SDK: no encoding is claimed and nothing here
    ever parses or interprets them."""
    content_hash: str
    """sha256, lowercase hex, over the verbatim bytes of ``content``."""
    name: str | None = None
    """Display name from LaunchDarkly metadata; never parsed from the content."""
    description: str | None = None
    """Description from LaunchDarkly metadata; never parsed from the content."""


# ---------------------------------------------------------------------------
# Model / graph options
# ---------------------------------------------------------------------------

ModelArgs = dict[str, Any]
RoutedModelArgs = dict[str, Any]
GraphOptions = dict[str, Any]
GraphArgs = dict[str, Any]
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
