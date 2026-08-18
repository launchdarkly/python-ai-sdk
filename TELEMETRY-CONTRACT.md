# Telemetry contract

What the Python SDK must emit to match the TypeScript SDK.

Derived from `js-ai-sdk` at commit `5178db1`, from these files:

- `packages/client/src/content.ts`
- `packages/client/src/utils.ts`
- `packages/{claude,openai,langchain}-{messages,agents}/src/handler.ts`

This file is the authority for the parity work. Where this file and a TypeScript source file
disagree, the TypeScript file wins and this file is wrong. Report the disagreement instead of
guessing.

Every name below is verbatim. Do not adjust the spelling to look more consistent. Several keys
look wrong and are not: `gen_ai.usage.cache_read.input_tokens` really does have two dots in the
middle, and `gen_ai.system` really does hold a different value from `gen_ai.provider.name` on the
two LangChain handlers.

---

## 1. Span tree

Every handler call produces this shape, in both the blocking and the streaming path:

```
invoke_agent                         one per handler call. The root.
├── chat {model}                     one per model turn.
│   └── (nothing)
└── execute_tool {tool_name}         one per tool call.
```

Tool spans are children of the same parent context as the `chat` span, not of the `chat` span
itself. They are siblings of `chat`, under the root. `startToolSpan` and `startModelSpan` both
receive the same `parentContext`, which is the root's context.

Pass the parent context explicitly. Do not rely on the ambient context, because these handlers
open a plain span rather than an active one on the streaming path, so there is no ambient span to
inherit.

Do not justify this with a claim about `asyncio.create_task`. An earlier draft of this file said
context does not survive a task boundary. That is wrong. `asyncio.create_task` copies the caller's
`contextvars` context, so ambient context does cross it. `threading.Thread` is the boundary that
loses context.

None of the six handlers create a task or a thread themselves. The boundaries that exist are
inside vendor libraries: `claude_agent_sdk` uses `anyio`, including one `anyio.to_thread.run_sync`,
and the `agents` and `langgraph` packages use plain `asyncio.create_task`, which inherits context
correctly. The thread hop inside `claude_agent_sdk` is the only place ambient context is genuinely
lost.

### Span names

| Span | Name | Notes |
|---|---|---|
| root | `invoke_agent` | Literal. No interpolation. |
| model turn | `chat {model}` | For example `chat claude-3-5-sonnet-20241022`. |
| tool call | `execute_tool {tool_name}` | For example `execute_tool get_weather`. |
| graph | `ld.ai.graph` | Already correct in Python. Do not touch. See section 8. |

The model goes in the `chat` span name because the semantic conventions name an inference span
`{gen_ai.operation.name} {gen_ai.request.model}`. A bare `chat` aggregates more neatly but tells a
reader nothing about which model ran, which is exactly what this span exists to answer for a
multi-turn run that switches models partway through.

### Blocking versus streaming

The blocking path opens the root as an active span. The streaming path opens it as a plain span and
ends it by hand, because an async generator cannot hold an active span across a yield.

Both paths emit the same spans with the same names and the same attributes. A consumer must not be
able to tell from the trace which path ran.

---

## 2. Attributes on the root span, `invoke_agent`

| Key | Value | Source |
|---|---|---|
| `gen_ai.operation.name` | `invoke_agent` | Literal. Set explicitly, not derived from the span name. |
| `gen_ai.system` | provider name, or `langchain` on the two LangChain handlers | `set_model_identity_attributes` |
| `gen_ai.provider.name` | the real provider name, always | `set_model_identity_attributes` |
| `gen_ai.request.model` | requested model name, except on `claude-agents`. See section 2a | `set_model_identity_attributes` |
| `gen_ai.response.model` | see section 2a. It differs by handler | `finish_root_span` |
| `gen_ai.usage.input_tokens` | run total | `set_usage_span_attributes` |
| `gen_ai.usage.output_tokens` | run total | `set_usage_span_attributes` |
| `gen_ai.usage.total_tokens` | run total | `set_usage_span_attributes` |
| `gen_ai.usage.cache_read.input_tokens` | run total | `set_usage_span_attributes` |
| `gen_ai.usage.cache_creation.input_tokens` | run total | `set_usage_span_attributes` |
| `gen_ai.usage.prompt_tokens` | same as `input_tokens` | `set_usage_span_attributes` |
| `gen_ai.usage.completion_tokens` | same as `output_tokens` | `set_usage_span_attributes` |
| `launchdarkly.operation.type` | `gen_ai` | `set_ld_span_attributes` |
| `launchdarkly.config.key` | `TrackData.configKey` | `set_ld_span_attributes` |
| `launchdarkly.variation.key` | `TrackData.variationKey` | `set_ld_span_attributes` |
| `launchdarkly.run.id` | `TrackData.runId` | `set_ld_span_attributes` |
| `launchdarkly.graph.key` | `TrackData.graphKey`, only when present | `set_ld_span_attributes` |
| `launchdarkly.stream.abandoned` | `True`, only when abandoned | `end_span_once` |

The root also carries one span event, `feature_flag`, with these event attributes:

| Event attribute | Value |
|---|---|
| `feature_flag.key` | config key |
| `feature_flag.provider.name` | `LaunchDarkly` |
| `feature_flag.set.id` | environment id, only when present |

The root is the only span that carries the config-association attributes and the `feature_flag`
event. Child spans carry neither. A test asserts this, so do not add them to children out of
tidiness.

The two lifecycle markers are the exception, and they are deliberate. `launchdarkly.stream.abandoned`
and `launchdarkly.run.cancelled` say why a span stopped, so they belong on whichever span was still
open when the teardown ran, root or child. A `chat` span that ends without a status and without a
marker is indistinguishable from a bug. Neither marker identifies a config, which is what the rule
above is protecting: a config-scoped query still finds exactly one span per run.

That is also why the root carries run-level token totals. It is the span a config-scoped query
finds. Without totals on it, that query returns nothing, because summing the children requires
having already found them.

`claude-agents` also writes `gen_ai.conversation.id` on the root, from the session id on the
CLI's `system` / `init` message, write-if-absent. A caller-supplied id from `conversation_id(...)`
(TypeScript: `withConversationId`) wins. When the caller supplies none, the session id is used.
An app that opens a fresh CLI session per turn and re-feeds history must pass its own id, or each
turn becomes its own conversation. See section 4 for the third place it appears.

When a conversation id is bound, every handler stamps it on root, `chat`, and `execute_tool` (and
on `ld.ai.graph`) via the shared span processor. No id is invented when the caller supplies none.

---

## 2a. Which model name goes in `gen_ai.response.model`

An earlier draft of this file said Anthropic and LangChain report the requested model and OpenAI
reports the model that answered. That was wrong. Only one handler reads the model back off the
response, and one other reports a real per-turn model. Copy this table exactly.

| Handler | On the root | On a `chat` span |
|---|---|---|
| `claude-messages` | requested name | requested name |
| `claude-agents` | requested name | the model the turn actually used, read off the streamed inference |
| `openai-messages` | the model that answered, falling back to the requested name | the model that answered |
| `openai-agents` | requested name | requested name |
| `langchain-messages` | requested name | requested name |
| `langchain-agents` | requested name | requested name |

So four of the six handlers write the requested name everywhere, and section 3's table must be
read through this one.

### `gen_ai.request.model` has one exception too

Five handlers write the requested name, which is what the caller asked for.

`claude-agents` writes the model the inference actually used, on its `chat` span, the same value it
writes to `gen_ai.response.model`. That is deliberate in the TypeScript source, which says the two
must not disagree: the CLI reports the model it really ran, and this handler has no separate
requested name per turn to report. Its root span still writes the requested name.

An earlier draft of this file claimed the requested name for all six, which reads as a defect in the
handler rather than in this document.

Do not "fix" `openai-agents` to resolve the answering model. It never has, no test pins it, and
adding it invents behaviour the TypeScript SDK does not have. The reason `openai-messages` differs
is that OpenAI resolves an alias such as `gpt-4o` to a dated snapshot and that handler has the
resolved value to hand.

---

## 3. Attributes on a `chat` span

| Key | Value |
|---|---|
| `gen_ai.operation.name` | `chat` |
| `gen_ai.system` | provider name, or `langchain` on the two LangChain handlers |
| `gen_ai.provider.name` | the real provider name |
| `gen_ai.request.model` | requested model name |
| `gen_ai.response.model` | per handler. See section 2a, not "the model that answered" |
| `gen_ai.response.finish_reasons` | a list, only when a reason exists. See section 5 |
| the seven `gen_ai.usage.*` keys | this turn's counts, not the run total |

On success, set the span status to OK and end it. On failure, see section 6.

On `claude-agents`, `gen_ai.request.model` is the model the turn actually used rather than the
requested name. See section 2a.

`claude-agents` sets three more, each only when the Claude Agent SDK reports it:

| Key | Value |
|---|---|
| `gen_ai.response.id` | provider request id |
| `gen_ai.conversation.id` | session id, write-if-absent against a caller-supplied id |
| `gen_ai.agent.name` | subagent type |

---

## 4. Attributes on an `execute_tool` span

| Key | Value |
|---|---|
| `gen_ai.operation.name` | `execute_tool` |
| `gen_ai.tool.name` | tool name as the model saw it |
| `gen_ai.tool.call.id` | provider's tool call id |
| `gen_ai.tool.call.arguments` | gated. See section 7 |
| `gen_ai.tool.call.result` | gated. See section 7 |

No usage attributes. A tool call spends no tokens.

`claude-agents` also writes `gen_ai.conversation.id` here, from the session id on the tool-use
hook input, when present, write-if-absent. That attribute therefore appears on all three span types
for this one handler: root, `chat`, and `execute_tool`. A caller-supplied id is already on the span
when `conversation_id(...)` is bound.

---

## 4a. Judge evaluation events

A judge run is itself a tracked AI call (`invoke_agent` + `chat`). After the score is parsed, the
SDK writes a `gen_ai.evaluation.result` span event on that `invoke_agent` span:

| Event attribute | Value |
|---|---|
| `gen_ai.evaluation.name` | judge config key |
| `gen_ai.evaluation.score.value` | numeric score |
| `gen_ai.evaluation.explanation` | judge reasoning, when present |

The same keys are mirrored as span attributes. `gen_ai.evaluation.score.label` is not invented.
The existing `track(evaluationMetricKey)` call is unchanged and still feeds AI Config Monitoring.

---

## 5. Finish reasons

One vocabulary across all six handlers: `stop`, `length`, `content_filter`, `tool_calls`, `error`.

Three different mechanisms produce it. An earlier draft of this file described one lookup table
for all six handlers. That was wrong, and building it would put a wrong or missing finish reason
on every tool-calling OpenAI turn.

| Handlers | Mechanism |
|---|---|
| `claude-messages`, `claude-agents` | map the provider's `stop_reason` through the table below |
| `langchain-messages`, `langchain-agents` | read the reason out of the LangChain result, then map it through the table below |
| `openai-messages`, `openai-agents` | do not use the table at all. Derive it. See 5a |

### The mapping table

Used by the Anthropic pair and the LangChain pair only. Compare keys lower-cased.

| Provider word | Becomes | Provider |
|---|---|---|
| `end_turn` | `stop` | Anthropic |
| `stop_sequence` | `stop` | Anthropic |
| `max_tokens` | `length` | Anthropic |
| `tool_use` | `tool_calls` | Anthropic |
| `refusal` | `content_filter` | Anthropic |
| `stop` | `stop` | OpenAI |
| `length` | `length` | OpenAI |
| `tool_calls` | `tool_calls` | OpenAI |
| `content_filter` | `content_filter` | OpenAI |
| `function_call` | `tool_calls` | OpenAI |

Anything not in that table passes through unchanged. Do not coerce it to `stop`. A word that
arrives unmapped is the signal to add a row here.

`pause_turn` is deliberately absent. Anthropic returns it when a long-running server-side tool
suspends a turn that has not finished. No value in the vocabulary means "did not finish", so it
must pass through rather than be flattened into `stop`.

An empty string or `None` produces no attribute at all. Do not write an empty list.

The attribute is always a list, even with one member, because one response may hold several
choices.

The OpenAI rows exist because the LangChain handlers can serve an OpenAI model and do see these
words. The two OpenAI handlers never do. See 5a.

### 5a. The two OpenAI handlers derive it instead

Both OpenAI handlers use the Responses API, which has no `finish_reason` field. No word from the
table above ever reaches them. Neither imports the mapper.

Derive the reason with a closed three-way check, in this precedence order:

1. If any output item is a function call, the reason is `tool_calls`.
2. Otherwise, if the response status is `incomplete`, the reason is `length` when the incomplete detail says the output token limit was hit, and `content_filter` otherwise.
3. Otherwise, if the response status is `completed`, the reason is `stop`.
4. Otherwise, write no attribute.

Step 1 comes first on purpose. A live seven-turn capture put status `completed` on all seven
turns, including the six that stopped to call a tool, so status alone made the attribute
worthless.

There is no passthrough here. An unrecognised status drops the attribute rather than emitting the
raw word. That is the opposite of the rule for the other four handlers, and it is correct.

### 5b. `claude-agents` almost never has one

Measured against Claude Agent SDK 0.3.220, `stop_reason` and `stop_details` are both null on
every assistant message, and only the run-level result message carries a reason. So a
`claude-agents` `chat` span usually carries no finish reason at all.

Guard the write and move on. Do not synthesise a value from the presence of a tool-call content
block. That would put a reason on the span the provider never returned.

LangChain does not normalise the field. Read it from every place providers put it, in this order:
`generation_info["finish_reason"]`, then `response_metadata["finish_reason"]`, then
`response_metadata["stop_reason"]`. Accept both an `LLMResult` (walk `generations`, flattened) and a
single `AIMessage`. Return `None` rather than an empty list when nothing is present, so the caller
leaves the attribute off instead of asserting that the turn finished for no reason.

---

## 6. Errors, and abandoned streams

### A turn or tool that raises

Record the exception on the innermost open span, set its status to ERROR with the exception
message, and end it. Then do the same to the root, but write the run's usage to the root first.

Partial spend survives a failed run. That is the whole point of tracking whether any turn reported
usage.

### The reported flag

The run accumulator tracks whether any turn reported usage at all, separately from whether the
totals are zero.

Only write usage attributes to the root when something was reported. All-zero attributes assert
that the run cost nothing, which a run whose first call died mid-flight cannot claim. An absent
attribute correctly says "unknown".

Count it. Do not derive it by testing the total for zero, or a provider that genuinely reports an
empty bag becomes indistinguishable from a provider that reported nothing.

Note the asymmetry: a `chat` span always writes the complete set of usage attributes, including
zeros, because an absent attribute drops that span from every query that groups on usage. Only the
root withholds them.

### An abandoned stream

A consumer that breaks out of the iteration, or throws inside the loop body, makes the generator
run its cleanup without ever entering the error path.

Wrap each streaming generator in `try/finally`. End every span the run opened through
`end_span_once`, which ends a span at most once and marks `launchdarkly.stream.abandoned` as
`True`.

`except Exception` does not catch this in Python. `GeneratorExit` inherits from `BaseException`.
This is the single most likely thing to get wrong in the port.

### Ending the span is not enough on two handlers

The span is ours. The vendor's generator is not, and abandoning it leaks something the span
lifecycle knows nothing about. Two handlers need more than `try/finally` plus `end_span_once`,
and a port that does only what the paragraph above says will pass every existing test and still
leak.

`claude-agents`. The blocking path already solved this. It holds the `query_fn(...)` generator in
a variable and awaits `aclose()` on it in a `finally`, because a bare `return` inside `async for`
abandons the generator, and asyncio's finalizer then raises `RuntimeError` when the generator is
suspended inside a real await in the vendor SDK. The streaming path does not do this. It iterates
`query_fn(...)` inline with no held reference. Give it the same treatment in the same `finally`
that ends the spans.

There is a test for the blocking path, `test_query_generator_closed_on_early_return`. There is
none for the streaming path. Write one.

`openai-agents`. The streaming path iterates `streamed.stream_events()` with no `finally` at all.
Abandoning it leaves the vendor's run going. Cancel the streamed run during teardown as well as
ending the spans.

The LangChain pair use `astream`. Whether their cleanup needs anything beyond `try/finally` was
not established. Check it while you are in each file, and say what you found rather than assuming
it is fine.

Leave an abandoned span at UNSET status. Do not set ERROR. Stopping early is a normal thing for a
consumer to do, such as rendering enough of a response and moving on. Nothing failed. Marking it
ERROR would also disagree with LaunchDarkly's own metrics, which record neither a success nor an
error for an abandoned stream, so two dashboards would describe the same run differently.

---

## 7. Content capture

Every handler factory takes `capture_content: bool = False`.

The default is off. Conversation content is personal data. A run emits only metadata, meaning
models, token counts, timings and tool names, until a caller asks for more. Turning it on sends the
text of every request and response to whatever collector the SDK points at.

Guard twice, on purpose. The helper returns without writing when capture is off, which makes a
forgotten call site harmless. The call site also checks before building the argument, which avoids
walking a conversation and serialising JSON that would then be thrown away, once per model turn,
inside a loop.

### What gets written when capture is on

Three carriers hold the same content. This is deliberate.

Canonical, from the OpenTelemetry GenAI semantic conventions:

| Key | Value |
|---|---|
| `gen_ai.system_instructions` | JSON, `[{"type": "text", "content": "..."}]`, only when non-empty |
| `gen_ai.input.messages` | JSON list of canonical messages, only when non-empty |
| `gen_ai.output.messages` | JSON list of canonical messages, only when non-empty |
| `gen_ai.tool.definitions` | JSON, `[{"type": "function", "name": ..., "description"?: ..., "parameters"?: ...}]` |

OpenLLMetry, one numbered attribute per field:

| Key | Value |
|---|---|
| `gen_ai.prompt.{i}.role` | role of message `i` |
| `gen_ai.prompt.{i}.content` | flattened text of message `i` |
| `gen_ai.completion.{i}.role` | role of output message `i` |
| `gen_ai.completion.{i}.content` | flattened text of output message `i` |

Legacy span events:

| Event | Event attribute |
|---|---|
| `gen_ai.content.prompt` | `gen_ai.prompt`, joined `role: content` lines |
| `gen_ai.content.completion` | `gen_ai.completion`, joined text |

Write all three. LaunchDarkly's trace view and conversation view read only the OpenLLMetry
carrier today, so canonical attributes alone render an empty transcript. The span events are
redundant and deprecated, but every published version has emitted them, and removing them would
silently break anyone who learned to read them.

The system prompt goes in twice. It gets its own canonical attribute, and it also goes in as
message 0 of the OpenLLMetry carrier with role `system`, because that shape has no separate slot
for it and dropping it there hides the system prompt from the only view that renders today.

An empty message list writes nothing to the canonical attributes and nothing to the OpenLLMetry
attributes. A reader treats a missing key and an empty list the same.

The two legacy events are asymmetric, and this is what the TypeScript source does rather than
something to tidy up. The output side skips the `gen_ai.content.completion` event entirely when
there are no messages. The input side adds the `gen_ai.content.prompt` event whenever capture is
on, even with no messages and no system prompt, giving it an empty string value. Match that. No
test pins it either way, and every handler always sends at least one message, so the case is
narrow, but implementing the two symmetrically diverges from the source.

### Canonical message shape

```
{
  "role": "system" | "user" | "assistant" | "tool" | <any string>,
  "parts": [ <part>, ... ],
  "finish_reason": <string>        # output messages only, omitted when absent
}
```

A part is one of:

```
{"type": "text",               "content": <string>}
{"type": "reasoning",          "content": <string>}
{"type": "tool_call",          "id"?: <string>, "name": <string>, "arguments"?: <any>}
{"type": "tool_call_response", "id"?: <string>, "result"?: <any>}
```

Keys are snake_case in the JSON. Omit absent members rather than writing null.

`arguments` on a `tool_call` part holds the object the provider means, not the encoding it chose.
Anthropic and LangChain hand over an object. OpenAI's Responses API and Agents SDK send a JSON
string, so the call site parses it before building the part. A string that does not parse goes on the
part verbatim rather than raising, because a truncated stream is worth reporting as it arrived and a
raise inside the telemetry path would end a run the provider has already billed.

This rule lives at the call site, never in the shared writer, for the same reason as the cache
folding in section 4. The writer cannot know whether a string is an encoding or a value.

Flattening a part for the OpenLLMetry carrier: text and reasoning contribute their text. A
`tool_call` becomes `{"name": ..., "arguments": ...}` as JSON. A `tool_call_response` contributes its
result, as a string if it is one and as JSON otherwise. Join the parts with a newline and drop
empty ones.

For tool arguments and results on an `execute_tool` span, a string passes through unchanged and
anything else becomes JSON. OpenTelemetry attributes hold only primitives and lists of primitives.

### LangChain message conversion

Narrow structurally, on `_get_type()`, `content`, `tool_calls`. Do not import LangChain in the
client package.

Lift `system` and `developer` messages out into `system_instructions`. Rename LangChain's roles:
`human` becomes `user`, `ai` becomes `assistant`. A `tool` message becomes one part of type
`tool_call_response` carrying `tool_call_id` and the text. Message content is either a string or a
list of typed blocks, in which case keep the blocks whose type is `text`.

---

## 8. Token accounting

This is the part most likely to be wrong in a way no test catches. Read this section twice.

### The rule that differs by provider

Providers describe the same call two different ways.

Anthropic and Bedrock Converse report cache reads and cache writes as buckets beside the input
total, so the input figure counts only the new tokens. A turn that read 19,971 tokens from cache
and wrote 3,580 more reports `input_tokens: 3`, when the model actually processed 23,554. The
caller must add the cache buckets into the input.

OpenAI and LangChain report one input figure that already contains the cached tokens, with a
subset breakdown alongside. The caller must pass the input through untouched, and surface the
cache figures for parity only.

The fold therefore lives at the call site, never inside the shared writer. Centralising it would
silently double-count for two providers out of three.

| Handler | Input | Cache read from | Cache creation |
|---|---|---|---|
| `claude-messages` | add cache buckets into input | `cache_read_input_tokens` | `cache_creation_input_tokens` |
| `claude-agents` | add cache buckets into input | `cache_read_input_tokens` | `cache_creation_input_tokens` |
| `openai-messages` | pass through | `input_tokens_details.cached_tokens` | always 0, OpenAI has no such concept |
| `openai-agents` | pass through | `input_tokens_details.cached_tokens` | always 0 |
| `langchain-messages` | pass through | `usage_metadata.input_token_details.cache_read` | `usage_metadata.input_token_details.cache_creation` |
| `langchain-agents` | pass through | `usage_metadata.input_token_details.cache_read` | `usage_metadata.input_token_details.cache_creation` |

### The span usage type

A `SpanUsage` carries four numbers: `input`, `output`, `cache_read`, `cache_creation`. Its `input`
is always cache-inclusive. The type existing at all means the folding is already done.

The two Anthropic handlers keep a second, separate accumulator in Anthropic's own field names with
the cache buckets unfolded, because that accumulator is also the handler's return value, and
`parse_usage` folds exactly once. Handing back a `SpanUsage` there would count the cache twice.
Name it so the two cannot be confused.

### Writing usage to a span

Always write all seven attributes, every time, including zeros:

```
gen_ai.usage.input_tokens
gen_ai.usage.output_tokens
gen_ai.usage.total_tokens
gen_ai.usage.cache_read.input_tokens
gen_ai.usage.cache_creation.input_tokens
gen_ai.usage.prompt_tokens          # alias of input_tokens
gen_ai.usage.completion_tokens      # alias of output_tokens
```

Consumers group and aggregate on the complete set. An absent attribute drops the span from those
queries entirely, which reads as "this call had no cached tokens" when it means "this handler
forgot to say". A provider with no cache-creation concept still reports 0.

`total_tokens` is always `input + output`, derived here. Never trust a provider's own total. A
provider-supplied figure can include tokens that appear in neither input nor output, which makes
the total derivable on five handlers and not on the sixth.

The two OpenLLMetry aliases are written here, next to the canonical numbers, and nowhere else.
Computed at a call site from the raw input field, the alias undercounted Anthropic by every cached
token, because on Anthropic that field excludes the cache. One writer, one number.

### Parsing usage for the LaunchDarkly metrics

`parse_usage` serves the `$ld:ai:tokens:*` metrics, not the span attributes. It takes a loosely
typed bag and returns totals.

Try these key pairs in order. The first row whose input key and output key are both present wins.

| Input key | Output key | Cache read keys | Cache creation keys |
|---|---|---|---|
| `input_tokens` | `output_tokens` | `cache_read_input_tokens` | `cache_creation_input_tokens` |
| `inputTokens` | `outputTokens` | `cacheReadInputTokens` | `cacheCreationInputTokens`, `cacheWriteInputTokens` |
| `input` | `output` | none | none |

Sum every accepted spelling of a cache field, so an alias never silently reads as 0. Add the cache
figures on top of the input figure. This fold is provider-blind, which is the contract handlers
must respect: a handler returns either raw usage with the cache fields intact, or an input figure
that already includes cache with the cache fields omitted. Returning a pre-folded input alongside
the cache fields double-counts.

Return `input`, `output`, and `total` as `input + output`. Include an `input_details` member with
`uncached`, `cache_read` and `cache_creation` only when the matched row defines cache keys and at
least one of them is present in the bag.

An unrecognised bag returns all zeros. It does not raise.

### Coercion

Every count that reaches a span or a usage total passes through one coercion helper first. It
returns 0 for anything that is absent, `None`, or not a finite number.

Python currently calls `int(...)` directly, which raises on `None`, and several handlers read
`response.usage.input_tokens` as a bare attribute, which raises when usage is absent. Both must
go.

An emitted 0 is bad. An emitted NaN is worse, because the metric guard tests whether the total is
greater than zero, and that test is false for NaN, so the metric is dropped silently rather than
reported low.

---

## 9. Model identity

Write both provider keys, always.

`gen_ai.system` is the pre-1.37 name and is what these handlers shipped before the span work.
`gen_ai.provider.name` is the current name. Emitting only the new key breaks dashboards written
against the old one. Emitting only the old one leaves the SDK off-spec. Both go out until the next
major version.

The two keys do not always hold the same value. In TypeScript the LangChain handlers put the
literal `langchain` on `gen_ai.system`, because `gen_ai.provider.name` means who served the model
and its enumeration has no `langchain` member.

Python does something different today, and this is a real behaviour change, not a port detail.
The Python LangChain handlers set `gen_ai.system` to the configured provider name, lower-cased,
falling back to the string `langchain` only when no provider is configured. So a Python span for
an Anthropic-backed LangChain config currently reads `anthropic` where TypeScript reads
`langchain`. Change Python to match TypeScript.

And `gen_ai.provider.name` on the two LangChain handlers is not a passthrough of the configured
provider. It is a binary choice: the configured name lower-cased if it equals `anthropic`,
otherwise `openai`. Anything else, including Bedrock, Azure, Cohere, a typo, or an unset value,
reports `openai`.

That looks wrong and is deliberate. It mirrors which model class the handler actually
instantiates, which is `ChatAnthropic` for Anthropic and `ChatOpenAI` for everything else. The
attribute names who served the model, so it has to follow the class, not the config. Do not
implement this as a passthrough.

---

## 10. What not to touch

The graph span is already at parity. `ld.ai.graph` with `ld.ai.graph.key` and `ld.ai.graph.path`
matches between `client/graph.ts` and `graph.py`, and across all three `native_graph` pairs.

`js-ai-sdk` PR #16 renames these three to `launchdarkly.graph`, `launchdarkly.graph.key` and
`launchdarkly.graph.path`. If that PR merges, make the same rename here, in `graph.py` and the
three `native_graph.py`. If it does not, change nothing.

Nothing else in `graph.py` or `native_graph.py` needs work.

---

## 11. Superseded, do not carry forward

`set_openllmetry_prompt` and `set_openllmetry_completion` in `utils.py` are the shape this SDK
emitted before the span work. In TypeScript they still exist and nothing calls them. The content
layer replaced them, and it handles any number of messages rather than only index 0.

Every Python handler calls them today, verified in all six. That is the clearest single statement
of the gap: Python is built on the layer TypeScript retired.

Move every call site onto the content helpers. Keep the two functions exported for one release,
then remove them.

Neither function has a test of its own in the client package today, and neither does
`set_ld_span_attributes`. The only coverage they have is indirect, through the handler tests. Give
their replacements direct tests.

---

## 12. Work this file used to leave implicit

Each item below is real work that the sections above assume and never name. Do not discover these
mid-port.

### New types and changed signatures

`SpanUsage` does not exist anywhere in Python. Add it in the client package, next to the usage
helpers, as a dataclass with the four fields named in section 8.

`UsageDict` is three integers today and cannot carry cache detail. It gains an `input_details`
member, which means `parse_usage`'s return shape changes.

`parse_usage` therefore has callers to check, not just tests to update. Find every one before
changing it. `graph.py` aggregates usage across nodes and reads the result by key.

### Exports

The client package's `__init__.py` needs the new content and usage helpers added. The barrel
package `packages/ai` re-exports the client with a wildcard import, so it does not need editing,
but `packages/ai/tests/test_reexports.py` asserts on what comes through and should be checked.

### Existing tests that must be rewritten, not just added to

The six `test_handler.py` files hold 348 test functions. At least 121 of them assert on a span
name, a `gen_ai.*` attribute, or a token count that this contract changes.

| Package | Test functions | Touching span, attribute or usage | Asserting the old span name |
|---|---|---|---|
| `claude-messages` | 64 | 20 | 2 |
| `claude-agents` | 56 | 20 | 2, plus the generator-lifecycle test |
| `openai-messages` | 62 | 22 | 2 |
| `openai-agents` | 53 | 18 | 2 |
| `langchain-messages` | 59 | 23 | 1 |
| `langchain-agents` | 54 | 18 | 2 |

These counts exclude `test_graph.py`, `test_native_graph.py` and `test_builtins.py`. Those are out
of scope for span work, but graph-level usage aggregation may still read the changed `parse_usage`
shape.

### Expect `claude-agents` to take longest

It is the largest handler, it is the one with a bespoke generator-teardown requirement, it is the
one whose finish reason is usually absent, and it is the only one carrying
`gen_ai.conversation.id` on three span types. Brief it first or give it the most time.

### One thing left open

Whether `langchain-messages` and `langchain-agents` need anything beyond `try/finally` to clean up
`astream`. Nobody has established this. Whoever takes those two packages must answer it in their
report.
