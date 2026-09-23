# Agent Guide — `launchdarkly-ai-google-adk-agents`

Wildcard agent handler (`provides_for = ("*", "agent")`) for Google ADK. An exact provider handler still wins when both are registered.

Gemini Developer API is the default. Vertex is opt-in (`use_vertexai=True`) and requires project and location at factory time. Do not send an API key in Vertex mode, and do not fall back from one transport to the other.

Python non-Gemini models go through ADK's `google.adk.models.lite_llm.LiteLlm` as `provider/model`. That is ADK's adapter, not `launchdarkly-ai-litellm`. TypeScript `@google/adk` 2.1 has no equivalent; inject `model` or the handler throws.

`google-adk` is imported lazily and is not a workspace dependency, because 2.9.2 caps `opentelemetry-sdk`. Install it in the app environment (`google-adk[extensions]` when you need LiteLLM).

## File map

| File | Responsibility |
|---|---|
| `handler.py` | Factory, session seeding, `LaunchDarklyTelemetryPlugin`, run and stream |
| `spans.py` | `invoke_agent` / `chat {model}` / `execute_tool {name}` |
| `graph.py` | `google_adk_graph()` |
| `native_graph.py` | `to_adk_agents()` |

Span calls go through the `spans` module attribute so tests can patch them. Tool spans are abandoned from a `finally` (`abandon_open_spans` does not call `fail_span`). `gen_ai.system` is `google_adk`. `gen_ai.provider.name` is the serving provider; Google, Gemini, and Vertex normalize to `gcp.gemini`.
