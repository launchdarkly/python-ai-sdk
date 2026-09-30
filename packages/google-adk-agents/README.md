# Google ADK agents

`launchdarkly-ai-google-adk-agents` is a wildcard agent handler for [Google ADK](https://google.github.io/adk-docs/). It serves any `agent` variation that does not have a more specific provider handler registered.

```python
from launchdarkly_ai_google_adk_agents import google_adk_agents

response = await google_adk_agents(key, user_input, context, tool_handlers=tools)
```

Gemini Developer API is the default. Set `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_GENAI_API_KEY`. Vertex is explicit: `create_google_adk_agents_handler(use_vertexai=True, project=..., location=...)`. A Gemini credential is never sent on the Vertex path, and a failed call is not retried on the other transport.

Non-Gemini models use ADK's `LiteLlm` adapter (`pip install 'google-adk[extensions]'`). The model id is `{provider}/{model}` unless the configured name already contains a slash. Pass `model=` to supply your own ADK model and skip both constructors.

`google-adk` is not declared on this package. Release 2.9.2 caps `opentelemetry-sdk<=1.42.1`, which conflicts with this workspace's exporter. Install ADK in the application environment.

`to_adk_agents(graph)` compiles a LaunchDarkly agent graph into an ADK `Workflow`. `google_adk_graph(key)` pre-wires this handler for `graph()`.
