# `launchdarkly-ai-vercel-messages`

Wildcard LaunchDarkly AI messages handler built on the official Vercel AI SDK
for Python. It builds Vercel's `creator/model` id from the evaluated provider
and model, while preserving already-qualified Gateway ids.

## Install

```bash
pip install launchdarkly-ai-server launchdarkly-server-sdk launchdarkly-ai-vercel-messages
```

Default routing is AI Gateway: for example, `xAI` + `grok-4.5` becomes
`spacexai/grok-4.5` before calling `ai.get_model`. Set `AI_GATEWAY_API_KEY`, or use Vercel OIDC with the
`ai[vercel]` extra. Provider keys such as `OPENAI_API_KEY` are ignored on that
path. This package does not import `langchain-openai` or other provider SDKs
from `config.provider.name`. To call a model directly, inject `model=` with an
already configured model or a sync/async factory:

```python
import ai
from launchdarkly_ai_vercel_messages import vercel_messages

result = await vercel_messages(
    key,
    user_input,
    context,
    model=lambda cfg: ai.Model(
        id=cfg["model"]["name"],
        provider=ai.get_provider("openai"),
    ),
)
```

All LaunchDarkly providers are handled explicitly. Anthropic, OpenAI, Azure,
Gemini, Cohere, DeepSeek, Meta, Mistral, Perplexity, and Vertex map directly.
Bedrock, Cortex, Cursor, Databricks, and Fireworks AI infer the creator from the
model and fail locally if ambiguous. AI21 Labs and IBM Watson fail locally
because Vercel's current catalog has no corresponding creator.

`experimental_evaluate` / `vercelEvaluate` is TypeScript-only until the official
Python `ai` package ships the same API.

## Use

```python
from launchdarkly_ai_server import config
from launchdarkly_ai_vercel_messages import create_vercel_messages_handler

result = await config(
    key="my-ai-config",
    handler=create_vercel_messages_handler(),
).invoke("Hello", {"kind": "user", "key": "user-123"})
```

The factory accepts `model=<model or sync/async factory>` and
`capture_content=False`. It advertises `("*", "messages")`; register only one
wildcard messages adapter (Vercel, LangChain, etc.) in a handler pool.

The adapter maps native text and multimodal history, callable tools, streaming,
structured JSON output, usage, and LaunchDarkly telemetry. Flag parameters
cannot replace handler-owned model, credentials, messages, tools, stream
lifecycle, output contract, or loop controls.
