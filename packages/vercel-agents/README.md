# `launchdarkly-ai-vercel-agents`

Wildcard LaunchDarkly AI agent handler built on the official Vercel `ai.Agent`
runtime for Python.

## Install

```bash
pip install launchdarkly-ai-server launchdarkly-server-sdk launchdarkly-ai-vercel-agents
```

Default routing is AI Gateway via a mapped `creator/model` id (for example,
`xAI` + `grok-4.5` becomes `spacexai/grok-4.5`). Already-qualified ids remain
unchanged. Set `AI_GATEWAY_API_KEY` or Vercel OIDC. This package does not load a provider SDK from
`config.provider.name`. Inject a model or model factory to call a provider
directly.

All LaunchDarkly providers are handled explicitly. Anthropic, OpenAI, Azure,
Gemini, Cohere, DeepSeek, Meta, Mistral, Perplexity, and Vertex map directly.
Bedrock, Cortex, Cursor, Databricks, and Fireworks AI infer the creator from the
model and fail locally if ambiguous. AI21 Labs and IBM Watson fail locally
because Vercel's current catalog has no corresponding creator.

## Use

```python
from launchdarkly_ai_server import config
from launchdarkly_ai_vercel_agents import create_vercel_agents_handler

result = await config(
    key="my-agent-config",
    handler=create_vercel_agents_handler(),
).invoke("Help me", {"kind": "user", "key": "user-123"})
```

The package also exports `vercel_graph()` and `to_vercel_agents()` for
LaunchDarkly and framework-native graph execution. The native adapter builds one
`ai.Agent` per graph node and follows only a `transfer_to_<target>` tool that the
current node actually selects.

This package advertises `("*", "agent")`; register only one wildcard agent
adapter in a handler pool.
