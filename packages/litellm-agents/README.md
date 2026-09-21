# `launchdarkly-ai-litellm-agents`

Provider-agnostic LaunchDarkly AI Config and graph handlers built on the OpenAI
Agents SDK's `LitellmModel`.

## Installation

```bash
pip install launchdarkly-ai-server launchdarkly-ai-litellm-agents
```

Python uses LiteLLM in-process; no proxy endpoint is required. Set the provider
credentials required by the models in your AI Config, such as
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Standard LiteLLM configuration,
gateways, and model aliases are supported.

## Usage

```python
from launchdarkly_ai_litellm_agents import create_litellm_agents_handler
from launchdarkly_ai_server import config

result = await config(
    key="assistant",
    handler=create_litellm_agents_handler(),
).invoke("Hello", {"kind": "user", "key": "user-123"})
```

Each evaluated config model is authoritative and creates its own `LitellmModel`.
The package supports callable tools, structured output, streaming, the
`litellm_graph` convenience wrapper, and `to_litellm_agents` for native Agents
SDK handoffs.

Because this is a wildcard agent handler, do not register it together with
another `["*", "agent"]` handler such as LangChain in the same registry.
