# `launchdarkly-ai-litellm-messages`

Provider-agnostic LaunchDarkly AI Config handler backed by LiteLLM's in-process
`acompletion` API.

## Installation

```bash
pip install launchdarkly-ai-server launchdarkly-ai-litellm-messages
```

Python uses LiteLLM in-process; no proxy endpoint is required. Set the provider
credentials required by the models in your AI Config, such as
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Standard LiteLLM configuration,
gateways, and model aliases are supported.

## Usage

```python
from launchdarkly_ai_litellm_messages import create_litellm_messages_handler
from launchdarkly_ai_server import config

result = await config(
    key="assistant",
    handler=create_litellm_messages_handler(),
).invoke("Hello", {"kind": "user", "key": "user-123"})
```

The evaluated AI Config model name is passed directly to LiteLLM, so standard
LiteLLM model prefixes, gateways, and aliases are supported. Tool calls,
structured output, multimodal history, and streaming are supported without
constructing a provider-native client.

Because this is a wildcard messages handler, do not register it together with
another `["*", "messages"]` handler such as LangChain in the same registry.
