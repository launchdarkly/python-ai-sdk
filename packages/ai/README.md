# `launchdarkly-ai`

Convenience barrel package for the LaunchDarkly AI Python SDK. Re-exports the complete public API of [`launchdarkly-ai-server`](../client/README.md) — install this instead of `launchdarkly-ai-server` for the simplest setup.

## Installation

```bash
pip install launchdarkly-ai launchdarkly-ai-openai-messages
```

To enable trace export to the LaunchDarkly Observability dashboard, install the `otel` extras group:

```bash
pip install "launchdarkly-ai[otel]"
```

`init_client()` detects the OTel packages at runtime and configures tracing automatically. If the extras are not installed, a single warning is logged and all AI calls continue normally with no-op spans.

## Usage

Import everything from `launchdarkly_ai` instead of `launchdarkly_ai_server`:

```python
import asyncio
from launchdarkly_ai import config, graph, resolve_graph
from launchdarkly_ai import init_client, shutdown, global_registry
from launchdarkly_ai_openai_messages import create_openai_messages_handler

async def main():
    result = await config(
        key="my-ai-config-flag",
        handler=create_openai_messages_handler(),
    ).invoke("What is feature flagging?", {"kind": "user", "key": "user-123"})

    print(result.response)
    await shutdown()

asyncio.run(main())
```

## `inspect_config(key, context)`

Reads an AI Config flag variation **without invoking any AI provider**. Re-exported from `launchdarkly-ai-server` — see the [full reference there](../client/README.md#inspect_configkey-context).

```python
from launchdarkly_ai import inspect_config

result = await inspect_config("my-ai-config-flag", {"kind": "user", "key": "user-123"})
if result["enabled"]:
    print(result["config"]["model"]["name"])
```

Never raises. Returns `{"enabled": bool, "config": dict | None, "meta": dict | None}`.

---

All exports, types, and behaviors are identical to `launchdarkly-ai-server`. See the [core client README](../client/README.md) for the full API reference.
