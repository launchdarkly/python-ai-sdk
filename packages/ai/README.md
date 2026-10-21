# `launchdarkly-ai-python`

Convenience barrel package for the LaunchDarkly AI Python SDK. Re-exports the complete public API of [`launchdarkly-ai-server`](https://github.com/launchdarkly/python-ai-sdk/blob/main/packages/client/README.md) — install this instead of `launchdarkly-ai-server` for the simplest setup.

## Installation

```bash
pip install launchdarkly-ai-python launchdarkly-ai-openai-messages
```

To enable trace export to the LaunchDarkly Observability dashboard, install the `otel` extras group:

```bash
pip install "launchdarkly-ai-python[otel]"
```

`init_client()` detects the OTel packages at runtime and configures tracing automatically. If the extras are not installed, a single warning is logged and all AI calls continue normally with no-op spans.

## Usage

Import everything from `launchdarkly_ai_python` instead of `launchdarkly_ai_server`:

```python
import asyncio
from launchdarkly_ai_python import config, graph, resolve_graph
from launchdarkly_ai_python import init_client, shutdown, global_registry
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

Reads an AI Config flag variation **without invoking any AI provider**. Re-exported from `launchdarkly-ai-server` — see the [full reference there](https://github.com/launchdarkly/python-ai-sdk/blob/main/packages/client/README.md#inspect_configkey-context).

```python
from launchdarkly_ai_python import inspect_config

result = await inspect_config("my-ai-config-flag", {"kind": "user", "key": "user-123"})
if result["enabled"]:
    print(result["config"]["model"]["name"])
```

Never raises. Returns `{"enabled": bool, "config": dict | None, "meta": dict | None}`.

## Experimental features

Experimental features are not re-exported from this package. They live in
`launchdarkly-ai-server`, the package this one depends on, and need a `launchdarkly-ai-server`
release that includes them. Import them from `launchdarkly_ai_server.experimental`; if that
import fails, upgrade `launchdarkly-ai-server` itself, since upgrading this package alone can
leave an older `launchdarkly-ai-server` in place. The experimental features are:

- **Agent Skills**, in `launchdarkly_ai_server.experimental.skills`; see the
  [Agent Skills guide](https://github.com/launchdarkly/python-ai-sdk/blob/main/packages/client/README.md#agent-skills-experimental).
- **Evaluations from Code**, in `launchdarkly_ai_server.experimental.evaluations`; see the
  [evaluations guide](https://github.com/launchdarkly/python-ai-sdk/blob/main/packages/client/README.md#run-an-evaluation-from-code-experimental).

---

All exports, types, and behaviors are identical to `launchdarkly-ai-server`. See the [core client README](https://github.com/launchdarkly/python-ai-sdk/blob/main/packages/client/README.md) for the full API reference.
