# `launchdarkly-ai-python`

Convenience barrel package for the LaunchDarkly AI Python SDK. Re-exports the complete public API of [`launchdarkly-ai-server`](../client/README.md) — install this instead of `launchdarkly-ai-server` for the simplest setup.

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

Reads an AI Config flag variation **without invoking any AI provider**. Re-exported from `launchdarkly-ai-server` — see the [full reference there](../client/README.md#inspect_configkey-context).

```python
from launchdarkly_ai_python import inspect_config

result = await inspect_config("my-ai-config-flag", {"kind": "user", "key": "user-123"})
if result["enabled"]:
    print(result["config"]["model"]["name"])
```

Never raises. Returns `{"enabled": bool, "config": dict | None, "meta": dict | None}`.

## Evaluations from code

`init_evaluations` and the evaluations result types are also re-exported:

```python
from launchdarkly_ai_python import init_evaluations

evals = init_evaluations()
result = await evals.run(
    project_key="my-project",
    key="unique-evaluation-key",
    dataset="golden-dataset",
    handler=my_handler,
    generation={"provider": "OpenAI", "model": "gpt-4o"},
)
```

`LD_API_TOKEN` is required. Configure `LD_SDK_KEY` — or initialize your own client with `init_client(client=...)` — to emit one `$ld:ai:offline-evals:generation` event per generated row through the standard SDK event transport. Use `LD_API_BASE_URI` for staging or local management API traffic; it is separate from the SDK delivery setting `LD_BASE_URI`. Evaluation-run links use the explicit `ui_base_uri` option or `LD_UI_BASE_URI` (for example, `https://ld-stg.launchdarkly.com` in staging), defaulting to `https://app.launchdarkly.com`. See the [core evaluations guide](../client/README.md#run-an-evaluation-from-code).

---

All exports, types, and behaviors are identical to `launchdarkly-ai-server`. See the [core client README](../client/README.md) for the full API reference.
