# Agent Guide — `launchdarkly-ai-python` (Convenience Wrapper)

This document describes the role, structure, and constraints of the `launchdarkly-ai-python` package for AI agents and contributors.

---

## Role and Tier

**Tier 0 — Convenience wrapper.**

This package is a pure re-export barrel that re-exports the entire public surface of `launchdarkly-ai-server` and carries `launchdarkly-server-sdk` as a hard (non-peer) dependency. No new logic lives here.

Its purpose: Python application developers install this single package and get both the LaunchDarkly AI SDK and the Python server SDK in one step, without managing `launchdarkly-server-sdk` as a peer dependency themselves.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_python/__init__.py` | Single `from launchdarkly_ai_server import *` — the entire public barrel |

---

## Public Exports

This package re-exports everything from `launchdarkly-ai-server` and nothing else:

```python
from launchdarkly_ai_server import *
```

Every symbol available from `launchdarkly-ai-server` is available from `launchdarkly-ai-python` under the same name. No additional symbols are added. When `launchdarkly-ai-server` gains a new export, this package automatically picks it up.

---

## Dependencies

| Dependency | Why |
|---|---|
| `launchdarkly-ai-server` | The package being re-exported |
| `launchdarkly-server-sdk` | Carried as a hard dep so consumers don't need to install it manually; auto-discovered by `init_client()` via dynamic import |

---

## OTel Setup

This package itself emits no spans. OTel is initialized and configured by `launchdarkly-ai-server` (re-exported through this package) during `init_client()`.

Install the OTel packages alongside this package:
```sh
pip install "launchdarkly-ai-python[otel]"
# or:
pip install launchdarkly-ai-python opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Once the OTel packages are installed, spans from all handler packages are automatically collected when `init_client()` runs.

For OTLP endpoint configuration see the [`launchdarkly-ai-server` agents.md](../client/agents.md#otel-setup).

---

## `init_client()` — When to Call It

`init_client()` is re-exported from `launchdarkly-ai-server`. Full details in the [`launchdarkly-ai-server` agents.md](../client/agents.md#lifecycle-invariants).

**Short answer for standard Python apps:**

- **You don't need to call it** — lazy init runs automatically on the first `config().invoke()` call as long as `LD_SDK_KEY` is set.
- **Call it explicitly** when you need custom `serviceName`/`environment`, a custom OTLP endpoint, or want to pre-warm the connection before the first user request:
  ```python
  from launchdarkly_ai_python import init_client
  await init_client({"serviceName": "my-service", "environment": "production"})
  ```
- **For BYOC / custom runtimes**, use `launchdarkly-ai-server` directly and pass a pre-initialized client: `await init_client(my_custom_client)`. Do not use this package for custom runtimes — it carries `launchdarkly-server-sdk` as a hard dependency which may conflict.

---

## Common Pitfalls

### 1. Do not add logic to this package

This package must remain a pure re-export barrel. Any new utility, type, or helper belongs in `launchdarkly-ai-server`, not here. Adding logic here creates a maintenance burden and violates the single-responsibility principle.

### 2. Do not import from both packages in the same application

Importing from both `launchdarkly-ai-python` and `launchdarkly-ai-server` in the same app can produce subtle issues if the dependency graph deduplication fails. Pick one: use `launchdarkly-ai-python` for standard Python apps, `launchdarkly-ai-server` for custom runtimes where you manage the SDK client yourself. The `get_client()` singleton is process-wide and shared regardless of which package path you import through.
