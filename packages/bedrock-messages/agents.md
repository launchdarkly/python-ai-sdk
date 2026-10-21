# Agent Guide — `launchdarkly-ai-bedrock-messages`

This Tier 1 package routes `('Bedrock', 'messages')` configs through Bedrock Runtime Converse.
`handler.py` owns prompt, history, tool, client, and streaming translation; `spans.py` owns the
provider-neutral `invoke_agent` / `chat {model}` / `execute_tool {name}` span tree.

Important invariants:

- `model.region` is an inference-profile prefix, never an AWS endpoint region.
- `converse_options(config)` runs for every provider turn, but generated model, messages, system,
  and tool-result fields remain authoritative.
- `model.custom` is never consumed.
- Injected async and sync clients are caller-owned. Sync methods run with `asyncio.to_thread`.
- Handler-created aioboto3 clients remain inside their async context for the full invocation.
- Bedrock cache read/write tokens are folded into telemetry input usage while retained as detail.
