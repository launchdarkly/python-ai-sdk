# Agent Guide — Vercel Messages

This Tier 1 package provides the wildcard `("*", "messages")` handler.

- `handler.py` owns model resolution, `InferenceRequestParams`, native message
  and file-part conversion, native tool declarations, structured output,
  streaming cleanup, usage, and spans.
- `__init__.py` exports the factory/wrapper and registers the package.
- Default models use a mapped Gateway `creator/model` id. Preserve
  slash-qualified ids, map `xAI` to `spacexai`, and split recognized LD dotted
  creator prefixes once before `ai.get_model`.
- Injected model instances and sync/async factories are the Gateway opt-out
  and are scoped to their handler.
- Only callable tool handlers are exposed; JSON Schema is preserved in
  `ToolSpec.params`.
- `ai.stream` must remain inside `async with`, including early consumer exit.
- Do not add a provider-native client or enable duplicate runtime telemetry.

`TESTING.md` §1.x and Appendix A.14 are authoritative.
