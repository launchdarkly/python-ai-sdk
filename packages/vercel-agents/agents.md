# Agent Guide — Vercel Agents

This Tier 1 package provides the wildcard `("*", "agent")` handler.

- Default models use a mapped Gateway `creator/model` id. Preserve
  slash-qualified ids, map `xAI` to `spacexai`, and split recognized LD dotted
  creator prefixes once. Injected models are the only Gateway opt-out.
- `handler.py` constructs the native `ai.Agent` and maps models, messages,
  `AgentTool`/`Tool`/`ToolSpec`, request parameters, streams, usage, and spans.
- `graph.py` pre-wires exactly one Vercel handler into the core graph API.
- `native_graph.py` creates one native agent per node. Handoff tools only select
  a target; execution follows the selected target after the current run ends.
- Root caller history is never replayed into child nodes.
- Native graph spans end in `finally`, including `CancelledError`.
- Runtime calls are always async context-managed and never enable duplicate
  runtime telemetry.

`TESTING.md` §2.x and Appendix A.14 are authoritative.
