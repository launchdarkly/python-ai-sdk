# Agent Guide — `launchdarkly-ai-typesafe`

This Tier 1 handler sends a judge config to TypeSafe Jev.

## Invariants

- Routing metadata is `("TypeSafe", "messages")`. Judge mode normalizes to `messages`.
- Questions come only from `extract_typesafe_questions`, which reads the variation's `classifiers` list.
- The handler does not ask for `{score, reasoning}`. That block is stripped from `message_history` before it becomes Jev state.
- One `system_one` call returns one result per label. The handler's `output` is `{"kind":"typesafe","results":[...]}`. The client expands that into `judge_results`.
- Auth is `TYPESAFE_API_KEY`. The factory does not take a client or a key.
- No tools and no streaming.
- Telemetry is `invoke_agent` → `chat <model>`. Content is gated by `capture_content`.

## Files

- `questions.py` — classifier extraction, state construction, and answer-to-score mapping.
- `handler.py` — Jev call and spans.
- `__init__.py` — package registration and exports.
