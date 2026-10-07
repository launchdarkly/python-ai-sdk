# `launchdarkly-ai-typesafe`

Judge handler for TypeSafe Jev. A judge AI Config whose provider is `TypeSafe` is routed here. Judge mode already normalizes to `messages`, so the handler registers as `("TypeSafe", "messages")`.

Jev does not return the `{score, reasoning}` JSON other judges use. The questions are the variation's `classifiers` list, sent as Jev `noul`, `choice`, and `score` questions. `extract_typesafe_questions` is the only reader of that list.

Each label becomes its own `judge_results` entry, keyed `judge_key.label_key`, and carries that label's `event_key`. The entry's `response` is the selected label value. Every entry carries the full token usage of the one Jev call. LaunchDarkly token telemetry is recorded once for that call. Each label is tracked on its own `eventKey`.

The client reads `TYPESAFE_API_KEY` from the environment.

```python
from launchdarkly_ai_typesafe import create_typesafe_handler
from launchdarkly_ai_server import config

result = await config(
    key="my-config",
    handler=create_typesafe_handler(),
).invoke(user_input, context)
```
