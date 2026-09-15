# Agent Guide — `launchdarkly-ai-bedrock-agents`

This Tier 1 package routes `('Bedrock', 'agent')` configs through Strands `Agent` and
`BedrockModel`. It deliberately does not expose AgentCore or a Strands-native graph adapter.

Important invariants:

- `model.region` is an idempotent inference-profile prefix; factory `region` selects the endpoint.
- `model_options(config)` and documented model parameters configure `BedrockModel`;
  `model.custom` is never consumed and generated `model_id` remains authoritative.
- A raw boto3 client is assigned to `model.client`; otherwise `boto_session` is forwarded.
- History uses Strands' structured message input rather than being flattened into the system prompt.
- `bedrock_graph()` pre-wires one Bedrock agent handler into the generic graph runner.
- Package telemetry uses the shared three-level span tree and folds Bedrock cache usage.
