# LaunchDarkly AI Bedrock Agents

Amazon Bedrock integration using Strands `Agent` and `BedrockModel`.

```python
from launchdarkly_ai_bedrock_agents import create_bedrock_agents_handler

handler = create_bedrock_agents_handler(region="us-east-1")
```

The factory also accepts a boto3 `client` or `boto_session`. Injected clients remain owned by the
caller. `bedrock_graph()` pre-wires the handler into LaunchDarkly's generic graph runner.
