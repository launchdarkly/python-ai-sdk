# LaunchDarkly AI Bedrock Messages

Amazon Bedrock Runtime Converse and ConverseStream integration for the LaunchDarkly AI SDK.

```python
from launchdarkly_ai_bedrock_messages import create_bedrock_messages_handler

handler = create_bedrock_messages_handler(region="us-east-1")
```

`model.region` is the Bedrock inference-profile prefix (`us`, `eu`, or `global`), while the
factory `region` selects the AWS endpoint. Authentication uses `api_key`,
`AWS_BEARER_TOKEN_BEDROCK`, or the normal AWS credential chain. A supplied aioboto3 or boto3
client is used as-is and is never closed by the handler.
