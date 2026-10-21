"""Example: Amazon Bedrock Runtime Converse."""

from __future__ import annotations

import json

from examples.tools import get_preferences
from examples.utils import new_context, write_output
from launchdarkly_ai_bedrock_messages import bedrock_messages


async def run(key: str, user_input: str) -> None:
    response = await bedrock_messages(
        key,
        user_input,
        new_context(),
        tool_handlers={"get-user-preferences": get_preferences},
        variables={"user_input": user_input},
    )
    print(json.dumps(response, indent=2, default=str))
    write_output(response)
