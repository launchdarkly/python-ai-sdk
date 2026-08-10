"""
Entrypoint for the Python SDK examples.

Usage:
    python main.py [example] [flag-key] [user-input]

Examples:
    python main.py agent                    launch-darkly-documentation-summarizer "What is the LaunchDarkly AI SDK?"
    python main.py streaming                launch-darkly-documentation-summarizer "Summarise feature flags in 3 bullets"
    python main.py judge                    launch-darkly-documentation-summarizer "What is the LaunchDarkly AI SDK?"
    python main.py graph                    my-agent-graph "What is the LaunchDarkly AI SDK?"
    python main.py graph-history            my-agent-graph ""
    python main.py openai-only              my-openai-flag "Tell me about feature flags"
    python main.py langchain                my-langchain-flag "Tell me about feature flags"
    python main.py claude-agents            launch-darkly-documentation-summarizer "What is the LaunchDarkly AI SDK?"
    python main.py openai-agents            launch-darkly-documentation-summarizer-open-ai-only "What is the LaunchDarkly AI SDK?"
    python main.py langchain-agents         launch-darkly-documentation-summarizer "What is the LaunchDarkly AI SDK?"
    python main.py native-graph             travel-agent-flow "Book me a flight to Paris"
    python main.py native-graph-langchain   travel-agent-flow "Book me a flight to Paris"
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from launchdarkly_ai_server import init_client, shutdown  # noqa: E402

# ---------------------------------------------------------------------------
# Registry of examples
# ---------------------------------------------------------------------------
# Imported lazily so that register.py side-effects only fire for the chosen
# example and the import-time cost is minimal.

ExampleFn = Callable[[str, str], Coroutine[Any, Any, None]]

EXAMPLES: dict[str, str] = {
    "agent": "examples.agent",
    "streaming": "examples.streaming",
    "graph": "examples.graph_example",
    "graph-history": "examples.graph_history",
    "history": "examples.history",
    "judge": "examples.judge_example",
    "claude-agents": "examples.claude_agents_example",
    "claude-messages": "examples.claude_messages_example",
    "openai-agents": "examples.openai_agents_example",
    "openai-messages": "examples.openai_messages_example",
    "openai-only": "examples.openai_only",
    "langchain": "examples.langchain_example",
    "langchain-agents": "examples.langchain_agents_example",
    "langchain-messages": "examples.langchain_messages_example",
    "native-graph": "examples.native_graph",
    "native-graph-langchain": "examples.native_graph_langchain",
}

DEFAULT_EXAMPLE = "agent"
DEFAULT_KEY = "launch-darkly-documentation-summarizer"
DEFAULT_USER_INPUT = "What is the LaunchDarkly AI SDK?"


def _parse_args() -> tuple[str, str, str]:
    argv = sys.argv[1:]
    example = argv[0].strip() if len(argv) > 0 else DEFAULT_EXAMPLE
    if example not in EXAMPLES:
        print(f"Unknown example '{example}'. Available: {', '.join(EXAMPLES)}")
        print(f"Falling back to '{DEFAULT_EXAMPLE}'.")
        example = DEFAULT_EXAMPLE
    key = argv[1].strip() if len(argv) > 1 else DEFAULT_KEY
    user_input = argv[2].strip() if len(argv) > 2 else DEFAULT_USER_INPUT
    return example, key, user_input


async def main() -> None:
    example_name, key, user_input = _parse_args()

    # Initialise the LaunchDarkly client (reads LD_SDK_KEY from env / .env)
    await init_client()

    print(f"Running example: {example_name!r}")
    print(f"  flag key   : {key}")
    print(f"  user input : {user_input}\n")

    import importlib

    mod = importlib.import_module(EXAMPLES[example_name])
    await mod.run(key, user_input)

    await shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        sys.stdout.flush()
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
