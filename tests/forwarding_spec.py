"""The cross-SDK forwarded-key lists (TESTING.md section 1.12), shared by every package's tests.

Each list names the ``model.parameters`` keys a handler forwards, in the config's own spelling, so
a key the handler renames (``effort``, ``max_tokens``, ``text``) is listed under the name the config
uses. The JS SDK is held to the same lists. Each package's tests probe its real call sites with
:func:`probe_forwarded_keys` and assert the result equals its list here exactly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from tests.never_forwarded import NEVER_FORWARDED_KEYS, leaked

CLAUDE_MESSAGES = frozenset(
    {
        "cache_control",
        "effort",
        "max_tokens",
        "output_config",
        "service_tier",
        "stop_sequences",
        "temperature",
        "thinking",
        "tool_choice",
        "top_k",
        "top_p",
    }
)

OPENAI_MESSAGES = frozenset(
    {
        "max_output_tokens",
        "max_completion_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "prompt_cache_key",
        "reasoning",
        "service_tier",
        "temperature",
        "tool_choice",
        "top_logprobs",
        "top_p",
    }
)

CLAUDE_AGENTS = frozenset(
    {
        "betas",
        "effort",
        "fallback_model",
        "max_budget_usd",
        "max_thinking_tokens",
        "max_turns",
        "output_format",
        "thinking",
    }
)

OPENAI_AGENTS = frozenset(
    {
        "frequency_penalty",
        "max_tokens",
        "max_turns",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning",
        "temperature",
        "text",
        "tool_choice",
        "top_p",
        "verbosity",
    }
)

LANGCHAIN_CHAT_OPENAI = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "presence_penalty",
        "prompt_cache_key",
        "service_tier",
        "stop",
        "stop_sequences",
        "temperature",
        "top_logprobs",
        "top_p",
        "verbosity",
    }
)

#: Canonical keys ``ChatOpenAI`` (langchain-openai 1.3.3) has no field for, so the handler cannot
#: set them and leaves them out.
LANGCHAIN_CHAT_OPENAI_UNSUPPORTED = frozenset({"prompt_cache_key"})

LANGCHAIN_CHAT_ANTHROPIC = frozenset(
    {
        "betas",
        "effort",
        "max_tokens",
        "max_tokens_to_sample",
        "output_config",
        "stop",
        "stop_sequences",
        "temperature",
        "thinking",
        "top_k",
        "top_p",
    }
)

LANGCHAIN_CHAT_BEDROCK_CONVERSE = frozenset(
    {"max_tokens", "performance_config", "service_tier", "temperature", "top_p"}
)

#: Keys the UI or another provider uses that are on none of the lists above for some handler, so
#: every probe also checks they are dropped.
_OFF_LIST_KEYS = frozenset(
    {
        "context_management",
        "frequency_penalty",
        "guard_last_turn_only",
        "guardrails",
        "include",
        "include_usage",
        "inference_geo",
        "instructions",
        "logit_bias",
        "logprobs",
        "max_output_tokens",
        "max_tokens",
        "metadata",
        "moderation",
        "n",
        "output_format",
        "prompt_cache_retention",
        "reasoning",
        "reasoning_effort",
        "request_metadata",
        "response_format",
        "response_include",
        "safety_identifier",
        "seed",
        "stop",
        "store",
        "stream",
        "system",
        "text",
        "top_logprobs",
        "truncation",
        "use_responses_api",
        "user",
        "user_profile_id",
        "verbose",
        "verbosity",
    }
)

#: A valid value for every canonical key, different from any default a handler applies, so
#: setting it alone always changes the provider call when the key is forwarded.
SAMPLE_VALUES: Mapping[str, Any] = {
    "betas": ["context-1m-2025-08-07"],
    "cache_control": {"type": "ephemeral"},
    "effort": "low",
    "fallback_model": "claude-fallback",
    "frequency_penalty": 0.3,
    "logit_bias": {"50256": -100},
    "logprobs": True,
    "max_budget_usd": 1.5,
    "max_completion_tokens": 222,
    "max_output_tokens": 333,
    "max_thinking_tokens": 444,
    "max_tokens": 111,
    "max_tokens_to_sample": 555,
    "max_turns": 4,
    "n": 2,
    "output_config": {"effort": "high"},
    "output_format": {"type": "json_schema", "schema": {"type": "object"}},
    "parallel_tool_calls": False,
    "performance_config": {"latency": "optimized"},
    "presence_penalty": 0.4,
    "prompt_cache_key": "cache-1",
    "reasoning": {"effort": "low"},
    "service_tier": "flex",
    "stop": ["STOP"],
    "stop_sequences": ["END"],
    "temperature": 0.2,
    "text": {"verbosity": "low"},
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "tool_choice": {"type": "auto"},
    "top_k": 7,
    "top_logprobs": 2,
    "top_p": 0.6,
    "verbosity": "high",
}


def candidate_keys(*accepted: Iterable[str]) -> frozenset[str]:
    """Every key worth probing: the provider's own accept-set(s), every canonical key, every
    never-forwarded key, and the off-list keys."""
    keys: set[str] = set(_OFF_LIST_KEYS | NEVER_FORWARDED_KEYS)
    for group in accepted:
        keys.update(group)
    return frozenset(keys)


def sample(key: str) -> Any:
    """The probe value for *key*: a valid value for a canonical key, a :func:`leaked` marker for
    any other."""
    return SAMPLE_VALUES[key] if key in SAMPLE_VALUES else leaked(key)


async def probe_forwarded_keys(
    candidates: Iterable[str],
    call: Callable[[dict[str, Any]], Awaitable[object]],
) -> frozenset[str]:
    """The keys among *candidates* that change what *call* hands its provider when set alone.

    *call* runs the handler with the given ``model.parameters`` and returns whatever it handed the
    provider. A key counts as forwarded when that result differs from the result with no
    parameters, or when setting it makes the call raise.
    """
    baseline = await call({})
    forwarded: set[str] = set()
    for key in sorted(set(candidates)):
        try:
            result = await call({key: sample(key)})
        except Exception:
            forwarded.add(key)
            continue
        if result != baseline:
            forwarded.add(key)
    return frozenset(forwarded)
