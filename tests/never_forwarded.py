"""The keys no handler may ever forward from ``model.parameters``, shared by every package's tests.

Each handler forwards only an explicit allowlist of model and run settings and drops every other
key. This module is the other half of that rule: the keys that must never be on any allowlist,
whatever the provider SDK accepts. ``test_never_forwarded_parameters.py`` checks every handler's
forwarded lists against it, and each package's own tests check their real call sites with
:data:`NEVER_FORWARDED_BAG`.
"""

from __future__ import annotations

#: Credentials, endpoints and connection settings, raw request injection, remote tools, the Claude
#: Agents host-process settings, and the categories no cross-SDK allowlist (TESTING.md section 1.12)
#: may hold: data retention, server-side state, identity and attribution, runtime wiring and
#: API-shape switches, prompt content beyond the instructions, and safety configuration.
NEVER_FORWARDED_KEYS = frozenset(
    {
        # Credentials.
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "bedrock_api_key",
        "credentials",
        "credentials_profile_name",
        "auth_token",
        "organization",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_region",
        # Endpoints and connection.
        "base_url",
        "openai_api_base",
        "anthropic_api_url",
        "endpoint",
        "endpoint_url",
        "region",
        "region_name",
        "inference_geo",
        "client_options",
        "client",
        "async_client",
        "http_client",
        "http_async_client",
        "default_headers",
        "default_query",
        "timeout",
        "request_timeout",
        "default_request_timeout",
        "max_retries",
        "retry",
        "proxies",
        "openai_proxy",
        "anthropic_proxy",
        # Request injection.
        "headers",
        "extra_headers",
        "extra_body",
        "extra_query",
        "extra_args",
        "model_kwargs",
        "provider_data",
        "additional_model_request_fields",
        # Remote tools.
        "mcp_servers",
        # Data retention.
        "store",
        "prompt_cache_retention",
        # Server-side state.
        "context_management",
        "truncation",
        # Identity and attribution.
        "metadata",
        "user",
        "safety_identifier",
        "request_metadata",
        # Runtime wiring and API-shape switches.
        "include",
        "include_usage",
        "response_include",
        # Prompt content beyond the instructions.
        "instructions",
        "system",
        # Safety configuration.
        "moderation",
        "guardrails",
        "guardrail_config",
        "guard_last_turn_only",
        # Claude Agents host-process settings.
        "cli_path",
        "env",
        "cwd",
        "add_dirs",
        "permission_mode",
        "settings",
        "setting_sources",
        "plugins",
        "sandbox",
        "resume",
        "session_id",
        "fork_session",
        "continue_conversation",
        "hooks",
        "can_use_tool",
        "stderr",
        "session_store",
    }
)


def leaked(key: str) -> str:
    """A value no handler sets itself, so finding it anywhere in a provider call means *key*
    leaked through from the config."""
    return f"leaked-from-config:{key}"


#: ``model.parameters`` holding every never-forwarded key, each set to its own :func:`leaked`
#: marker.
NEVER_FORWARDED_BAG = {key: leaked(key) for key in NEVER_FORWARDED_KEYS}


def find_leaks(value: object) -> set[str]:
    """Every :func:`leaked` marker found anywhere inside *value* (dicts, lists, tuples, and object
    attributes are searched)."""
    found: set[str] = set()
    seen: set[int] = set()

    def walk(v: object) -> None:
        if id(v) in seen:
            return
        seen.add(id(v))
        if isinstance(v, str):
            if v.startswith("leaked-from-config:"):
                found.add(v.split(":", 1)[1])
        elif isinstance(v, dict):
            for k, item in v.items():
                walk(k)
                walk(item)
        elif isinstance(v, (list, tuple, set, frozenset)):
            for item in v:
                walk(item)
        elif hasattr(v, "__dict__") and not isinstance(v, type):
            for item in vars(v).values():
                walk(item)

    walk(value)
    return found
