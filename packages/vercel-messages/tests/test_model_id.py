import pytest

from launchdarkly_ai_vercel_messages.model_id import gateway_model_id


def config(provider: str, model: str = "model") -> dict:
    return {"provider": {"name": provider}, "model": {"name": model}}


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("Anthropic", "model", "anthropic/model"),
        ("OpenAI", "model", "openai/model"),
        ("Bedrock", "anthropic.claude-sonnet-4", "anthropic/claude-sonnet-4"),
        ("Azure", "model", "openai/model"),
        ("Gemini", "model", "google/model"),
        ("Cohere", "model", "cohere/model"),
        ("Cortex", "llama-4-scout", "meta/llama-4-scout"),
        ("Cursor", "claude-sonnet-4", "anthropic/claude-sonnet-4"),
        ("Databricks", "llama-4-maverick", "meta/llama-4-maverick"),
        ("DeepSeek", "model", "deepseek/model"),
        ("Fireworks AI", "qwen-3-235b", "alibaba/qwen-3-235b"),
        ("Meta", "model", "meta/model"),
        ("Mistral", "model", "mistral/model"),
        ("Perplexity", "model", "perplexity/model"),
        ("Vertex", "model", "google/model"),
    ],
)
def test_maps_every_supported_launchdarkly_provider(
    provider: str, model: str, expected: str
) -> None:
    assert gateway_model_id(config(provider, model)) == expected


@pytest.mark.parametrize("provider", ["AI21 Labs", "IBM Watson"])
def test_fails_locally_for_unsupported_creator(provider: str) -> None:
    with pytest.raises(ValueError, match="currently exposes no models created by"):
        gateway_model_id(config(provider))


def test_preserves_explicit_gateway_id() -> None:
    assert gateway_model_id(config("Bedrock", "amazon/nova-pro")) == "amazon/nova-pro"


def test_removes_bedrock_region_and_dotted_creator_prefix() -> None:
    assert (
        gateway_model_id(config("Bedrock", "us.anthropic.claude-sonnet-4"))
        == "anthropic/claude-sonnet-4"
    )


def test_fails_locally_for_unidentifiable_multi_model_host() -> None:
    with pytest.raises(ValueError, match="hosts models from multiple creators"):
        gateway_model_id(config("Databricks", "custom-deployment"))


#: Each expected id resolved in https://ai-gateway.vercel.sh/v1/models/<id> on 2026-11-09.
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic/claude-sonnet-4-5-20250929",
        ),
        (
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic/claude-sonnet-4-5-20250929",
        ),
        (
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic/claude-haiku-4-5-20251001",
        ),
        (
            "eu.anthropic.claude-opus-4-5-20251101-v1:0",
            "anthropic/claude-opus-4-5-20251101",
        ),
        ("us.anthropic.claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
        ("us.amazon.nova-pro-v1:0", "amazon/nova-pro"),
        (
            "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic/claude-sonnet-4-5-20250929",
        ),
    ],
)
def test_strips_the_bedrock_profile_prefix_and_version_suffix(
    model: str, expected: str
) -> None:
    assert gateway_model_id(config("Bedrock", model)) == expected


def test_keeps_a_version_suffix_on_an_explicit_gateway_id() -> None:
    assert gateway_model_id(config("Bedrock", "anthropic/claude-sonnet-4-5-v1:0")) == (
        "anthropic/claude-sonnet-4-5-v1:0"
    )
