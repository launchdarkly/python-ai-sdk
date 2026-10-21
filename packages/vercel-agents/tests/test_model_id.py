import pytest

from launchdarkly_ai_vercel_agents.model_id import gateway_model_id


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


def test_preserves_explicit_and_dotted_gateway_ids() -> None:
    assert gateway_model_id(config("Bedrock", "amazon/nova-pro")) == "amazon/nova-pro"
    assert (
        gateway_model_id(config("Bedrock", "us.anthropic.claude-sonnet-4"))
        == "anthropic/claude-sonnet-4"
    )
