"""A Bedrock ``model.region`` reaches the AI Gateway as ``providerOptions.gateway.inferenceRegion``.

The last test runs the handler against the real ``ai`` library and captures the request it sends,
so it proves the option is on the wire, not only in ``InferenceRequestParams``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from launchdarkly_ai_vercel_messages import create_vercel_messages_handler
from launchdarkly_ai_vercel_messages.handler import _request_params
from launchdarkly_ai_vercel_messages.model_id import gateway_inference_region


def bedrock(
    model: str, region: str | None = None, provider: str = "Bedrock"
) -> dict[str, Any]:
    model_rep: dict[str, Any] = {"name": model}
    if region is not None:
        model_rep["region"] = region
    return {
        "provider": {"name": provider},
        "model": model_rep,
        "instructions": "Be brief.",
    }


def bedrock_only(geo_region: str) -> dict[str, Any]:
    return {"providers": {"bedrock": {"scope": "zone", "geoRegion": geo_region}}}


class TestGatewayInferenceRegion:
    @pytest.mark.parametrize("region", ["us", "eu", "au", "jp"])
    def test_maps_model_region(self, region: str) -> None:
        config = bedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", region)
        assert gateway_inference_region(config) == bedrock_only(region)

    def test_reads_the_profile_prefix_of_the_model_id(self) -> None:
        config = bedrock("eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert gateway_inference_region(config) == bedrock_only("eu")

    def test_profile_prefix_wins_over_model_region(self) -> None:
        config = bedrock("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "us")
        assert gateway_inference_region(config) == bedrock_only("eu")

    def test_normalizes_case_whitespace_and_trailing_dot(self) -> None:
        config = bedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", " US. ")
        assert gateway_inference_region(config) == bedrock_only("us")

    def test_applies_to_an_explicit_gateway_id(self) -> None:
        assert gateway_inference_region(
            bedrock("anthropic/claude-sonnet-4.5", "eu")
        ) == (bedrock_only("eu"))

    def test_global_sends_no_option(self) -> None:
        assert gateway_inference_region(bedrock("anthropic.claude-x", "global")) is None
        assert gateway_inference_region(bedrock("global.anthropic.claude-x")) is None

    @pytest.mark.parametrize("region", [None, "", "  "])
    def test_no_region_sends_no_option(self, region: str | None) -> None:
        assert gateway_inference_region(bedrock("anthropic.claude-x", region)) is None

    def test_ignores_a_region_on_a_provider_other_than_bedrock(self) -> None:
        config = bedrock("claude-sonnet-4.5", "eu", provider="Anthropic")
        assert gateway_inference_region(config) is None

    @pytest.mark.parametrize("region", ["apac", "us-gov", "ca", "in", "us-east-1"])
    def test_rejects_a_region_the_gateway_cannot_honor(self, region: str) -> None:
        with pytest.raises(
            ValueError, match=f'Bedrock region "{region}" has no Vercel AI Gateway'
        ):
            gateway_inference_region(bedrock("anthropic.claude-x", region))

    def test_rejects_an_unmappable_profile_prefix(self) -> None:
        with pytest.raises(ValueError, match='Bedrock region "apac"'):
            gateway_inference_region(bedrock("apac.anthropic.claude-x"))


class _GatewayProvider:
    provider_class_id = "gateway"


class _GatewayModel:
    provider = _GatewayProvider()


class TestBuildRequestParams:
    def test_gateway_model_gets_the_region_in_extra_body(self) -> None:
        params = _request_params(
            bedrock("anthropic.claude-x", "eu"), model=_GatewayModel()
        )
        assert params.extra_body == {
            "providerOptions": {"gateway": {"inferenceRegion": bedrock_only("eu")}}
        }

    def test_injected_non_gateway_model_gets_nothing(self) -> None:
        params = _request_params(bedrock("anthropic.claude-x", "eu"), model=object())
        assert params.extra_body is None

    def test_injected_non_gateway_model_skips_the_unmappable_region_check(self) -> None:
        params = _request_params(bedrock("anthropic.claude-x", "apac"), model=object())
        assert params.extra_body is None


async def test_region_is_on_the_outbound_gateway_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    sent: list[httpx2.Request] = []

    async def send(self: httpx2.AsyncClient, request: httpx2.Request, **_: Any) -> Any:
        sent.append(request)
        raise httpx2.ConnectError("captured", request=request)

    monkeypatch.setattr(httpx2.AsyncClient, "send", send)
    config = bedrock("us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    with pytest.raises(Exception):  # noqa: B017 - the captured request never gets a response
        await create_vercel_messages_handler()(config, "hi")

    body = json.loads(sent[0].content)
    assert sent[0].url.path.endswith("/language-model")
    assert (
        sent[0].headers["ai-language-model-id"]
        == "anthropic/claude-sonnet-4-5-20250929"
    )
    assert body["providerOptions"]["gateway"]["inferenceRegion"] == bedrock_only("us")
