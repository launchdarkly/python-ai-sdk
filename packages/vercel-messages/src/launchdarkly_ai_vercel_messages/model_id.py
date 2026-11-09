from __future__ import annotations

import re
from typing import Any

from launchdarkly_ai_server import AiConfigRep

GATEWAY_CREATORS = {
    "alibaba": "alibaba",
    "amazon": "amazon",
    "anthropic": "anthropic",
    "arceeai": "arcee-ai",
    "bfl": "bfl",
    "bytedance": "bytedance",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "fishaudio": "fish-audio",
    "gemini": "google",
    "google": "google",
    "googleai": "google",
    "inception": "inception",
    "inclusionai": "inclusionai",
    "inferencenet": "inference-net",
    "interfaze": "interfaze",
    "klingai": "klingai",
    "meta": "meta",
    "minimax": "minimax",
    "mistral": "mistral",
    "mistralai": "mistral",
    "mixedbread": "mixedbread",
    "moonshotai": "moonshotai",
    "morph": "morph",
    "nvidia": "nvidia",
    "openai": "openai",
    "perplexity": "perplexity",
    "poolside": "poolside",
    "prodia": "prodia",
    "quiverai": "quiverai",
    "recraft": "recraft",
    "sakana": "sakana",
    "spacexai": "spacexai",
    "stepfun": "stepfun",
    "tencent": "tencent",
    "thinkingmachines": "thinkingmachines",
    "typesafe": "typesafe-ai",
    "typesafeai": "typesafe-ai",
    "voyage": "voyage",
    "xiaomi": "xiaomi",
    "xai": "spacexai",
    "zai": "zai",
}

PROVIDER_CREATORS: dict[str, str | None] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "bedrock": None,
    "azure": "openai",
    "gemini": "google",
    "ai21labs": None,
    "cohere": "cohere",
    "cortex": None,
    "cursor": None,
    "databricks": None,
    "deepseek": "deepseek",
    "fireworksai": None,
    "ibmwatson": None,
    "meta": "meta",
    "mistral": "mistral",
    "perplexity": "perplexity",
    "vertex": "google",
    # Compatibility aliases found in existing configs.
    "google": "google",
    "googleai": "google",
    "mistralai": "mistral",
    "spacexai": "spacexai",
    "typesafe": "typesafe-ai",
    "typesafeai": "typesafe-ai",
    "xai": "spacexai",
}

MODEL_FAMILY_CREATORS = (
    (re.compile(r"^(?:gpt|o[1-9])(?:[-.]|$)", re.I), "openai"),
    (re.compile(r"^claude(?:[-.]|$)", re.I), "anthropic"),
    (re.compile(r"^gemini(?:[-.]|$)", re.I), "google"),
    (re.compile(r"^grok(?:[-.]|$)", re.I), "spacexai"),
    (re.compile(r"^(?:command|aya)(?:[-.]|$)", re.I), "cohere"),
    (re.compile(r"^deepseek(?:[-.]|$)", re.I), "deepseek"),
    (re.compile(r"^llama(?:[-.]|$)", re.I), "meta"),
    (
        re.compile(r"^(?:mistral|mixtral|codestral|pixtral)(?:[-.]|$)", re.I),
        "mistral",
    ),
    (re.compile(r"^sonar(?:[-.]|$)", re.I), "perplexity"),
    (re.compile(r"^(?:nova|titan)(?:[-.]|$)", re.I), "amazon"),
    (re.compile(r"^qwen(?:[-.]|$)", re.I), "alibaba"),
)


#: Bedrock cross-region inference-profile prefixes
#: (https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html).
BEDROCK_INFERENCE_PROFILE_PREFIXES = (
    "global",
    "us",
    "us-gov",
    "eu",
    "apac",
    "au",
    "jp",
    "ca",
    "in",
)

#: The Bedrock profile prefixes that have an AI Gateway ``inferenceRegion`` equivalent
#: (https://vercel.com/docs/ai-gateway/security-and-compliance/regional-inference). ``global`` is
#: the Gateway's default, and sending it explicitly fails under a team-wide region pin, so it maps
#: to no option at all.
_GATEWAY_GEO_REGIONS: dict[str, str | None] = {
    "global": None,
    "us": "us",
    "eu": "eu",
    "au": "au",
    "jp": "jp",
}

#: Bedrock's model-version suffix: ``-v1:0``, ``-v1``, and ``-1:0`` (``openai.gpt-oss-120b-1:0``).
_BEDROCK_VERSION_SUFFIX = re.compile(r"-(?:v\d+(?::\d+)?|\d+:\d+)$")


def _provider_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _creator_from_model(model: str) -> tuple[str, str] | None:
    parts = model.split(".")
    for index, part in enumerate(parts[:-1]):
        creator = GATEWAY_CREATORS.get(_provider_key(part))
        if creator:
            rest = ".".join(parts[index + 1 :])
            return creator, _BEDROCK_VERSION_SUFFIX.sub("", rest)
    for pattern, creator in MODEL_FAMILY_CREATORS:
        if pattern.search(model):
            return creator, model
    return None


def _bedrock_region(config: AiConfigRep) -> str | None:
    """The configured Bedrock inference-profile prefix: the model id's own prefix, which is what
    Bedrock itself would run, else ``model.region``."""
    model = config.get("model") or {}
    name = str(model.get("name") or "")
    prefix, dot, _ = name.partition(".")
    if dot and "/" not in name and prefix.lower() in BEDROCK_INFERENCE_PROFILE_PREFIXES:
        return prefix.lower()
    region = model.get("region")
    if isinstance(region, str) and region.strip().rstrip("."):
        return region.strip().rstrip(".").lower()
    return None


def gateway_inference_region(config: AiConfigRep) -> dict[str, Any] | None:
    """The AI Gateway ``inferenceRegion`` for a Bedrock config, scoped to the ``bedrock`` provider.

    Returns ``None`` for a config that is not Bedrock, has no region, or asks for ``global``.
    Raises ``ValueError`` for a region the Gateway has no equivalent for.
    """
    provider = str((config.get("provider") or {}).get("name") or "")
    if _provider_key(provider) not in {"bedrock", "amazonbedrock"}:
        return None
    region = _bedrock_region(config)
    if region is None:
        return None
    if region not in _GATEWAY_GEO_REGIONS:
        supported = ", ".join(sorted(_GATEWAY_GEO_REGIONS))
        raise ValueError(
            f'Bedrock region "{region}" has no Vercel AI Gateway inferenceRegion equivalent. '
            f"Use one of: {supported}, or pass the handler a model that calls Bedrock directly."
        )
    geo_region = _GATEWAY_GEO_REGIONS[region]
    if geo_region is None:
        return None
    return {"providers": {"bedrock": {"scope": "zone", "geoRegion": geo_region}}}


def gateway_model_id(config: AiConfigRep) -> str:
    """Build the AI Gateway ``creator/model`` id for an evaluated config."""
    model = str(config["model"]["name"])
    if "/" in model:
        return model

    provider = str((config.get("provider") or {}).get("name") or "")
    provider_name = _provider_key(provider)
    inferred = _creator_from_model(model)
    if inferred:
        return f"{inferred[0]}/{inferred[1]}"

    creator = PROVIDER_CREATORS.get(provider_name)
    if creator:
        return f"{creator}/{model}"
    if provider_name in {"ai21labs", "ibmwatson"}:
        raise ValueError(
            f'Vercel AI Gateway currently exposes no models created by "{provider}". '
            "Inject a direct provider model with model/model_factory instead."
        )
    if provider_name in PROVIDER_CREATORS:
        raise ValueError(
            f'LaunchDarkly provider "{provider}" hosts models from multiple '
            f'creators, so "{model}" cannot be converted to a Vercel creator/model '
            "id. Store an explicit creator/model id or inject a model/model_factory."
        )
    else:
        raise ValueError(
            f'Cannot map LaunchDarkly provider "{provider or "unknown"}" to a '
            "Vercel AI Gateway creator. Pass a creator/model id in "
            "config.model.name or inject a model/model factory."
        )
