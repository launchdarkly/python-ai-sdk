# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-server-0.1.3...launchdarkly-ai-server-0.2.0) (2026-09-03)


### ⚠ BREAKING CHANGES

* **client:** once the handlers move onto this layer, prompt and completion content will be absent from spans unless the caller passes capture_content=True. Anyone reading gen_ai.prompt.0.content today will need to opt in.

### Features

* add evaluations module scaffold, credentials, LD API client, result types ([80fd4e4](https://github.com/launchdarkly/python-ai-sdk/commit/80fd4e46351a4640970d53212eac62579a61ff66))
* **AIC-3230:** emit evaluation context identity ([#65](https://github.com/launchdarkly/python-ai-sdk/issues/65)) ([ca30e83](https://github.com/launchdarkly/python-ai-sdk/commit/ca30e835ae758c084a85ae39e4301be4b1d73643))
* **client:** cache-aware token accounting and a reusable span lifecycle ([f54ef59](https://github.com/launchdarkly/python-ai-sdk/commit/f54ef596c8d5c8dea5bdb555af49538e9da7b0cd))
* **client:** end spans a cancelled run would otherwise strand ([dff27f6](https://github.com/launchdarkly/python-ai-sdk/commit/dff27f60915f6ea6fb24e0351c9eb5faec83ff27))
* **client:** put conversation content on spans behind an opt-in flag ([2bb8310](https://github.com/launchdarkly/python-ai-sdk/commit/2bb8310bad322ebf9915c2214b8e7ca367d5878b))
* **client:** tell a cancelled stream apart from an abandoned one ([14ed066](https://github.com/launchdarkly/python-ai-sdk/commit/14ed0668708bac31f110451cdf25d748e4076063))
* emit $ld:ai:sdk:info event per AI package ([#62](https://github.com/launchdarkly/python-ai-sdk/issues/62)) ([65136a3](https://github.com/launchdarkly/python-ai-sdk/commit/65136a3aa07d1245b28c38a7f38946a3c75a4516))
* emit conversation id and judge evals ([1b0f5bd](https://github.com/launchdarkly/python-ai-sdk/commit/1b0f5bd926e2f0583d330f313b2af35cba1128f5))
* emit evaluation context identity on feature_flag spans ([97528e3](https://github.com/launchdarkly/python-ai-sdk/commit/97528e35d94bdc3cba621d2ffeb057b4f7eab60a))
* emit gen_ai.conversation.id ([#42](https://github.com/launchdarkly/python-ai-sdk/issues/42)) ([c372e56](https://github.com/launchdarkly/python-ai-sdk/commit/c372e56df9ff0e32f6a6196a35cc8fc909daba01))
* **evaluations:** configure run link UI base ([4cbcbd6](https://github.com/launchdarkly/python-ai-sdk/commit/4cbcbd66e2dd4eed0be931924ee7246d430474df))
* **evaluations:** emit generation events ([311b012](https://github.com/launchdarkly/python-ai-sdk/commit/311b012937a8d4862165ee35b038597df6c7badb))
* **evaluations:** event-based offline evaluations runner ([#39](https://github.com/launchdarkly/python-ai-sdk/issues/39)) ([4239739](https://github.com/launchdarkly/python-ai-sdk/commit/4239739fc8fd62bfac7113c9d7778e4146a8bb28))
* **evaluations:** make summary poll interval and timeout configurable per run ([cd84352](https://github.com/launchdarkly/python-ai-sdk/commit/cd843522fcf0e233e0b9cb548cb06e5c880ade57))
* **evaluations:** print event emission timestamps ([0079289](https://github.com/launchdarkly/python-ai-sdk/commit/00792895517c7442f3c17d0591af22d49992112a))
* **evaluations:** require an SDK key and fail fast when it is missing ([a1ba71e](https://github.com/launchdarkly/python-ai-sdk/commit/a1ba71edb5c8c027cfe7d978b100c4cd848a60c9))
* gate evaluation generation result ingest ([0ea442f](https://github.com/launchdarkly/python-ai-sdk/commit/0ea442f96b6de6963916419bd8b4ce767f2314c8))
* gate the judge explanation on capture_content ([8a66365](https://github.com/launchdarkly/python-ai-sdk/commit/8a66365059386b637d43c96c56b659867decb047))
* record judge scores as gen_ai.evaluation.result ([2604816](https://github.com/launchdarkly/python-ai-sdk/commit/26048166736941e59cff13197b61a74611ae39c3))
* record judge scores as gen_ai.evaluation.result ([#43](https://github.com/launchdarkly/python-ai-sdk/issues/43)) ([7c8b0c9](https://github.com/launchdarkly/python-ai-sdk/commit/7c8b0c9820655c4a909366ad05d20c78050c4d17))
* run client-side evaluations from the SDK ([50fee37](https://github.com/launchdarkly/python-ai-sdk/commit/50fee37c88b2bc207544bd7a28bf3d4d00a34875))


### Bug Fixes

* **AIC-3230:** align user context canonical identity ([68a5c2b](https://github.com/launchdarkly/python-ai-sdk/commit/68a5c2b01a1cc64cb33e7cdc1e463c29ac4a41d6))
* align evaluations with staging dataset and run APIs ([28f8eb1](https://github.com/launchdarkly/python-ai-sdk/commit/28f8eb19770adcff0d856368f53d7b731298fc02))
* bind conversation id at stream() call time ([bda8393](https://github.com/launchdarkly/python-ai-sdk/commit/bda839312c7372bb55ad13bb92b0e3a0a6618738))
* **client:** a tool that returned nothing does not return null ([eb61eb2](https://github.com/launchdarkly/python-ai-sdk/commit/eb61eb2000d51fb8dd523eab6f7cfbaeafb1db99))
* **client:** carry the cache breakdown into the public UsageDict ([c60fa7c](https://github.com/launchdarkly/python-ai-sdk/commit/c60fa7c6ade65542cbbbaefd440eca13e56088a4))
* **client:** guard the fourth content writer, and assert the family rule ([d637b95](https://github.com/launchdarkly/python-ai-sdk/commit/d637b955290f0c9c671af62774018a8cb343e454))
* **client:** keep the plain text a LangChain message holds in a list ([c11adca](https://github.com/launchdarkly/python-ai-sdk/commit/c11adca31cfe9a165441dca992496b10ab28a501))
* **client:** make end_span_once a no-op on a None span ([a1772c1](https://github.com/launchdarkly/python-ai-sdk/commit/a1772c1a59c4f45e322d9805986ad494c2ef5bcc))
* **client:** read a LangChain ChatMessage's own speaker ([d56103a](https://github.com/launchdarkly/python-ai-sdk/commit/d56103a615b7d6e4c93004209d257e3fe5771fa0))
* **client:** read a LangChain message's role from the field, not only the method ([641f8e1](https://github.com/launchdarkly/python-ai-sdk/commit/641f8e194cfaf705e38cdd8820f4f9d90f8e1110))
* **client:** stop the content carriers disagreeing, and stop them crashing ([b32d3fc](https://github.com/launchdarkly/python-ai-sdk/commit/b32d3fc7116abd35e9db090668962365c21fd932))
* delay only LaunchDarkly invoke_agent spans during judge evals ([af5a92c](https://github.com/launchdarkly/python-ai-sdk/commit/af5a92c27e6bc34d16c0b630cf5c21ecf9107d8a))
* **evaluations:** accept a BYOC client without an SDK key and reject NaN poll values ([ab5a4e5](https://github.com/launchdarkly/python-ai-sdk/commit/ab5a4e5d9cb6d29b189f3efde6cea1f0336ba0b4))
* **evaluations:** avoid replaying non-idempotent POSTs and blocking the event loop ([485dc85](https://github.com/launchdarkly/python-ai-sdk/commit/485dc85ee0eb1d4a6caf9f614716824e04c680de))
* **evaluations:** derive result from summary ([a7acc68](https://github.com/launchdarkly/python-ai-sdk/commit/a7acc68cef558e03d580382f5fe7597e613da157))
* **evaluations:** emit direct generation output ([b98b643](https://github.com/launchdarkly/python-ai-sdk/commit/b98b6434f41cbfb4d9d2b4a6d6d78cf099ec4a01))
* **evaluations:** flatten generation event payload ([a0254d0](https://github.com/launchdarkly/python-ai-sdk/commit/a0254d0a9c8d0ffe378c4eb042397965c2b7f8e1))
* **evaluations:** include evaluation run event ID ([8cae0e4](https://github.com/launchdarkly/python-ai-sdk/commit/8cae0e4b092c475dab61481659cb21102c3e956d))
* **evaluations:** nest generation usage tokens ([c88e9ae](https://github.com/launchdarkly/python-ai-sdk/commit/c88e9aee533caeb36caeca1a4ea5c2adce9bfa72))
* **evaluations:** poll run summary to terminal state ([75d3072](https://github.com/launchdarkly/python-ai-sdk/commit/75d3072b9343e87adeaf2ff861639093d579637c))
* **evaluations:** return pending summaries promptly ([ec3e290](https://github.com/launchdarkly/python-ai-sdk/commit/ec3e290ad606c2a7362e6c7b1aec57f82ee9d58c))
* **evaluations:** use API run source ([433273a](https://github.com/launchdarkly/python-ai-sdk/commit/433273aa71dc5c65714228f2e4cc2fbb0e15b2cf))
* return JudgeResult objects from inline judges ([c9e20a2](https://github.com/launchdarkly/python-ai-sdk/commit/c9e20a2fdb3cdea5763ef9c19928ddeca3c7ec7a))
* scope the processor, keep streaming parenting, accept a nullish id ([9794d36](https://github.com/launchdarkly/python-ai-sdk/commit/9794d367b80961f17aecea14f06188f32160f7e6))
* stop exporting judge reasoning, validate the score, freeze the end time ([8fa4f6f](https://github.com/launchdarkly/python-ai-sdk/commit/8fa4f6fd81543bce5a5aae2448f4eabdac3dce37))
* use tool keys in tool call telemetry ([841ba77](https://github.com/launchdarkly/python-ai-sdk/commit/841ba7792475f91a9335104d9cb1304ac8e17cfb))
* use tool keys in tool call telemetry ([#56](https://github.com/launchdarkly/python-ai-sdk/issues/56)) ([ae3130f](https://github.com/launchdarkly/python-ai-sdk/commit/ae3130f973d23e6f52bb9646fbf54c88671ebb11))


### Documentation

* describe the span tree the SDK now emits ([#37](https://github.com/launchdarkly/python-ai-sdk/issues/37)) ([6a82ef5](https://github.com/launchdarkly/python-ai-sdk/commit/6a82ef5d1fdbae2b1228f9325d960979bcf50881))

## [0.1.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-server-0.1.2...launchdarkly-ai-server-0.1.3) (2026-08-07)


### Bug Fixes

* update module docstrings across all packages ([1c53ea3](https://github.com/launchdarkly/python-ai-sdk/commit/1c53ea3c248c5d3b4f5a553ae71d9a6fa9144bcc))
* update module docstrings across all packages ([#24](https://github.com/launchdarkly/python-ai-sdk/issues/24)) ([3b5f82b](https://github.com/launchdarkly/python-ai-sdk/commit/3b5f82b7b65eee735fee5a49d28c8bd6a7feb3dc))

## [0.1.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-server-0.1.1...launchdarkly-ai-server-0.1.2) (2026-08-07)


### Features

* initial commit — LaunchDarkly AI SDK for Python ([0c74677](https://github.com/launchdarkly/python-ai-sdk/commit/0c7467797a86b3346631c1289941df0f6ac6595b))
* initial commit — LaunchDarkly AI SDK for Python ([#1](https://github.com/launchdarkly/python-ai-sdk/issues/1)) ([1cfbf42](https://github.com/launchdarkly/python-ai-sdk/commit/1cfbf4259aa3e75f4c30a0594636b889626eb6a6))


### Bug Fixes

* add module docstring to client package ([e27264b](https://github.com/launchdarkly/python-ai-sdk/commit/e27264b2980da7b8b78f17c112c6c54b3211435c))
* add module docstring to client package ([#18](https://github.com/launchdarkly/python-ai-sdk/issues/18)) ([34d314b](https://github.com/launchdarkly/python-ai-sdk/commit/34d314b5e658eb6cf6c3dfceeab361078be029fa))

## 0.1.0 (2026-07-31)


### Features

* initial commit — LaunchDarkly AI SDK for Python ([0c74677](https://github.com/launchdarkly/python-ai-sdk/commit/0c7467797a86b3346631c1289941df0f6ac6595b))
* initial commit — LaunchDarkly AI SDK for Python ([#1](https://github.com/launchdarkly/python-ai-sdk/issues/1)) ([1cfbf42](https://github.com/launchdarkly/python-ai-sdk/commit/1cfbf4259aa3e75f4c30a0594636b889626eb6a6))

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
