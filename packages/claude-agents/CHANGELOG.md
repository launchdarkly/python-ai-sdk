# Changelog

## [1.0.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.3.0...launchdarkly-ai-claude-agents-1.0.0) (2026-10-30)


### ⚠ BREAKING CHANGES

* **AIC-3598:** make Python packaging and public types GA-ready ([#218](https://github.com/launchdarkly/python-ai-sdk/issues/218))
* **AIC-3568:** give the barrel core's __all__, and drop private-spec references ([#234](https://github.com/launchdarkly/python-ai-sdk/issues/234))
* **claude-agents:** honour the config's max_tokens and disable thinking unless the config asks ([#237](https://github.com/launchdarkly/python-ai-sdk/issues/237))
* **client:** capture conversation content by default ([#213](https://github.com/launchdarkly/python-ai-sdk/issues/213))
* **AIC-3685:** make the internal provider helpers private ([#224](https://github.com/launchdarkly/python-ai-sdk/issues/224))
* **skills:** rename FDv2SkillStore to LaunchDarklySkillStore ([#223](https://github.com/launchdarkly/python-ai-sdk/issues/223))

### Features

* **AIC-3685:** make the internal provider helpers private ([#224](https://github.com/launchdarkly/python-ai-sdk/issues/224)) ([9afdb94](https://github.com/launchdarkly/python-ai-sdk/commit/9afdb94d8fa000063c694d54e8f702eb605d4b5e))
* **client:** add result variant and value to the feature_flag span event ([#240](https://github.com/launchdarkly/python-ai-sdk/issues/240)) ([040d487](https://github.com/launchdarkly/python-ai-sdk/commit/040d487b49f159ddf3ab1a7dcc44d2b74baa7197))
* **client:** capture conversation content by default ([#213](https://github.com/launchdarkly/python-ai-sdk/issues/213)) ([77a7cf5](https://github.com/launchdarkly/python-ai-sdk/commit/77a7cf5a88be6dbfff66ea94e0e99780b2cd03df))
* **skills:** rename FDv2SkillStore to LaunchDarklySkillStore ([#223](https://github.com/launchdarkly/python-ai-sdk/issues/223)) ([11f2c0b](https://github.com/launchdarkly/python-ai-sdk/commit/11f2c0bbb21934bdc79db64e9bb6409870bf2ba4))


### Bug Fixes

* **AIC-3568:** give the barrel core's __all__, and drop private-spec references ([#234](https://github.com/launchdarkly/python-ai-sdk/issues/234)) ([88e57e6](https://github.com/launchdarkly/python-ai-sdk/commit/88e57e627d0b01b6fe89961e75d9efe08a3d9fdb))
* **AIC-3598:** make Python packaging and public types GA-ready ([#218](https://github.com/launchdarkly/python-ai-sdk/issues/218)) ([5860b40](https://github.com/launchdarkly/python-ai-sdk/commit/5860b40078f69febc522680ec76519c25274a606))
* **claude-agents:** honour the config's max_tokens and disable thinking unless the config asks ([#237](https://github.com/launchdarkly/python-ai-sdk/issues/237)) ([a7ded1b](https://github.com/launchdarkly/python-ai-sdk/commit/a7ded1be07bd49a9a64802e08812f49d12ca6aa2))

## [0.3.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.2.6...launchdarkly-ai-claude-agents-0.3.0) (2026-10-27)


### ⚠ BREAKING CHANGES

* **skills:** promote Agent Skills to core ([#210](https://github.com/launchdarkly/python-ai-sdk/issues/210))
* **claude-agents:** switch the built-ins off by default and load no host settings ([#201](https://github.com/launchdarkly/python-ai-sdk/issues/201))
* handlers now forward the AI Config's model parameters to the provider, limited to each handler's list in TESTING.md 1.12. Configs that set parameters which used to be ignored now take effect.

### Features

* **skills:** export the Claude Agents skills handler from the package root ([#217](https://github.com/launchdarkly/python-ai-sdk/issues/217)) ([5d46dbb](https://github.com/launchdarkly/python-ai-sdk/commit/5d46dbbf55a3c008a28dfa72345dd9d542d00603))
* **skills:** promote Agent Skills to core ([#210](https://github.com/launchdarkly/python-ai-sdk/issues/210)) ([715815f](https://github.com/launchdarkly/python-ai-sdk/commit/715815fb37b1fd3f5a537bbd5b3a6237b90fc9fb))


### Bug Fixes

* **claude-agents:** switch the built-ins off by default and load no host settings ([#201](https://github.com/launchdarkly/python-ai-sdk/issues/201)) ([f99d538](https://github.com/launchdarkly/python-ai-sdk/commit/f99d5384353b439a221c556b4103dde783e6a150))
* pass AI Config model parameters through to every provider handler ([#107](https://github.com/launchdarkly/python-ai-sdk/issues/107)) ([e3afb3f](https://github.com/launchdarkly/python-ai-sdk/commit/e3afb3f2c19ebf7085f84e94cd5a53f2bbe929ae))

## [0.2.6](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.2.5...launchdarkly-ai-claude-agents-0.2.6) (2026-10-23)


### Features

* **claude:** emit AG-UI text and tool events ([#165](https://github.com/launchdarkly/python-ai-sdk/issues/165)) ([3f3a0ce](https://github.com/launchdarkly/python-ai-sdk/commit/3f3a0cea17806beba21d839847b3158821600d26))
* **client:** add configure for SDK defaults ([#158](https://github.com/launchdarkly/python-ai-sdk/issues/158)) ([3736767](https://github.com/launchdarkly/python-ai-sdk/commit/373676797cd810c04548a7c0da32b8ea5f7c02c2))


### Bug Fixes

* **AIC-3571:** visit every child before its parent in reverse_traverse ([#189](https://github.com/launchdarkly/python-ai-sdk/issues/189)) ([87c73d9](https://github.com/launchdarkly/python-ai-sdk/commit/87c73d9857ced5b916753f80caa943f43e3b31ff))
* **AIC-3577:** send object tool results to the model as JSON ([#188](https://github.com/launchdarkly/python-ai-sdk/issues/188)) ([dead4bd](https://github.com/launchdarkly/python-ai-sdk/commit/dead4bdce43a1e68416a148073058da81030ab73))
* **claude-agents:** fail the run when the CLI reports an API error as success ([#174](https://github.com/launchdarkly/python-ai-sdk/issues/174)) ([3d1a768](https://github.com/launchdarkly/python-ai-sdk/commit/3d1a7683f12e547ecf9c03aac44ab1dc588cd555))
* give native graph spans their LaunchDarkly identity ([#121](https://github.com/launchdarkly/python-ai-sdk/issues/121)) ([e863b6d](https://github.com/launchdarkly/python-ai-sdk/commit/e863b6d140bcc52ef058e1df374d40116bbaf762))

## [0.2.5](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.2.4...launchdarkly-ai-claude-agents-0.2.5) (2026-10-09)


### Features

* **AIC-3495:** record which public helpers an application calls ([#145](https://github.com/launchdarkly/python-ai-sdk/issues/145)) ([19d2c2b](https://github.com/launchdarkly/python-ai-sdk/commit/19d2c2baf35440af57ed57f12f0413e524de27fa))

## [0.2.4](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.2.3...launchdarkly-ai-claude-agents-0.2.4) (2026-09-30)


### Features

* **AIC-3211:** record each graph node as its own tracking event ([#114](https://github.com/launchdarkly/python-ai-sdk/issues/114)) ([c3c439e](https://github.com/launchdarkly/python-ai-sdk/commit/c3c439ede989ea98c43d238b80467351c5920c56))

## [0.2.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.2.2...launchdarkly-ai-claude-agents-0.2.3) (2026-09-21)


### Features

* **AIC-3106:** add multimodal history support to graph().invoke() ([#26](https://github.com/launchdarkly/python-ai-sdk/issues/26)) ([c91132d](https://github.com/launchdarkly/python-ai-sdk/commit/c91132de81f846326457c6d397be609b7a510e35))


### Documentation

* replace stale launchdarkly-ai package name with launchdarkly-ai-python ([#40](https://github.com/launchdarkly/python-ai-sdk/issues/40)) ([3256db7](https://github.com/launchdarkly/python-ai-sdk/commit/3256db77a8f3e54c08635ea3a8157e60e5aa0f8b))

## [0.2.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.2.1...launchdarkly-ai-claude-agents-0.2.2) (2026-09-08)


### Bug Fixes

* document docs link in package docstrings ([#76](https://github.com/launchdarkly/python-ai-sdk/issues/76)) ([b75366d](https://github.com/launchdarkly/python-ai-sdk/commit/b75366df3a740753d45995f93d649d46e1ddf6b7))

## [0.2.1](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.2.0...launchdarkly-ai-claude-agents-0.2.1) (2026-09-08)


### Bug Fixes

* pin hatchling build backend to a metadata-compatible range ([#73](https://github.com/launchdarkly/python-ai-sdk/issues/73)) ([1f293a6](https://github.com/launchdarkly/python-ai-sdk/commit/1f293a60a772de1f7c7b3f05caba0c0b22c5591e))

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.1.4...launchdarkly-ai-claude-agents-0.2.0) (2026-09-08)


### ⚠ BREAKING CHANGES

* **claude-agents:** stop an empty tool list switching off the built-ins
* **claude-agents:** the span this handler emits is renamed from `claude.query` and `claude.query.stream` to `invoke_agent`. Queries selecting on the old names will not match. Prompt and completion content is no longer on spans unless the caller passes capture_content=True.

### Features

* **claude-agents:** emit invoke_agent, chat and execute_tool spans ([7831c81](https://github.com/launchdarkly/python-ai-sdk/commit/7831c8129bdeedb6440685a55a40280daf63b35d))
* emit $ld:ai:sdk:info event per AI package ([#62](https://github.com/launchdarkly/python-ai-sdk/issues/62)) ([65136a3](https://github.com/launchdarkly/python-ai-sdk/commit/65136a3aa07d1245b28c38a7f38946a3c75a4516))
* emit conversation id and judge evals ([1b0f5bd](https://github.com/launchdarkly/python-ai-sdk/commit/1b0f5bd926e2f0583d330f313b2af35cba1128f5))
* emit gen_ai.conversation.id ([#42](https://github.com/launchdarkly/python-ai-sdk/issues/42)) ([c372e56](https://github.com/launchdarkly/python-ai-sdk/commit/c372e56df9ff0e32f6a6196a35cc8fc909daba01))
* gate the judge explanation on capture_content ([8a66365](https://github.com/launchdarkly/python-ai-sdk/commit/8a66365059386b637d43c96c56b659867decb047))
* record judge scores as gen_ai.evaluation.result ([#43](https://github.com/launchdarkly/python-ai-sdk/issues/43)) ([7c8b0c9](https://github.com/launchdarkly/python-ai-sdk/commit/7c8b0c9820655c4a909366ad05d20c78050c4d17))


### Bug Fixes

* bind conversation id at stream() call time ([bda8393](https://github.com/launchdarkly/python-ai-sdk/commit/bda839312c7372bb55ad13bb92b0e3a0a6618738))
* **claude-agents:** an abandoned stream is not an error for its tool spans ([10b8158](https://github.com/launchdarkly/python-ai-sdk/commit/10b81583650ae5356045cfa444313d8adde79bb5))
* **claude-agents:** an empty run must not claim it cost nothing ([85c73ff](https://github.com/launchdarkly/python-ai-sdk/commit/85c73ffb9cb0af85304f38d93e3a896855f2c61a))
* **claude-agents:** keep a tool span trackable when its content will not serialise ([0738606](https://github.com/launchdarkly/python-ai-sdk/commit/07386063f5e8157b44819fbadb2ebbd882ed859b))
* **claude-agents:** keep the spans of a cancelled run ([b7dddfa](https://github.com/launchdarkly/python-ai-sdk/commit/b7dddfac2099fafed29870790cd48f48f96d10b1))
* **claude-agents:** put the input content write inside the guard too ([f0160c2](https://github.com/launchdarkly/python-ai-sdk/commit/f0160c29f11b6b8965c07f209e09463414bca48a))
* **claude-agents:** report a cancelled stream as cancelled, not abandoned ([d4f1b6f](https://github.com/launchdarkly/python-ai-sdk/commit/d4f1b6f76bf703e2bbfd24d4cfc007de0f9be4ba))
* **claude-agents:** stop an empty tool list switching off the built-ins ([50e82fd](https://github.com/launchdarkly/python-ai-sdk/commit/50e82fd8903aa4fbd9c83697fa48f4ddb8c05d8d))
* scope the processor, keep streaming parenting, accept a nullish id ([9794d36](https://github.com/launchdarkly/python-ai-sdk/commit/9794d367b80961f17aecea14f06188f32160f7e6))
* use tool keys in tool call telemetry ([841ba77](https://github.com/launchdarkly/python-ai-sdk/commit/841ba7792475f91a9335104d9cb1304ac8e17cfb))
* use tool keys in tool call telemetry ([#56](https://github.com/launchdarkly/python-ai-sdk/issues/56)) ([ae3130f](https://github.com/launchdarkly/python-ai-sdk/commit/ae3130f973d23e6f52bb9646fbf54c88671ebb11))


### Documentation

* describe the span tree the SDK now emits ([bc7815a](https://github.com/launchdarkly/python-ai-sdk/commit/bc7815a61e46c96df9cd786b20bfceb7d8b13334))
* describe the span tree the SDK now emits ([#37](https://github.com/launchdarkly/python-ai-sdk/issues/37)) ([6a82ef5](https://github.com/launchdarkly/python-ai-sdk/commit/6a82ef5d1fdbae2b1228f9325d960979bcf50881))

## [0.1.4](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.1.3...launchdarkly-ai-claude-agents-0.1.4) (2026-08-07)


### Bug Fixes

* update module docstrings across all packages ([1c53ea3](https://github.com/launchdarkly/python-ai-sdk/commit/1c53ea3c248c5d3b4f5a553ae71d9a6fa9144bcc))
* update module docstrings across all packages ([#24](https://github.com/launchdarkly/python-ai-sdk/issues/24)) ([3b5f82b](https://github.com/launchdarkly/python-ai-sdk/commit/3b5f82b7b65eee735fee5a49d28c8bd6a7feb3dc))

## [0.1.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.1.2...launchdarkly-ai-claude-agents-0.1.3) (2026-08-07)


### Bug Fixes

* add missing contents:read permission to release jobs ([b33d50f](https://github.com/launchdarkly/python-ai-sdk/commit/b33d50f2d6726b88d6515c9e9d334ae11cb5e159))
* add missing contents:read permission to release jobs ([#21](https://github.com/launchdarkly/python-ai-sdk/issues/21)) ([da72e1a](https://github.com/launchdarkly/python-ai-sdk/commit/da72e1a079b02a14e62d615b53cb08e2b44e7ba8))

## [0.1.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.1.1...launchdarkly-ai-claude-agents-0.1.2) (2026-08-07)


### Features

* initial commit — LaunchDarkly AI SDK for Python ([0c74677](https://github.com/launchdarkly/python-ai-sdk/commit/0c7467797a86b3346631c1289941df0f6ac6595b))
* initial commit — LaunchDarkly AI SDK for Python ([#1](https://github.com/launchdarkly/python-ai-sdk/issues/1)) ([1cfbf42](https://github.com/launchdarkly/python-ai-sdk/commit/1cfbf4259aa3e75f4c30a0594636b889626eb6a6))

## 0.1.0 (2026-07-31)


### Features

* initial commit — LaunchDarkly AI SDK for Python ([0c74677](https://github.com/launchdarkly/python-ai-sdk/commit/0c7467797a86b3346631c1289941df0f6ac6595b))
* initial commit — LaunchDarkly AI SDK for Python ([#1](https://github.com/launchdarkly/python-ai-sdk/issues/1)) ([1cfbf42](https://github.com/launchdarkly/python-ai-sdk/commit/1cfbf4259aa3e75f4c30a0594636b889626eb6a6))

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
