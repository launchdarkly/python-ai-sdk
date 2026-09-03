# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-agents-0.1.4...launchdarkly-ai-claude-agents-0.2.0) (2026-09-03)


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
