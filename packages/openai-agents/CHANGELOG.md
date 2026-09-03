# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-agents-0.1.4...launchdarkly-ai-openai-agents-0.2.0) (2026-09-03)


### ⚠ BREAKING CHANGES

* **openai-agents:** the span this handler emits is renamed from `openai.agent.run` and `openai.agent.run.stream` to `invoke_agent`. Queries selecting on the old names will not match. Prompt and completion content is no longer on spans unless the caller passes capture_content=True.

### Features

* emit $ld:ai:sdk:info event per AI package ([#62](https://github.com/launchdarkly/python-ai-sdk/issues/62)) ([65136a3](https://github.com/launchdarkly/python-ai-sdk/commit/65136a3aa07d1245b28c38a7f38946a3c75a4516))
* gate the judge explanation on capture_content ([8a66365](https://github.com/launchdarkly/python-ai-sdk/commit/8a66365059386b637d43c96c56b659867decb047))
* **openai-agents:** emit invoke_agent, chat and execute_tool spans ([c5331eb](https://github.com/launchdarkly/python-ai-sdk/commit/c5331eb7d251ceec09b3802ef8955c5af6a9c20b))
* record judge scores as gen_ai.evaluation.result ([#43](https://github.com/launchdarkly/python-ai-sdk/issues/43)) ([7c8b0c9](https://github.com/launchdarkly/python-ai-sdk/commit/7c8b0c9820655c4a909366ad05d20c78050c4d17))


### Bug Fixes

* **openai-agents:** cancel the Runner even when there are no spans ([d9caaac](https://github.com/launchdarkly/python-ai-sdk/commit/d9caaac524cf9f3d26d989a43fcea241a22fd00e))
* **openai-agents:** close the tool span when the SDK reports no call id ([1d66254](https://github.com/launchdarkly/python-ai-sdk/commit/1d66254e0435f1340cd34e88a41046b8b1ad9d83))
* **openai-agents:** end the chat span the hook already stopped tracking ([9aecf05](https://github.com/launchdarkly/python-ai-sdk/commit/9aecf0575e27d573e0bc481ea1e7e0dedbe827e8))
* **openai-agents:** forward capture_content from the convenience wrapper ([a2440e4](https://github.com/launchdarkly/python-ai-sdk/commit/a2440e45131997bfbe518ee7cff231b6b1f1c65d))
* **openai-agents:** keep a tool span trackable when its content will not serialise ([32eb830](https://github.com/launchdarkly/python-ai-sdk/commit/32eb830899c464947e6dbd669806b7951402273d))
* **openai-agents:** keep the spans of a cancelled run ([063a392](https://github.com/launchdarkly/python-ai-sdk/commit/063a39271ec979c5977e7a0022d903299cebc614))
* **openai-agents:** put the input content write inside the guard too ([ead8ed5](https://github.com/launchdarkly/python-ai-sdk/commit/ead8ed586f3b71793249f24d1b9ba2197bae73ef))
* **openai-agents:** report a cancelled stream as cancelled, not abandoned ([2b246d0](https://github.com/launchdarkly/python-ai-sdk/commit/2b246d0f7483f4e7f9d4fe06658cdc028afdeed7))
* **openai-agents:** report tool arguments as an object, not a JSON string ([c646b6a](https://github.com/launchdarkly/python-ai-sdk/commit/c646b6a8a98831a2c64252886385a25473d8ec65))
* **openai-agents:** stop double-counting a failed run's token spend ([8991dd5](https://github.com/launchdarkly/python-ai-sdk/commit/8991dd5233915f4e18e142d561a53174c7075c53))
* **openai-agents:** stop tying token accounting to span creation ([0aa7241](https://github.com/launchdarkly/python-ai-sdk/commit/0aa72413907743d8e21c0f5616bd29b51ee1e789))
* **openai-agents:** take the run total from the Runner, not from the hooks ([64cd458](https://github.com/launchdarkly/python-ai-sdk/commit/64cd4588a50c1333a3e9681c5ec34ea214b82f53))


### Documentation

* describe the span tree the SDK now emits ([bc7815a](https://github.com/launchdarkly/python-ai-sdk/commit/bc7815a61e46c96df9cd786b20bfceb7d8b13334))
* describe the span tree the SDK now emits ([#37](https://github.com/launchdarkly/python-ai-sdk/issues/37)) ([6a82ef5](https://github.com/launchdarkly/python-ai-sdk/commit/6a82ef5d1fdbae2b1228f9325d960979bcf50881))

## [0.1.4](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-agents-0.1.3...launchdarkly-ai-openai-agents-0.1.4) (2026-08-07)


### Bug Fixes

* update module docstrings across all packages ([1c53ea3](https://github.com/launchdarkly/python-ai-sdk/commit/1c53ea3c248c5d3b4f5a553ae71d9a6fa9144bcc))
* update module docstrings across all packages ([#24](https://github.com/launchdarkly/python-ai-sdk/issues/24)) ([3b5f82b](https://github.com/launchdarkly/python-ai-sdk/commit/3b5f82b7b65eee735fee5a49d28c8bd6a7feb3dc))

## [0.1.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-agents-0.1.2...launchdarkly-ai-openai-agents-0.1.3) (2026-08-07)


### Bug Fixes

* add missing contents:read permission to release jobs ([b33d50f](https://github.com/launchdarkly/python-ai-sdk/commit/b33d50f2d6726b88d6515c9e9d334ae11cb5e159))
* add missing contents:read permission to release jobs ([#21](https://github.com/launchdarkly/python-ai-sdk/issues/21)) ([da72e1a](https://github.com/launchdarkly/python-ai-sdk/commit/da72e1a079b02a14e62d615b53cb08e2b44e7ba8))

## [0.1.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-agents-0.1.1...launchdarkly-ai-openai-agents-0.1.2) (2026-08-07)


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
