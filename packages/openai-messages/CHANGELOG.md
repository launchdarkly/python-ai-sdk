# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-messages-0.1.4...launchdarkly-ai-openai-messages-0.2.0) (2026-09-03)


### ⚠ BREAKING CHANGES

* **openai-messages:** the span this handler emits is renamed from `openai.response` and `openai.response.stream` to `invoke_agent`. Queries selecting on the old names will not match. Prompt and completion content is no longer on spans unless the caller passes capture_content=True.

### Features

* emit $ld:ai:sdk:info event per AI package ([#62](https://github.com/launchdarkly/python-ai-sdk/issues/62)) ([65136a3](https://github.com/launchdarkly/python-ai-sdk/commit/65136a3aa07d1245b28c38a7f38946a3c75a4516))
* gate the judge explanation on capture_content ([8a66365](https://github.com/launchdarkly/python-ai-sdk/commit/8a66365059386b637d43c96c56b659867decb047))
* **openai-messages:** emit invoke_agent, chat and execute_tool spans ([17ad375](https://github.com/launchdarkly/python-ai-sdk/commit/17ad3751d8ef286826140c86116d1c978f3bf264))
* record judge scores as gen_ai.evaluation.result ([#43](https://github.com/launchdarkly/python-ai-sdk/issues/43)) ([7c8b0c9](https://github.com/launchdarkly/python-ai-sdk/commit/7c8b0c9820655c4a909366ad05d20c78050c4d17))


### Bug Fixes

* **openai-messages:** close the tool span a cancelled tool leaves open ([35a254d](https://github.com/launchdarkly/python-ai-sdk/commit/35a254d0660a84c0eff606bd907efdc922635990))
* **openai-messages:** drop output items that produced no content ([66bacd0](https://github.com/launchdarkly/python-ai-sdk/commit/66bacd06c0eaf0a0b13d39c55924b7266730e7ad))
* **openai-messages:** fail the streaming chat span, and keep its tokens ([83f8684](https://github.com/launchdarkly/python-ai-sdk/commit/83f8684af2a01ab3111b9eea2cfc4aa6a3df5b26))
* **openai-messages:** forward capture_content from the convenience wrapper ([495391f](https://github.com/launchdarkly/python-ai-sdk/commit/495391f2acd5752de3fe232162fc2b8936f6376f))
* **openai-messages:** keep every chat-span write inside the guard that ends it ([a4fed91](https://github.com/launchdarkly/python-ai-sdk/commit/a4fed91da4c6d2239efef0f6c5ff8fedf0526c23))
* **openai-messages:** keep the spans of a cancelled run ([c93a902](https://github.com/launchdarkly/python-ai-sdk/commit/c93a902a7d077adb50e561b5ebb01fd69b5335b2))
* **openai-messages:** keep the tokens a turn already cost when its content fails ([8501379](https://github.com/launchdarkly/python-ai-sdk/commit/85013797c826d93247ac3dedc236d85f1f20321e))
* **openai-messages:** record a tool result inside the guard that ends its span ([6b9813b](https://github.com/launchdarkly/python-ai-sdk/commit/6b9813bfee06cd647321ca8cafa4d665c0406efd))
* **openai-messages:** report a cancelled stream as cancelled, not abandoned ([76c4cef](https://github.com/launchdarkly/python-ai-sdk/commit/76c4cefc7b6e476b91540b0e3397bdba2f384a23))
* **openai-messages:** report tool arguments as an object, not a JSON string ([a6b9da9](https://github.com/launchdarkly/python-ai-sdk/commit/a6b9da9e0b268fa47d8c0b6dbf92de066ede3eaa))


### Documentation

* describe the span tree the SDK now emits ([bc7815a](https://github.com/launchdarkly/python-ai-sdk/commit/bc7815a61e46c96df9cd786b20bfceb7d8b13334))
* describe the span tree the SDK now emits ([#37](https://github.com/launchdarkly/python-ai-sdk/issues/37)) ([6a82ef5](https://github.com/launchdarkly/python-ai-sdk/commit/6a82ef5d1fdbae2b1228f9325d960979bcf50881))

## [0.1.4](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-messages-0.1.3...launchdarkly-ai-openai-messages-0.1.4) (2026-08-07)


### Bug Fixes

* update module docstrings across all packages ([1c53ea3](https://github.com/launchdarkly/python-ai-sdk/commit/1c53ea3c248c5d3b4f5a553ae71d9a6fa9144bcc))
* update module docstrings across all packages ([#24](https://github.com/launchdarkly/python-ai-sdk/issues/24)) ([3b5f82b](https://github.com/launchdarkly/python-ai-sdk/commit/3b5f82b7b65eee735fee5a49d28c8bd6a7feb3dc))

## [0.1.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-messages-0.1.2...launchdarkly-ai-openai-messages-0.1.3) (2026-08-07)


### Bug Fixes

* add missing contents:read permission to release jobs ([b33d50f](https://github.com/launchdarkly/python-ai-sdk/commit/b33d50f2d6726b88d6515c9e9d334ae11cb5e159))
* add missing contents:read permission to release jobs ([#21](https://github.com/launchdarkly/python-ai-sdk/issues/21)) ([da72e1a](https://github.com/launchdarkly/python-ai-sdk/commit/da72e1a079b02a14e62d615b53cb08e2b44e7ba8))

## [0.1.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-openai-messages-0.1.1...launchdarkly-ai-openai-messages-0.1.2) (2026-08-07)


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
