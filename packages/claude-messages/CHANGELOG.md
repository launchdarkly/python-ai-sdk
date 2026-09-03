# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-messages-0.1.4...launchdarkly-ai-claude-messages-0.2.0) (2026-09-03)


### ⚠ BREAKING CHANGES

* **claude-messages:** the span this handler emits is renamed from `claude.messages` and `claude.messages.stream` to `invoke_agent`. Queries that select on the old names will not match. Prompt and completion content is no longer on spans unless the caller passes capture_content=True.

### Features

* **claude-messages:** emit invoke_agent, chat and execute_tool spans ([d4b5b6e](https://github.com/launchdarkly/python-ai-sdk/commit/d4b5b6eccb4714fd78384d78d8c1647c356d4c9c))
* emit $ld:ai:sdk:info event per AI package ([#62](https://github.com/launchdarkly/python-ai-sdk/issues/62)) ([65136a3](https://github.com/launchdarkly/python-ai-sdk/commit/65136a3aa07d1245b28c38a7f38946a3c75a4516))
* gate the judge explanation on capture_content ([8a66365](https://github.com/launchdarkly/python-ai-sdk/commit/8a66365059386b637d43c96c56b659867decb047))
* record judge scores as gen_ai.evaluation.result ([#43](https://github.com/launchdarkly/python-ai-sdk/issues/43)) ([7c8b0c9](https://github.com/launchdarkly/python-ai-sdk/commit/7c8b0c9820655c4a909366ad05d20c78050c4d17))


### Bug Fixes

* **claude-messages:** close the tool span a cancelled tool leaves open ([4f23cf4](https://github.com/launchdarkly/python-ai-sdk/commit/4f23cf439c449f33e13fce6b78681233f85f9ba2))
* **claude-messages:** do not invent the cache figures Anthropic did not send ([db417b1](https://github.com/launchdarkly/python-ai-sdk/commit/db417b1ce2d47aef4ffa10f0bf2ac4056e2d008a))
* **claude-messages:** fail the streaming chat span instead of abandoning it ([2f0e565](https://github.com/launchdarkly/python-ai-sdk/commit/2f0e565230c5130f0dc6206c634284ad146a74b3))
* **claude-messages:** forward capture_content from the convenience wrapper ([b7531d3](https://github.com/launchdarkly/python-ai-sdk/commit/b7531d39ff3a60337c8a3552c34b9428bdfed573))
* **claude-messages:** guard the streaming chat span's input write ([6e358dd](https://github.com/launchdarkly/python-ai-sdk/commit/6e358ddf7e5ac134737a590655ff6663063c7201))
* **claude-messages:** keep every blocking chat-span write inside the guard ([21fb591](https://github.com/launchdarkly/python-ai-sdk/commit/21fb5918ee0508dd1dbc658c9fb212723e64ca81))
* **claude-messages:** keep the spans of a cancelled run ([bb2e52f](https://github.com/launchdarkly/python-ai-sdk/commit/bb2e52f13f96d40d080ce51cf4025189bb83d3ac))
* **claude-messages:** keep the streaming turn's billed tokens on a content failure ([1933374](https://github.com/launchdarkly/python-ai-sdk/commit/19333749371fd10551a76b2bae19f7aa1e4b6ccb))
* **claude-messages:** put the input content write inside the guard too ([4aaa497](https://github.com/launchdarkly/python-ai-sdk/commit/4aaa497250db5c19c25ade189298418050347eae))
* **claude-messages:** record a tool result inside the guard that ends its span ([19d932e](https://github.com/launchdarkly/python-ai-sdk/commit/19d932ebffeb23e2059a407b35a7acd9688a27a1))
* **claude-messages:** report a cancelled stream as cancelled, not abandoned ([2d3379a](https://github.com/launchdarkly/python-ai-sdk/commit/2d3379ab2085b0dce53bda4a9f729bd50918b47d))


### Documentation

* describe the span tree the SDK now emits ([bc7815a](https://github.com/launchdarkly/python-ai-sdk/commit/bc7815a61e46c96df9cd786b20bfceb7d8b13334))
* describe the span tree the SDK now emits ([#37](https://github.com/launchdarkly/python-ai-sdk/issues/37)) ([6a82ef5](https://github.com/launchdarkly/python-ai-sdk/commit/6a82ef5d1fdbae2b1228f9325d960979bcf50881))

## [0.1.4](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-messages-0.1.3...launchdarkly-ai-claude-messages-0.1.4) (2026-08-07)


### Bug Fixes

* update module docstrings across all packages ([1c53ea3](https://github.com/launchdarkly/python-ai-sdk/commit/1c53ea3c248c5d3b4f5a553ae71d9a6fa9144bcc))
* update module docstrings across all packages ([#24](https://github.com/launchdarkly/python-ai-sdk/issues/24)) ([3b5f82b](https://github.com/launchdarkly/python-ai-sdk/commit/3b5f82b7b65eee735fee5a49d28c8bd6a7feb3dc))

## [0.1.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-messages-0.1.2...launchdarkly-ai-claude-messages-0.1.3) (2026-08-07)


### Bug Fixes

* add missing contents:read permission to release jobs ([b33d50f](https://github.com/launchdarkly/python-ai-sdk/commit/b33d50f2d6726b88d6515c9e9d334ae11cb5e159))
* add missing contents:read permission to release jobs ([#21](https://github.com/launchdarkly/python-ai-sdk/issues/21)) ([da72e1a](https://github.com/launchdarkly/python-ai-sdk/commit/da72e1a079b02a14e62d615b53cb08e2b44e7ba8))

## [0.1.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-claude-messages-0.1.1...launchdarkly-ai-claude-messages-0.1.2) (2026-08-07)


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
