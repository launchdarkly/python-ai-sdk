# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-messages-0.1.4...launchdarkly-ai-langchain-messages-0.2.0) (2026-09-03)


### ⚠ BREAKING CHANGES

* **langchain-messages:** the span this handler emits is renamed from `langchain.invoke` and `langchain.stream` to `invoke_agent`. Queries selecting on the old names will not match. Prompt and completion content is no longer on spans unless the caller passes capture_content=True. gen_ai.system changes value; see below.

### Features

* emit $ld:ai:sdk:info event per AI package ([#62](https://github.com/launchdarkly/python-ai-sdk/issues/62)) ([65136a3](https://github.com/launchdarkly/python-ai-sdk/commit/65136a3aa07d1245b28c38a7f38946a3c75a4516))
* gate the judge explanation on capture_content ([8a66365](https://github.com/launchdarkly/python-ai-sdk/commit/8a66365059386b637d43c96c56b659867decb047))
* **langchain-messages:** emit invoke_agent, chat and execute_tool spans ([51032b5](https://github.com/launchdarkly/python-ai-sdk/commit/51032b5f004d0977d4e38971822b38f28eb3c6b4))
* record judge scores as gen_ai.evaluation.result ([#43](https://github.com/launchdarkly/python-ai-sdk/issues/43)) ([7c8b0c9](https://github.com/launchdarkly/python-ai-sdk/commit/7c8b0c9820655c4a909366ad05d20c78050c4d17))


### Bug Fixes

* **langchain-messages:** close the tool span a cancelled tool leaves open ([53c7f9a](https://github.com/launchdarkly/python-ai-sdk/commit/53c7f9a65006f06217e69830f2c7cf30a6e20eec))
* **langchain-messages:** do not serialise the output nobody asked for ([0c9a5dd](https://github.com/launchdarkly/python-ai-sdk/commit/0c9a5ddba5b0bd2be4bae9bc36aa43351de7d225))
* **langchain-messages:** fail the streaming chat span, and keep its tokens ([b084536](https://github.com/launchdarkly/python-ai-sdk/commit/b084536655b27799b957a595615b44bf9b4e2b5e))
* **langchain-messages:** forward capture_content from the convenience wrapper ([79d8030](https://github.com/launchdarkly/python-ai-sdk/commit/79d8030b583f02ecb1e1d7a5d64cedbb266443be))
* **langchain-messages:** guard the per-turn chat span's input write too ([2b901b4](https://github.com/launchdarkly/python-ai-sdk/commit/2b901b4349b34d276018d544a0b4c041ad23f4bb))
* **langchain-messages:** guard the streaming root's output write ([178891a](https://github.com/launchdarkly/python-ai-sdk/commit/178891a337cf12abfdc7f931ff1615d10821b54d))
* **langchain-messages:** keep every structured-turn write inside the guard ([de76f8b](https://github.com/launchdarkly/python-ai-sdk/commit/de76f8be4621abed38112a99fc3a2186a54bfffa))
* **langchain-messages:** keep the spans of a cancelled run ([9c30fb3](https://github.com/launchdarkly/python-ai-sdk/commit/9c30fb377499f0f9218ec134c124e42e16118a19))
* **langchain-messages:** one structured strategy, and don't report zeros as spend ([d7129d4](https://github.com/launchdarkly/python-ai-sdk/commit/d7129d44363a3ce2c41c2d3d0cd276ca8553538c))
* **langchain-messages:** put the input content write inside the guard too ([a68fdff](https://github.com/launchdarkly/python-ai-sdk/commit/a68fdff9271565f9fa2dbc4bbb22691619588c5b))
* **langchain-messages:** record a tool result inside the guard that ends its span ([72c7481](https://github.com/launchdarkly/python-ai-sdk/commit/72c7481d9fbde2d5bfc2f7dad7ce24593be12675))
* **langchain-messages:** report a cancelled stream as cancelled, not abandoned ([9b8e10d](https://github.com/launchdarkly/python-ai-sdk/commit/9b8e10dafe6328505e5273ff16464dfd5e8141ab))
* **langchain-messages:** two more ways a run misreported itself ([ad90d70](https://github.com/launchdarkly/python-ai-sdk/commit/ad90d709c36357ce4183503e074cba4269ead9d2))
* **langchain-messages:** two ways a run could vanish from its own trace ([b8bdfaf](https://github.com/launchdarkly/python-ai-sdk/commit/b8bdfaf6813e33b4e7fc25c6b2930ffd1371458d))


### Documentation

* describe the span tree the SDK now emits ([bc7815a](https://github.com/launchdarkly/python-ai-sdk/commit/bc7815a61e46c96df9cd786b20bfceb7d8b13334))
* describe the span tree the SDK now emits ([#37](https://github.com/launchdarkly/python-ai-sdk/issues/37)) ([6a82ef5](https://github.com/launchdarkly/python-ai-sdk/commit/6a82ef5d1fdbae2b1228f9325d960979bcf50881))

## [0.1.4](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-messages-0.1.3...launchdarkly-ai-langchain-messages-0.1.4) (2026-08-07)


### Bug Fixes

* update module docstrings across all packages ([1c53ea3](https://github.com/launchdarkly/python-ai-sdk/commit/1c53ea3c248c5d3b4f5a553ae71d9a6fa9144bcc))
* update module docstrings across all packages ([#24](https://github.com/launchdarkly/python-ai-sdk/issues/24)) ([3b5f82b](https://github.com/launchdarkly/python-ai-sdk/commit/3b5f82b7b65eee735fee5a49d28c8bd6a7feb3dc))

## [0.1.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-messages-0.1.2...launchdarkly-ai-langchain-messages-0.1.3) (2026-08-07)


### Bug Fixes

* add missing contents:read permission to release jobs ([b33d50f](https://github.com/launchdarkly/python-ai-sdk/commit/b33d50f2d6726b88d6515c9e9d334ae11cb5e159))
* add missing contents:read permission to release jobs ([#21](https://github.com/launchdarkly/python-ai-sdk/issues/21)) ([da72e1a](https://github.com/launchdarkly/python-ai-sdk/commit/da72e1a079b02a14e62d615b53cb08e2b44e7ba8))

## [0.1.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-messages-0.1.1...launchdarkly-ai-langchain-messages-0.1.2) (2026-08-07)


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
