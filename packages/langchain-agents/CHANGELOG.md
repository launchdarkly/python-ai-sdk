# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-agents-0.1.4...launchdarkly-ai-langchain-agents-0.2.0) (2026-09-03)


### ⚠ BREAKING CHANGES

* **langchain-agents:** the span this handler emits is renamed from `langchain.agent` and `langchain.agent.stream` to `invoke_agent`. Queries selecting on the old names will not match. Prompt and completion content is no longer on spans unless the caller passes capture_content=True. gen_ai.system changes value; see below.

### Features

* emit $ld:ai:sdk:info event per AI package ([#62](https://github.com/launchdarkly/python-ai-sdk/issues/62)) ([65136a3](https://github.com/launchdarkly/python-ai-sdk/commit/65136a3aa07d1245b28c38a7f38946a3c75a4516))
* gate the judge explanation on capture_content ([8a66365](https://github.com/launchdarkly/python-ai-sdk/commit/8a66365059386b637d43c96c56b659867decb047))
* **langchain-agents:** emit invoke_agent, chat and execute_tool spans ([7683c9d](https://github.com/launchdarkly/python-ai-sdk/commit/7683c9df9cae4f9f4a7f2a5cdb0b3af07d322721))
* record judge scores as gen_ai.evaluation.result ([#43](https://github.com/launchdarkly/python-ai-sdk/issues/43)) ([7c8b0c9](https://github.com/launchdarkly/python-ai-sdk/commit/7c8b0c9820655c4a909366ad05d20c78050c4d17))


### Bug Fixes

* **langchain-agents:** a callback that raises must not leave a span unreachable ([2169a05](https://github.com/launchdarkly/python-ai-sdk/commit/2169a05147511bc72e0f1ab81fa9f6fbeb65ad2a))
* **langchain-agents:** a reported zero token count is not a missing one ([0d5e38c](https://github.com/launchdarkly/python-ai-sdk/commit/0d5e38c1d4ed44dbca0e2729d72314e6e4483647))
* **langchain-agents:** an abandoned stream marks its open spans, it does not fail them ([d087d73](https://github.com/launchdarkly/python-ai-sdk/commit/d087d737382b6dd3660aa0a32ff36bb5a5be57f5))
* **langchain-agents:** forward capture_content from the convenience wrapper ([06b210d](https://github.com/launchdarkly/python-ai-sdk/commit/06b210d0bab68e72b2734ea07c2a980af8b8c5e8))
* **langchain-agents:** keep a turn's billed tokens when its content write fails ([485db67](https://github.com/launchdarkly/python-ai-sdk/commit/485db678ed9769a2f9750f0e2736040e15167218))
* **langchain-agents:** keep the spans of a cancelled run ([ff53cc6](https://github.com/launchdarkly/python-ai-sdk/commit/ff53cc6d22c103ca524734839e7470ee5c6ee874))
* **langchain-agents:** make a successful run's total agree with its own spans ([6333808](https://github.com/launchdarkly/python-ai-sdk/commit/6333808dea92c312643deceef2cabbf0a18d1c45))
* **langchain-agents:** put the input content write inside the guard too ([0fa4ed1](https://github.com/launchdarkly/python-ai-sdk/commit/0fa4ed1c1a827315aa7f554cfea9279afde5d27c))
* **langchain-agents:** record the root completion the way the chat span does ([1f4242b](https://github.com/launchdarkly/python-ai-sdk/commit/1f4242bf6baed11349466f4bce7df30e997d6558))
* **langchain-agents:** report a cancelled stream as cancelled, not abandoned ([010a9ca](https://github.com/launchdarkly/python-ai-sdk/commit/010a9ca3f760b9cd8ed708a4e8be3e4a5142a7e5))
* **langchain-agents:** total the streaming root from the callbacks when the messages report nothing ([6ff5a94](https://github.com/launchdarkly/python-ai-sdk/commit/6ff5a94586b889d205794abda60d44a646b5967f))
* **langchain-agents:** track the chat span before writing content that can raise ([883da7b](https://github.com/launchdarkly/python-ai-sdk/commit/883da7b6ba09faf279a06d9e5d047b4cc8ec83a4))


### Documentation

* describe the span tree the SDK now emits ([bc7815a](https://github.com/launchdarkly/python-ai-sdk/commit/bc7815a61e46c96df9cd786b20bfceb7d8b13334))
* describe the span tree the SDK now emits ([#37](https://github.com/launchdarkly/python-ai-sdk/issues/37)) ([6a82ef5](https://github.com/launchdarkly/python-ai-sdk/commit/6a82ef5d1fdbae2b1228f9325d960979bcf50881))

## [0.1.4](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-agents-0.1.3...launchdarkly-ai-langchain-agents-0.1.4) (2026-08-07)


### Bug Fixes

* update module docstrings across all packages ([1c53ea3](https://github.com/launchdarkly/python-ai-sdk/commit/1c53ea3c248c5d3b4f5a553ae71d9a6fa9144bcc))
* update module docstrings across all packages ([#24](https://github.com/launchdarkly/python-ai-sdk/issues/24)) ([3b5f82b](https://github.com/launchdarkly/python-ai-sdk/commit/3b5f82b7b65eee735fee5a49d28c8bd6a7feb3dc))

## [0.1.3](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-agents-0.1.2...launchdarkly-ai-langchain-agents-0.1.3) (2026-08-07)


### Bug Fixes

* add missing contents:read permission to release jobs ([b33d50f](https://github.com/launchdarkly/python-ai-sdk/commit/b33d50f2d6726b88d6515c9e9d334ae11cb5e159))
* add missing contents:read permission to release jobs ([#21](https://github.com/launchdarkly/python-ai-sdk/issues/21)) ([da72e1a](https://github.com/launchdarkly/python-ai-sdk/commit/da72e1a079b02a14e62d615b53cb08e2b44e7ba8))

## [0.1.2](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-langchain-agents-0.1.1...launchdarkly-ai-langchain-agents-0.1.2) (2026-08-07)


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
