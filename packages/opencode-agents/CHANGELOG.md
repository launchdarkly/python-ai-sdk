# Changelog

## [0.3.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-opencode-agents-0.2.0...launchdarkly-ai-opencode-agents-0.3.0) (2026-10-27)


### ⚠ BREAKING CHANGES

* **opencode-agents:** honor AI Config tools ([#216](https://github.com/launchdarkly/python-ai-sdk/issues/216))
* handlers now forward the AI Config's model parameters to the provider, limited to each handler's list in TESTING.md 1.12. Configs that set parameters which used to be ignored now take effect.

### Experimental

* **evaluations:** normalize text behaviors ([7cee8c8](https://github.com/launchdarkly/python-ai-sdk/commit/7cee8c8667ecfa2ec4b939accbcb1c633071e81a))


### Features

* **opencode-agents:** honor AI Config tools ([#216](https://github.com/launchdarkly/python-ai-sdk/issues/216)) ([b5a63ec](https://github.com/launchdarkly/python-ai-sdk/commit/b5a63ecc9fb45ae2c65e89bc3a57167bce2ce14c))


### Bug Fixes

* **client:** show a judge an output of 0 or false, and render non-string values as JSON ([7cee8c8](https://github.com/launchdarkly/python-ai-sdk/commit/7cee8c8667ecfa2ec4b939accbcb1c633071e81a))
* pass AI Config model parameters through to every provider handler ([#107](https://github.com/launchdarkly/python-ai-sdk/issues/107)) ([e3afb3f](https://github.com/launchdarkly/python-ai-sdk/commit/e3afb3f2c19ebf7085f84e94cd5a53f2bbe929ae))

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-opencode-agents-0.1.0...launchdarkly-ai-opencode-agents-0.2.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* **opencode:** start and stop opencode serve in the handler ([#198](https://github.com/launchdarkly/python-ai-sdk/issues/198))

### Features

* **opencode-agents:** add an OpenCode handler ([#159](https://github.com/launchdarkly/python-ai-sdk/issues/159)) ([fc08e8c](https://github.com/launchdarkly/python-ai-sdk/commit/fc08e8c383bfb6e5e505b7a07d57f5bd9dd79a11))
* **opencode-agents:** emit the AG-UI text and tool subset ([#162](https://github.com/launchdarkly/python-ai-sdk/issues/162)) ([7677621](https://github.com/launchdarkly/python-ai-sdk/commit/7677621a88ba433a2117b1b5a007631cbf7d74d0))
* **opencode-agents:** open one OpenCode session per call ([#179](https://github.com/launchdarkly/python-ai-sdk/issues/179)) ([b225113](https://github.com/launchdarkly/python-ai-sdk/commit/b225113d66bbc980cb72c3007620bca061ab6035))
* **opencode-agents:** render agent instruction templates ([#182](https://github.com/launchdarkly/python-ai-sdk/issues/182)) ([83175aa](https://github.com/launchdarkly/python-ai-sdk/commit/83175aa680dc18697ddcc1d17863cdfdaf08d7e6))
* **opencode:** start and stop opencode serve in the handler ([#198](https://github.com/launchdarkly/python-ai-sdk/issues/198)) ([0748dcf](https://github.com/launchdarkly/python-ai-sdk/commit/0748dcf209076f98cffa38fc59d1987293d5015f))


### Bug Fixes

* **opencode-agents:** own the HTTP client ([#194](https://github.com/launchdarkly/python-ai-sdk/issues/194)) ([ed2d419](https://github.com/launchdarkly/python-ai-sdk/commit/ed2d41914ee3f843fe087d754914cd4f2ae1dac8))
* **opencode-agents:** skip judges and record structured history ([#177](https://github.com/launchdarkly/python-ai-sdk/issues/177)) ([6fadf1d](https://github.com/launchdarkly/python-ai-sdk/commit/6fadf1da3abc49c0a82ffc4c2532484aceac8d76))

## Changelog

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
