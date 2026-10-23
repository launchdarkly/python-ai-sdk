# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-litellm-agents-0.1.1...launchdarkly-ai-litellm-agents-0.2.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* handlers now forward the AI Config's model parameters to the provider, limited to each handler's list in TESTING.md 1.12. Configs that set parameters which used to be ignored now take effect.

### Bug Fixes

* pass AI Config model parameters through to every provider handler ([#107](https://github.com/launchdarkly/python-ai-sdk/issues/107)) ([e3afb3f](https://github.com/launchdarkly/python-ai-sdk/commit/e3afb3f2c19ebf7085f84e94cd5a53f2bbe929ae))

## [0.1.1](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-litellm-agents-0.1.0...launchdarkly-ai-litellm-agents-0.1.1) (2026-10-23)


### Features

* add LiteLLM messages and agents integrations ([#102](https://github.com/launchdarkly/python-ai-sdk/issues/102)) ([f2e9886](https://github.com/launchdarkly/python-ai-sdk/commit/f2e9886194d701f635b0c5074c6b5b6dd1427a73))
