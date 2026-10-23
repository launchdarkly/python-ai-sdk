# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-vercel-agents-0.1.1...launchdarkly-ai-vercel-agents-0.2.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* handlers now forward the AI Config's model parameters to the provider, limited to each handler's list in TESTING.md 1.12. Configs that set parameters which used to be ignored now take effect.

### Bug Fixes

* pass AI Config model parameters through to every provider handler ([#107](https://github.com/launchdarkly/python-ai-sdk/issues/107)) ([e3afb3f](https://github.com/launchdarkly/python-ai-sdk/commit/e3afb3f2c19ebf7085f84e94cd5a53f2bbe929ae))

## [0.1.1](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-vercel-agents-0.1.0...launchdarkly-ai-vercel-agents-0.1.1) (2026-10-23)


### Features

* **AIC-3408:** add Vercel AI SDK messages and agents adapters ([#103](https://github.com/launchdarkly/python-ai-sdk/issues/103)) ([64d679e](https://github.com/launchdarkly/python-ai-sdk/commit/64d679e6dbf7125a08a4728a1acb5bbac3a2f39e))
