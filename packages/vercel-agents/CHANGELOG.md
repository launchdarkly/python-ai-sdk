# Changelog

## [1.0.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-vercel-agents-0.2.0...launchdarkly-ai-vercel-agents-1.0.0) (2026-10-30)


### ⚠ BREAKING CHANGES

* **AIC-3598:** make Python packaging and public types GA-ready ([#218](https://github.com/launchdarkly/python-ai-sdk/issues/218))
* **AIC-3568:** give the barrel core's __all__, and drop private-spec references ([#234](https://github.com/launchdarkly/python-ai-sdk/issues/234))

### Features

* **client:** add result variant and value to the feature_flag span event ([#240](https://github.com/launchdarkly/python-ai-sdk/issues/240)) ([040d487](https://github.com/launchdarkly/python-ai-sdk/commit/040d487b49f159ddf3ab1a7dcc44d2b74baa7197))


### Bug Fixes

* **AIC-3568:** give the barrel core's __all__, and drop private-spec references ([#234](https://github.com/launchdarkly/python-ai-sdk/issues/234)) ([88e57e6](https://github.com/launchdarkly/python-ai-sdk/commit/88e57e627d0b01b6fe89961e75d9efe08a3d9fdb))
* **AIC-3573:** never send a NativeTool to the model as a function tool ([#248](https://github.com/launchdarkly/python-ai-sdk/issues/248)) ([73caeae](https://github.com/launchdarkly/python-ai-sdk/commit/73caeae69c1eda516e3ae3e144878f2b517e5f61))
* **AIC-3598:** make Python packaging and public types GA-ready ([#218](https://github.com/launchdarkly/python-ai-sdk/issues/218)) ([5860b40](https://github.com/launchdarkly/python-ai-sdk/commit/5860b40078f69febc522680ec76519c25274a606))

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-vercel-agents-0.1.1...launchdarkly-ai-vercel-agents-0.2.0) (2026-10-27)


### ⚠ BREAKING CHANGES

* handlers now forward the AI Config's model parameters to the provider, limited to each handler's list in TESTING.md 1.12. Configs that set parameters which used to be ignored now take effect.

### Bug Fixes

* pass AI Config model parameters through to every provider handler ([#107](https://github.com/launchdarkly/python-ai-sdk/issues/107)) ([e3afb3f](https://github.com/launchdarkly/python-ai-sdk/commit/e3afb3f2c19ebf7085f84e94cd5a53f2bbe929ae))

## [0.1.1](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-vercel-agents-0.1.0...launchdarkly-ai-vercel-agents-0.1.1) (2026-10-23)


### Features

* **AIC-3408:** add Vercel AI SDK messages and agents adapters ([#103](https://github.com/launchdarkly/python-ai-sdk/issues/103)) ([64d679e](https://github.com/launchdarkly/python-ai-sdk/commit/64d679e6dbf7125a08a4728a1acb5bbac3a2f39e))
