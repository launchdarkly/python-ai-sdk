# Changelog

## [0.2.0](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-typesafe-0.1.1...launchdarkly-ai-typesafe-0.2.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* handlers now forward the AI Config's model parameters to the provider, limited to each handler's list in TESTING.md 1.12. Configs that set parameters which used to be ignored now take effect.

### Bug Fixes

* pass AI Config model parameters through to every provider handler ([#107](https://github.com/launchdarkly/python-ai-sdk/issues/107)) ([e3afb3f](https://github.com/launchdarkly/python-ai-sdk/commit/e3afb3f2c19ebf7085f84e94cd5a53f2bbe929ae))

## [0.1.1](https://github.com/launchdarkly/python-ai-sdk/compare/launchdarkly-ai-typesafe-0.1.0...launchdarkly-ai-typesafe-0.1.1) (2026-10-23)


### Features

* **AIC-3534:** add a TypeSafe Jev judge handler ([#149](https://github.com/launchdarkly/python-ai-sdk/issues/149)) ([1b41ea4](https://github.com/launchdarkly/python-ai-sdk/commit/1b41ea40a13580a5024b3d478e59effcdaf3e9db))

## 0.1.0

* add a TypeSafe Jev handler for judge configs
