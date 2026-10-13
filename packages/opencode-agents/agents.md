# Agent Guide — `launchdarkly-ai-opencode-agents`

```python
provides_for = ('*', 'agent')
```

The handler reduces OpenCode session events the caller supplies. It does not import or construct `anomalyco/opencode-sdk-python`, and it does not start an OpenCode process.

`create_opencode_handler(*, capture_content=None, events=None)` returns a `ProviderHandler`. `events` is called per invocation and returns an async iterable of bus events (`message.part.delta`, `message.part.updated`, `message.updated`, `session.error`).

Assistant text is emitted only after `message.updated` marks that message `role == "assistant"`. A part snapshot is the full text of that part so far; a later delta appends. The public stream is `chunk` then `done`.
