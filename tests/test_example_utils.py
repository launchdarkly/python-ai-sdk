from examples.utils import new_multi_context


def test_new_multi_context_is_complex_and_unique() -> None:
    first = new_multi_context()
    second = new_multi_context()

    assert first["kind"] == "multi"
    assert first["organization"] == {"key": "example-org:west%region"}
    assert first["user"]["key"].startswith("example-user-")
    assert len(first["user"]["key"]) == len("example-user-") + 8
    assert second["user"]["key"] != first["user"]["key"]
