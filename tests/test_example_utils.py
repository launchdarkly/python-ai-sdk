import re

from examples.utils import new_multi_context


def test_new_multi_context_is_complex_and_unique() -> None:
    first = new_multi_context()
    second = new_multi_context()

    assert first["kind"] == "multi"
    assert first["organization"] == {"key": "example-org:west%region"}
    assert re.fullmatch(r"example-user-[0-9a-f]{8}", first["user"]["key"])
    assert second["user"]["key"] != first["user"]["key"]
