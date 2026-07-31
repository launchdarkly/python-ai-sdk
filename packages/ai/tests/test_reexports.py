"""
Tests for §5 launchdarkly-ai re-export barrel.
Reference: TESTING.md §5
"""

import importlib


class TestReexports:
    def test_all_named_exports_from_server_are_reexported(self) -> None:
        server = importlib.import_module("launchdarkly_ai_server")
        barrel = importlib.import_module("launchdarkly_ai")

        for name in server.__all__:
            assert hasattr(barrel, name), (
                f"launchdarkly_ai is missing re-export: {name!r}"
            )

    def test_reexported_values_are_identical_references(self) -> None:
        server = importlib.import_module("launchdarkly_ai_server")
        barrel = importlib.import_module("launchdarkly_ai")

        for name in server.__all__:
            server_obj = getattr(server, name)
            barrel_obj = getattr(barrel, name)
            assert server_obj is barrel_obj, (
                f"launchdarkly_ai.{name} is not the same object as launchdarkly_ai_server.{name}"
            )
