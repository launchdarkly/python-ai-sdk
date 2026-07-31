"""
Tests for §3.12 resolve_graph() and graph().
Reference: TESTING.md §3.12
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, graph, resolve_graph

CONTEXT = {"kind": "user", "key": "u1"}


def _make_client(graph_variation: dict | None = None) -> MagicMock:
    c = MagicMock()
    c.track = MagicMock()
    c.flush = AsyncMock()
    c.close = AsyncMock()

    node_variation = {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "Be helpful.",
        "_ldMeta": {
            "enabled": True,
            "variationKey": "v1",
            "version": 1,
            "mode": "messages",
        },
    }

    graph_var = graph_variation or {
        "root": "root-node",
        "edges": {"root-node": [{"key": "leaf-node"}]},
    }

    async def variation_side_effect(key: str, ctx: dict, default: Any) -> Any:
        if key == "graph-key":
            return graph_var
        return node_variation

    c.variation = AsyncMock(side_effect=variation_side_effect)
    return c


def _make_handler(response: str = "ok") -> ProviderHandler:
    async def fn(config, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": response, "usage": {"input_tokens": 1, "output_tokens": 1}}

    return ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]


@pytest.fixture
def mock_ld_client() -> MagicMock:
    client = _make_client()
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


# ---------------------------------------------------------------------------
# resolve_graph
# ---------------------------------------------------------------------------


class TestResolveGraph:
    async def test_returns_graph_definition_enabled_true(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        assert gd.enabled is True

    async def test_returns_enabled_false_without_root(
        self, mock_ld_client: MagicMock
    ) -> None:
        mock_ld_client.variation = AsyncMock(return_value={"edges": {}})
        gd = await resolve_graph("graph-key", context=CONTEXT)
        assert gd.enabled is False

    async def test_get_node_returns_correct_node(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        node = gd.get_node("root-node")
        assert node is not None
        assert node.key == "root-node"

    async def test_get_child_nodes_returns_correct_children(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        children = gd.get_child_nodes("root-node")
        assert len(children) == 1
        assert children[0].key == "leaf-node"

    async def test_get_parent_nodes_returns_correct_parents(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        parents = gd.get_parent_nodes("leaf-node")
        assert len(parents) == 1
        assert parents[0].key == "root-node"

    async def test_terminal_nodes_returns_leaf_nodes(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        terminals = gd.terminal_nodes()
        assert any(n.key == "leaf-node" for n in terminals)

    async def test_is_terminal_correct_per_node(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        assert gd.is_terminal("leaf-node") is True
        assert gd.is_terminal("root-node") is False

    async def test_edges_from_returns_correct_edges(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        edges = gd.edges_from("root-node")
        assert len(edges) == 1
        assert edges[0].target_key == "leaf-node"

    async def test_disabled_node_disables_whole_graph(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def failing_variation(key: str, ctx: dict, default: Any) -> Any:
            if key == "graph-key":
                return {
                    "root": "root-node",
                    "edges": {"root-node": [{"key": "bad-node"}]},
                }
            if key == "bad-node":
                return None  # disabled
            return {
                "model": {"name": "gpt-4"},
                "provider": {"name": "TestProvider"},
                "instructions": "hi",
                "_ldMeta": {"enabled": True, "variationKey": "v1", "version": 1},
            }

        mock_ld_client.variation = AsyncMock(side_effect=failing_variation)
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        assert gd.enabled is False


# ---------------------------------------------------------------------------
# graph().invoke()
# ---------------------------------------------------------------------------


class TestGraphInvoke:
    async def test_throws_when_graph_disabled(self, mock_ld_client: MagicMock) -> None:
        mock_ld_client.variation = AsyncMock(return_value={"edges": {}})
        g = graph("graph-key", handlers=[_make_handler()])
        with pytest.raises((ValueError, RuntimeError), match="disabled"):
            await g.invoke("hi", CONTEXT)

    async def test_throws_when_no_handlers_supplied(
        self, mock_ld_client: MagicMock
    ) -> None:
        g = graph("graph-key")
        with pytest.raises((ValueError, RuntimeError)):
            await g.invoke("hi", CONTEXT)

    async def test_traverses_root_leaf_returns_aggregated_result(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler("leaf-result")
        g = graph("graph-key", handlers=[h])
        result = await g.invoke("hi", CONTEXT)
        assert result.response is not None

    async def test_graph_duration_total_tracked(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler()
        g = graph("graph-key", handlers=[h])
        await g.invoke("hi", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:graph:duration:total" in events

    async def test_graph_invocation_success_tracked(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler()
        g = graph("graph-key", handlers=[h])
        await g.invoke("hi", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:graph:invocation_success" in events

    async def test_graph_invocation_failure_tracked_on_error(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def bad_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            raise RuntimeError("fail")

        h = ProviderHandler(fn=bad_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        g = graph("graph-key", handlers=[h])
        with pytest.raises(RuntimeError):
            await g.invoke("hi", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:graph:invocation_failure" in events

    async def test_graph_path_tracked(self, mock_ld_client: MagicMock) -> None:
        g = graph("graph-key", handlers=[_make_handler()])
        await g.invoke("hi", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:graph:path" in events

    async def test_node_variations_resolved_only_once(
        self, mock_ld_client: MagicMock
    ) -> None:
        g = graph("graph-key", handlers=[_make_handler()])
        await g.invoke("hi", CONTEXT)
        await g.invoke("hi again", CONTEXT)
        # Second call should use the cache — variation calls should not double
        assert mock_ld_client.variation.call_count <= 4  # graph + 2 nodes × 1 cache hit

    async def test_handoff_success_tracked_on_each_handoff(
        self, mock_ld_client: MagicMock
    ) -> None:
        # Create a handler that invokes the __handoff_ tool to trigger routing
        async def routing_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            for name, fn in (tool_handlers or {}).items():
                if name.startswith("__handoff_"):
                    fn()  # trigger the handoff
                    break
            return {
                "output": "routed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        h = ProviderHandler(fn=routing_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        g = graph("graph-key", handlers=[h])
        await g.invoke("hi", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:graph:handoff_success" in events

    async def test_cycle_guard_terminates(self, mock_ld_client: MagicMock) -> None:
        """A graph where every node routes back to itself must terminate."""
        cyclic_graph_var = {
            "root": "root-node",
            "edges": {"root-node": [{"key": "root-node"}]},  # self-loop
        }
        client = _make_client(cyclic_graph_var)
        lifecycle_module._set_client_for_testing(client)
        try:
            call_count = 0

            async def counting_fn(
                config, user_input, tool_handlers, variables, history=None
            ) -> dict:  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                # Invoke the handoff to trigger cycle
                for name, fn in (tool_handlers or {}).items():
                    if name.startswith("__handoff_"):
                        fn()
                        break
                return {
                    "output": "cycle",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }

            h = ProviderHandler(
                fn=counting_fn, provides_for=("TestProvider", "messages")
            )  # type: ignore[arg-type]
            g = graph("graph-key", handlers=[h])
            result = await g.invoke("hi", CONTEXT)
            # Must terminate (not loop forever)
            assert result.response is not None
        finally:
            lifecycle_module._reset_for_testing()

    async def test_per_call_variables_isolated(self, mock_ld_client: MagicMock) -> None:
        """Second call with omitted variables must not see first call's variables."""
        received_variables: list[Any] = []

        async def capturing_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_variables.append(variables)
            return {"output": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}

        h = ProviderHandler(fn=capturing_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        g = graph("graph-key", handlers=[h])
        await g.invoke("first", CONTEXT, variables={"x": 1})
        received_variables.clear()
        await g.invoke("second", CONTEXT)
        # Second call's handler should not receive x=1
        assert all(
            vars is None or "x" not in (vars or {}) for vars in received_variables
        )

    async def test_traverse_visits_root_before_leaf(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        visited: list[str] = []

        async def visitor(node: Any, ctx: Any) -> None:
            visited.append(node.key)

        await gd.traverse(visitor)
        assert visited.index("root-node") < visited.index("leaf-node")

    async def test_reverse_traverse_visits_leaf_before_root(
        self, mock_ld_client: MagicMock
    ) -> None:
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        visited: list[str] = []

        async def visitor(node: Any, ctx: Any) -> None:
            visited.append(node.key)

        await gd.reverse_traverse(visitor)
        assert visited.index("leaf-node") < visited.index("root-node")

    async def test_reverse_traverse_key_is_snake_case(
        self, mock_ld_client: MagicMock
    ) -> None:
        """Python GraphDefinition must expose 'reverse_traverse' (snake_case), not
        'reverseTraverse' (camelCase). See TESTING.md Appendix A.2."""
        gd = await resolve_graph(
            "graph-key", context=CONTEXT, handlers=[_make_handler()]
        )
        assert hasattr(gd, "reverse_traverse"), (
            "GraphDefinition must have attribute 'reverse_traverse' (snake_case) in Python"
        )
        assert not hasattr(gd, "reverseTraverse"), (
            "GraphDefinition must not expose camelCase 'reverseTraverse' in Python"
        )

    async def test_disabled_graph_reverse_traverse_key_is_snake_case(
        self, mock_ld_client: MagicMock
    ) -> None:
        """Disabled GraphDefinition stub must also use 'reverse_traverse', not
        'reverseTraverse'. See TESTING.md Appendix A.2."""
        mock_ld_client.variation = AsyncMock(return_value={"edges": {}})
        gd = await resolve_graph("graph-key", context=CONTEXT)
        assert hasattr(gd, "reverse_traverse")
        assert not hasattr(gd, "reverseTraverse")

    async def test_cache_is_bounded(self, mock_ld_client: MagicMock) -> None:
        """GraphInstance._cache must not grow beyond MAX_GRAPH_CACHE_SIZE.
        See TESTING.md §3.11."""
        from launchdarkly_ai_server.graph import MAX_GRAPH_CACHE_SIZE

        g = graph("graph-key", handlers=[_make_handler()])
        # Invoke with more distinct contexts than the max
        limit = MAX_GRAPH_CACHE_SIZE + 10
        for i in range(limit):
            await g.invoke("hi", {"kind": "user", "key": f"u{i}"})
        assert len(g._cache) <= MAX_GRAPH_CACHE_SIZE

    async def test_equal_content_contexts_share_cache_entry(
        self, mock_ld_client: MagicMock
    ) -> None:
        """Two distinct dict objects with identical JSON content must share one
        cache entry. The cache key must be json.dumps(context, sort_keys=True),
        not id(context). See TESTING.md §3.11."""
        g = graph("graph-key", handlers=[_make_handler()])
        ctx_a = {"kind": "user", "key": "u1"}
        ctx_b = {"kind": "user", "key": "u1"}  # distinct object, same content
        assert ctx_a is not ctx_b, "test requires two different dict objects"
        await g.invoke("first call", ctx_a)
        await g.invoke("second call", ctx_b)
        assert len(g._cache) == 1, (
            "Equal-content contexts must share a single cache entry — "
            "the cache key must be json.dumps(context), not id(context)."
        )

    async def test_error_logging_when_node_variation_fails(
        self, mock_ld_client: MagicMock
    ) -> None:
        """When extractVariation throws for a node, the error must be logged."""
        node_error = RuntimeError("node-var-fail")

        async def failing_variation(key: str, ctx: dict, default: Any) -> Any:
            if key == "graph-key":
                return {
                    "root": "root-node",
                    "edges": {"root-node": [{"key": "leaf-node"}]},
                }
            raise node_error

        mock_ld_client.variation = AsyncMock(side_effect=failing_variation)

        with patch("launchdarkly_ai_server.graph.logger") as mock_logger:
            gd = await resolve_graph(
                "graph-key", context=CONTEXT, handlers=[_make_handler()]
            )

        assert gd.enabled is False
        mock_logger.error.assert_called()
