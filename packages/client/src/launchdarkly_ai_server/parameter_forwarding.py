"""
Shared filtering for ``model.parameters`` before it reaches a provider SDK call.

``model.parameters`` is a free-form dict the LaunchDarkly UI writes, offering keys that make
sense across providers (``temperature``, ``top_p``, ``max_tokens``, ``tool_choice``, ...). No
single provider SDK accepts all of them, and each handler package classifies every key its own
provider entry point accepts into exactly one written-down list: forwarded (model and run settings
only), handler-owned (set by the call site itself), or excluded (accepted by the provider but never
forwarded: credentials, endpoints and connection settings, raw request injection such as extra
headers or ``model_kwargs``, remote tools, host-process settings, and anything that would break
the handler). Only the forwarded list is read at runtime; every other key is dropped. Those lists are
literal, next to each handler's own call sites, not derived from the provider SDK at runtime: a
hand-maintained list is reviewable and a runtime-derived one is not, and each handler package has a
drift test asserting its lists still cover everything its provider SDK accepts.

This module is the one shared piece: given a params dict and the literal forwarded-keys list, keep
only the keys on that list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def select_forwarded_parameters(
    params: Mapping[str, Any], forwarded_keys: frozenset[str]
) -> dict[str, Any]:
    """Returns the subset of *params* whose key is in *forwarded_keys*.

    *forwarded_keys* is a handler's literal, hand-maintained list of the keys it forwards to its
    provider call: never the provider's full accept-set, since a handler's own owned and excluded
    keys (including client/connection configuration) are already left off that list. A key present
    in *params* but not in *forwarded_keys* is silently dropped rather than raising, matching this
    SDK's behaviour before ``model.parameters`` forwarding existed: an unrecognised tuning value is
    ignored, not a hard failure.
    """
    return {k: v for k, v in params.items() if k in forwarded_keys}
