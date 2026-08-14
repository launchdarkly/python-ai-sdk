"""Tests for the span usage and identity helpers.

Covers TELEMETRY-CONTRACT.md sections 6 (span lifecycle), 8 (token accounting) and 9 (model
identity).

The load-bearing tests here are the cache-direction ones. Folding cache tokens for a provider that
already includes them, or failing to fold for a provider that does not, produces a number that is
wrong by exactly the cached portion of every call, and nothing else catches it.
"""

from __future__ import annotations

from typing import Any

import pytest

from launchdarkly_ai_server.utils import (
    SpanUsage,
    add_cached_tokens_to_input,
    create_run_usage,
    end_span_once,
    lang_chain_span_usage,
    number_or_zero,
    parse_usage,
    set_model_identity_attributes,
    set_usage_span_attributes,
    to_usage_dict,
)


class FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.ended = 0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self) -> None:
        self.ended += 1


# ─── number_or_zero ──────────────────────────────────────────────────────────


class TestNumberOrZero:
    def test_passes_a_number_through(self) -> None:
        assert number_or_zero(42) == 42

    def test_none_becomes_zero_rather_than_raising(self) -> None:
        # The old bare int(...) raised here, taking the whole call down with it.
        assert number_or_zero(None) == 0

    def test_a_non_numeric_string_becomes_zero(self) -> None:
        assert number_or_zero("abc") == 0

    def test_a_numeric_string_is_read(self) -> None:
        assert number_or_zero("17") == 17

    def test_nan_becomes_zero(self) -> None:
        # NaN is worse than 0: the metric guard tests `> 0`, which is false for NaN, so the metric
        # would be dropped silently rather than reported low.
        assert number_or_zero(float("nan")) == 0

    def test_infinity_becomes_zero(self) -> None:
        assert number_or_zero(float("inf")) == 0
        assert number_or_zero(float("-inf")) == 0

    def test_a_bool_is_not_a_token_count(self) -> None:
        assert number_or_zero(True) == 0

    def test_a_float_truncates(self) -> None:
        assert number_or_zero(3.7) == 3


# ─── parse_usage ─────────────────────────────────────────────────────────────


class TestParseUsage:
    def test_reads_the_snake_case_pair(self) -> None:
        assert parse_usage({"input_tokens": 10, "output_tokens": 5}) == {
            "input": 10,
            "output": 5,
            "total": 15,
        }

    def test_reads_the_camel_case_pair(self) -> None:
        assert parse_usage({"inputTokens": 10, "outputTokens": 5})["total"] == 15

    def test_reads_the_bare_pair(self) -> None:
        assert parse_usage({"input": 10, "output": 5})["total"] == 15

    def test_first_matching_pair_wins(self) -> None:
        result = parse_usage(
            {"input_tokens": 1, "output_tokens": 2, "input": 100, "output": 200}
        )
        assert result["total"] == 3

    def test_never_trusts_a_provider_total(self) -> None:
        result = parse_usage(
            {"input_tokens": 3, "output_tokens": 7, "total_tokens": 999}
        )
        assert result["total"] == 10

    def test_an_unrecognised_bag_returns_zeros(self) -> None:
        assert parse_usage({"foo": 1}) == {"input": 0, "output": 0, "total": 0}

    def test_a_none_field_does_not_raise(self) -> None:
        assert parse_usage({"input_tokens": None, "output_tokens": 5})["input"] == 0

    def test_folds_anthropic_cache_tokens_into_input(self) -> None:
        # Anthropic reports cache beside input, so the real input is the sum of all three.
        result = parse_usage(
            {
                "input_tokens": 3,
                "output_tokens": 10,
                "cache_read_input_tokens": 19971,
                "cache_creation_input_tokens": 3580,
            }
        )
        assert result["input"] == 23554
        assert result["total"] == 23564

    def test_reports_the_cache_breakdown_when_present(self) -> None:
        result = parse_usage(
            {
                "input_tokens": 3,
                "output_tokens": 1,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 20,
            }
        )
        assert result["input_details"] == {
            "uncached": 3,
            "cache_read": 10,
            "cache_creation": 20,
        }

    def test_omits_the_breakdown_when_no_cache_field_is_present(self) -> None:
        assert "input_details" not in parse_usage(
            {"input_tokens": 1, "output_tokens": 2}
        )

    def test_omits_the_breakdown_for_the_bare_pair_which_has_no_cache_keys(
        self,
    ) -> None:
        assert "input_details" not in parse_usage({"input": 1, "output": 2})

    def test_accepts_the_bedrock_cache_write_alias(self) -> None:
        # An unlisted spelling is silently dropped from the total, so every alias must be summed.
        result = parse_usage(
            {"inputTokens": 5, "outputTokens": 1, "cacheWriteInputTokens": 100}
        )
        assert result["input"] == 105

    def test_sums_both_camel_case_creation_spellings(self) -> None:
        result = parse_usage(
            {
                "inputTokens": 0,
                "outputTokens": 0,
                "cacheCreationInputTokens": 10,
                "cacheWriteInputTokens": 5,
            }
        )
        assert result["input"] == 15


# ─── add_cached_tokens_to_input ──────────────────────────────────────────────


class TestAddCachedTokensToInput:
    def test_adds_both_cache_buckets_on_top_of_input(self) -> None:
        usage = add_cached_tokens_to_input(
            {
                "input_tokens": 3,
                "output_tokens": 10,
                "cache_read_input_tokens": 19971,
                "cache_creation_input_tokens": 3580,
            }
        )
        assert usage.input == 23554
        assert usage.output == 10
        assert usage.cache_read == 19971
        assert usage.cache_creation == 3580

    def test_an_absent_cache_field_is_zero_not_an_error(self) -> None:
        usage = add_cached_tokens_to_input({"input_tokens": 5, "output_tokens": 2})
        assert (usage.input, usage.cache_read, usage.cache_creation) == (5, 0, 0)

    def test_an_absent_usage_bag_is_all_zeros(self) -> None:
        usage = add_cached_tokens_to_input({})
        assert (usage.input, usage.output) == (0, 0)


# ─── lang_chain_span_usage ───────────────────────────────────────────────────


class TestLangChainSpanUsage:
    def test_does_not_add_cache_on_top_of_input(self) -> None:
        # LangChain already counts cached tokens inside input_tokens. Adding them would double-count.
        usage = lang_chain_span_usage(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_token_details": {"cache_read": 80, "cache_creation": 5},
            }
        )
        assert usage is not None
        assert usage.input == 100
        assert usage.cache_read == 80
        assert usage.cache_creation == 5

    def test_an_empty_bag_is_none_not_zeros(self) -> None:
        # None keeps a turn the provider said nothing about from registering as reported.
        assert lang_chain_span_usage({}) is None
        assert lang_chain_span_usage(None) is None

    def test_a_bag_with_no_token_keys_is_none(self) -> None:
        assert lang_chain_span_usage({"input_token_details": {"cache_read": 5}}) is None

    def test_a_zero_count_is_still_reported(self) -> None:
        usage = lang_chain_span_usage({"input_tokens": 0, "output_tokens": 0})
        assert usage is not None
        assert usage.input == 0

    def test_missing_details_default_to_zero(self) -> None:
        usage = lang_chain_span_usage({"input_tokens": 5, "output_tokens": 1})
        assert usage is not None
        assert usage.cache_read == 0


# ─── The run accumulator ─────────────────────────────────────────────────────


class TestRunUsage:
    def test_starts_unreported_and_at_zero(self) -> None:
        run = create_run_usage()
        assert run.reported is False
        assert run.total.input == 0

    def test_sums_across_turns(self) -> None:
        run = create_run_usage()
        run.add(SpanUsage(input=10, output=1, cache_read=2, cache_creation=3))
        run.add(SpanUsage(input=20, output=2, cache_read=4, cache_creation=5))
        assert run.total.input == 30
        assert run.total.output == 3
        assert run.total.cache_read == 6
        assert run.total.cache_creation == 8

    def test_none_is_a_no_op_and_does_not_count_as_reported(self) -> None:
        run = create_run_usage()
        run.add(None)
        assert run.reported is False

    def test_a_genuinely_empty_turn_still_counts_as_reported(self) -> None:
        # This is the distinction the failure path needs: "reported zero" is not "reported nothing".
        run = create_run_usage()
        run.add(SpanUsage())
        assert run.reported is True
        assert run.total.input == 0

    def test_two_accumulators_do_not_share_state(self) -> None:
        # A mutable default on the dataclass would make this fail.
        first = create_run_usage()
        first.add(SpanUsage(input=5))
        assert create_run_usage().total.input == 0


# ─── set_usage_span_attributes ───────────────────────────────────────────────


class TestSetUsageSpanAttributes:
    def test_writes_all_seven_attributes(self) -> None:
        span = FakeSpan()
        set_usage_span_attributes(
            span, SpanUsage(input=10, output=4, cache_read=6, cache_creation=1)
        )
        assert span.attributes == {
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 4,
            "gen_ai.usage.total_tokens": 14,
            "gen_ai.usage.cache_read.input_tokens": 6,
            "gen_ai.usage.cache_creation.input_tokens": 1,
            "gen_ai.usage.prompt_tokens": 10,
            "gen_ai.usage.completion_tokens": 4,
        }

    def test_writes_zeros_rather_than_omitting_them(self) -> None:
        # An absent attribute drops the span from every query that groups on usage, which reads as
        # "no cached tokens" when it means "this handler forgot to say".
        span = FakeSpan()
        set_usage_span_attributes(span, SpanUsage())
        assert len(span.attributes) == 7
        assert span.attributes["gen_ai.usage.cache_creation.input_tokens"] == 0

    def test_total_is_derived_not_taken(self) -> None:
        span = FakeSpan()
        set_usage_span_attributes(span, SpanUsage(input=7, output=3))
        assert span.attributes["gen_ai.usage.total_tokens"] == 10

    def test_the_openllmetry_aliases_agree_with_the_canonical_keys(self) -> None:
        # They disagreed once, because the alias was computed at a call site off Anthropic's
        # cache-excluding input field. One writer, one number.
        span = FakeSpan()
        set_usage_span_attributes(span, SpanUsage(input=100, output=5, cache_read=80))
        assert (
            span.attributes["gen_ai.usage.prompt_tokens"]
            == span.attributes["gen_ai.usage.input_tokens"]
        )
        assert (
            span.attributes["gen_ai.usage.completion_tokens"]
            == span.attributes["gen_ai.usage.output_tokens"]
        )


# ─── set_model_identity_attributes ───────────────────────────────────────────


class TestSetModelIdentityAttributes:
    def test_writes_both_provider_keys_with_the_same_value_by_default(self) -> None:
        span = FakeSpan()
        set_model_identity_attributes(span, "anthropic", "claude-3-5-sonnet")
        assert span.attributes["gen_ai.system"] == "anthropic"
        assert span.attributes["gen_ai.provider.name"] == "anthropic"
        assert span.attributes["gen_ai.request.model"] == "claude-3-5-sonnet"

    def test_the_legacy_key_can_differ_from_the_current_one(self) -> None:
        # The LangChain handlers keep the framework name on the old key, because the new key's
        # semconv enum has no 'langchain' member.
        span = FakeSpan()
        set_model_identity_attributes(
            span, "anthropic", "claude-3-5-sonnet", "langchain"
        )
        assert span.attributes["gen_ai.system"] == "langchain"
        assert span.attributes["gen_ai.provider.name"] == "anthropic"


# ─── end_span_once ───────────────────────────────────────────────────────────


class TestEndSpanOnce:
    def test_ends_the_span(self) -> None:
        span: FakeSpan = FakeSpan()
        tracker: set[int] = set()
        end_span_once(span, tracker)
        assert span.ended == 1

    def test_a_second_call_is_ignored(self) -> None:
        # The streaming finally and the success path can both reach the same span.
        span: FakeSpan = FakeSpan()
        tracker: set[int] = set()
        end_span_once(span, tracker)
        end_span_once(span, tracker)
        assert span.ended == 1

    def test_marks_abandonment_without_asserting_failure(self) -> None:
        span: FakeSpan = FakeSpan()
        tracker: set[int] = set()
        end_span_once(span, tracker, abandoned=True)
        assert span.attributes["launchdarkly.stream.abandoned"] is True
        assert span.ended == 1

    def test_does_not_mark_a_normal_end(self) -> None:
        span: FakeSpan = FakeSpan()
        tracker: set[int] = set()
        end_span_once(span, tracker)
        assert "launchdarkly.stream.abandoned" not in span.attributes

    def test_tracks_each_span_separately(self) -> None:
        first, second = FakeSpan(), FakeSpan()
        tracker: set[int] = set()
        end_span_once(first, tracker)
        end_span_once(second, tracker)
        assert (first.ended, second.ended) == (1, 1)

    def test_a_none_span_is_a_no_op(self) -> None:
        # Handlers hold None whenever the OTel SDK is absent, and a cleanup path in a `finally` is
        # the last place that should have to remember it. Every sibling helper already no-ops.
        end_span_once(None, set())
        end_span_once(None, set(), abandoned=True)

    def test_an_unhashable_span_is_still_tracked(self) -> None:
        # The tracker holds id(span), because an OTel span is not guaranteed hashable.
        class Unhashable(FakeSpan):
            __hash__ = None  # type: ignore[assignment]

        span = Unhashable()
        tracker: set[int] = set()
        end_span_once(span, tracker)
        end_span_once(span, tracker)
        assert span.ended == 1


# ─── to_usage_dict ───────────────────────────────────────────────────────────


class TestToUsageDict:
    """The public UsageDict must carry everything parse_usage reported.

    Both call sites used to build the dataclass by hand from three keys, so the cache breakdown
    vanished the moment parse_usage started reporting one: a caller reading input_details off a
    blocking call got None while the streaming path handed back the nested dict.
    """

    def test_carries_the_three_totals(self) -> None:
        usage = to_usage_dict({"input": 10, "output": 5, "total": 15})
        assert (usage.input, usage.output, usage.total) == (10, 5, 15)

    def test_carries_the_cache_breakdown_when_present(self) -> None:
        usage = to_usage_dict(
            {
                "input": 23554,
                "output": 10,
                "total": 23564,
                "input_details": {
                    "uncached": 3,
                    "cache_read": 19971,
                    "cache_creation": 3580,
                },
            }
        )
        assert usage.input_details is not None
        assert usage.input_details.uncached == 3
        assert usage.input_details.cache_read == 19971
        assert usage.input_details.cache_creation == 3580

    def test_is_none_when_the_provider_reported_no_cache(self) -> None:
        assert (
            to_usage_dict({"input": 1, "output": 2, "total": 3}).input_details is None
        )

    def test_round_trips_a_parse_usage_result(self) -> None:
        # The two functions are used together, so pin them together.
        raw = {
            "input_tokens": 3,
            "output_tokens": 10,
            "cache_read_input_tokens": 19971,
            "cache_creation_input_tokens": 3580,
        }
        usage = to_usage_dict(parse_usage(raw))
        assert usage.input == 23554
        assert usage.input_details is not None
        assert usage.input_details.cache_read == 19971

    def test_a_missing_field_defaults_rather_than_raising(self) -> None:
        usage = to_usage_dict({})
        assert (usage.input, usage.output, usage.total) == (0, 0, 0)
        assert usage.input_details is None


@pytest.fixture
def tracer_and_exporter():
    """A real tracer and exporter, not a mock.

    A mock span cannot tell a span that was ended from one that was not, which is the only thing these
    tests are about.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


class TestEndUnfinishedSpans:
    """TELEMETRY-CONTRACT.md section 6: a `finally` owns every end."""

    def test_ends_a_span_a_cancelled_run_left_open(self, tracer_and_exporter) -> None:
        # asyncio.CancelledError is a BaseException, so `except Exception` never sees it. Without this
        # helper a cancelled run exported no span at all, and the root carries the feature_flag event.
        from launchdarkly_ai_server import end_unfinished_spans

        tracer, exporter = tracer_and_exporter
        span = tracer.start_span("invoke_agent")

        end_unfinished_spans(span)

        [finished] = exporter.get_finished_spans()
        assert finished.name == "invoke_agent"
        assert finished.attributes["launchdarkly.run.cancelled"] is True

    def test_leaves_the_span_unset_rather_than_error(self, tracer_and_exporter) -> None:
        # Nothing failed. The caller went away. Marking ERROR would disagree with LaunchDarkly's own
        # metrics, which record neither a success nor an error for a run that never finished.
        from opentelemetry.trace import StatusCode

        from launchdarkly_ai_server import end_unfinished_spans

        tracer, exporter = tracer_and_exporter
        end_unfinished_spans(tracer.start_span("chat"))

        [finished] = exporter.get_finished_spans()
        assert finished.status.status_code is StatusCode.UNSET

    def test_skips_a_span_another_path_already_ended(
        self, tracer_and_exporter, caplog
    ) -> None:
        # This runs on the success path too, where every span is already closed. The OTel SDK makes a
        # second end idempotent but logs it, and that log is the only observable difference: it is how
        # a real double-end shows up in production, and it would bury a genuine leak in noise.
        import logging

        from launchdarkly_ai_server import end_unfinished_spans

        tracer, exporter = tracer_and_exporter
        span = tracer.start_span("chat")
        span.end()

        with caplog.at_level(logging.WARNING, logger="opentelemetry.sdk.trace"):
            end_unfinished_spans(span)

        assert caplog.records == []
        [finished] = exporter.get_finished_spans()
        assert "launchdarkly.run.cancelled" not in (finished.attributes or {})

    def test_ignores_none_so_a_finally_need_not_check(self) -> None:
        from launchdarkly_ai_server import end_unfinished_spans

        end_unfinished_spans(None, None)
