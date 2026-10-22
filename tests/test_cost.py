"""Tests for cost estimation and spend projection."""

from __future__ import annotations

import pytest

from modelrouter import (
    CostProjection,
    ModelArm,
    arm_cost_for_request,
    compare_projections,
    estimate_cost,
    project_policy,
    project_spend,
)
from tests.conftest import make_arm


def test_estimate_cost_sums_input_and_output_prices() -> None:
    """Cost is input price plus output price, per 1000 tokens."""
    arm = make_arm("x", input_cost=0.001, output_cost=0.002, cache_read_cost=0.0001)
    cost = estimate_cost(arm, prompt_tokens=1000, completion_tokens=500)
    assert cost == pytest.approx(0.001 + 0.001)


def test_cache_read_price_is_applied() -> None:
    """Regression test for LiteLLM #38064: cache reads are priced separately.

    Ignoring the cache-read price inflates the cost of a cache-warm arm and
    can flip a routing decision on a price tie.
    """
    arm = make_arm(
        "cached",
        input_cost=0.010,
        output_cost=0.010,
        cache_read_cost=0.001,
    )
    cold = estimate_cost(arm, prompt_tokens=1000, completion_tokens=0, cached_input_tokens=0)
    warm = estimate_cost(arm, prompt_tokens=1000, completion_tokens=0, cached_input_tokens=1000)
    expected = 0.001 * 1000 / 1000
    assert warm == pytest.approx(expected)
    assert warm < cold
    # The cold price must reflect the *input* sheet, not the cache sheet.
    assert cold == pytest.approx(0.010)


def test_cached_path_is_strictly_cheaper() -> None:
    """A fully cached prompt costs exactly the cache-read price."""
    arm = make_arm("cached", input_cost=0.010, output_cost=0.010, cache_read_cost=0.001)
    cold = estimate_cost(arm, 2000, 100, cached_input_tokens=0)
    warm = estimate_cost(arm, 2000, 100, cached_input_tokens=2000)
    assert warm < cold
    assert warm == pytest.approx(0.001 * 2 + 0.010 * 0.1)


def test_cache_read_price_breaks_an_input_output_tie() -> None:
    """The cache price is what makes the two providers distinguishable.

    On a pure input+output price tie the cached provider must be the cheaper
    one -- otherwise "always cheapest" picks wrong.
    """
    cached_provider = make_arm(
        "cached-provider",
        input_cost=0.003,
        output_cost=0.015,
        cache_read_cost=0.0003,
    )
    plain_provider = make_arm(
        "plain-provider",
        input_cost=0.003,
        output_cost=0.015,
        cache_read_cost=0.003,
    )
    cached_cost = estimate_cost(cached_provider, 1000, 250, cached_input_tokens=800)
    plain_cost = estimate_cost(plain_provider, 1000, 250, cached_input_tokens=800)
    assert cached_cost < plain_cost


def test_estimate_cost_rejects_cached_tokens_above_prompt() -> None:
    """A cache hit count larger than the prompt is a bug."""
    arm = make_arm("x")
    with pytest.raises(ValueError, match="cannot exceed prompt_tokens"):
        estimate_cost(arm, 100, 10, cached_input_tokens=200)


def test_estimate_cost_clamps_negative_token_counts() -> None:
    """Negative token counts are treated as zero rather than crediting money."""
    arm = make_arm("x", input_cost=0.001, output_cost=0.002, cache_read_cost=0.0001)
    assert estimate_cost(arm, -5, -5) == 0.0


def test_estimate_cost_of_zero_tokens_is_zero() -> None:
    """An empty call costs nothing."""
    assert estimate_cost(make_arm("x"), 0, 0) == 0.0


def test_arm_cost_for_request_uses_cache_hit_ratio() -> None:
    """The ratio helper agrees with the absolute helper."""
    arm = make_arm("x", input_cost=0.010, output_cost=0.0, cache_read_cost=0.001)
    assert arm_cost_for_request(arm, 1000, 0, cache_hit_ratio=0.5) == pytest.approx(
        estimate_cost(arm, 1000, 0, cached_input_tokens=500)
    )


def test_arm_cost_for_request_rejects_bad_ratio() -> None:
    """The cache hit ratio must be a fraction."""
    with pytest.raises(ValueError, match="cache_hit_ratio"):
        arm_cost_for_request(make_arm("x"), 100, 0, cache_hit_ratio=1.5)


def test_project_spend_totals_are_arithmetically_correct() -> None:
    """The projection total equals the sum of the per-arm costs."""
    arms = [
        make_arm("a", input_cost=0.001, output_cost=0.001, cache_read_cost=0.0001),
        make_arm("b", input_cost=0.010, output_cost=0.010, cache_read_cost=0.001),
    ]
    projection = project_spend("even", arms, 100, prompt_tokens=1000, completion_tokens=100)
    assert projection.total_requests == 100
    assert sum(projection.per_arm.values()) == pytest.approx(projection.total_cost_usd)
    expected = 50 * estimate_cost(arms[0], 1000, 100) + 50 * estimate_cost(arms[1], 1000, 100)
    assert projection.total_cost_usd == pytest.approx(expected)
    assert projection.avg_cost_usd == pytest.approx(projection.total_cost_usd / 100)


def test_project_spend_respects_the_mix() -> None:
    """A 100/0 mix sends every request to one arm."""
    arms = [
        make_arm("a", input_cost=0.001, output_cost=0.001, cache_read_cost=0.0001),
        make_arm("b", input_cost=0.010, output_cost=0.010, cache_read_cost=0.001),
    ]
    projection = project_spend("all-a", arms, 100, mix={"a": 1.0, "b": 0.0})
    assert projection.per_arm_requests["a"] == 100
    assert projection.per_arm_requests["b"] == 0
    assert projection.per_arm["b"] == pytest.approx(0.0)


def test_always_cheapest_is_cheaper_than_the_router_mix() -> None:
    """Routing away from the cheapest arm must cost more than always-cheapest."""
    arms = [
        make_arm("cheap", input_cost=0.0001, output_cost=0.0002, cache_read_cost=0.00001),
        make_arm("nice", input_cost=0.010, output_cost=0.020, cache_read_cost=0.001),
    ]
    always_cheapest = project_spend("always-cheapest", arms, 100, mix={"cheap": 1.0})
    router = project_spend("router", arms, 100, mix={"cheap": 0.8, "nice": 0.2})
    assert always_cheapest.total_cost_usd < router.total_cost_usd


def test_always_best_is_more_expensive_than_always_cheapest() -> None:
    """The best arm costs more than the cheapest one over the same traffic."""
    arms = [
        make_arm("cheap", input_cost=0.0001, output_cost=0.0002, cache_read_cost=0.00001),
        make_arm("nice", input_cost=0.010, output_cost=0.020, cache_read_cost=0.001),
    ]
    always_cheapest = project_spend("always-cheapest", arms, 100, mix={"cheap": 1.0})
    always_best = project_spend("always-best", arms, 100, mix={"nice": 1.0})
    assert always_best.total_cost_usd > always_cheapest.total_cost_usd


def test_project_spend_validates_inputs() -> None:
    """Bad mixes and empty arm lists are rejected."""
    arms = [make_arm("a")]
    with pytest.raises(ValueError, match="at least one arm"):
        project_spend("x", [], 10)
    with pytest.raises(ValueError, match="requests must be"):
        project_spend("x", arms, -1)
    with pytest.raises(ValueError, match="unknown arms"):
        project_spend("x", arms, 10, mix={"ghost": 1.0})
    with pytest.raises(ValueError, match="must sum to a positive"):
        project_spend("x", arms, 10, mix={"a": 0.0})


def test_project_policy_scales_the_mix_to_the_request_count() -> None:
    """``project_policy`` re-scales an observed mix onto the requested volume."""
    arms = [
        make_arm("a", input_cost=0.001, output_cost=0.001, cache_read_cost=0.0001),
        make_arm("b", input_cost=0.002, output_cost=0.002, cache_read_cost=0.0002),
    ]
    projection = project_policy("router", arms, 1000, mix={"a": 0.5, "b": 0.5})
    assert projection.total_requests == 1000
    assert sum(projection.per_arm.values()) == pytest.approx(projection.total_cost_usd)


def test_cost_projection_repr_and_dict() -> None:
    """The projection renders readably and serialises to scalars."""
    projection = CostProjection(policy="p", total_requests=10, total_cost_usd=1.5)
    assert "p" in repr(projection)
    snapshot = projection.as_dict()
    assert snapshot["total_requests"] == 10
    assert projection.avg_cost_usd == pytest.approx(0.15)


def test_compare_projections_indexes_by_policy_name() -> None:
    """``compare_projections`` returns a name-keyed mapping."""
    a = CostProjection(policy="a", total_requests=1, total_cost_usd=1.0)
    b = CostProjection(policy="b", total_requests=1, total_cost_usd=2.0)
    compared = compare_projections(a, b)
    assert set(compared) == {"a", "b"}
    assert compared["b"].total_cost_usd == 2.0


def test_cost_projection_with_zero_requests_is_safe() -> None:
    """A zero-request projection does not divide by zero."""
    projection = CostProjection(policy="p", total_requests=0, total_cost_usd=0.0)
    assert projection.avg_cost_usd == 0.0


def test_model_arm_cache_price_is_independent_of_input_price() -> None:
    """The three prices are stored independently, not derived from each other."""
    arm = ModelArm(
        name="x",
        provider="p",
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.02,
        cache_read_cost_per_1k=0.001,
        capabilities=frozenset({"chat"}),
        max_context_tokens=1000,
        avg_latency_ms=100.0,
    )
    assert arm.cache_read_cost_per_1k == 0.001
    assert arm.input_cost_per_1k == 0.01
