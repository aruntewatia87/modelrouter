"""Tests for the arm registry, capability filtering and per-arm statistics."""

from __future__ import annotations

import dataclasses

import pytest

from modelrouter import ArmRegistry, ArmStats
from tests.conftest import CHAT, make_arm


def test_capability_filter_excludes_incompatible_arms(registry: ArmRegistry) -> None:
    """An arm missing ANY required capability must be filtered out."""
    eligible = registry.eligible_for({"tools"})
    assert [arm.name for arm in eligible] == ["tools"]


def test_capability_filter_requires_every_capability(registry: ArmRegistry) -> None:
    """Requiring tools + json excludes each arm that only has one of them."""
    eligible = registry.eligible_for({"tools", "json_mode"})
    assert eligible == []
    both = make_arm(
        "both",
        capabilities=frozenset({"chat", "tools", "json_mode", "vision"}),
    )
    registry.register(both)
    assert registry.eligible_for({"tools", "json_mode", "vision"}) == [both]


def test_empty_capability_requirement_matches_every_arm(registry: ArmRegistry) -> None:
    """No required capabilities means every arm is eligible."""
    assert len(registry.eligible_for([])) == len(registry)


def test_context_window_filter(registry: ArmRegistry) -> None:
    """Arms too small for the required context are excluded."""
    small = make_arm("small", max_context_tokens=1000)
    registry.register(small)
    eligible = registry.eligible_for({"chat"}, min_context_tokens=10_000)
    assert small not in eligible
    assert len(eligible) == len(registry) - 1


def test_cheapest_for_is_deterministic(registry: ArmRegistry) -> None:
    """``cheapest_for`` is stable and returns a single arm."""
    cheapest = registry.cheapest_for({"chat"})
    assert cheapest is not None
    assert cheapest.name == "cheap"
    assert registry.cheapest_for({"chat"}) is cheapest


def test_best_for_prefers_quality_then_price(registry: ArmRegistry) -> None:
    """``best_for`` ranks by quality prior first."""
    best = registry.best_for({"chat"})
    assert best is not None
    assert best.name == "best"


def test_cheapest_and_best_for_empty_registry_is_none() -> None:
    """With no arms registered the helpers return ``None``."""
    empty = ArmRegistry()
    assert empty.cheapest_for({"chat"}) is None
    assert empty.best_for({"chat"}) is None


def test_registry_lookup_and_contains(registry: ArmRegistry) -> None:
    """Arms can be looked up by name, and unknown names raise."""
    assert registry.get("mid").provider == "beta"
    assert "mid" in registry
    assert "nope" not in registry
    with pytest.raises(KeyError, match="unknown arm"):
        registry.get("nope")


def test_registry_register_replaces_same_name(registry: ArmRegistry) -> None:
    """Registering an existing name replaces the arm rather than duplicating it."""
    before = len(registry)
    registry.register(make_arm("cheap", input_cost=0.9))
    assert len(registry) == before
    assert registry.get("cheap").input_cost_per_1k == 0.9


def test_model_arm_rejects_invalid_values() -> None:
    """Bad prices, priors or capabilities are rejected at construction time."""
    with pytest.raises(ValueError, match="name must be"):
        make_arm("")
    with pytest.raises(ValueError, match="input_cost_per_1k"):
        make_arm("x", input_cost=-1.0)
    with pytest.raises(ValueError, match="output_cost_per_1k"):
        make_arm("x", output_cost=-1.0)
    with pytest.raises(ValueError, match="cache_read_cost_per_1k"):
        make_arm("x", cache_read_cost=-1.0)
    with pytest.raises(ValueError, match="max_context_tokens"):
        make_arm("x", max_context_tokens=0)
    with pytest.raises(ValueError, match="avg_latency_ms"):
        make_arm("x", avg_latency_ms=0.0)
    with pytest.raises(ValueError, match="quality_prior"):
        make_arm("x", quality_prior=1.5)
    with pytest.raises(ValueError, match="reliability"):
        make_arm("x", reliability=-0.1)


def test_model_arm_rejects_unknown_capability() -> None:
    """Only the documented capability tokens are accepted."""
    with pytest.raises(ValueError, match="unknown capabilities"):
        make_arm("x", capabilities=frozenset({"chat", "telepathy"}))


def test_model_arm_supports_helpers() -> None:
    """``supports`` and ``supports_all`` reflect the capability set."""
    arm = make_arm("x", capabilities=frozenset({"chat", "tools"}))
    assert arm.supports("tools")
    assert not arm.supports("vision")
    assert arm.supports_all({"chat", "tools"})
    assert not arm.supports_all({"chat", "vision"})


def test_baseline_cost_blends_prompt_and_completion() -> None:
    """The blended price uses a 3:1 prompt:completion ratio."""
    arm = make_arm("x", input_cost=0.004, output_cost=0.0, cache_read_cost=0.0)
    assert arm.baseline_cost_per_1k() == pytest.approx(0.003)


def test_arm_stats_tracks_success_rate_and_cost() -> None:
    """Per-arm stats aggregate pulls, successes, spend and latency."""
    stats = ArmStats()
    stats.record(success=True, quality=0.8, latency_ms=100.0, cost_usd=0.01)
    stats.record(success=False, quality=0.0, latency_ms=300.0, cost_usd=0.02, error_type="timeout")
    assert stats.pulls == 2
    assert stats.successes == 1
    assert stats.success_rate == pytest.approx(0.5)
    assert stats.total_cost_usd == pytest.approx(0.03)
    assert stats.avg_latency_ms == pytest.approx(200.0)
    assert stats.rolling_avg_latency_ms == pytest.approx(200.0)
    assert stats.error_count == 1
    assert stats.errors_by_type == {"timeout": 1}


def test_arm_stats_empty_defaults() -> None:
    """A fresh stats object reports zeros rather than dividing by zero."""
    stats = ArmStats()
    assert stats.pulls == 0
    assert stats.success_rate == 0.0
    assert stats.avg_latency_ms == 0.0
    assert stats.rolling_avg_latency_ms == 0.0
    assert stats.avg_cost_usd == 0.0


def test_arm_stats_rolling_latency_window_is_bounded() -> None:
    """The rolling latency deque keeps only the most recent samples."""
    stats = ArmStats()
    stats._window = 3
    for i in range(10):
        stats.record(success=True, quality=0.5, latency_ms=float(i), cost_usd=0.0)
    assert len(stats.rolling_latency_ms) == 3
    assert stats.rolling_avg_latency_ms == pytest.approx((7 + 8 + 9) / 3)


def test_arm_stats_ewma_tracks_quality() -> None:
    """The quality EWMA moves towards the most recent sample."""
    stats = ArmStats()
    stats.record(success=True, quality=1.0, latency_ms=1.0, cost_usd=0.0)
    assert stats.quality_score == pytest.approx(1.0)
    stats.record(success=True, quality=0.0, latency_ms=1.0, cost_usd=0.0)
    assert stats.quality_score == pytest.approx(0.7)


def test_arm_stats_as_dict_is_json_friendly() -> None:
    """``as_dict`` returns only scalar values."""
    stats = ArmStats()
    stats.record(success=True, quality=0.5, latency_ms=10.0, cost_usd=0.001)
    snapshot = stats.as_dict()
    assert snapshot["pulls"] == 1
    assert all(isinstance(v, (int, float)) for v in snapshot.values())


def test_registry_iteration_and_names(registry: ArmRegistry) -> None:
    """The registry iterates in registration order and exposes names."""
    assert registry.names() == ["cheap", "mid", "best", "tools", "json"]
    assert [arm.name for arm in registry] == registry.names()


def test_registry_stats_creates_entries(registry: ArmRegistry) -> None:
    """``stats`` returns one entry per registered arm."""
    assert set(registry.stats()) == set(registry.names())


def test_capability_tokens_are_the_documented_set() -> None:
    """The exported capability vocabulary matches the documented set."""
    from modelrouter import CAPABILITIES

    assert frozenset({"chat", "tools", "json_mode", "vision", "long_context"}) == CAPABILITIES


def test_model_arm_is_hashable_and_frozen() -> None:
    """Arms are frozen dataclasses, so they can be used as dict keys."""
    arm = make_arm("x", capabilities=CHAT)
    assert hash(arm) == hash(make_arm("x", capabilities=CHAT))
    with pytest.raises(dataclasses.FrozenInstanceError):
        arm.name = "y"  # type: ignore[misc]
