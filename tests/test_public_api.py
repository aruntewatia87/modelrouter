"""Tests for the public API surface: exports, version and cross-module wiring."""

from __future__ import annotations

import modelrouter
from modelrouter import (
    UCB1,
    ArmRegistry,
    ModelArm,
    Outcome,
    RequestContext,
    RewardShaper,
    Router,
)


def test_version_is_exported() -> None:
    """The package exposes a semantic version."""
    assert modelrouter.__version__ == "0.1.0"


def test_all_is_sorted_and_resolvable() -> None:
    """Every name in ``__all__`` exists and is importable from the package."""
    assert modelrouter.__all__ == sorted(modelrouter.__all__, key=_ruff022_key)
    for name in modelrouter.__all__:
        assert hasattr(modelrouter, name), name


def _ruff022_key(name: str) -> tuple[int, int, str]:
    """Reproduce ruff's RUF022 ordering: constants, classes, then functions."""
    if name.isupper():
        return (0, 0, name)
    if name[:1].isupper():
        return (1, 0, name)
    return (2, 0, name)


def test_public_api_covers_the_documented_surface() -> None:
    """The documented public names are all exported."""
    expected = {
        "ArmHealth",
        "ArmRegistry",
        "ArmStats",
        "Bandit",
        "CAPABILITIES",
        "CostProjection",
        "EpsilonGreedy",
        "HealthSignal",
        "HealthTracker",
        "ModelArm",
        "Outcome",
        "RequestContext",
        "RewardShaper",
        "RoutingDecision",
        "RoutingError",
        "Router",
        "RouterStats",
        "SoftmaxRouter",
        "ThompsonSampling",
        "UCB1",
        "arm_cost_for_request",
        "compare_projections",
        "estimate_cost",
        "estimate_prompt_tokens",
        "is_complex_prompt",
        "project_policy",
        "project_spend",
        "select_arms_by_capability",
    }
    assert expected == set(modelrouter.__all__)


def test_module_import_does_not_touch_the_network() -> None:
    """Importing the package must not open a socket."""
    import socket

    original = socket.socket

    class Guard(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("network access during import")

    socket.socket = Guard  # type: ignore[assignment]
    try:
        import importlib

        importlib.reload(modelrouter)
    finally:
        socket.socket = original  # type: ignore[assignment]


def test_select_arms_by_capability_helper() -> None:
    """The convenience helper mirrors ``ArmRegistry.eligible_for``."""
    registry = ArmRegistry(
        [
            ModelArm(
                name="a",
                provider="p",
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                cache_read_cost_per_1k=0.0,
                capabilities=frozenset({"chat", "tools"}),
                max_context_tokens=1000,
                avg_latency_ms=1.0,
            )
        ]
    )
    assert [arm.name for arm in modelrouter.select_arms_by_capability(registry, ["tools"])] == ["a"]


def test_end_to_end_minimal_workflow() -> None:
    """The shortest possible route -> observe -> stats loop works."""
    registry = ArmRegistry(
        [
            ModelArm(
                name="only",
                provider="p",
                input_cost_per_1k=0.001,
                output_cost_per_1k=0.002,
                cache_read_cost_per_1k=0.0001,
                capabilities=frozenset({"chat"}),
                max_context_tokens=8000,
                avg_latency_ms=100.0,
            )
        ]
    )
    router = Router(registry, UCB1(seed=1), shaper=RewardShaper(lambda_cost=0.5))
    decision = router.route(RequestContext(request_id="r1", prompt_tokens=10))
    assert decision.arm == "only"
    reward = router.observe(
        "r1", Outcome(success=True, quality=1.0, latency_ms=50.0, cost_usd=0.01)
    )
    assert reward > 0.0
    assert router.stats().total_requests == 1


def test_registry_and_router_share_the_stats_table(registry: ArmRegistry) -> None:
    """The registry owns the statistics, so both views agree."""
    router = Router(registry, UCB1(seed=1))
    router.route(RequestContext(request_id="r1", prompt_tokens=100))
    router.observe("r1", Outcome(success=True, quality=0.9, latency_ms=10.0, cost_usd=0.01))
    for name in registry.names():
        assert registry.stats()[name] is router.arm_stats(name)
    assert sum(s.pulls for s in registry.stats().values()) == 1


def test_router_does_not_clobber_pre_registered_health() -> None:
    """A pre-seeded health tracker keeps its observations."""
    from modelrouter import HealthTracker

    registry = ArmRegistry(
        [
            ModelArm(
                name="a",
                provider="p",
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                cache_read_cost_per_1k=0.0,
                capabilities=frozenset({"chat"}),
                max_context_tokens=1000,
                avg_latency_ms=100.0,
            )
        ]
    )
    health = HealthTracker()
    health.register("a", 100.0)
    health.record_outcome("a", success=False, latency_ms=1.0)
    Router(registry, UCB1(seed=1), health=health)
    assert health.health("a").total_observations == 1
    assert health.health("a").baseline_latency_ms == 100.0
