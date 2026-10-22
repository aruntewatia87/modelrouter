"""Tests for the router: capability filter, budget filter, kill switch, feedback."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelrouter import (
    UCB1,
    ArmRegistry,
    EpsilonGreedy,
    HealthTracker,
    Outcome,
    RequestContext,
    Router,
    RoutingError,
)
from tests.conftest import make_arm


class FixedArm:
    """A test bandit that always selects one arm, to isolate the kill switch."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.updates: list[tuple[str, float]] = []

    def select(self, arms: list[str], context: object = None) -> str:
        """Always return the pinned arm."""
        return self.arm

    def update(self, arm: str, reward: float, context: object = None) -> None:
        """Record the update without acting on it."""
        self.updates.append((arm, reward))


def make_router(registry: ArmRegistry, **kwargs: object) -> Router:
    """Build a router over ``registry`` with a deterministic UCB1 policy."""
    kwargs.setdefault("bandit", UCB1(seed=42))
    return Router(registry, **kwargs)  # type: ignore[arg-type]


def test_capability_filter_is_enforced_end_to_end(registry: ArmRegistry) -> None:
    """A tool request never lands on an arm without tool support."""
    router = make_router(registry)
    request = RequestContext(request_id="r1", needs_tools=True)
    for _ in range(60):
        decision = router.route(request)
        assert registry.get(decision.arm).supports("tools")
        router.observe("r1", Outcome(success=True, quality=0.8, latency_ms=100.0, cost_usd=0.01))


def test_tool_request_never_routes_to_a_non_tool_arm(registry: ArmRegistry) -> None:
    """Even under a policy that would happily pick the cheap arm."""
    router = make_router(registry)
    request = RequestContext(request_id="r1", needs_tools=True, prompt_tokens=100)
    candidates = router.candidate_arms(request)
    assert [arm.name for arm in candidates] == ["tools"]
    assert router.route(request).arm == "tools"


def test_json_request_only_reaches_json_capable_arms(registry: ArmRegistry) -> None:
    """JSON mode is a hard filter too."""
    router = make_router(registry)
    request = RequestContext(request_id="r1", needs_json=True)
    assert [arm.name for arm in router.candidate_arms(request)] == ["json"]


def test_budget_filter_excludes_an_over_ceiling_arm(registry: ArmRegistry) -> None:
    """An arm whose estimated cost exceeds the ceiling is dropped."""
    # "best" costs 0.015/1k input + 0.075/1k output -> ~0.034 for 1000+250 tokens.
    router = make_router(registry, budget_per_request_usd=0.005)
    request = RequestContext(request_id="r1", prompt_tokens=1000)
    names = [arm.name for arm in router.candidate_arms(request)]
    assert "best" not in names
    assert "cheap" in names


def test_budget_filter_can_exclude_every_arm(registry: ArmRegistry) -> None:
    """An impossible budget leaves no candidates and routing raises."""
    router = make_router(registry, budget_per_request_usd=1e-9)
    request = RequestContext(request_id="r1", prompt_tokens=1000)
    assert router.candidate_arms(request) == []
    with pytest.raises(RoutingError, match="no eligible arm"):
        router.route(request)


def test_budget_of_zero_disables_the_filter(registry: ArmRegistry) -> None:
    """A zero or ``None`` budget means "unlimited"."""
    router = make_router(registry, budget_per_request_usd=0.0)
    request = RequestContext(request_id="r1", prompt_tokens=1000)
    assert len(router.candidate_arms(request)) == len(registry)


def test_observe_updates_stats(registry: ArmRegistry) -> None:
    """Observing an outcome updates the arm statistics and total spend."""
    router = make_router(registry)
    request = RequestContext(request_id="r1", prompt_tokens=100)
    decision = router.route(request)
    router.observe("r1", Outcome(success=True, quality=0.9, latency_ms=250.0, cost_usd=0.004))
    stats = router.arm_stats(decision.arm)
    assert stats.pulls == 1
    assert stats.successes == 1
    assert stats.success_rate == pytest.approx(1.0)
    assert stats.total_cost_usd == pytest.approx(0.004)
    assert stats.quality_score == pytest.approx(0.9)
    summary = router.stats()
    assert summary.total_requests == 1
    assert summary.total_successes == 1
    assert summary.total_spend_usd == pytest.approx(0.004)


def test_observe_returns_the_shaped_reward(registry: ArmRegistry) -> None:
    """``observe`` returns exactly the reward that was fed to the bandit."""
    router = make_router(registry)
    router.route(RequestContext(request_id="r1", prompt_tokens=100))
    reward = router.observe(
        "r1", Outcome(success=True, quality=0.8, latency_ms=0.0, cost_usd=0.002)
    )
    assert reward == pytest.approx(
        router.shaper.shape(success=True, quality=0.8, cost_usd=0.002, latency_ms=0.0)
    )


def test_observe_unknown_request_raises(registry: ArmRegistry) -> None:
    """Observing a request that was never routed is an error."""
    router = make_router(registry)
    with pytest.raises(KeyError, match="unknown request_id"):
        router.observe("never-routed", Outcome(success=True))


def test_kill_switch_excludes_a_failing_arm_after_min_pulls() -> None:
    """An arm that keeps failing is killed once it has enough pulls.

    The health breaker is relaxed here so the *cumulative* kill switch is
    exercised in isolation: it needs ``min_pulls`` observations before it
    trips, whereas the breaker reacts to the sliding window immediately.
    """
    registry = ArmRegistry([make_arm("bad"), make_arm("good")])
    router = Router(
        registry,
        FixedArm("bad"),
        health=HealthTracker(min_success_rate=0.0),
        min_pulls=5,
        kill_threshold=0.5,
        kill_cooldown=1000,
    )
    request = RequestContext(request_id="r", prompt_tokens=100)
    for i in range(12):
        router.route(replace(request, request_id=f"r-{i}"))
        router.observe(f"r-{i}", Outcome(success=False, latency_ms=100.0, cost_usd=0.001))
    assert "bad" in router.killed_arms()
    assert "bad" not in [arm.name for arm in router.candidate_arms(request)]
    assert "good" in [arm.name for arm in router.candidate_arms(request)]


def test_kill_switch_respects_min_pulls(registry: ArmRegistry) -> None:
    """An arm is not killed before it reaches ``min_pulls``."""
    router = make_router(
        registry,
        health=HealthTracker(min_success_rate=0.0),
        min_pulls=10,
        kill_threshold=0.5,
    )
    decision = router.route(RequestContext(request_id="r0", prompt_tokens=100))
    for _ in range(3):
        router.observe("r0", Outcome(success=False, latency_ms=1.0))
    assert decision.arm not in router.killed_arms()


def test_killed_arm_is_excluded_but_others_remain() -> None:
    """Killing one arm leaves the others routable."""
    registry = ArmRegistry([make_arm("bad"), make_arm("good")])
    router = Router(
        registry,
        FixedArm("bad"),
        health=HealthTracker(min_success_rate=0.0),
        min_pulls=3,
        kill_threshold=0.1,
        kill_cooldown=1000,
    )
    request = RequestContext(request_id="r", prompt_tokens=100)
    for i in range(5):
        router.route(replace(request, request_id=f"r-{i}"))
        router.observe(f"r-{i}", Outcome(success=False, latency_ms=1.0))
    killed = router.killed_arms()
    assert killed
    remaining = [arm.name for arm in router.candidate_arms(request)]
    assert not set(remaining) & set(killed)


def test_health_breaker_excludes_an_unhealthy_arm(registry: ArmRegistry) -> None:
    """An arm whose health fails is excluded from routing."""
    health = HealthTracker(min_success_rate=0.5)
    router = make_router(registry, health=health)
    request = RequestContext(request_id="r", prompt_tokens=100)
    decision = router.route(request)
    for _ in range(6):
        router.observe(
            "r",
            Outcome(success=False, latency_ms=5000.0, cost_usd=0.0, error="timeout"),
        )
    assert health.is_healthy(decision.arm) is False
    assert decision.arm not in [arm.name for arm in router.candidate_arms(request)]


def test_router_stats_reports_excluded_arms() -> None:
    """``stats`` lists the arms currently excluded."""
    registry = ArmRegistry([make_arm("bad"), make_arm("good")])
    router = Router(
        registry,
        FixedArm("bad"),
        health=HealthTracker(min_success_rate=0.0),
        min_pulls=3,
        kill_threshold=0.1,
        kill_cooldown=1000,
    )
    for i in range(5):
        router.route(RequestContext(request_id=f"r-{i}", prompt_tokens=100))
        router.observe(f"r-{i}", Outcome(success=False, latency_ms=1.0))
    assert router.stats().excluded_arms == ["bad"]


def test_traffic_mix_sums_to_one(registry: ArmRegistry) -> None:
    """The observed traffic mix is a probability distribution."""
    router = make_router(registry)
    for i in range(20):
        router.route(RequestContext(request_id=f"r-{i}", prompt_tokens=100))
        router.observe(f"r-{i}", Outcome(success=True, quality=0.5, latency_ms=1.0, cost_usd=0.0))
    mix = router.traffic_mix()
    assert sum(mix.values()) == pytest.approx(1.0)


def test_traffic_mix_is_zero_before_any_traffic(registry: ArmRegistry) -> None:
    """An idle router reports an all-zero mix rather than dividing by zero."""
    router = make_router(registry)
    assert sum(router.traffic_mix().values()) == 0.0


def test_router_validates_constructor_arguments(registry: ArmRegistry) -> None:
    """Invalid router configuration is rejected."""
    with pytest.raises(ValueError, match="budget_per_request_usd"):
        make_router(registry, budget_per_request_usd=-1.0)
    with pytest.raises(ValueError, match="min_pulls"):
        make_router(registry, min_pulls=-1)
    with pytest.raises(ValueError, match="kill_threshold"):
        make_router(registry, kill_threshold=1.5)


def test_router_repr_is_informative(registry: ArmRegistry) -> None:
    """The repr shows the policy and arm count."""
    router = make_router(registry)
    assert "UCB1" in repr(router)
    assert "arms=5" in repr(router)


def test_decision_records_the_estimate(registry: ArmRegistry) -> None:
    """The decision carries a pre-flight cost estimate."""
    router = make_router(registry)
    decision = router.route(RequestContext(request_id="r1", prompt_tokens=1000))
    assert decision.estimated_cost_usd > 0.0
    assert decision.candidate_count >= 1
    assert router.decision("r1").arm == decision.arm


def test_decision_unknown_request_raises(registry: ArmRegistry) -> None:
    """Looking up an unknown decision raises."""
    router = make_router(registry)
    with pytest.raises(KeyError, match="unknown request_id"):
        router.decision("ghost")


def test_arm_stats_unknown_arm_raises(registry: ArmRegistry) -> None:
    """Looking up stats for an unknown arm raises."""
    router = make_router(registry)
    with pytest.raises(KeyError, match="unknown arm"):
        router.arm_stats("ghost")


def test_router_uses_the_requested_bandit(registry: ArmRegistry) -> None:
    """The router drives the policy it was given."""
    bandit = EpsilonGreedy(epsilon_start=0.0, epsilon_end=0.0, decay_steps=0, seed=1)
    router = make_router(registry, bandit=bandit)
    request = RequestContext(request_id="r", prompt_tokens=100)
    router.route(request)
    router.observe("r", Outcome(success=True, quality=1.0, latency_ms=1.0, cost_usd=0.0))
    assert bandit.pulls(registry.names()[0]) >= 0
    assert bandit.total_pulls() == 1


def test_request_context_requires_an_id() -> None:
    """A request must carry an id so outcomes can be correlated."""
    with pytest.raises(ValueError, match="request_id"):
        RequestContext(request_id="")


def test_request_context_rejects_negative_tokens() -> None:
    """Negative token counts are rejected."""
    with pytest.raises(ValueError, match="prompt_tokens"):
        RequestContext(request_id="r", prompt_tokens=-1)


def test_outcome_validates_its_fields() -> None:
    """Quality, latency and cost must be in range."""
    with pytest.raises(ValueError, match="quality"):
        Outcome(success=True, quality=1.5)
    with pytest.raises(ValueError, match="latency_ms"):
        Outcome(success=True, latency_ms=-1.0)
    with pytest.raises(ValueError, match="cost_usd"):
        Outcome(success=True, cost_usd=-0.1)


def test_required_capability_set_combines_flags() -> None:
    """Tools, JSON and long context all land in the requirement set."""
    request = RequestContext(
        request_id="r", needs_tools=True, needs_json=True, prompt_tokens=50_000
    )
    caps = request.required_capability_set()
    assert {"tools", "json_mode", "long_context"} <= caps


def test_long_context_request_only_reaches_long_context_arms() -> None:
    """A very long prompt needs an arm that supports ``long_context``."""
    registry = ArmRegistry(
        [
            make_arm("small", capabilities=frozenset({"chat"})),
            make_arm("big", capabilities=frozenset({"chat", "long_context"})),
        ]
    )
    router = make_router(registry)
    request = RequestContext(request_id="r", prompt_tokens=100_000)
    assert [arm.name for arm in router.candidate_arms(request)] == ["big"]
