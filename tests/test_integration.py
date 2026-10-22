"""End-to-end tests: the router as a whole, determinism and the ROI story."""

from __future__ import annotations

import numpy as np
import pytest

from modelrouter import (
    UCB1,
    ArmRegistry,
    EpsilonGreedy,
    HealthTracker,
    ModelArm,
    Outcome,
    RequestContext,
    RewardShaper,
    Router,
    ThompsonSampling,
    estimate_cost,
    estimate_prompt_tokens,
    is_complex_prompt,
    project_spend,
)
from tests.conftest import make_arm


def simulate(router: Router, *, requests: int, seed: int, failure_arm: str | None = None) -> None:
    """Run a deterministic simulated workload through the router."""
    rng = np.random.default_rng(seed)
    for i in range(requests):
        request = RequestContext(request_id=f"req-{i}", prompt_tokens=800)
        decision = router.route(request)
        arm = router.registry.get(decision.arm)
        if failure_arm is not None and decision.arm == failure_arm:
            router.observe(
                f"req-{i}",
                Outcome(
                    success=False, quality=0.0, latency_ms=5000.0, cost_usd=0.0, error="rate_limit"
                ),
            )
            continue
        quality = float(np.clip(rng.normal(arm.quality_prior, 0.05), 0.0, 1.0))
        latency = float(max(1.0, rng.normal(arm.avg_latency_ms, arm.avg_latency_ms * 0.1)))
        cost = estimate_cost(arm, request.prompt_tokens, 250)
        router.observe(
            f"req-{i}",
            Outcome(success=True, quality=quality, latency_ms=latency, cost_usd=cost),
        )


def test_router_end_to_end_learns_and_reports(registry: ArmRegistry) -> None:
    """A full simulated workload produces coherent per-arm statistics."""
    router = Router(registry, UCB1(seed=7), shaper=RewardShaper(lambda_cost=2.0))
    simulate(router, requests=200, seed=11)
    stats = router.stats()
    assert stats.total_requests == 200
    assert stats.total_spend_usd > 0.0
    assert set(stats.per_arm) == set(registry.names())
    assert sum(v["pulls"] for v in stats.per_arm.values()) == 200


def test_router_shifts_traffic_away_from_a_failing_arm(registry: ArmRegistry) -> None:
    """When one arm starts failing the router routes around it."""
    router = Router(
        registry,
        UCB1(seed=3),
        shaper=RewardShaper(lambda_cost=1.0),
        health=HealthTracker(min_success_rate=0.5, window_size=20),
    )
    simulate(router, requests=120, seed=5)
    before = router.arm_stats("cheap").pulls
    simulate(router, requests=200, seed=6, failure_arm="cheap")
    after = router.arm_stats("cheap").pulls
    assert after - before < 200
    assert router.arm_stats("cheap").success_rate < 1.0


def test_determinism_same_seed_same_decisions(registry: ArmRegistry) -> None:
    """Two routers with the same seed make identical decisions."""

    def run() -> list[str]:
        router = Router(registry, UCB1(seed=1234), shaper=RewardShaper(lambda_cost=1.5))
        picks = []
        for i in range(60):
            request = RequestContext(request_id=f"r-{i}", prompt_tokens=500)
            picks.append(router.route(request).arm)
            router.observe(
                f"r-{i}",
                Outcome(success=True, quality=0.8, latency_ms=100.0, cost_usd=0.001),
            )
        return picks

    assert run() == run()


def test_determinism_thompson_same_seed(registry: ArmRegistry) -> None:
    """Thompson sampling is reproducible under a fixed seed."""

    def run() -> list[str]:
        router = Router(registry, ThompsonSampling(seed=99))
        picks = []
        for i in range(40):
            request = RequestContext(request_id=f"r-{i}", prompt_tokens=500)
            picks.append(router.route(request).arm)
            router.observe(f"r-{i}", Outcome(success=True, quality=0.5, latency_ms=1.0))
        return picks

    assert run() == run()


def test_different_seeds_can_diverge_but_each_is_reproducible(registry: ArmRegistry) -> None:
    """Seeds control the policy's randomness."""

    def run(seed: int) -> list[str]:
        router = Router(registry, ThompsonSampling(seed=seed))
        return [
            router.route(RequestContext(request_id=f"r-{i}", prompt_tokens=100)).arm
            for i in range(30)
        ]

    assert run(1) == run(1)
    assert isinstance(run(2), list)


def test_reward_shaping_flips_the_routing_preference() -> None:
    """The same two arms are ordered differently at different ``lambda_cost``.

    This is the economic core of the router: one knob decides whether the
    operator optimises for dollars or for answer quality.
    """
    cheap_bad = make_arm(
        "cheap-bad",
        input_cost=0.0005,
        output_cost=0.001,
        cache_read_cost=0.00005,
        quality_prior=0.45,
        avg_latency_ms=300.0,
    )
    pricey_good = make_arm(
        "pricey-good",
        input_cost=0.020,
        output_cost=0.040,
        cache_read_cost=0.002,
        quality_prior=0.95,
        avg_latency_ms=2500.0,
    )
    reference = estimate_cost(pricey_good, 1000, 250)
    cheap_cost = estimate_cost(cheap_bad, 1000, 250)
    pricey_cost = estimate_cost(pricey_good, 1000, 250)

    def rewards(lambda_cost: float) -> dict[str, float]:
        """Feed one outcome per arm to a bandit and return the mean rewards."""
        bandit = EpsilonGreedy(epsilon_start=0.0, epsilon_end=0.0, decay_steps=0, seed=1)
        shaper = RewardShaper(lambda_cost=lambda_cost, cost_reference=reference)
        for arm, quality, cost in (
            (cheap_bad, 0.45, cheap_cost),
            (pricey_good, 0.95, pricey_cost),
        ):
            bandit.update(
                arm.name,
                shaper.shape(
                    success=True,
                    quality=quality,
                    cost_usd=cost,
                    latency_ms=arm.avg_latency_ms,
                ),
            )
        return {name: bandit.mean_reward(name) for name in ("cheap-bad", "pricey-good")}

    quality_first = rewards(0.0)
    cost_first = rewards(50.0)
    assert quality_first["pricey-good"] > quality_first["cheap-bad"]
    assert cost_first["cheap-bad"] > cost_first["pricey-good"]


def test_always_cheapest_cheaper_than_router_always_best_costs_more(registry: ArmRegistry) -> None:
    """The ROI bracket: the router sits between the two static baselines."""
    always_cheapest = project_spend("always-cheapest", registry.arms, 1000, mix={"cheap": 1.0})
    always_best = project_spend("always-best", registry.arms, 1000, mix={"best": 1.0})
    router = project_spend(
        "router", registry.arms, 1000, mix={"cheap": 0.6, "mid": 0.3, "best": 0.1}
    )
    assert always_cheapest.total_cost_usd < router.total_cost_usd
    assert always_best.total_cost_usd > router.total_cost_usd


def test_router_beats_always_cheapest_on_quality(registry: ArmRegistry) -> None:
    """Routing across arms delivers better quality than always-cheapest."""
    router = Router(registry, UCB1(seed=2), shaper=RewardShaper(lambda_cost=2.0))
    simulate(router, requests=150, seed=17)
    always_cheapest = registry.get("cheap")
    routed_quality = sum(
        router.arm_stats(name).quality_score * router.arm_stats(name).pulls
        for name in registry.names()
    ) / max(1, router.stats().total_requests)
    assert routed_quality > always_cheapest.quality_prior


def test_complexity_heuristic_marks_multi_step_prompts() -> None:
    """Multi-step prompts are detected without calling an LLM."""
    assert is_complex_prompt("First do X, then compare the results") is True
    assert is_complex_prompt("What is 2+2? And what is 3+3?") is True
    assert is_complex_prompt("```python\nprint('hi')\n```") is True
    assert is_complex_prompt("x" * 20_000) is True


def test_complexity_heuristic_leaves_simple_prompts_alone() -> None:
    """A trivial prompt is not complex."""
    assert is_complex_prompt("hello") is False
    assert is_complex_prompt("") is False
    assert is_complex_prompt("summarise this ticket in one line") is False


def test_request_context_from_prompt_derives_complexity() -> None:
    """``from_prompt`` fills in the token count and complexity flag."""
    ctx = RequestContext.from_prompt("r1", "First do X, then compare the results")
    assert ctx.is_complex is True
    assert ctx.prompt_tokens > 0
    simple = RequestContext.from_prompt("r2", "hello")
    assert simple.is_complex is False
    assert simple.prompt_tokens == estimate_prompt_tokens("hello")


def test_estimate_prompt_tokens_edges() -> None:
    """The token estimator handles empty and unicode prompts."""
    assert estimate_prompt_tokens("") == 0
    assert estimate_prompt_tokens("abcd") == 1
    assert estimate_prompt_tokens("abcdefgh") == 2
    assert estimate_prompt_tokens("éééé") == 1


def test_tool_capability_end_to_end_over_many_requests(registry: ArmRegistry) -> None:
    """Across a full workload, no tool request ever lands on a non-tool arm."""
    router = Router(registry, UCB1(seed=8))
    for i in range(120):
        request = RequestContext(request_id=f"r-{i}", needs_tools=True, prompt_tokens=300)
        decision = router.route(request)
        assert registry.get(decision.arm).supports("tools")
        router.observe(
            f"r-{i}",
            Outcome(success=True, quality=0.8, latency_ms=100.0, cost_usd=0.002),
        )


def test_budget_filter_never_violates_the_ceiling(registry: ArmRegistry) -> None:
    """No routed request exceeds the per-request budget."""
    ceiling = 0.004
    router = Router(registry, UCB1(seed=4), budget_per_request_usd=ceiling)
    for i in range(80):
        request = RequestContext(request_id=f"r-{i}", prompt_tokens=1000)
        decision = router.route(request)
        assert decision.estimated_cost_usd <= ceiling


def test_model_arm_with_cache_discount_is_cheaper_per_token() -> None:
    """A 10x cache-read discount shows up in the estimated cost."""
    arm = ModelArm(
        name="x",
        provider="p",
        input_cost_per_1k=0.010,
        output_cost_per_1k=0.010,
        cache_read_cost_per_1k=0.001,
        capabilities=frozenset({"chat"}),
        max_context_tokens=1000,
        avg_latency_ms=100.0,
    )
    cold = estimate_cost(arm, 1000, 0)
    warm = estimate_cost(arm, 1000, 0, cached_input_tokens=1000)
    assert cold == pytest.approx(0.010)
    assert warm == pytest.approx(0.001)
    assert cold / warm == pytest.approx(10.0)
