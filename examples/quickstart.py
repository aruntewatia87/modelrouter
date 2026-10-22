#!/usr/bin/env python3
"""End-to-end demo: a health-aware, cost-aware router in a feedback loop.

Four simulated model arms serve 200 identical requests. The cheapest arm
starts failing part-way through, and the router has to notice (through the
health breaker and the kill switch) and route around it -- without ever
paying an LLM to classify the request.

The demo compares three policies over the *same* request stream:

* ``always-cheapest`` -- the static routing table everybody starts with,
* ``always-best``     -- the "just use the good model" baseline,
* ``router``          -- UCB1 + reward shaping + a health breaker.
"""

from __future__ import annotations

import numpy as np

from modelrouter import (
    UCB1,
    ArmRegistry,
    HealthTracker,
    ModelArm,
    Outcome,
    RequestContext,
    RewardShaper,
    Router,
    estimate_cost,
    project_spend,
)

PROMPT_TOKENS = 1200
COMPLETION_TOKENS = 300
TOTAL_REQUESTS = 200
FAILURE_STARTS_AT = 90
FAILURE_RATE = 0.9
FALLBACK_RETRIES = 1


def build_arms() -> list[ModelArm]:
    """Four arms: cheap/fast, mid, expensive/best, and one that starts failing."""
    return [
        ModelArm(
            name="legacy-mini",
            provider="openai",
            input_cost_per_1k=0.00008,
            output_cost_per_1k=0.00030,
            cache_read_cost_per_1k=0.000008,
            capabilities=frozenset({"chat", "json_mode"}),
            max_context_tokens=128_000,
            avg_latency_ms=380.0,
            quality_prior=0.55,
            reliability=0.90,
        ),
        ModelArm(
            name="cheap-mini",
            provider="openai",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.00060,
            cache_read_cost_per_1k=0.000015,
            capabilities=frozenset({"chat", "json_mode"}),
            max_context_tokens=128_000,
            avg_latency_ms=420.0,
            quality_prior=0.62,
            reliability=0.97,
        ),
        ModelArm(
            name="mid-haiku",
            provider="anthropic",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.005,
            cache_read_cost_per_1k=0.0001,
            capabilities=frozenset({"chat", "tools", "json_mode", "vision"}),
            max_context_tokens=200_000,
            avg_latency_ms=780.0,
            quality_prior=0.80,
            reliability=0.99,
        ),
        ModelArm(
            name="flagship-sonnet",
            provider="anthropic",
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            cache_read_cost_per_1k=0.0003,
            capabilities=frozenset({"chat", "tools", "json_mode", "vision", "long_context"}),
            max_context_tokens=200_000,
            avg_latency_ms=1450.0,
            quality_prior=0.93,
            reliability=0.995,
        ),
    ]


def call(arm: ModelArm, rng: np.random.Generator, *, failing: bool) -> Outcome:
    """Simulate one call against ``arm``."""
    if failing and rng.random() < FAILURE_RATE:
        return Outcome(
            success=False,
            quality=0.0,
            latency_ms=float(rng.normal(3000.0, 400.0)),
            cost_usd=0.0,
            error="rate_limit",
        )
    quality = float(np.clip(rng.normal(arm.quality_prior, 0.05), 0.0, 1.0))
    latency = float(max(1.0, rng.normal(arm.avg_latency_ms, arm.avg_latency_ms * 0.08)))
    cost = estimate_cost(arm, PROMPT_TOKENS, COMPLETION_TOKENS)
    return Outcome(success=True, quality=quality, latency_ms=latency, cost_usd=cost)


def run_static_policy(
    arms: list[ModelArm],
    primary_index: int,
    *,
    seed: int,
) -> tuple[float, float, float]:
    """Run a fixed-primary policy with one fallback retry.

    A failed attempt is billed for its input tokens (the provider did the
    work) and the request is retried on the next-best arm. This is what makes
    "always cheapest" expensive in practice: the cheap model *and* the
    fallback both get paid for.

    Returns
    -------
    tuple[float, float, float]
        ``(total_cost_usd, success_rate, mean_quality)``
    """
    rng = np.random.default_rng(seed)
    total_cost = 0.0
    successes = 0
    quality_sum = 0.0
    order = sorted(
        range(len(arms)),
        key=lambda i: (arms[i].baseline_cost_per_1k(), arms[i].name),
    )
    primary = order[primary_index]
    fallback = order[(primary_index + 1) % len(order)]
    for i in range(TOTAL_REQUESTS):
        arm = arms[primary]
        failing = arm.name == "legacy-mini" and i >= FAILURE_STARTS_AT
        outcome = call(arm, rng, failing=failing)
        if not outcome.success:
            total_cost += arm.input_cost_per_1k * PROMPT_TOKENS / 1000.0
            if FALLBACK_RETRIES:
                fb = arms[fallback]
                outcome = call(fb, rng, failing=False)
        total_cost += outcome.cost_usd
        if outcome.success:
            successes += 1
            quality_sum += outcome.quality
    return total_cost, successes / TOTAL_REQUESTS, quality_sum / TOTAL_REQUESTS


def main() -> None:
    """Run the simulation and print the router's report."""
    arms = build_arms()
    registry = ArmRegistry(arms)
    by_name = {arm.name: arm for arm in arms}

    # The cost reference is the most expensive arm's price for one request:
    # normalizing against it makes lambda_cost directly comparable.
    cost_reference = max(estimate_cost(arm, PROMPT_TOKENS, COMPLETION_TOKENS) for arm in arms)
    latency_reference = max(arm.avg_latency_ms for arm in arms)

    router = Router(
        registry,
        UCB1(c=1.4, seed=2026),
        shaper=RewardShaper(
            lambda_cost=0.35,
            lambda_latency=0.1,
            cost_reference=cost_reference,
            latency_reference=latency_reference,
        ),
        health=HealthTracker(window_size=30, min_success_rate=0.7, max_latency_multiplier=3.0),
        budget_per_request_usd=cost_reference,
        min_pulls=15,
        kill_threshold=0.5,
        kill_cooldown=40,
    )

    rng = np.random.default_rng(7)
    print(f"Simulating {TOTAL_REQUESTS} requests across {len(arms)} model arms")
    print(f"  cost reference   : ${cost_reference:.6f} per request")
    print(f"  latency reference: {latency_reference:.0f} ms")
    print(
        "  policy           : UCB1(c=1.4) + reward shaping (lambda_cost=0.35, lambda_latency=0.1)"
    )
    print(f"  'legacy-mini' (the cheapest arm) starts failing at request {FAILURE_STARTS_AT}\n")

    for i in range(TOTAL_REQUESTS):
        request = RequestContext(
            request_id=f"req-{i:04d}",
            prompt_tokens=PROMPT_TOKENS,
            needs_tools=(i % 5 == 0),
            tenant_id="acme",
        )
        decision = router.route(request)
        arm = by_name[decision.arm]
        failing = arm.name == "legacy-mini" and i >= FAILURE_STARTS_AT
        router.observe(request.request_id, call(arm, rng, failing=failing))

    stats = router.stats()
    print("Per-arm results")
    print(
        f"  {'arm':<18} {'pulls':>6} {'share':>7} {'ok%':>7} {'quality':>8} "
        f"{'avg $/req':>11} {'p95 ms':>8} {'health':>7}"
    )
    print(f"  {'-' * 18} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 11} {'-' * 8} {'-' * 7}")
    for name in registry.names():
        row = stats.per_arm[name]
        health = router.health.health_score(name)
        p95 = router.health.p95_latency_ms(name)
        share = row["pulls"] / stats.total_requests if stats.total_requests else 0.0
        avg_cost = row["total_cost_usd"] / row["pulls"] if row["pulls"] else 0.0
        print(
            f"  {name:<18} {row['pulls']:>6} {share:>6.1%} "
            f"{row['success_rate']:>6.1%} {row['quality_score']:>8.3f} "
            f"{avg_cost:>11.6f} {p95:>8.0f} {health:>7.3f}"
        )

    print(f"\nTotal spend      : ${stats.total_spend_usd:.6f} over {stats.total_requests} requests")
    print(f"Overall success  : {stats.success_rate:.1%}")
    print(f"Excluded arms    : {', '.join(stats.excluded_arms) or '(none)'}")

    mix = router.traffic_mix()
    router_projection = project_spend(
        "router (learned mix)",
        arms,
        TOTAL_REQUESTS,
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        mix=mix,
    )
    cheapest_arm = min(arms, key=lambda a: a.baseline_cost_per_1k())
    best_arm = max(arms, key=lambda a: a.quality_prior)
    cheapest_projection = project_spend(
        "always-cheapest",
        arms,
        TOTAL_REQUESTS,
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        mix={cheapest_arm.name: 1.0},
    )
    best_projection = project_spend(
        "always-best",
        arms,
        TOTAL_REQUESTS,
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        mix={best_arm.name: 1.0},
    )

    print("\nCost projection (200 requests, same token volume, list price)")
    print(f"  {'policy':<22} {'total $':>12} {'avg $/req':>12}")
    print(f"  {'-' * 22} {'-' * 12} {'-' * 12}")
    for projection in (cheapest_projection, router_projection, best_projection):
        print(
            f"  {projection.policy:<22} {projection.total_cost_usd:>12.6f} "
            f"{projection.avg_cost_usd:>12.6f}"
        )
    savings = (
        best_projection.total_cost_usd - router_projection.total_cost_usd
    ) / best_projection.total_cost_usd
    print(
        f"\n  router saves {savings:.1%} vs always-best on list price "
        f"(${best_projection.total_cost_usd - router_projection.total_cost_usd:.6f})"
    )

    # Honest comparison: the same 200 requests through all three policies,
    # with a fallback retry when the primary arm fails.
    cheapest_cost, cheapest_ok, cheapest_quality = run_static_policy(arms, 0, seed=21)
    best_cost, best_ok, best_quality = run_static_policy(arms, len(arms) - 1, seed=21)
    router_quality = (
        sum(
            router.arm_stats(name).quality_score * router.arm_stats(name).pulls
            for name in registry.names()
        )
        / stats.total_requests
    )
    print("\nRealistic comparison (same 200 requests, 1 fallback retry on failure)")
    print(f"  {'policy':<16} {'total $':>11} {'success':>9} {'avg quality':>12}")
    print(f"  {'-' * 16} {'-' * 11} {'-' * 9} {'-' * 12}")
    print(
        f"  {'always-cheapest':<16} {cheapest_cost:>11.6f} {cheapest_ok:>8.1%} "
        f"{cheapest_quality:>12.3f}"
    )
    print(
        f"  {'router':<16} {stats.total_spend_usd:>11.6f} {stats.success_rate:>8.1%} "
        f"{router_quality:>12.3f}"
    )
    print(f"  {'always-best':<16} {best_cost:>11.6f} {best_ok:>8.1%} {best_quality:>12.3f}")
    print("\n  always-cheapest is the cheapest on paper but collapses when the cheap arm degrades;")
    print(
        "  always-best is the most reliable and the most expensive; the router sits between them."
    )

    # The cost/quality knob: the SAME two arms, ordered differently depending
    # on how much a dollar is worth to the operator.
    print("\nReward-shaping knob (same outcome, different lambda_cost)")
    cheap = by_name["cheap-mini"]
    pricey = by_name["flagship-sonnet"]
    cheap_cost = estimate_cost(cheap, PROMPT_TOKENS, COMPLETION_TOKENS)
    pricey_cost = estimate_cost(pricey, PROMPT_TOKENS, COMPLETION_TOKENS)
    print(f"  {'lambda_cost':<12} {'cheap-mini':>12} {'flagship':>12}  preferred")
    for lc in (0.0, 0.35, 5.0):
        shaper = RewardShaper(
            lambda_cost=lc,
            cost_reference=cost_reference,
            latency_reference=latency_reference,
        )
        r_cheap = shaper.shape(
            success=True,
            quality=cheap.quality_prior,
            cost_usd=cheap_cost,
            latency_ms=cheap.avg_latency_ms,
        )
        r_pricey = shaper.shape(
            success=True,
            quality=pricey.quality_prior,
            cost_usd=pricey_cost,
            latency_ms=pricey.avg_latency_ms,
        )
        preferred = "flagship-sonnet" if r_pricey > r_cheap else "cheap-mini"
        print(f"  {lc:<12} {r_cheap:>12.4f} {r_pricey:>12.4f}  {preferred}")


if __name__ == "__main__":
    main()
