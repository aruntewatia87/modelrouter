"""modelrouter - cost-aware, health-aware model routing with multi-armed bandits.

Route LLM requests across providers by continuously balancing **price,
latency, quality and reliability** -- without paying an LLM to classify the
request, and without a static "always cheapest" table that breaks the moment
the cheap model degrades.

The public API is intentionally small::

    from modelrouter import ArmRegistry, ModelArm, RewardShaper, Router, UCB1

    registry = ArmRegistry([ModelArm(...), ...])
    router = Router(registry, UCB1(seed=7), shaper=RewardShaper(lambda_cost=2.0))
    decision = router.route(RequestContext.from_prompt("req-1", "hello"))
    router.observe("req-1", Outcome(success=True, quality=0.9, latency_ms=800,
                                     cost_usd=0.002))
"""

from modelrouter.bandits import (
    UCB1,
    Bandit,
    EpsilonGreedy,
    RewardShaper,
    SoftmaxRouter,
    ThompsonSampling,
)
from modelrouter.cost import (
    CostProjection,
    arm_cost_for_request,
    compare_projections,
    estimate_cost,
    project_policy,
    project_spend,
)
from modelrouter.health import ArmHealth, HealthSignal, HealthTracker
from modelrouter.models import CAPABILITIES, ArmRegistry, ArmStats, ModelArm
from modelrouter.router import (
    Outcome,
    RequestContext,
    Router,
    RouterStats,
    RoutingDecision,
    RoutingError,
    estimate_prompt_tokens,
    is_complex_prompt,
    select_arms_by_capability,
)

__version__ = "0.1.0"

__all__ = [
    "CAPABILITIES",
    "UCB1",
    "ArmHealth",
    "ArmRegistry",
    "ArmStats",
    "Bandit",
    "CostProjection",
    "EpsilonGreedy",
    "HealthSignal",
    "HealthTracker",
    "ModelArm",
    "Outcome",
    "RequestContext",
    "RewardShaper",
    "Router",
    "RouterStats",
    "RoutingDecision",
    "RoutingError",
    "SoftmaxRouter",
    "ThompsonSampling",
    "arm_cost_for_request",
    "compare_projections",
    "estimate_cost",
    "estimate_prompt_tokens",
    "is_complex_prompt",
    "project_policy",
    "project_spend",
    "select_arms_by_capability",
]
