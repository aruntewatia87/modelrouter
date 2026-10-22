"""Cost estimation and spend projection.

The pricing model here deliberately keeps **three separate prices** per arm:
input, output, and *cache read*. Collapsing cache reads into the input price
is the exact bug that makes "always cheapest" routing pick the wrong provider
on an input+output tie -- see LiteLLM issue #38064, where a 10x mis-price
flipped the chosen provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from modelrouter.models import ModelArm


def estimate_cost(
    arm: ModelArm,
    prompt_tokens: int,
    completion_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Estimate the USD cost of one call against ``arm``.

    The cost model is::

        cost = (uncached_input / 1000) * input_cost_per_1k
             + (cached_input  / 1000) * cache_read_cost_per_1k
             + (completion     / 1000) * output_cost_per_1k

    ``cached_input_tokens`` is the portion of the prompt served from the
    provider's prompt cache. It is billed at ``cache_read_cost_per_1k``, which
    is usually 10x cheaper than the input price -- ignoring it inflates the
    estimated cost of a cache-warm arm and can invert a routing decision.

    Parameters
    ----------
    arm:
        The model arm to price.
    prompt_tokens:
        Total prompt tokens, including any cached portion.
    completion_tokens:
        Tokens produced by the model.
    cached_input_tokens:
        How many of the prompt tokens were served from cache.

    Returns
    -------
    float
        Estimated cost in USD.
    """
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    cached = max(0, int(cached_input_tokens))
    if cached > prompt:
        raise ValueError("cached_input_tokens cannot exceed prompt_tokens")
    uncached = prompt - cached
    return (
        uncached / 1000.0 * arm.input_cost_per_1k
        + cached / 1000.0 * arm.cache_read_cost_per_1k
        + completion / 1000.0 * arm.output_cost_per_1k
    )


def arm_cost_for_request(
    arm: ModelArm,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_hit_ratio: float = 0.0,
) -> float:
    """Estimate cost with a cache hit *ratio* instead of an absolute count.

    Parameters
    ----------
    arm:
        The model arm to price.
    prompt_tokens:
        Total prompt tokens.
    completion_tokens:
        Tokens produced by the model.
    cache_hit_ratio:
        Fraction of the prompt served from cache, in ``[0, 1]``.

    Returns
    -------
    float
        Estimated cost in USD.
    """
    if not 0.0 <= cache_hit_ratio <= 1.0:
        raise ValueError("cache_hit_ratio must be within [0, 1]")
    prompt = max(0, int(prompt_tokens))
    cached = round(prompt * cache_hit_ratio)
    return estimate_cost(arm, prompt, completion_tokens, cached)


@dataclass(slots=True)
class CostProjection:
    """Projected spend for a routing policy over a number of requests.

    Attributes
    ----------
    policy:
        Name of the policy that was projected.
    total_requests:
        Number of requests the projection covers.
    total_cost_usd:
        Sum of the projected per-request costs.
    per_arm:
        Mapping of arm name -> projected cost for that arm.
    per_arm_requests:
        Mapping of arm name -> number of requests routed to that arm.
    quality:
        Expected quality per request, in ``[0, 1]``.
    success_rate:
        Expected success rate per request, in ``[0, 1]``.
    """

    policy: str
    total_requests: int
    total_cost_usd: float
    per_arm: dict[str, float] = field(default_factory=dict)
    per_arm_requests: dict[str, int] = field(default_factory=dict)
    quality: float = 0.0
    success_rate: float = 0.0

    @property
    def avg_cost_usd(self) -> float:
        """Mean projected cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost_usd / self.total_requests

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot of the projection."""
        return {
            "policy": self.policy,
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "avg_cost_usd": round(self.avg_cost_usd, 10),
            "per_arm": {k: round(v, 8) for k, v in self.per_arm.items()},
            "per_arm_requests": dict(self.per_arm_requests),
            "quality": round(self.quality, 6),
            "success_rate": round(self.success_rate, 6),
        }

    def __repr__(self) -> str:
        return (
            f"CostProjection(policy={self.policy!r}, requests={self.total_requests}, "
            f"total=${self.total_cost_usd:.6f}, avg=${self.avg_cost_usd:.8f})"
        )


def project_spend(
    policy_name: str,
    arms: Sequence[ModelArm],
    requests: int,
    *,
    prompt_tokens: int = 1000,
    completion_tokens: int = 250,
    mix: dict[str, float] | None = None,
) -> CostProjection:
    """Project spend for a *static* routing mix over ``requests`` requests.

    Parameters
    ----------
    policy_name:
        Label for the policy being projected (e.g. ``"always-cheapest"``).
    arms:
        The arms the policy may choose from.
    requests:
        Number of requests to project.
    prompt_tokens:
        Prompt tokens per request.
    completion_tokens:
        Completion tokens per request.
    mix:
        Optional mapping of arm name -> share of traffic in ``[0, 1]``. Shares
        are normalised; arms not mentioned receive zero traffic. When ``mix``
        is ``None`` the traffic is split evenly.

    Returns
    -------
    CostProjection
        The projected spend, broken down per arm.
    """
    if requests < 0:
        raise ValueError("requests must be >= 0")
    if not arms:
        raise ValueError("at least one arm is required")
    by_name = {arm.name: arm for arm in arms}

    if mix is None:
        raw = dict.fromkeys(by_name, 1.0)
    else:
        unknown = set(mix) - set(by_name)
        if unknown:
            raise ValueError(f"mix references unknown arms: {sorted(unknown)}")
        raw = {name: float(mix.get(name, 0.0)) for name in by_name}
    total_share = sum(raw.values())
    if total_share <= 0.0:
        raise ValueError("mix shares must sum to a positive number")

    per_arm: dict[str, float] = {}
    per_arm_requests: dict[str, int] = {}
    total = 0.0
    for name, share in raw.items():
        arm = by_name[name]
        per_request = estimate_cost(arm, prompt_tokens, completion_tokens)
        share_norm = share / total_share
        arm_requests = share_norm * requests
        arm_cost = per_request * arm_requests
        per_arm[name] = arm_cost
        per_arm_requests[name] = round(arm_requests)
        total += arm_cost

    return CostProjection(
        policy=policy_name,
        total_requests=requests,
        total_cost_usd=total,
        per_arm=per_arm,
        per_arm_requests=per_arm_requests,
    )


def project_policy(
    policy_name: str,
    arms: Sequence[ModelArm],
    requests: int,
    *,
    prompt_tokens: int = 1000,
    completion_tokens: int = 250,
    mix: dict[str, float] | None = None,
) -> CostProjection:
    """Project spend under the router's *observed* traffic mix.

    This is the honest counterpart to :func:`project_spend`: it prices the mix
    the router actually produced instead of a hypothetical one.
    """
    projection = project_spend(
        policy_name,
        arms,
        requests,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        mix=mix,
    )
    total = sum(projection.per_arm_requests.values())
    if total > 0:
        scaled = {name: cost * (requests / total) for name, cost in projection.per_arm.items()}
        projection.total_cost_usd = sum(scaled.values())
        projection.per_arm = scaled
    return projection


def compare_projections(*projections: CostProjection) -> dict[str, CostProjection]:
    """Return a name -> projection mapping for side-by-side comparison."""
    return {p.policy: p for p in projections}
