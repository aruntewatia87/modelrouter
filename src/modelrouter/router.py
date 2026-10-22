"""The router: capability filter -> budget filter -> bandit -> feedback.

This is the module that turns the rest of the package into something a
production gateway can call. Every routing decision is recorded against the
request id, and every outcome feeds back into the bandit, the per-arm
statistics, and the health tracker.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from modelrouter.bandits import Bandit, RewardShaper
from modelrouter.cost import estimate_cost
from modelrouter.health import HealthTracker
from modelrouter.models import ArmRegistry, ArmStats, ModelArm

#: Keywords that suggest a multi-step or reasoning-heavy request.
COMPLEXITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "then",
        "after",
        "first",
        "finally",
        "step",
        "steps",
        "analyze",
        "analyse",
        "compare",
        "design",
        "architect",
        "why",
        "explain",
        "derive",
        "optimize",
        "optimise",
        "trade-off",
        "tradeoff",
        "refactor",
        "debug",
        "plan",
        "strategy",
    }
)

_CODE_BLOCK_RE = re.compile(r"```")
_CODE_HINT_RE = re.compile(
    r"\b(def|class|function|import|sudo|npm|pip|docker|kubectl|git|api|sql|SELECT)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything the router knows about a request before routing it.

    Attributes
    ----------
    request_id:
        Unique id, used to correlate the later :meth:`Router.observe` call.
    prompt_tokens:
        Estimated prompt tokens.
    needs_tools:
        Whether the request requires tool/function calling.
    needs_json:
        Whether the response must be strict JSON.
    is_complex:
        Whether the request looks multi-step or reasoning-heavy.
    tenant_id:
        Tenant the request belongs to (for per-tenant accounting).
    priority:
        Priority of the request; higher values should be served by better arms.
    required_capabilities:
        Extra capabilities the request needs beyond tools/json.
    """

    request_id: str
    prompt_tokens: int = 0
    needs_tools: bool = False
    needs_json: bool = False
    is_complex: bool = False
    tenant_id: str = "default"
    priority: int = 0
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")

    def required_capability_set(self) -> frozenset[str]:
        """Return the full set of capabilities this request requires."""
        caps = set(self.required_capabilities)
        if self.needs_tools:
            caps.add("tools")
        if self.needs_json:
            caps.add("json_mode")
        if self.prompt_tokens > 32_000:
            caps.add("long_context")
        return frozenset(caps)

    @classmethod
    def from_prompt(
        cls,
        request_id: str,
        prompt: str,
        *,
        needs_tools: bool = False,
        needs_json: bool = False,
        tenant_id: str = "default",
        priority: int = 0,
        required_capabilities: Iterable[str] = (),
    ) -> RequestContext:
        """Build a context from a raw prompt, deriving complexity heuristically.

        The complexity heuristic is deterministic and cheap -- no LLM call, no
        extra spend, and it can be computed inline in the request path:

        * a prompt above the token threshold is complex,
        * a prompt containing a fenced code block is complex,
        * a prompt with multiple question marks is complex,
        * a prompt containing multi-step keywords is complex.
        """
        tokens = estimate_prompt_tokens(prompt)
        return cls(
            request_id=request_id,
            prompt_tokens=tokens,
            needs_tools=needs_tools,
            needs_json=needs_json,
            is_complex=is_complex_prompt(prompt, tokens),
            tenant_id=tenant_id,
            priority=priority,
            required_capabilities=frozenset(required_capabilities),
        )


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of a routed request, fed back into the router.

    Attributes
    ----------
    success:
        Whether the call succeeded.
    quality:
        Observed answer quality in ``[0, 1]``.
    latency_ms:
        Observed latency in milliseconds.
    cost_usd:
        Observed spend in USD.
    error:
        Optional error message or type.
    """

    success: bool
    quality: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be within [0, 1]")
        if self.latency_ms < 0.0:
            raise ValueError("latency_ms must be >= 0")
        if self.cost_usd < 0.0:
            raise ValueError("cost_usd must be >= 0")


@dataclass(slots=True)
class RoutingDecision:
    """The router's choice for a request, kept so outcomes can be correlated.

    Attributes
    ----------
    request:
        The request context that was routed.
    arm:
        Name of the arm that was selected.
    estimated_cost_usd:
        Pre-flight cost estimate for the call.
    candidate_count:
        How many arms survived the capability and budget filters.
    reason:
        Short label describing why the arm was chosen.
    """

    request: RequestContext
    arm: str
    estimated_cost_usd: float
    candidate_count: int
    reason: str


@dataclass(slots=True)
class RouterStats:
    """Aggregate view of the router's traffic.

    Attributes
    ----------
    per_arm:
        Arm name -> statistics snapshot.
    total_spend_usd:
        Total observed spend.
    total_requests:
        Number of observed outcomes.
    total_successes:
        Number of successful outcomes.
    excluded_arms:
        Arms currently excluded by the health breaker or kill switch.
    """

    per_arm: dict[str, dict[str, float | int]] = field(default_factory=dict)
    total_spend_usd: float = 0.0
    total_requests: int = 0
    total_successes: int = 0
    excluded_arms: list[str] = field(default_factory=list)

    @property
    def avg_cost_usd(self) -> float:
        """Mean observed spend per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_spend_usd / self.total_requests

    @property
    def success_rate(self) -> float:
        """Fraction of observed requests that succeeded."""
        if self.total_requests == 0:
            return 0.0
        return self.total_successes / self.total_requests

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot of the router statistics."""
        return {
            "per_arm": self.per_arm,
            "total_spend_usd": round(self.total_spend_usd, 8),
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "success_rate": round(self.success_rate, 6),
            "avg_cost_usd": round(self.avg_cost_usd, 10),
            "excluded_arms": list(self.excluded_arms),
        }


class Router:
    """Route requests across model arms using a bandit policy.

    The routing pipeline is, in order:

    1. **Capability hard filter** -- an arm must support *every* capability
       the request needs. A tool-call request never lands on a model without
       tool support, no matter how attractive it looks.
    2. **Budget filter** -- an arm whose estimated cost exceeds
       ``budget_per_request_usd`` is dropped.
    3. **Kill switch** -- an arm whose observed success rate collapses is
       excluded until it recovers.
    4. **Health breaker** -- an arm whose sliding-window health fails is
       excluded from routing.
    5. **Bandit selection** -- the policy picks among the survivors.

    Parameters
    ----------
    registry:
        The arms available to the router.
    bandit:
        The policy used to select among eligible arms.
    shaper:
        Turns raw outcomes into the scalar reward the bandit learns from.
    health:
        The health tracker acting as a circuit breaker.
    budget_per_request_usd:
        Optional per-request spend ceiling. ``0`` or ``None`` disables it.
    min_pulls:
        Minimum number of pulls before the kill switch may trip an arm.
    kill_threshold:
        Success rate below which an arm is considered broken.
    estimated_completion_tokens:
        Completion tokens assumed when pre-flighting the cost.
    """

    def __init__(
        self,
        registry: ArmRegistry,
        bandit: Bandit,
        *,
        shaper: RewardShaper | None = None,
        health: HealthTracker | None = None,
        budget_per_request_usd: float | None = None,
        min_pulls: int = 20,
        kill_threshold: float = 0.5,
        kill_cooldown: int = 50,
        estimated_completion_tokens: int = 250,
    ) -> None:
        if budget_per_request_usd is not None and budget_per_request_usd < 0.0:
            raise ValueError("budget_per_request_usd must be >= 0")
        if min_pulls < 0:
            raise ValueError("min_pulls must be >= 0")
        if not 0.0 <= kill_threshold <= 1.0:
            raise ValueError("kill_threshold must be within [0, 1]")
        if kill_cooldown < 0:
            raise ValueError("kill_cooldown must be >= 0")
        self._registry = registry
        self._bandit = bandit
        self._shaper = shaper or RewardShaper()
        self._health = health or HealthTracker()
        self._budget = budget_per_request_usd
        self._min_pulls = min_pulls
        self._kill_threshold = kill_threshold
        self._kill_cooldown = kill_cooldown
        self._completion_tokens = estimated_completion_tokens
        self._decisions: dict[str, RoutingDecision] = {}
        self._stats: dict[str, ArmStats] = {}
        self._killed_at: dict[str, int] = {}
        self._probe_arms: set[str] = set()
        for arm in registry:
            self._stats[arm.name] = registry.stats_for(arm.name)
            self._health.register_if_absent(arm.name, arm.avg_latency_ms)

    @property
    def registry(self) -> ArmRegistry:
        """The arm registry backing this router."""
        return self._registry

    @property
    def bandit(self) -> Bandit:
        """The bandit policy used for selection."""
        return self._bandit

    @property
    def shaper(self) -> RewardShaper:
        """The reward shaper used to convert outcomes into rewards."""
        return self._shaper

    @property
    def health(self) -> HealthTracker:
        """The health tracker acting as the breaker."""
        return self._health

    @property
    def budget_per_request_usd(self) -> float | None:
        """The per-request spend ceiling, or ``None`` when unlimited."""
        return self._budget

    @property
    def min_pulls(self) -> int:
        """Minimum pulls before the kill switch may trip an arm."""
        return self._min_pulls

    @property
    def kill_threshold(self) -> float:
        """Rolling success rate below which an arm is killed."""
        return self._kill_threshold

    @property
    def kill_cooldown(self) -> int:
        """Number of pulls after which a killed arm is re-probed."""
        return self._kill_cooldown

    def eligible_arms(self, request: RequestContext) -> list[ModelArm]:
        """Return the arms that pass the capability and budget filters.

        The kill switch and the health breaker are applied separately by
        :meth:`candidate_arms` because they are transient exclusions, while
        this method answers the "is this arm even usable" question.
        """
        required = request.required_capability_set()
        candidates = self._registry.eligible_for(required)
        if self._budget is None or self._budget <= 0.0:
            return candidates
        within_budget = []
        for arm in candidates:
            estimate = estimate_cost(arm, request.prompt_tokens, self._completion_tokens)
            if estimate <= self._budget:
                within_budget.append(arm)
        return within_budget

    def killed_arms(self) -> list[str]:
        """Return the arms currently tripped by the kill switch.

        An arm is killed when it has been pulled at least ``min_pulls`` times
        and its *rolling* success rate is below ``kill_threshold``. Using the
        rolling rate (rather than the lifetime rate) means a transient outage
        does not permanently condemn an arm.

        A killed arm is re-probed after ``kill_cooldown`` further pulls have
        happened, giving it a chance to recover: the probe request is served
        by the arm itself and its outcome decides whether it is reinstated.
        """
        killed = []
        for name, stats in self._stats.items():
            if stats.pulls < self._min_pulls:
                continue
            if stats.rolling_success_rate >= self._kill_threshold:
                continue
            if self._killed_at.get(name, 0) + self._kill_cooldown <= stats.pulls:
                # Cool-down elapsed: the next pull of this arm is a probe.
                self._killed_at[name] = stats.pulls
                self._probe_arms.add(name)
                continue
            killed.append(name)
        return killed

    def candidate_arms(self, request: RequestContext) -> list[ModelArm]:
        """Return every arm that may serve ``request`` right now.

        Applies the capability filter, the budget filter, the kill switch and
        the health breaker. A killed arm that is due for a recovery probe is
        still eligible, so the router can learn that it works again.
        """
        eligible = self.eligible_arms(request)
        killed = set(self.killed_arms())
        return [
            arm
            for arm in eligible
            if (arm.name not in killed or arm.name in self._probe_arms)
            and self._health.is_healthy(arm.name)
        ]

    def route(self, request: RequestContext) -> RoutingDecision:
        """Select an arm for ``request`` and record the decision.

        Returns
        -------
        RoutingDecision
            The arm that was chosen and the context needed to correlate the
            later :meth:`observe` call.

        Raises
        ------
        RoutingError
            If no arm can serve the request.
        """
        candidates = self.candidate_arms(request)
        if not candidates:
            raise RoutingError(
                f"no eligible arm for request {request.request_id!r}: "
                f"capabilities={sorted(request.required_capability_set())}, "
                f"budget={self._budget}"
            )
        names = [arm.name for arm in candidates]
        probe = next((n for n in names if n in self._probe_arms), None)
        if probe is not None:
            # A killed arm that is due for a recovery probe gets the request,
            # so the router can confirm the arm is healthy again.
            chosen = probe
            self._probe_arms.discard(probe)
            reason = "probe"
        else:
            chosen = self._bandit.select(names, request)
            reason = "bandit"
        arm = self._registry.get(chosen)
        estimate = estimate_cost(arm, request.prompt_tokens, self._completion_tokens)
        decision = RoutingDecision(
            request=request,
            arm=chosen,
            estimated_cost_usd=estimate,
            candidate_count=len(candidates),
            reason=reason,
        )
        self._decisions[request.request_id] = decision
        return decision

    def route_arm(self, request: RequestContext) -> str:
        """Select an arm for ``request`` and return just its name."""
        return self.route(request).arm

    def observe(self, request_id: str, outcome: Outcome) -> float:
        """Feed an observed outcome back into the router.

        Updates the per-arm statistics, the health tracker, and the bandit
        (via the reward shaper).

        Parameters
        ----------
        request_id:
            The id of the request that was routed.
        outcome:
            The observed result.

        Returns
        -------
        float
            The shaped reward that was fed to the bandit.

        Raises
        ------
        KeyError
            If ``request_id`` was never routed.
        """
        decision = self._decisions.get(request_id)
        if decision is None:
            raise KeyError(f"unknown request_id: {request_id!r}")
        arm_name = decision.arm
        stats = self._stats[arm_name]
        stats.record(
            success=outcome.success,
            quality=outcome.quality,
            latency_ms=outcome.latency_ms,
            cost_usd=outcome.cost_usd,
            error_type=outcome.error,
        )
        self._health.record_outcome(
            arm_name,
            success=outcome.success,
            latency_ms=outcome.latency_ms,
            error_type=outcome.error,
        )
        reward = self._shaper.shape(
            success=outcome.success,
            quality=outcome.quality,
            cost_usd=outcome.cost_usd,
            latency_ms=outcome.latency_ms,
        )
        self._bandit.update(arm_name, reward, decision.request)
        return reward

    def decision(self, request_id: str) -> RoutingDecision:
        """Return the recorded decision for ``request_id``."""
        try:
            return self._decisions[request_id]
        except KeyError:
            raise KeyError(f"unknown request_id: {request_id!r}") from None

    def arm_stats(self, arm: str) -> ArmStats:
        """Return the running statistics for ``arm``."""
        try:
            return self._stats[arm]
        except KeyError:
            raise KeyError(f"unknown arm: {arm!r}") from None

    def stats(self) -> RouterStats:
        """Return per-arm aggregates plus total spend."""
        per_arm = {name: stats.as_dict() for name, stats in self._stats.items()}
        return RouterStats(
            per_arm=per_arm,
            total_spend_usd=sum(s.total_cost_usd for s in self._stats.values()),
            total_requests=sum(s.pulls for s in self._stats.values()),
            total_successes=sum(s.successes for s in self._stats.values()),
            excluded_arms=sorted(
                set(self.killed_arms()) | set(self._health.unhealthy_arms(list(self._stats)))
            ),
        )

    def traffic_mix(self) -> dict[str, float]:
        """Return the share of traffic each arm received, in ``[0, 1]``."""
        total = sum(s.pulls for s in self._stats.values())
        if total == 0:
            return dict.fromkeys(self._stats, 0.0)
        return {name: s.pulls / total for name, s in self._stats.items()}

    def __repr__(self) -> str:
        return (
            f"Router(arms={len(self._registry)}, bandit={type(self._bandit).__name__}, "
            f"budget={self._budget})"
        )


class RoutingError(RuntimeError):
    """Raised when no arm can serve a request."""


def estimate_prompt_tokens(prompt: str) -> int:
    """Roughly estimate the token count of a raw prompt string."""
    if not prompt:
        return 0
    return max(1, (len(prompt) + 3) // 4)


def is_complex_prompt(prompt: str, prompt_tokens: int | None = None) -> bool:
    """Deterministically decide whether a prompt is "complex".

    A prompt is complex when any of these hold:

    * it is longer than 2000 estimated tokens,
    * it contains a fenced code block,
    * it contains more than one question mark,
    * it contains a multi-step keyword such as ``"then"``, ``"analyze"`` or
      ``"compare"``.

    The heuristic is intentionally boring: it must be cheap, deterministic and
    free, because paying an LLM to classify a request is exactly the cost bug
    LiteLLM issue #46469 is about.
    """
    if not prompt:
        return False
    tokens = prompt_tokens if prompt_tokens is not None else estimate_prompt_tokens(prompt)
    if tokens > 2000:
        return True
    if _CODE_BLOCK_RE.search(prompt):
        return True
    if prompt.count("?") > 1:
        return True
    lowered = prompt.lower()
    words = set(re.findall(r"[a-z_]+", lowered))
    if words & COMPLEXITY_KEYWORDS:
        return True
    return bool(_CODE_HINT_RE.search(prompt))


def select_arms_by_capability(registry: ArmRegistry, capabilities: Sequence[str]) -> list[ModelArm]:
    """Convenience helper: the arms supporting every capability in ``capabilities``."""
    return registry.eligible_for(capabilities)
