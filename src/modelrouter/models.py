"""Model arm definitions, capability filtering and per-arm running statistics.

A *model arm* is one concrete (provider, model) pair that a request could be
sent to. :class:`ArmRegistry` holds the arms a router is allowed to choose
from, and :class:`ArmStats` accumulates the per-arm feedback the router learns
from.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

#: Capability tokens understood by :class:`ModelArm`.
CAPABILITIES: frozenset[str] = frozenset({"chat", "tools", "json_mode", "vision", "long_context"})


@dataclass(frozen=True, slots=True)
class ModelArm:
    """One routable model: its price sheet, capabilities and priors.

    Attributes
    ----------
    name:
        Unique identifier of the arm, e.g. ``"anthropic/claude-sonnet-4"``.
    provider:
        Vendor serving the model, e.g. ``"anthropic"``.
    input_cost_per_1k:
        USD per 1000 uncached input tokens.
    output_cost_per_1k:
        USD per 1000 output (completion) tokens.
    cache_read_cost_per_1k:
        USD per 1000 tokens read from the provider's prompt cache. This is a
        *separate* price sheet from ``input_cost_per_1k`` and omitting it is
        what makes "always cheapest" routing pick the wrong provider
        (see LiteLLM issue #38064).
    capabilities:
        Set of supported capabilities. Must be a subset of :data:`CAPABILITIES`.
    max_context_tokens:
        Largest prompt+completion the model accepts.
    avg_latency_ms:
        Baseline latency prior in milliseconds, used by the health tracker as
        the reference point for "is this arm slower than usual?".
    quality_prior:
        Prior belief about answer quality in ``[0, 1]``.
    reliability:
        Prior belief about success probability in ``[0, 1]``.
    """

    name: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    cache_read_cost_per_1k: float
    capabilities: frozenset[str]
    max_context_tokens: int
    avg_latency_ms: float
    quality_prior: float = 0.7
    reliability: float = 0.99

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if self.input_cost_per_1k < 0.0:
            raise ValueError("input_cost_per_1k must be >= 0")
        if self.output_cost_per_1k < 0.0:
            raise ValueError("output_cost_per_1k must be >= 0")
        if self.cache_read_cost_per_1k < 0.0:
            raise ValueError("cache_read_cost_per_1k must be >= 0")
        unknown = set(self.capabilities) - CAPABILITIES
        if unknown:
            raise ValueError(f"unknown capabilities: {sorted(unknown)}")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.avg_latency_ms <= 0.0:
            raise ValueError("avg_latency_ms must be positive")
        for label, value in (
            ("quality_prior", self.quality_prior),
            ("reliability", self.reliability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be within [0, 1]")

    def supports(self, capability: str) -> bool:
        """Return ``True`` if this arm supports ``capability``."""
        return capability in self.capabilities

    def supports_all(self, capabilities: Iterable[str]) -> bool:
        """Return ``True`` if this arm supports **every** required capability."""
        required = set(capabilities)
        return required.issubset(self.capabilities)

    def baseline_cost_per_1k(self) -> float:
        """Blended input+output price per 1000 tokens, used for cheapness ranking.

        Uses a 3:1 prompt-to-completion token ratio, the common shape of chat
        traffic. Cache reads are deliberately excluded: a cached request is not
        comparable to a cold one on price alone.
        """
        return (3.0 * self.input_cost_per_1k + self.output_cost_per_1k) / 4.0

    def __str__(self) -> str:
        return self.name


@dataclass(slots=True)
class ArmStats:
    """Running per-arm statistics fed by the router's feedback loop.

    Attributes
    ----------
    pulls:
        Number of times the arm has been selected.
    successes:
        Number of successful outcomes observed.
    total_cost_usd:
        Sum of observed spend for this arm.
    total_latency_ms:
        Sum of observed latency for this arm.
    rolling_latency_ms:
        Deque of recent latencies used for the rolling average.
    quality_score:
        Exponentially weighted moving average of observed quality in ``[0, 1]``.
    error_count:
        Number of failed outcomes.
    errors_by_type:
        Mapping of error type -> count.
    """

    pulls: int = 0
    successes: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    rolling_latency_ms: list[float] = field(default_factory=list)
    rolling_success: list[bool] = field(default_factory=list)
    quality_score: float = 0.0
    error_count: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    _ewma_alpha: float = 0.3
    _window: int = 100

    def record(
        self,
        *,
        success: bool,
        quality: float,
        latency_ms: float,
        cost_usd: float,
        error_type: str | None = None,
    ) -> None:
        """Fold one observed outcome into the running statistics.

        Parameters
        ----------
        success:
            Whether the call succeeded.
        quality:
            Observed answer quality in ``[0, 1]``.
        latency_ms:
            Observed latency in milliseconds.
        cost_usd:
            Observed spend in USD.
        error_type:
            Optional error label recorded when ``success`` is ``False``.
        """
        self.pulls += 1
        self.total_cost_usd += cost_usd
        self.total_latency_ms += latency_ms
        if success:
            self.successes += 1
        else:
            self.error_count += 1
            key = error_type or "unknown"
            self.errors_by_type[key] = self.errors_by_type.get(key, 0) + 1
        self.rolling_latency_ms.append(latency_ms)
        self.rolling_success.append(bool(success))
        if len(self.rolling_latency_ms) > self._window:
            del self.rolling_latency_ms[: len(self.rolling_latency_ms) - self._window]
        if len(self.rolling_success) > self._window:
            del self.rolling_success[: len(self.rolling_success) - self._window]
        self.quality_score = self._ewma(self.quality_score, quality)

    def _ewma(self, previous: float, sample: float) -> float:
        """Update the exponentially weighted moving average of quality."""
        if self.pulls == 1:
            return float(sample)
        alpha = self._ewma_alpha
        return (1.0 - alpha) * previous + alpha * float(sample)

    @property
    def success_rate(self) -> float:
        """Fraction of observed calls that succeeded (``0.0`` before any pull)."""
        if self.pulls == 0:
            return 0.0
        return self.successes / self.pulls

    @property
    def rolling_success_rate(self) -> float:
        """Success rate over the most recent window of calls.

        This is the signal the router's kill switch uses: unlike the lifetime
        :attr:`success_rate` it can recover, so a transient outage does not
        permanently condemn an arm.
        """
        if not self.rolling_success:
            return 0.0
        return sum(self.rolling_success) / len(self.rolling_success)

    @property
    def avg_latency_ms(self) -> float:
        """Mean latency over all observed calls."""
        if self.pulls == 0:
            return 0.0
        return self.total_latency_ms / self.pulls

    @property
    def rolling_avg_latency_ms(self) -> float:
        """Mean latency over the most recent window of calls."""
        if not self.rolling_latency_ms:
            return 0.0
        return sum(self.rolling_latency_ms) / len(self.rolling_latency_ms)

    @property
    def avg_cost_usd(self) -> float:
        """Mean spend per observed call."""
        if self.pulls == 0:
            return 0.0
        return self.total_cost_usd / self.pulls

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly snapshot of the statistics."""
        return {
            "pulls": self.pulls,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 6),
            "error_count": self.error_count,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "avg_cost_usd": round(self.avg_cost_usd, 8),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "rolling_avg_latency_ms": round(self.rolling_avg_latency_ms, 3),
            "quality_score": round(self.quality_score, 6),
        }

    def __repr__(self) -> str:
        return (
            f"ArmStats(pulls={self.pulls}, success_rate={self.success_rate:.3f}, "
            f"quality={self.quality_score:.3f})"
        )


class ArmRegistry:
    """A collection of :class:`ModelArm` objects with capability filtering.

    The registry is the *hard filter* in front of the router: an arm that does
    not support every capability a request needs must never be selected, no
    matter how good the bandit thinks it is.
    """

    def __init__(self, arms: Iterable[ModelArm] = ()) -> None:
        self._arms: dict[str, ModelArm] = {}
        for arm in arms:
            self.register(arm)

    def register(self, arm: ModelArm) -> ModelArm:
        """Register (or replace) an arm and return it."""
        self._arms[arm.name] = arm
        return arm

    def get(self, name: str) -> ModelArm:
        """Look up an arm by name.

        Raises
        ------
        KeyError
            If no arm is registered under ``name``.
        """
        try:
            return self._arms[name]
        except KeyError:
            raise KeyError(f"unknown arm: {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._arms

    def __len__(self) -> int:
        return len(self._arms)

    def __iter__(self) -> Iterator[ModelArm]:
        return iter(self._arms.values())

    @property
    def arms(self) -> list[ModelArm]:
        """All registered arms, in registration order."""
        return list(self._arms.values())

    def names(self) -> list[str]:
        """Names of all registered arms, in registration order."""
        return list(self._arms)

    def eligible_for(
        self, capabilities: Iterable[str], *, min_context_tokens: int = 0
    ) -> list[ModelArm]:
        """Return the arms supporting **every** required capability.

        Parameters
        ----------
        capabilities:
            Required capabilities. An arm qualifies only if it supports all of
            them (this is the tool-call safety rule: a request needing tools
            never lands on an arm without tool support).
        min_context_tokens:
            Optional minimum context window required by the request.

        Returns
        -------
        list[ModelArm]
            Eligible arms in registration order.
        """
        required = set(capabilities)
        return [
            arm
            for arm in self._arms.values()
            if arm.supports_all(required) and arm.max_context_tokens >= min_context_tokens
        ]

    def cheapest_for(self, capabilities: Iterable[str]) -> ModelArm | None:
        """Return the eligible arm with the lowest blended price, if any."""
        eligible = self.eligible_for(capabilities)
        if not eligible:
            return None
        return min(eligible, key=lambda arm: (arm.baseline_cost_per_1k(), arm.name))

    def best_for(self, capabilities: Iterable[str]) -> ModelArm | None:
        """Return the eligible arm with the highest quality prior, if any.

        Ties break on the lower blended price and then on the name so the
        result is deterministic.
        """
        eligible = self.eligible_for(capabilities)
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda arm: (-arm.quality_prior, arm.baseline_cost_per_1k(), arm.name),
        )

    def stats(self) -> dict[str, ArmStats]:
        """Return the per-arm statistics map, creating entries on demand."""
        return {name: self.stats_for(name) for name in self._arms}

    def stats_for(self, name: str) -> ArmStats:
        """Return the shared statistics object for ``name``.

        The registry owns the statistics table so that every consumer (the
        router, a dashboard, a report) sees the same numbers.

        Raises
        ------
        KeyError
            If no arm is registered under ``name``.
        """
        if not hasattr(self, "_stats_table"):
            self._stats_table: dict[str, ArmStats] = {}
        try:
            return self._stats_table.setdefault(name, ArmStats())
        except KeyError:  # pragma: no cover - defensive
            raise KeyError(f"unknown arm: {name!r}") from None

    def __repr__(self) -> str:
        return f"ArmRegistry({len(self._arms)} arms: {', '.join(self._arms)})"


def estimate_tokens(text: str) -> int:
    """Roughly estimate the token count of ``text``.

    Uses the standard ~4 characters per token heuristic, which is good enough
    for cost pre-flight checks and for the complexity heuristic.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
