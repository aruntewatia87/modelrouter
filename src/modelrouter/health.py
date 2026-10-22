"""Health tracking: sliding-window success rate, p95 latency and a breaker.

An unhealthy arm is **excluded from routing**, exactly like an open circuit
breaker. Because the window slides, a recovered arm becomes eligible again
without any manual intervention.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class HealthSignal:
    """A single observation about an arm's health.

    Attributes
    ----------
    arm:
        Name of the arm the signal refers to.
    timestamp:
        Unix timestamp (seconds) of the observation.
    success:
        Whether the call succeeded.
    latency_ms:
        Observed latency in milliseconds.
    error_type:
        Optional error label, e.g. ``"rate_limit"`` or ``"timeout"``.
    """

    arm: str
    timestamp: float
    success: bool
    latency_ms: float
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not self.arm:
            raise ValueError("arm must be a non-empty string")
        if self.latency_ms < 0.0:
            raise ValueError("latency_ms must be >= 0")


@dataclass(slots=True)
class ArmHealth:
    """Rolling health state for a single arm."""

    arm: str
    baseline_latency_ms: float
    window_size: int = 50
    signals: deque[HealthSignal] = field(default_factory=deque)
    total_observations: int = 0
    total_failures: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.baseline_latency_ms <= 0.0:
            raise ValueError("baseline_latency_ms must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not self.signals:
            self.signals = deque(maxlen=self.window_size)

    def record(self, signal: HealthSignal) -> None:
        """Fold a health signal into the rolling window."""
        if self.signals.maxlen is None:
            self.signals = deque(self.signals, maxlen=self.window_size)
        self.signals.append(signal)
        self.total_observations += 1
        if not signal.success:
            self.total_failures += 1
            key = signal.error_type or "unknown"
            self.errors_by_type[key] = self.errors_by_type.get(key, 0) + 1

    @property
    def success_rate(self) -> float:
        """Success rate over the sliding window (``1.0`` when no signals yet)."""
        if not self.signals:
            return 1.0
        successes = sum(1 for s in self.signals if s.success)
        return successes / len(self.signals)

    @property
    def error_rate(self) -> float:
        """Failure rate over the sliding window."""
        return 1.0 - self.success_rate

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency over the sliding window.

        Uses :func:`numpy.percentile` with linear interpolation, matching the
        standard "p95" definition. Returns ``0.0`` when there are no signals.
        """
        if not self.signals:
            return 0.0
        latencies = np.array([s.latency_ms for s in self.signals], dtype=np.float64)
        return float(np.percentile(latencies, 95))

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency over the sliding window."""
        if not self.signals:
            return 0.0
        return float(np.mean([s.latency_ms for s in self.signals]))

    @property
    def latency_ratio(self) -> float:
        """Window p95 latency divided by the arm's baseline latency."""
        return self.p95_latency_ms / self.baseline_latency_ms

    @property
    def health_score(self) -> float:
        """Combined health in ``[0, 1]``.

        The score is the product of the window success rate and a latency
        factor. The latency factor is ``1.0`` while p95 stays within the
        arm's baseline and decays towards ``0.0`` as p95 climbs past it.
        """
        latency_factor = 1.0 / (1.0 + max(0.0, self.latency_ratio - 1.0))
        score = self.success_rate * latency_factor
        return float(min(1.0, max(0.0, score)))

    def as_dict(self) -> dict[str, float | int | dict[str, int]]:
        """Return a JSON-friendly snapshot of the health state."""
        return {
            "arm": self.arm,
            "observations": self.total_observations,
            "window": len(self.signals),
            "success_rate": round(self.success_rate, 6),
            "error_rate": round(self.error_rate, 6),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "latency_ratio": round(self.latency_ratio, 4),
            "health_score": round(self.health_score, 6),
            "errors_by_type": dict(self.errors_by_type),
        }


class HealthTracker:
    """Track per-arm health over a sliding window and act as a breaker.

    Parameters
    ----------
    window_size:
        Number of recent signals kept per arm.
    min_success_rate:
        Minimum window success rate for an arm to be considered healthy.
    max_latency_multiplier:
        How many times the arm's baseline latency its p95 may reach before
        the arm is considered unhealthy.
    """

    def __init__(
        self,
        *,
        window_size: int = 50,
        min_success_rate: float = 0.8,
        max_latency_multiplier: float = 3.0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be within [0, 1]")
        if max_latency_multiplier <= 0.0:
            raise ValueError("max_latency_multiplier must be positive")
        self._window_size = window_size
        self._min_success_rate = min_success_rate
        self._max_latency_multiplier = max_latency_multiplier
        self._arms: dict[str, ArmHealth] = {}

    @property
    def min_success_rate(self) -> float:
        """Success-rate threshold below which an arm is unhealthy."""
        return self._min_success_rate

    @property
    def max_latency_multiplier(self) -> float:
        """Latency multiplier above which an arm is unhealthy."""
        return self._max_latency_multiplier

    def register(self, arm: str, baseline_latency_ms: float) -> ArmHealth:
        """Register an arm with its baseline latency, creating its health state."""
        health = ArmHealth(
            arm=arm,
            baseline_latency_ms=baseline_latency_ms,
            window_size=self._window_size,
        )
        self._arms[arm] = health
        return health

    def ensure(self, arm: str, baseline_latency_ms: float = 1000.0) -> ArmHealth:
        """Return the health state for ``arm``, creating it if needed."""
        health = self._arms.get(arm)
        if health is None:
            health = self.register(arm, baseline_latency_ms)
        return health

    def register_if_absent(self, arm: str, baseline_latency_ms: float) -> ArmHealth:
        """Register ``arm`` only if it is not already being tracked.

        Unlike :meth:`register` this never clobbers existing health state, so
        a caller can safely offer a baseline without discarding observations
        that are already in the window.

        Returns
        -------
        ArmHealth
            The health state for ``arm``, existing or newly created.
        """
        health = self._arms.get(arm)
        if health is None:
            health = self.register(arm, baseline_latency_ms)
        return health

    def record(self, signal: HealthSignal) -> None:
        """Record a health signal against its arm."""
        health = self.ensure(signal.arm)
        health.record(signal)

    def record_outcome(
        self,
        arm: str,
        *,
        success: bool,
        latency_ms: float,
        timestamp: float = 0.0,
        error_type: str | None = None,
    ) -> None:
        """Convenience wrapper around :meth:`record`."""
        self.record(
            HealthSignal(
                arm=arm,
                timestamp=timestamp,
                success=success,
                latency_ms=latency_ms,
                error_type=error_type,
            )
        )

    def health(self, arm: str) -> ArmHealth:
        """Return the health state for ``arm``.

        Raises
        ------
        KeyError
            If the arm has never been seen.
        """
        try:
            return self._arms[arm]
        except KeyError:
            raise KeyError(f"unknown arm: {arm!r}") from None

    def success_rate(self, arm: str) -> float:
        """Window success rate for ``arm``."""
        return self.ensure(arm).success_rate

    def p95_latency_ms(self, arm: str) -> float:
        """Window p95 latency for ``arm``."""
        return self.ensure(arm).p95_latency_ms

    def health_score(self, arm: str) -> float:
        """Combined health score for ``arm``."""
        return self.ensure(arm).health_score

    def error_rate_by_type(self, arm: str) -> dict[str, int]:
        """Counts of each error type observed for ``arm``."""
        return dict(self.ensure(arm).errors_by_type)

    def is_healthy(
        self,
        arm: str,
        *,
        min_success_rate: float | None = None,
        max_latency_multiplier: float | None = None,
    ) -> bool:
        """Return ``True`` if ``arm`` is healthy enough to receive traffic.

        An arm with no observations is considered healthy (it has not had a
        chance to fail yet). Otherwise the window success rate must be at
        least ``min_success_rate`` and the window p95 latency must not exceed
        ``max_latency_multiplier`` times the arm's baseline latency.

        Parameters
        ----------
        arm:
            Arm to check.
        min_success_rate:
            Override for the tracker's default success-rate threshold.
        max_latency_multiplier:
            Override for the tracker's default latency multiplier.
        """
        health = self.ensure(arm)
        if not health.signals:
            return True
        threshold = self._min_success_rate if min_success_rate is None else min_success_rate
        multiplier = (
            self._max_latency_multiplier
            if max_latency_multiplier is None
            else max_latency_multiplier
        )
        if health.success_rate < threshold:
            return False
        return health.p95_latency_ms <= multiplier * health.baseline_latency_ms

    def healthy_arms(self, arms: Iterable[str]) -> list[str]:
        """Return only the arms that currently pass the breaker."""
        return [arm for arm in arms if self.is_healthy(arm)]

    def unhealthy_arms(self, arms: Iterable[str]) -> list[str]:
        """Return only the arms that are currently excluded by the breaker."""
        return [arm for arm in arms if not self.is_healthy(arm)]

    def as_dict(self) -> dict[str, dict[str, float | int | dict[str, int]]]:
        """Return a JSON-friendly snapshot of every tracked arm."""
        return {name: health.as_dict() for name, health in self._arms.items()}

    def __repr__(self) -> str:
        return f"HealthTracker(arms={len(self._arms)}, window={self._window_size})"
