"""Multi-armed bandit policies implemented from scratch.

Every policy implements the :class:`Bandit` protocol: ``select`` picks an arm
name given the candidate arms and an optional request context, and ``update``
feeds a (already shaped) scalar reward back into the policy's internal state.

The policies are deliberately simple and inspectable: a portfolio reviewer
should be able to read the maths and check it by hand.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: Context type accepted by the bandit callbacks. Kept intentionally loose so
#: callers can pass a request context object, a dict, or ``None``.
Context = Mapping[str, Any] | Any | None


@runtime_checkable
class Bandit(Protocol):
    """The contract shared by every routing policy."""

    def select(self, arms: Sequence[str], context: Context = None) -> str:
        """Choose the name of an arm to use for the next request."""
        ...

    def update(self, arm: str, reward: float, context: Context = None) -> None:
        """Record the observed ``reward`` for ``arm``."""
        ...


class _BaseBandit:
    """Shared bookkeeping: arm registry, reward history and pull counts."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._pulls: dict[str, int] = {}
        self._reward_sum: dict[str, float] = {}
        self._last_arm: str | None = None

    @property
    def rng(self) -> np.random.Generator:
        """The seeded random generator backing any stochastic decisions."""
        return self._rng

    def pulls(self, arm: str) -> int:
        """Number of times ``arm`` has been pulled."""
        return self._pulls.get(arm, 0)

    def mean_reward(self, arm: str) -> float:
        """Mean observed reward for ``arm`` (``0.0`` if never pulled)."""
        n = self._pulls.get(arm, 0)
        if n == 0:
            return 0.0
        return self._reward_sum.get(arm, 0.0) / n

    def total_pulls(self) -> int:
        """Total number of pulls across every arm."""
        return sum(self._pulls.values())

    def _record(self, arm: str, reward: float) -> None:
        """Store an observed reward and bump the pull count."""
        self._pulls[arm] = self._pulls.get(arm, 0) + 1
        self._reward_sum[arm] = self._reward_sum.get(arm, 0.0) + float(reward)

    def _require_arms(self, arms: Sequence[str]) -> list[str]:
        """Validate and return the candidate arm names."""
        if not arms:
            raise ValueError("select() requires at least one candidate arm")
        return list(arms)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pulls={self.total_pulls()})"


class EpsilonGreedy(_BaseBandit):
    """Epsilon-greedy: exploit the best-known arm, or explore with probability epsilon.

    Parameters
    ----------
    epsilon_start:
        Initial exploration probability.
    epsilon_end:
        Final exploration probability once ``decay_steps`` have elapsed.
    decay_steps:
        Number of pulls over which epsilon decays linearly from
        ``epsilon_start`` to ``epsilon_end``. Use ``0`` to disable decay.
    seed:
        Seed for the exploration coin flips.
    """

    def __init__(
        self,
        *,
        epsilon_start: float = 0.1,
        epsilon_end: float = 0.01,
        decay_steps: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= epsilon_start <= 1.0:
            raise ValueError("epsilon_start must be within [0, 1]")
        if not 0.0 <= epsilon_end <= 1.0:
            raise ValueError("epsilon_end must be within [0, 1]")
        if decay_steps < 0:
            raise ValueError("decay_steps must be >= 0")
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._decay_steps = decay_steps
        self._explorations = 0

    @property
    def epsilon(self) -> float:
        """Current exploration probability, after any decay."""
        if self._decay_steps <= 0:
            return self._epsilon_start
        progress = min(1.0, self.total_pulls() / self._decay_steps)
        return self._epsilon_start + (self._epsilon_end - self._epsilon_start) * progress

    @property
    def explorations(self) -> int:
        """How many times the policy chose to explore rather than exploit."""
        return self._explorations

    def select(self, arms: Sequence[str], context: Context = None) -> str:
        """Pick an arm, exploring with probability :attr:`epsilon`."""
        candidates = self._require_arms(arms)
        if self.rng.random() < self.epsilon:
            self._explorations += 1
            return candidates[int(self.rng.integers(0, len(candidates)))]
        return self._exploit(candidates)

    def _exploit(self, candidates: Sequence[str]) -> str:
        """Return the arm with the highest mean reward (ties break by name)."""
        return min(candidates, key=lambda name: (-self.mean_reward(name), name))

    def update(self, arm: str, reward: float, context: Context = None) -> None:
        """Record the observed reward for ``arm``."""
        self._record(arm, reward)


class UCB1(_BaseBandit):
    """Upper Confidence Bound 1.

    The score of arm ``i`` after ``t`` total pulls is::

        score_i = mean_reward_i + c * sqrt(ln(t) / n_i)

    Arms that have never been pulled get an infinite score, so they are always
    tried first. Ties are broken deterministically by arm name.
    """

    def __init__(self, *, c: float = 1.4, seed: int = 42) -> None:
        super().__init__(seed=seed)
        if c < 0.0:
            raise ValueError("c must be >= 0")
        self._c = c

    @property
    def c(self) -> float:
        """The exploration constant."""
        return self._c

    def score(self, arm: str, total_pulls: int | None = None) -> float:
        """Return the UCB1 score of ``arm``.

        ``math.inf`` is returned for an arm that has never been pulled.
        """
        n = self.pulls(arm)
        if n == 0:
            return math.inf
        t = self.total_pulls() if total_pulls is None else total_pulls
        if t <= 0:
            t = n
        return self.mean_reward(arm) + self._c * math.sqrt(math.log(t) / n)

    def select(self, arms: Sequence[str], context: Context = None) -> str:
        """Pick the arm with the highest UCB1 score."""
        candidates = self._require_arms(arms)
        return min(candidates, key=lambda name: (-self.score(name), name))

    def update(self, arm: str, reward: float, context: Context = None) -> None:
        """Record the observed reward for ``arm``."""
        self._record(arm, reward)


class ThompsonSampling(_BaseBandit):
    """Thompson sampling with a Beta-Bernoulli posterior.

    Each arm keeps an ``(alpha, beta)`` Beta posterior over its success
    probability. Selection samples from each posterior and takes the arm with
    the largest sample; ``update`` increments alpha on success and beta on
    failure.
    """

    def __init__(self, *, alpha: float = 1.0, beta: float = 1.0, seed: int = 42) -> None:
        super().__init__(seed=seed)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        self._alpha0 = alpha
        self._beta0 = beta
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}

    def alpha(self, arm: str) -> float:
        """Current Beta posterior alpha for ``arm``."""
        return self._alpha.get(arm, self._alpha0)

    def beta(self, arm: str) -> float:
        """Current Beta posterior beta for ``arm``."""
        return self._beta.get(arm, self._beta0)

    def posterior_mean(self, arm: str) -> float:
        """Posterior mean success probability for ``arm``."""
        a, b = self.alpha(arm), self.beta(arm)
        return a / (a + b)

    def select(self, arms: Sequence[str], context: Context = None) -> str:
        """Sample each posterior and return the arm with the largest draw."""
        candidates = self._require_arms(arms)
        samples = {
            name: float(self.rng.beta(self.alpha(name), self.beta(name))) for name in candidates
        }
        return min(candidates, key=lambda name: (-samples[name], name))

    def update(self, arm: str, reward: float, context: Context = None) -> None:
        """Update the Beta posterior from a binary reward.

        A reward of ``1.0`` counts as a success, ``0.0`` as a failure, and
        anything in between is treated as a partial success (a fractional
        alpha increment), which keeps the posterior valid for noisy signals.
        """
        value = float(reward)
        if value < 0.0 or value > 1.0:
            raise ValueError("Thompson sampling rewards must be within [0, 1]")
        self._alpha[arm] = self.alpha(arm) + value
        self._beta[arm] = self.beta(arm) + (1.0 - value)
        self._record(arm, value)


class SoftmaxRouter(_BaseBandit):
    """Softmax policy over per-arm scores.

    Arm scores are converted to a probability distribution with a temperature
    controlled softmax; the arm is then sampled from that distribution. A
    lower temperature concentrates the distribution on the best arm.
    """

    def __init__(self, *, temperature: float = 0.1, seed: int = 42) -> None:
        super().__init__(seed=seed)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature

    @property
    def temperature(self) -> float:
        """Softmax temperature."""
        return self._temperature

    def probabilities(self, arms: Sequence[str]) -> dict[str, float]:
        """Return the softmax selection probability for each arm."""
        if not arms:
            return {}
        scores = np.array([self.mean_reward(name) for name in arms], dtype=np.float64)
        scaled = scores / self._temperature
        scaled -= scaled.max()
        weights = np.exp(scaled)
        total = float(weights.sum())
        if total <= 0.0:
            uniform = 1.0 / len(arms)
            return dict.fromkeys(arms, uniform)
        return {name: float(p) for name, p in zip(arms, weights / total, strict=True)}

    def select(self, arms: Sequence[str], context: Context = None) -> str:
        """Sample an arm from the softmax distribution."""
        candidates = self._require_arms(arms)
        probs = self.probabilities(candidates)
        names = list(probs)
        weights = np.array([probs[name] for name in names], dtype=np.float64)
        weights = weights / weights.sum()
        idx = int(self.rng.choice(len(names), p=weights))
        return names[idx]

    def update(self, arm: str, reward: float, context: Context = None) -> None:
        """Record the observed reward for ``arm``."""
        self._record(arm, reward)


class RewardShaper:
    """Turn a raw request outcome into the scalar reward the bandits optimise.

    The shaped reward is::

        reward = quality * success
                 - lambda_cost * normalized_cost
                 - lambda_latency * normalized_latency

    where ``normalized_cost`` and ``normalized_latency`` are the observed cost
    and latency expressed as a fraction of the reference (typically the most
    expensive eligible arm and the slowest eligible arm). This is the piece
    that makes the router trade dollars against quality: raising
    ``lambda_cost`` makes cheap arms win more often, and lowering it lets the
    expensive-but-better arm take over.
    """

    def __init__(
        self,
        *,
        lambda_cost: float = 1.0,
        lambda_latency: float = 0.0,
        cost_reference: float = 1.0,
        latency_reference: float = 1.0,
    ) -> None:
        if lambda_cost < 0.0:
            raise ValueError("lambda_cost must be >= 0")
        if lambda_latency < 0.0:
            raise ValueError("lambda_latency must be >= 0")
        if cost_reference <= 0.0:
            raise ValueError("cost_reference must be positive")
        if latency_reference <= 0.0:
            raise ValueError("latency_reference must be positive")
        self._lambda_cost = lambda_cost
        self._lambda_latency = lambda_latency
        self._cost_reference = cost_reference
        self._latency_reference = latency_reference

    @property
    def lambda_cost(self) -> float:
        """Weight applied to normalized cost."""
        return self._lambda_cost

    @property
    def lambda_latency(self) -> float:
        """Weight applied to normalized latency."""
        return self._lambda_latency

    @property
    def cost_reference(self) -> float:
        """Cost value that normalizes to ``1.0``."""
        return self._cost_reference

    @property
    def latency_reference(self) -> float:
        """Latency value that normalizes to ``1.0``."""
        return self._latency_reference

    def normalize_cost(self, cost_usd: float) -> float:
        """Express ``cost_usd`` as a fraction of the cost reference."""
        return max(0.0, float(cost_usd)) / self._cost_reference

    def normalize_latency(self, latency_ms: float) -> float:
        """Express ``latency_ms`` as a fraction of the latency reference."""
        return max(0.0, float(latency_ms)) / self._latency_reference

    def shape(
        self,
        *,
        success: bool,
        quality: float,
        cost_usd: float,
        latency_ms: float,
    ) -> float:
        """Compute the shaped reward for one outcome.

        Parameters
        ----------
        success:
            Whether the call succeeded. A failure zeroes the quality term.
        quality:
            Observed quality in ``[0, 1]``.
        cost_usd:
            Observed spend in USD.
        latency_ms:
            Observed latency in milliseconds.

        Returns
        -------
        float
            The shaped scalar reward.
        """
        quality_term = float(quality) if success else 0.0
        return (
            quality_term
            - self._lambda_cost * self.normalize_cost(cost_usd)
            - self._lambda_latency * self.normalize_latency(latency_ms)
        )

    def __repr__(self) -> str:
        return (
            f"RewardShaper(lambda_cost={self._lambda_cost}, lambda_latency={self._lambda_latency})"
        )
