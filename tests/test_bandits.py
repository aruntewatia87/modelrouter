"""Tests for the multi-armed bandit policies and the reward shaper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from modelrouter import (
    UCB1,
    Bandit,
    EpsilonGreedy,
    RewardShaper,
    SoftmaxRouter,
    ThompsonSampling,
)


def test_all_policies_satisfy_the_bandit_protocol() -> None:
    """Every policy implements ``select`` and ``update``."""
    for bandit in (
        EpsilonGreedy(seed=1),
        UCB1(seed=1),
        ThompsonSampling(seed=1),
        SoftmaxRouter(seed=1),
    ):
        assert isinstance(bandit, Bandit)
        assert bandit.select(["a", "b"]) in {"a", "b"}
        bandit.update("a", 0.5)


def test_select_rejects_empty_arm_list() -> None:
    """Selecting from no arms is a programming error."""
    with pytest.raises(ValueError, match="at least one candidate arm"):
        UCB1().select([])


def test_epsilon_greedy_with_zero_epsilon_always_exploits() -> None:
    """With epsilon=0 the policy always plays the best-known arm."""
    bandit = EpsilonGreedy(epsilon_start=0.0, epsilon_end=0.0, decay_steps=0, seed=7)
    bandit.update("good", 1.0)
    bandit.update("bad", 0.0)
    for _ in range(50):
        assert bandit.select(["good", "bad"]) == "good"
    assert bandit.explorations == 0


def test_epsilon_greedy_with_zero_epsilon_breaks_ties_by_name() -> None:
    """Equal rewards are broken deterministically by arm name."""
    bandit = EpsilonGreedy(epsilon_start=0.0, epsilon_end=0.0, decay_steps=0, seed=7)
    bandit.update("beta", 0.5)
    bandit.update("alpha", 0.5)
    assert bandit.select(["alpha", "beta"]) == "alpha"


def test_epsilon_greedy_decays_towards_epsilon_end() -> None:
    """Epsilon falls linearly from start to end over ``decay_steps`` pulls."""
    bandit = EpsilonGreedy(epsilon_start=0.5, epsilon_end=0.1, decay_steps=100, seed=3)
    assert bandit.epsilon == pytest.approx(0.5)
    for _ in range(50):
        bandit.update("a", 1.0)
    assert bandit.epsilon == pytest.approx(0.3)
    for _ in range(50):
        bandit.update("a", 1.0)
    assert bandit.epsilon == pytest.approx(0.1)
    for _ in range(100):
        bandit.update("a", 1.0)
    assert bandit.epsilon == pytest.approx(0.1)


def test_epsilon_greedy_explores_sometimes() -> None:
    """With a non-zero epsilon the policy does explore."""
    bandit = EpsilonGreedy(epsilon_start=0.5, epsilon_end=0.5, decay_steps=0, seed=11)
    chosen = {bandit.select(["a", "b", "c"]) for _ in range(200)}
    assert len(chosen) > 1
    assert bandit.explorations > 0


def test_epsilon_greedy_validates_parameters() -> None:
    """Out-of-range parameters are rejected."""
    with pytest.raises(ValueError, match="epsilon_start"):
        EpsilonGreedy(epsilon_start=1.5)
    with pytest.raises(ValueError, match="epsilon_end"):
        EpsilonGreedy(epsilon_end=-0.1)
    with pytest.raises(ValueError, match="decay_steps"):
        EpsilonGreedy(decay_steps=-5)


def test_ucb1_selects_untried_arm_first() -> None:
    """An arm with n_i == 0 has an infinite score and is explored first."""
    bandit = UCB1(seed=1)
    assert bandit.score("fresh") == math.inf
    assert bandit.select(["a", "b", "c"]) == "a"
    bandit.update("a", 0.5)
    assert bandit.select(["a", "b", "c"]) == "b"


def test_ucb1_matches_hand_computed_score() -> None:
    """The UCB1 score is exactly ``mean + c * sqrt(ln(t) / n)``."""
    bandit = UCB1(c=2.0, seed=1)
    for _ in range(4):
        bandit.update("a", 1.0)
    bandit.update("a", 0.0)
    # a: n=5, mean=0.8, t=5
    expected = 0.8 + 2.0 * math.sqrt(math.log(5) / 5)
    assert bandit.score("a") == pytest.approx(expected)


def test_ucb1_eventually_picks_the_best_arm() -> None:
    """A clearly better arm wins once every arm has been sampled."""
    bandit = UCB1(c=1.4, seed=5)
    for _ in range(200):
        chosen = bandit.select(["weak", "strong"])
        reward = 1.0 if chosen == "strong" else 0.0
        bandit.update(chosen, reward)
    assert bandit.select(["weak", "strong"]) == "strong"


def test_ucb1_breaks_ties_deterministically() -> None:
    """Identical arms resolve to the same arm every time."""
    bandit = UCB1(seed=2)
    bandit.update("alpha", 0.5)
    bandit.update("beta", 0.5)
    picks = {bandit.select(["alpha", "beta"]) for _ in range(20)}
    assert picks == {"alpha"}


def test_ucb1_exposes_pull_counts_and_mean_reward() -> None:
    """The shared bookkeeping exposes pulls and mean reward."""
    bandit = UCB1(seed=1)
    bandit.update("a", 1.0)
    bandit.update("a", 0.0)
    assert bandit.pulls("a") == 2
    assert bandit.mean_reward("a") == pytest.approx(0.5)
    assert bandit.mean_reward("never") == 0.0
    assert bandit.total_pulls() == 2


def test_ucb1_rejects_negative_c() -> None:
    """The exploration constant must be non-negative."""
    with pytest.raises(ValueError, match="c must be"):
        UCB1(c=-1.0)


def test_thompson_alpha_increments_on_success() -> None:
    """A success increments the Beta posterior's alpha."""
    bandit = ThompsonSampling(seed=1)
    assert bandit.alpha("a") == 1.0
    assert bandit.beta("a") == 1.0
    bandit.update("a", 1.0)
    assert bandit.alpha("a") == pytest.approx(2.0)
    assert bandit.beta("a") == pytest.approx(1.0)
    assert bandit.posterior_mean("a") == pytest.approx(2 / 3)


def test_thompson_beta_increments_on_failure() -> None:
    """A failure increments the Beta posterior's beta."""
    bandit = ThompsonSampling(seed=1)
    bandit.update("a", 0.0)
    assert bandit.alpha("a") == pytest.approx(1.0)
    assert bandit.beta("a") == pytest.approx(2.0)
    assert bandit.posterior_mean("a") == pytest.approx(1 / 3)


def test_thompson_rejects_out_of_range_rewards() -> None:
    """Beta-Bernoulli updates require a reward in [0, 1]."""
    bandit = ThompsonSampling(seed=1)
    with pytest.raises(ValueError, match="within"):
        bandit.update("a", 1.5)
    with pytest.raises(ValueError, match="within"):
        bandit.update("a", -0.5)


def test_thompson_prefers_the_better_arm_statistically() -> None:
    """The better arm is chosen far more often than the worse one."""
    bandit = ThompsonSampling(seed=99)
    rng = np.random.default_rng(99)
    counts = {"good": 0, "bad": 0}
    for _ in range(400):
        chosen = bandit.select(["good", "bad"])
        counts[chosen] += 1
        # "good" succeeds 90% of the time, "bad" 20% of the time.
        p = 0.9 if chosen == "good" else 0.2
        bandit.update(chosen, 1.0 if rng.random() < p else 0.0)
    assert counts["good"] > counts["bad"] * 3


def test_thompson_is_deterministic_under_a_seed() -> None:
    """The same seed produces the same sequence of decisions."""

    def run() -> list[str]:
        bandit = ThompsonSampling(seed=2024)
        rng = np.random.default_rng(7)
        picks = []
        for _ in range(30):
            chosen = bandit.select(["a", "b"])
            picks.append(chosen)
            bandit.update(chosen, 1.0 if rng.random() < 0.6 else 0.0)
        return picks

    assert run() == run()


def test_softmax_probabilities_sum_to_one() -> None:
    """The softmax distribution is a proper probability distribution."""
    bandit = SoftmaxRouter(temperature=0.5, seed=1)
    bandit.update("a", 1.0)
    bandit.update("b", 0.0)
    probs = bandit.probabilities(["a", "b"])
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["a"] > probs["b"]


def test_softmax_uniform_when_no_rewards_observed() -> None:
    """Without observations every arm is equally likely."""
    bandit = SoftmaxRouter(seed=1)
    probs = bandit.probabilities(["a", "b", "c"])
    assert all(p == pytest.approx(1 / 3) for p in probs.values())


def test_softmax_lower_temperature_concentrates_probability() -> None:
    """A colder softmax puts more mass on the best arm."""
    hot = SoftmaxRouter(temperature=1.0, seed=1)
    cold = SoftmaxRouter(temperature=0.1, seed=1)
    for bandit in (hot, cold):
        bandit.update("good", 1.0)
        bandit.update("bad", 0.0)
    assert cold.probabilities(["good", "bad"])["good"] > hot.probabilities(["good", "bad"])["good"]


def test_softmax_rejects_non_positive_temperature() -> None:
    """Temperature must be strictly positive."""
    with pytest.raises(ValueError, match="temperature must be positive"):
        SoftmaxRouter(temperature=0.0)


def test_reward_shaper_rewards_quality_and_penalises_cost() -> None:
    """The shaped reward is quality minus weighted cost and latency."""
    shaper = RewardShaper(
        lambda_cost=2.0, lambda_latency=1.0, cost_reference=0.01, latency_reference=1000.0
    )
    reward = shaper.shape(success=True, quality=0.9, cost_usd=0.005, latency_ms=500.0)
    assert reward == pytest.approx(0.9 - 2.0 * 0.5 - 1.0 * 0.5)


def test_reward_shaper_zeroes_quality_on_failure() -> None:
    """A failed call earns no quality credit."""
    shaper = RewardShaper(lambda_cost=0.0)
    assert shaper.shape(success=False, quality=1.0, cost_usd=0.0, latency_ms=0.0) == 0.0


def test_reward_ordering_flips_as_lambda_cost_changes() -> None:
    """A cheap-but-bad arm loses to an expensive-but-good arm once cost matters.

    This is the core economic claim of the router: the same two arms can be
    ordered either way depending on how much the operator values a dollar.
    """
    cheap_bad = {"success": True, "quality": 0.5, "cost_usd": 0.0001, "latency_ms": 300.0}
    expensive_good = {"success": True, "quality": 0.95, "cost_usd": 0.05, "latency_ms": 3000.0}
    reference = 0.05

    quality_only = RewardShaper(lambda_cost=0.0, cost_reference=reference)
    assert quality_only.shape(**expensive_good) > quality_only.shape(**cheap_bad)

    cost_sensitive = RewardShaper(lambda_cost=10.0, cost_reference=reference)
    assert cost_sensitive.shape(**cheap_bad) > cost_sensitive.shape(**expensive_good)


def test_reward_shaper_normalizes_cost_and_latency() -> None:
    """Normalization expresses values as a fraction of the reference."""
    shaper = RewardShaper(cost_reference=0.02, latency_reference=2000.0)
    assert shaper.normalize_cost(0.01) == pytest.approx(0.5)
    assert shaper.normalize_latency(1000.0) == pytest.approx(0.5)
    assert shaper.normalize_cost(-5.0) == 0.0


def test_reward_shaper_validates_parameters() -> None:
    """Negative weights and non-positive references are rejected."""
    with pytest.raises(ValueError, match="lambda_cost"):
        RewardShaper(lambda_cost=-1.0)
    with pytest.raises(ValueError, match="lambda_latency"):
        RewardShaper(lambda_latency=-1.0)
    with pytest.raises(ValueError, match="cost_reference"):
        RewardShaper(cost_reference=0.0)
    with pytest.raises(ValueError, match="latency_reference"):
        RewardShaper(latency_reference=-1.0)
