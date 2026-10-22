"""Tests for the health tracker, sliding window and circuit-breaker behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from modelrouter import HealthSignal, HealthTracker


def test_new_tracker_treats_unseen_arms_as_healthy() -> None:
    """An arm with no observations has not failed yet."""
    tracker = HealthTracker()
    assert tracker.is_healthy("never-seen") is True


def test_success_rate_over_the_sliding_window() -> None:
    """The window success rate only counts the most recent signals."""
    tracker = HealthTracker(window_size=4)
    for _ in range(4):
        tracker.record_outcome("a", success=True, latency_ms=100.0)
    assert tracker.success_rate("a") == pytest.approx(1.0)
    tracker.record_outcome("a", success=False, latency_ms=100.0)
    # The window holds 4 of the 5 signals: 3 successes and 1 failure.
    assert tracker.success_rate("a") == pytest.approx(0.75)
    assert tracker.health("a").total_observations == 5
    assert len(tracker.health("a").signals) == 4


def test_p95_latency_matches_numpy_percentile() -> None:
    """p95 uses ``numpy.percentile`` with linear interpolation."""
    tracker = HealthTracker(window_size=100)
    latencies = [100.0, 200.0, 300.0, 400.0, 5000.0]
    for latency in latencies:
        tracker.record_outcome("a", success=True, latency_ms=latency)
    assert tracker.p95_latency_ms("a") == pytest.approx(float(np.percentile(latencies, 95)))


def test_p95_of_empty_window_is_zero() -> None:
    """No signals means no latency to report."""
    tracker = HealthTracker()
    assert tracker.p95_latency_ms("a") == 0.0


def test_unhealthy_arm_is_excluded() -> None:
    """A failing arm is excluded by the breaker."""
    tracker = HealthTracker(min_success_rate=0.8)
    for _ in range(5):
        tracker.record_outcome("bad", success=False, latency_ms=100.0, error_type="rate_limit")
    for _ in range(5):
        tracker.record_outcome("good", success=True, latency_ms=100.0)
    assert tracker.is_healthy("bad") is False
    assert tracker.healthy_arms(["bad", "good"]) == ["good"]
    assert tracker.unhealthy_arms(["bad", "good"]) == ["bad"]


def test_recovered_arm_becomes_eligible_again() -> None:
    """A recovered arm returns to the healthy set with no manual reset."""
    tracker = HealthTracker(window_size=5, min_success_rate=0.8)
    for _ in range(5):
        tracker.record_outcome("flaky", success=False, latency_ms=100.0)
    assert tracker.is_healthy("flaky") is False
    for _ in range(5):
        tracker.record_outcome("flaky", success=True, latency_ms=100.0)
    assert tracker.is_healthy("flaky") is True


def test_slow_arm_is_unhealthy() -> None:
    """Latency above the multiplier of baseline trips the breaker."""
    tracker = HealthTracker(max_latency_multiplier=2.0)
    tracker.register("slow", baseline_latency_ms=100.0)
    for _ in range(5):
        tracker.record_outcome("slow", success=True, latency_ms=500.0)
    assert tracker.is_healthy("slow") is False


def test_health_score_is_bounded_and_combines_signals() -> None:
    """The health score lies in [0, 1] and drops with failures."""
    tracker = HealthTracker()
    tracker.register("a", baseline_latency_ms=1000.0)
    assert tracker.health_score("a") == pytest.approx(1.0)
    for _ in range(4):
        tracker.record_outcome("a", success=False, latency_ms=1000.0)
    score = tracker.health_score("a")
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.0)


def test_health_score_drops_when_latency_degrades() -> None:
    """A slow-but-successful arm scores below a fast-and-successful arm."""
    tracker = HealthTracker()
    tracker.register("fast", baseline_latency_ms=1000.0)
    tracker.register("slow", baseline_latency_ms=1000.0)
    for _ in range(5):
        tracker.record_outcome("fast", success=True, latency_ms=1000.0)
        tracker.record_outcome("slow", success=True, latency_ms=3000.0)
    assert tracker.health_score("fast") > tracker.health_score("slow")
    assert tracker.health_score("fast") == pytest.approx(1.0)


def test_error_rates_by_type_are_tracked() -> None:
    """Errors are bucketed by their type."""
    tracker = HealthTracker()
    tracker.record_outcome("a", success=False, latency_ms=1.0, error_type="timeout")
    tracker.record_outcome("a", success=False, latency_ms=1.0, error_type="timeout")
    tracker.record_outcome("a", success=False, latency_ms=1.0, error_type="rate_limit")
    assert tracker.error_rate_by_type("a") == {"timeout": 2, "rate_limit": 1}
    assert tracker.health("a").error_rate == pytest.approx(1.0)


def test_health_signal_validates_its_fields() -> None:
    """Signals reject empty arm names and negative latency."""
    with pytest.raises(ValueError, match="arm must be"):
        HealthSignal(arm="", timestamp=0.0, success=True, latency_ms=1.0)
    with pytest.raises(ValueError, match="latency_ms must be"):
        HealthSignal(arm="a", timestamp=0.0, success=True, latency_ms=-1.0)


def test_health_signal_requires_arm_timestamp_success_latency() -> None:
    """The signal dataclass exposes the documented fields."""
    signal = HealthSignal(arm="a", timestamp=1.5, success=True, latency_ms=42.0, error_type="x")
    assert signal.arm == "a"
    assert signal.timestamp == 1.5
    assert signal.success is True
    assert signal.latency_ms == 42.0
    assert signal.error_type == "x"


def test_is_healthy_accepts_per_call_overrides() -> None:
    """Per-call thresholds override the tracker defaults."""
    tracker = HealthTracker(min_success_rate=0.99)
    tracker.record_outcome("a", success=True, latency_ms=1.0)
    tracker.record_outcome("a", success=False, latency_ms=1.0)
    assert tracker.is_healthy("a") is False
    assert tracker.is_healthy("a", min_success_rate=0.4) is True


def test_health_tracker_validates_configuration() -> None:
    """Bad tracker configuration is rejected."""
    with pytest.raises(ValueError, match="window_size"):
        HealthTracker(window_size=0)
    with pytest.raises(ValueError, match="min_success_rate"):
        HealthTracker(min_success_rate=1.5)
    with pytest.raises(ValueError, match="max_latency_multiplier"):
        HealthTracker(max_latency_multiplier=0.0)


def test_health_of_unknown_arm_raises() -> None:
    """``health`` raises for an arm that was never registered."""
    tracker = HealthTracker()
    with pytest.raises(KeyError, match="unknown arm"):
        tracker.health("ghost")


def test_health_tracker_as_dict_snapshot() -> None:
    """The snapshot contains the documented health fields."""
    tracker = HealthTracker()
    tracker.record_outcome("a", success=True, latency_ms=100.0)
    snapshot = tracker.as_dict()
    assert snapshot["a"]["success_rate"] == pytest.approx(1.0)
    assert "p95_latency_ms" in snapshot["a"]
    assert "health_score" in snapshot["a"]
