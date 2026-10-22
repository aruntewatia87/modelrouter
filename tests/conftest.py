"""Shared fixtures for the modelrouter test suite."""

from __future__ import annotations

import numpy as np
import pytest

from modelrouter import ArmRegistry, ModelArm

CHAT = frozenset({"chat"})


def make_arm(
    name: str,
    *,
    provider: str = "test-provider",
    input_cost: float = 0.001,
    output_cost: float = 0.002,
    cache_read_cost: float = 0.0001,
    capabilities: frozenset[str] = CHAT,
    max_context_tokens: int = 200_000,
    avg_latency_ms: float = 1000.0,
    quality_prior: float = 0.7,
    reliability: float = 0.99,
) -> ModelArm:
    """Build a :class:`ModelArm` with sensible defaults for tests."""
    return ModelArm(
        name=name,
        provider=provider,
        input_cost_per_1k=input_cost,
        output_cost_per_1k=output_cost,
        cache_read_cost_per_1k=cache_read_cost,
        capabilities=capabilities,
        max_context_tokens=max_context_tokens,
        avg_latency_ms=avg_latency_ms,
        quality_prior=quality_prior,
        reliability=reliability,
    )


@pytest.fixture
def arms() -> list[ModelArm]:
    """Four arms spanning cheap/fast, mid, expensive/best and tool-capable."""
    return [
        make_arm(
            "cheap",
            provider="alpha",
            input_cost=0.00015,
            output_cost=0.0006,
            cache_read_cost=0.000015,
            avg_latency_ms=450.0,
            quality_prior=0.55,
        ),
        make_arm(
            "mid",
            provider="beta",
            input_cost=0.003,
            output_cost=0.015,
            cache_read_cost=0.0003,
            avg_latency_ms=1200.0,
            quality_prior=0.82,
        ),
        make_arm(
            "best",
            provider="gamma",
            input_cost=0.015,
            output_cost=0.075,
            cache_read_cost=0.0015,
            avg_latency_ms=2800.0,
            quality_prior=0.96,
        ),
        make_arm(
            "tools",
            provider="delta",
            input_cost=0.002,
            output_cost=0.008,
            cache_read_cost=0.0002,
            capabilities=frozenset({"chat", "tools"}),
            avg_latency_ms=900.0,
            quality_prior=0.78,
        ),
        make_arm(
            "json",
            provider="epsilon",
            input_cost=0.001,
            output_cost=0.004,
            cache_read_cost=0.0001,
            capabilities=frozenset({"chat", "json_mode"}),
            avg_latency_ms=700.0,
            quality_prior=0.72,
        ),
    ]


@pytest.fixture
def registry(arms: list[ModelArm]) -> ArmRegistry:
    """A registry populated with the :func:`arms` fixture."""
    return ArmRegistry(arms)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator for deterministic simulation in tests."""
    return np.random.default_rng(1234)
