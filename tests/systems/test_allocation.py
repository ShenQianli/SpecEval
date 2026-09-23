from __future__ import annotations

import numpy as np

from async_hbn.allocation import HBNAllocator, PartialHBNAllocator
from speculative_eval.core import (
    exact_integer_allocation,
    posterior_fresh_variance,
    quadrature,
)


TASKS = tuple(str(index) for index in range(60, 90))


def test_allocator_matches_release_statistical_implementation() -> None:
    rewards = {
        task_id: tuple((task_index + ordinal) % 3 == 0 for ordinal in range(10))
        for task_index, task_id in enumerate(TASKS)
    }
    allocator = HBNAllocator(TASKS, 10, 660, quadrature_order=16)
    result = allocator.allocate(rewards)

    counts = np.asarray([[sum(rewards[task_id]) for task_id in TASKS]], dtype=np.int16)
    mu, dispersion, weights = quadrature(order=16)
    scores = posterior_fresh_variance(counts, 10, mu, dispersion, weights)
    expected = exact_integer_allocation(scores, 660)[0]

    assert list(result.allocation.values()) == expected.tolist()
    assert sum(result.allocation.values()) == 660
    assert min(result.allocation.values()) >= 1
    assert result.compute_ns > 0


def test_allocator_rejects_incomplete_pilot_vector() -> None:
    allocator = HBNAllocator(TASKS, 10, 660)
    rewards = {task_id: [0] * 10 for task_id in TASKS}
    rewards[TASKS[-1]] = [0] * 9
    try:
        allocator.allocate(rewards)
    except ValueError as exc:
        assert "expected 10" in str(exc)
    else:
        raise AssertionError("incomplete pilots must not trigger allocation")


def test_partial_allocator_matches_official_allocator_when_complete() -> None:
    rewards = {
        task_id: tuple((task_index * 7 + ordinal) % 5 < 2 for ordinal in range(10))
        for task_index, task_id in enumerate(TASKS)
    }
    official = HBNAllocator(TASKS, 10, 660).allocate(rewards)
    partial = PartialHBNAllocator(TASKS, 10, 660).allocate(rewards)
    assert partial.observed_total == 300
    assert partial.allocation == official.allocation
    np.testing.assert_allclose(
        list(partial.posterior_variances.values()),
        list(official.posterior_variances.values()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_partial_allocator_accepts_unequal_observation_counts() -> None:
    rewards = {
        task_id: tuple(
            (task_index + ordinal) % 4 == 0
            for ordinal in range(task_index % 11)
        )
        for task_index, task_id in enumerate(TASKS)
    }
    result = PartialHBNAllocator(TASKS, 10, 660).allocate(rewards)
    assert result.observed_total == sum(map(len, rewards.values()))
    assert sum(result.allocation.values()) == 660
    assert min(result.allocation.values()) >= 1
    assert result.compute_ns > 0
