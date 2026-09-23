"""Frozen HBN posterior and exact positive-integer continuation allocation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Mapping, Sequence

import numpy as np
from scipy.special import betaln, logsumexp

from .hbn_math import (
    exact_integer_allocation,
    posterior_fresh_variance,
    quadrature,
)


@dataclass(frozen=True)
class AllocationResult:
    allocation: Mapping[str, int]
    pilot_counts: Mapping[str, int]
    posterior_variances: Mapping[str, float]
    compute_ns: int


@dataclass(frozen=True)
class PartialAllocationResult:
    allocation: Mapping[str, int]
    pilot_successes: Mapping[str, int]
    pilot_observations: Mapping[str, int]
    posterior_variances: Mapping[str, float]
    observed_total: int
    compute_ns: int


class HBNAllocator:
    """Precompute static quadrature; time only pilot-dependent work."""

    def __init__(
        self,
        task_ids: Sequence[str],
        pilot_per_task: int,
        continuation_total: int,
        quadrature_order: int = 16,
    ) -> None:
        self.task_ids = tuple(map(str, task_ids))
        self.pilot_per_task = int(pilot_per_task)
        self.continuation_total = int(continuation_total)
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        if self.continuation_total < len(self.task_ids):
            raise ValueError("at least one continuation per task is required")
        # Static, pilot-independent design work; callers construct this before T0.
        self._mu, self._dispersion, self._weights = quadrature(order=quadrature_order)

    def allocate(self, rewards: Mapping[str, Sequence[int]]) -> AllocationResult:
        start = perf_counter_ns()
        counts = []
        counts_by_task: dict[str, int] = {}
        for task_id in self.task_ids:
            values = tuple(int(value) for value in rewards.get(task_id, ()))
            if len(values) != self.pilot_per_task:
                raise ValueError(
                    f"task {task_id} has {len(values)} pilot rewards; "
                    f"expected {self.pilot_per_task}"
                )
            if any(value not in (0, 1) for value in values):
                raise ValueError(f"task {task_id} has a non-Bernoulli reward")
            count = sum(values)
            counts.append(count)
            counts_by_task[task_id] = count
        pilot_counts = np.asarray([counts], dtype=np.int16)
        variances = posterior_fresh_variance(
            pilot_counts,
            self.pilot_per_task,
            self._mu,
            self._dispersion,
            self._weights,
        )
        allocated = exact_integer_allocation(variances, self.continuation_total)[0]
        elapsed = perf_counter_ns() - start
        allocation = {
            task_id: int(value) for task_id, value in zip(self.task_ids, allocated)
        }
        posterior = {
            task_id: float(value) for task_id, value in zip(self.task_ids, variances[0])
        }
        if sum(allocation.values()) != self.continuation_total:
            raise AssertionError("continuation budget mismatch")
        if min(allocation.values()) < 1:
            raise AssertionError("HBN allocation lost the positive lower bound")
        return AllocationResult(allocation, counts_by_task, posterior, elapsed)


class PartialHBNAllocator:
    """Scheduling-only HBN plug-in for unequal partial pilot sample sizes."""

    def __init__(
        self,
        task_ids: Sequence[str],
        pilot_per_task: int,
        continuation_total: int,
        quadrature_order: int = 16,
    ) -> None:
        self.task_ids = tuple(map(str, task_ids))
        self.pilot_per_task = int(pilot_per_task)
        self.continuation_total = int(continuation_total)
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        if self.continuation_total < len(self.task_ids):
            raise ValueError("at least one continuation per task is required")
        self._mu, self._dispersion, self._weights = quadrature(
            order=quadrature_order
        )

    def allocate(
        self, rewards: Mapping[str, Sequence[int]]
    ) -> PartialAllocationResult:
        start = perf_counter_ns()
        successes = []
        observations = []
        successes_by_task: dict[str, int] = {}
        observations_by_task: dict[str, int] = {}
        for task_id in self.task_ids:
            values = tuple(int(value) for value in rewards.get(task_id, ()))
            if len(values) > self.pilot_per_task:
                raise ValueError(
                    f"task {task_id} has {len(values)} pilot rewards; "
                    f"maximum is {self.pilot_per_task}"
                )
            if any(value not in (0, 1) for value in values):
                raise ValueError(f"task {task_id} has a non-Bernoulli reward")
            success = sum(values)
            successes.append(success)
            observations.append(len(values))
            successes_by_task[task_id] = success
            observations_by_task[task_id] = len(values)

        success_array = np.asarray(successes, dtype=np.float64)
        observation_array = np.asarray(observations, dtype=np.float64)
        concentration = (1.0 - self._dispersion) / self._dispersion
        prior_a = self._mu * concentration
        prior_b = (1.0 - self._mu) * concentration
        # Each task can have a different number of observed pilots. Broadcast
        # task statistics against the frozen hyperparameter quadrature grid.
        log_marginal = (
            betaln(
                prior_a[None, :] + success_array[:, None],
                prior_b[None, :]
                + observation_array[:, None]
                - success_array[:, None],
            )
            - betaln(prior_a, prior_b)[None, :]
        )
        log_weights = np.sum(log_marginal, axis=0) + np.log(self._weights)
        log_weights -= logsumexp(log_weights)
        posterior_weights = np.exp(log_weights)
        variance_by_task = (
            (prior_a[None, :] + success_array[:, None])
            * (
                prior_b[None, :]
                + observation_array[:, None]
                - success_array[:, None]
            )
            / (
                (concentration[None, :] + observation_array[:, None])
                * (concentration[None, :] + observation_array[:, None] + 1.0)
            )
        )
        variances = variance_by_task @ posterior_weights
        allocated = exact_integer_allocation(
            variances[None, :], self.continuation_total
        )[0]
        elapsed = perf_counter_ns() - start
        allocation = {
            task_id: int(value) for task_id, value in zip(self.task_ids, allocated)
        }
        posterior = {
            task_id: float(value) for task_id, value in zip(self.task_ids, variances)
        }
        if sum(allocation.values()) != self.continuation_total:
            raise AssertionError("partial continuation budget mismatch")
        if min(allocation.values()) < 1:
            raise AssertionError("partial allocation lost the positive lower bound")
        return PartialAllocationResult(
            allocation=allocation,
            pilot_successes=successes_by_task,
            pilot_observations=observations_by_task,
            posterior_variances=posterior,
            observed_total=int(np.sum(observation_array)),
            compute_ns=elapsed,
        )
