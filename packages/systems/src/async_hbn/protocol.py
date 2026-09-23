"""Immutable request universe and fixed-ID recovery rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Iterable, Mapping

from .identities import Phase, RequestKey, stable_task_permutation


class Policy(StrEnum):
    UNIFORM_FLAT = "uniform_flat"
    HBN_SYNC = "hbn_sync"
    HBN_ASYNC = "hbn_spec_fill256_partial_plugin"


HBN_POLICIES = frozenset({
    Policy.HBN_SYNC,
    Policy.HBN_ASYNC,
})


@dataclass(frozen=True)
class ProtocolPlan:
    task_ids: tuple[str, ...]
    budget_per_task: int
    pilot_per_task: int
    continuation_total: int
    speculative_depth: int
    master_seed: int

    def __post_init__(self) -> None:
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        if self.pilot_per_task >= self.budget_per_task:
            raise ValueError("pilot_per_task must be below budget_per_task")
        expected = len(self.task_ids) * (self.budget_per_task - self.pilot_per_task)
        if self.continuation_total != expected:
            raise ValueError(f"continuation_total must equal {expected}")
        if self.speculative_depth < 1:
            raise ValueError("speculative_depth must be positive")

    @property
    def pilot_total(self) -> int:
        return len(self.task_ids) * self.pilot_per_task

    @property
    def accepted_total(self) -> int:
        return len(self.task_ids) * self.budget_per_task

    @property
    def uniform_continuations_per_task(self) -> int:
        return self.budget_per_task - self.pilot_per_task

    @property
    def max_continuations_per_task(self) -> int:
        """Largest positive allocation one task can receive."""

        return self.continuation_total - len(self.task_ids) + 1

    def pilots(self) -> tuple[RequestKey, ...]:
        return tuple(
            RequestKey(task_id, Phase.PILOT, ordinal)
            for ordinal in range(1, self.pilot_per_task + 1)
            for task_id in self.task_ids
        )

    def uniform_continuations(self) -> tuple[RequestKey, ...]:
        return tuple(
            RequestKey(task_id, Phase.CONTINUATION, ordinal)
            for ordinal in range(1, self.uniform_continuations_per_task + 1)
            for task_id in self.task_ids
        )

    def selected_continuations(
        self,
        policy: Policy,
        allocation: Mapping[str, int] | None = None,
    ) -> frozenset[RequestKey]:
        if policy is Policy.UNIFORM_FLAT:
            return frozenset(self.uniform_continuations())
        if policy not in HBN_POLICIES:
            raise ValueError(f"unknown policy {policy}")
        if allocation is None:
            raise ValueError("HBN policies require an allocation")
        if set(allocation) != set(self.task_ids):
            raise ValueError("allocation task universe mismatch")
        if sum(map(int, allocation.values())) != self.continuation_total:
            raise ValueError("allocation continuation budget mismatch")
        if min(map(int, allocation.values())) < 1:
            raise ValueError("allocation must retain at least one continuation per task")
        return frozenset(
            RequestKey(task_id, Phase.CONTINUATION, ordinal)
            for task_id in self.task_ids
            for ordinal in range(1, int(allocation[task_id]) + 1)
        )

    def partial_plugin_priority(
        self, allocation: Mapping[str, int]
    ) -> tuple[RequestKey, ...]:
        """Return a deterministic proportional fixed-prefix priority.

        The provisional allocation is scheduling-only. Sorting prefix IDs by
        ``ordinal / allocation`` makes every truncation approximately
        proportional while preserving ordinal order within each task.
        """

        if set(allocation) != set(self.task_ids):
            raise ValueError("partial allocation task universe mismatch")
        values = {task_id: int(allocation[task_id]) for task_id in self.task_ids}
        if sum(values.values()) != self.continuation_total:
            raise ValueError("partial allocation continuation budget mismatch")
        if min(values.values()) < 1:
            raise ValueError("partial allocation must remain positive")
        task_order = stable_task_permutation(self.task_ids, self.master_seed)
        task_rank = {task_id: rank for rank, task_id in enumerate(task_order)}
        priority = [
            RequestKey(task_id, Phase.CONTINUATION, ordinal)
            for task_id in self.task_ids
            for ordinal in range(1, values[task_id] + 1)
        ]
        priority.sort(key=lambda key: (
            Fraction(key.ordinal, values[key.task_id]),
            key.ordinal,
            task_rank[key.task_id],
        ))
        return tuple(priority)

    def initial_requests(
        self,
        policy: Policy,
    ) -> tuple[RequestKey, ...]:
        if policy is Policy.UNIFORM_FLAT:
            return self.pilots() + self.uniform_continuations()
        return self.pilots()

    def accepted_requests(
        self,
        policy: Policy,
        allocation: Mapping[str, int] | None = None,
    ) -> frozenset[RequestKey]:
        return frozenset(self.pilots()) | self.selected_continuations(policy, allocation)


def classify_prefetch(
    prefetched: Iterable[RequestKey],
    selected: frozenset[RequestKey],
) -> tuple[frozenset[RequestKey], frozenset[RequestKey]]:
    """Classify only by fixed ID; completion order is intentionally absent."""

    prefetched_set = frozenset(prefetched)
    return prefetched_set & selected, prefetched_set - selected
