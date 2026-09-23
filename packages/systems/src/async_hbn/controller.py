"""Policy state machine shared by fake and real request-level routers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from .allocation import AllocationResult, HBNAllocator
from .identities import Phase, RequestKey
from .protocol import (
    HBN_POLICIES,
    Policy,
    ProtocolPlan,
)


@dataclass(frozen=True)
class BarrierResult:
    selected: frozenset[RequestKey]
    newly_eligible: tuple[RequestKey, ...]
    allocation_result: AllocationResult | None


class PolicyController:
    """Central serialized state for eligibility and fixed-ID recovery.

    Engine scheduling and completion events may be asynchronous, but all calls
    into this object must be serialized by the router event loop.
    """

    def __init__(
        self,
        plan: ProtocolPlan,
        policy: Policy,
        allocator: HBNAllocator | None,
    ) -> None:
        self.plan = plan
        self.policy = policy
        self.allocator = allocator
        if policy in HBN_POLICIES and allocator is None:
            raise ValueError("HBN policy requires an allocator")
        self.initial_queue = deque(plan.initial_requests(policy))
        self.post_barrier_queue: deque[RequestKey] = deque()
        self.partial_priority_queue: deque[RequestKey] = deque()
        self.partial_priority_version = 0
        self.partial_priority_updates_applied = 0
        self.partial_priority_observed_pilots = 0
        self.partial_priority_allocation: Mapping[str, int] | None = None
        self.submitted: set[RequestKey] = set()
        self.completed_rewards: dict[RequestKey, int] = {}
        self.prefetched: set[RequestKey] = set()
        self.barrier_released = policy is Policy.UNIFORM_FLAT
        self.allocation_result: AllocationResult | None = None
        if policy is Policy.UNIFORM_FLAT:
            self.selected: frozenset[RequestKey] | None = plan.accepted_requests(
                policy
            )
        else:
            self.selected = None

    @property
    def all_pilots_submitted(self) -> bool:
        return all(key in self.submitted for key in self.plan.pilots())

    @property
    def all_pilot_rewards_ready(self) -> bool:
        return all(key in self.completed_rewards for key in self.plan.pilots())

    def take_initial(self, limit: int) -> tuple[RequestKey, ...]:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        values = tuple(self.initial_queue.popleft() for _ in range(min(limit, len(self.initial_queue))))
        self.submitted.update(values)
        return values

    def take_post_barrier(self, limit: int) -> tuple[RequestKey, ...]:
        if not self.barrier_released:
            return ()
        values = tuple(
            self.post_barrier_queue.popleft()
            for _ in range(min(limit, len(self.post_barrier_queue)))
        )
        self.submitted.update(values)
        return values

    def take_prefetch(self, limit: int) -> tuple[RequestKey, ...]:
        if self.policy is not Policy.HBN_ASYNC or self.barrier_released:
            return ()
        if not self.all_pilots_submitted:
            return ()
        if self.partial_priority_version == 0:
            return ()
        queues = (self.partial_priority_queue,)
        output: list[RequestKey] = []
        for queue in queues:
            while queue and len(output) < limit:
                key = queue.popleft()
                if key in self.submitted:
                    continue
                output.append(key)
                self.submitted.add(key)
                self.prefetched.add(key)
            if len(output) >= limit:
                break
        return tuple(output)

    def pilot_rewards_snapshot(self) -> dict[str, tuple[int, ...]]:
        return {
            task_id: tuple(
                self.completed_rewards[key]
                for ordinal in range(1, self.plan.pilot_per_task + 1)
                if (
                    key := RequestKey(task_id, Phase.PILOT, ordinal)
                ) in self.completed_rewards
            )
            for task_id in self.plan.task_ids
        }

    def apply_partial_priority(
        self,
        *,
        version: int,
        observed_pilots: int,
        allocation: Mapping[str, int],
        priority: Sequence[RequestKey],
    ) -> bool:
        if self.policy is not Policy.HBN_ASYNC:
            raise ValueError("partial priority is only valid for HBN-async")
        if version <= self.partial_priority_version or self.barrier_released:
            return False
        if len(priority) != self.plan.continuation_total:
            raise ValueError("partial priority must cover the provisional budget")
        if len(set(priority)) != len(priority):
            raise ValueError("partial priority contains duplicate fixed IDs")
        self.partial_priority_queue = deque(priority)
        self.partial_priority_version = int(version)
        self.partial_priority_updates_applied += 1
        self.partial_priority_observed_pilots = int(observed_pilots)
        self.partial_priority_allocation = dict(allocation)
        return True

    def record_reward(self, key: RequestKey, reward: int) -> BarrierResult | None:
        if key not in self.submitted:
            raise ValueError(f"completion for unsubmitted request {key}")
        if key in self.completed_rewards:
            raise ValueError(f"duplicate completion for {key}")
        if reward not in (0, 1):
            raise ValueError("reward must be Bernoulli")
        self.completed_rewards[key] = int(reward)
        if key.phase is not Phase.PILOT or self.barrier_released:
            return None
        if not self.all_pilot_rewards_ready:
            return None
        return self._release_barrier()

    def _release_barrier(self) -> BarrierResult:
        if self.barrier_released:
            raise RuntimeError("barrier already released")
        allocation: Mapping[str, int] | None = None
        if self.policy in HBN_POLICIES:
            assert self.allocator is not None
            rewards = {
                task_id: tuple(
                    self.completed_rewards[RequestKey(task_id, Phase.PILOT, ordinal)]
                    for ordinal in range(1, self.plan.pilot_per_task + 1)
                )
                for task_id in self.plan.task_ids
            }
            self.allocation_result = self.allocator.allocate(rewards)
            allocation = self.allocation_result.allocation
        selected_continuations = self.plan.selected_continuations(self.policy, allocation)
        self.selected = frozenset(self.plan.pilots()) | selected_continuations
        missing = tuple(
            key for key in selected_continuations if key not in self.submitted
        )
        # Sets have no scheduling semantics: restore deterministic breadth-first
        # ordering when releasing the selected continuations.
        missing_set = set(missing)
        max_ordinal = max(key.ordinal for key in selected_continuations)
        ordered_missing = tuple(
            key
            for ordinal in range(1, max_ordinal + 1)
            for task_id in self.plan.task_ids
            if (key := RequestKey(task_id, Phase.CONTINUATION, ordinal)) in missing_set
        )
        self.post_barrier_queue.extend(ordered_missing)
        self.barrier_released = True
        return BarrierResult(self.selected, ordered_missing, self.allocation_result)

    def result_ready(self) -> bool:
        return self.selected is not None and all(
            key in self.completed_rewards for key in self.selected
        )

    def selected_rewards(self) -> dict[RequestKey, int]:
        if not self.result_ready():
            raise RuntimeError("selected result is not ready")
        assert self.selected is not None
        return {key: self.completed_rewards[key] for key in self.selected}

    def surplus_submitted(self) -> frozenset[RequestKey]:
        if self.selected is None:
            return frozenset()
        return frozenset(self.submitted - set(self.selected))
