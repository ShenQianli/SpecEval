from __future__ import annotations

import pytest

from async_hbn.allocation import HBNAllocator
from async_hbn.controller import PolicyController
from async_hbn.identities import Phase, RequestKey
from async_hbn.protocol import Policy, ProtocolPlan, classify_prefetch


TASKS = tuple(str(index) for index in range(60, 90))
PLAN = ProtocolPlan(
    task_ids=TASKS,
    budget_per_task=32,
    pilot_per_task=10,
    continuation_total=660,
    speculative_depth=4,
    master_seed=42,
)


def _allocator() -> HBNAllocator:
    return HBNAllocator(TASKS, 10, 660)


def test_initial_eligibility_separates_flat_and_barrier() -> None:
    assert len(PLAN.initial_requests(Policy.UNIFORM_FLAT)) == 960
    assert len(PLAN.initial_requests(Policy.HBN_SYNC)) == 300
    assert (
        len(PLAN.initial_requests(Policy.HBN_ASYNC))
        == 300
    )


def test_request_universe_covers_the_largest_positive_allocation() -> None:
    assert PLAN.max_continuations_per_task == 631
    allocation = {task: 1 for task in TASKS}
    allocation[TASKS[0]] = PLAN.max_continuations_per_task
    selected = PLAN.selected_continuations(Policy.HBN_ASYNC, allocation)
    assert len(selected) == PLAN.continuation_total
    assert max(k.ordinal for k in selected) == 631


def test_partial_plugin_priority_is_proportional_and_prefix_preserving() -> None:
    allocation = {task_id: 22 for task_id in TASKS}
    allocation[TASKS[0]] = 42
    allocation[TASKS[1]] = 2
    priority = PLAN.partial_plugin_priority(allocation)
    assert len(priority) == 660
    assert len(set(priority)) == 660
    first = priority[:330]
    by_task = {
        task_id: sorted(key.ordinal for key in first if key.task_id == task_id)
        for task_id in TASKS
    }
    for ordinals in by_task.values():
        assert ordinals == list(range(1, len(ordinals) + 1))
    assert len(by_task[TASKS[0]]) > len(by_task[TASKS[2]])
    assert len(by_task[TASKS[1]]) < len(by_task[TASKS[2]])


def test_controller_atomically_replaces_partial_priority() -> None:
    controller = PolicyController(
        PLAN, Policy.HBN_ASYNC, _allocator()
    )
    controller.take_initial(300)
    allocation = {task_id: 22 for task_id in TASKS}
    allocation[TASKS[0]] = 42
    allocation[TASKS[1]] = 2
    priority = PLAN.partial_plugin_priority(allocation)
    assert controller.apply_partial_priority(
        version=3,
        observed_pilots=150,
        allocation=allocation,
        priority=priority,
    )
    assert not controller.apply_partial_priority(
        version=2,
        observed_pilots=100,
        allocation=allocation,
        priority=priority,
    )
    assert controller.take_prefetch(40) == priority[:40]
    assert controller.partial_priority_version == 3
    assert controller.partial_priority_observed_pilots == 150


def test_partial_plugin_allocation_is_an_admission_gate() -> None:
    controller = PolicyController(
        PLAN, Policy.HBN_ASYNC, _allocator()
    )
    controller.take_initial(300)
    assert controller.take_prefetch(256) == ()

    allocation = {task_id: 22 for task_id in TASKS}
    priority = PLAN.partial_plugin_priority(allocation)
    assert controller.apply_partial_priority(
        version=1,
        observed_pilots=0,
        allocation=allocation,
        priority=priority,
    )
    assert controller.take_prefetch(700) == priority
    # The full breadth-first universe is deliberately not a fallback once the
    # current provisional allocation has been dispatched.
    assert controller.take_prefetch(1) == ()
    assert len(controller.prefetched) == PLAN.continuation_total


def test_fixed_id_recovery_does_not_observe_completion_order() -> None:
    prefetched = [
        RequestKey(TASKS[0], Phase.CONTINUATION, 3),
        RequestKey(TASKS[0], Phase.CONTINUATION, 2),
        RequestKey(TASKS[0], Phase.CONTINUATION, 1),
    ]
    allocation = {task_id: 22 for task_id in TASKS}
    allocation[TASKS[0]] = 1
    allocation[TASKS[1]] = 43
    selected = PLAN.selected_continuations(Policy.HBN_ASYNC, allocation)
    recovered, discarded = classify_prefetch(prefetched, selected)
    assert recovered == {RequestKey(TASKS[0], Phase.CONTINUATION, 1)}
    assert discarded == {
        RequestKey(TASKS[0], Phase.CONTINUATION, 2),
        RequestKey(TASKS[0], Phase.CONTINUATION, 3),
    }


def test_barrier_requires_all_300_rewards_not_only_submission() -> None:
    controller = PolicyController(PLAN, Policy.HBN_SYNC, _allocator())
    pilots = controller.take_initial(300)
    assert controller.all_pilots_submitted
    for index, key in enumerate(pilots[:-1]):
        assert controller.record_reward(key, index % 2) is None
    assert not controller.all_pilot_rewards_ready
    released = controller.record_reward(pilots[-1], 0)
    assert released is not None
    assert controller.all_pilot_rewards_ready
    assert sum(released.allocation_result.allocation.values()) == 660
    assert len(released.selected) == 960


def test_only_three_paper_strategies_are_available() -> None:
    assert {p.value for p in Policy} == {
        "uniform_flat", "hbn_sync", "hbn_spec_fill256_partial_plugin"
    }


def test_async_waits_for_pilot_submission_and_never_resubmits_ids() -> None:
    controller = PolicyController(PLAN, Policy.HBN_ASYNC, _allocator())
    allocation = {task: 22 for task in TASKS}
    priority = PLAN.partial_plugin_priority(allocation)
    controller.apply_partial_priority(version=1, observed_pilots=0,
                                      allocation=allocation, priority=priority)
    controller.take_initial(299)
    assert controller.take_prefetch(10) == ()
    controller.take_initial(1)
    first = controller.take_prefetch(10)
    controller.apply_partial_priority(version=2, observed_pilots=1,
                                      allocation=allocation, priority=priority)
    assert set(first).isdisjoint(controller.take_prefetch(20))


def test_partial_update_preserves_submitted_work_until_final_selection() -> None:
    controller = PolicyController(PLAN, Policy.HBN_ASYNC, _allocator())
    pilots = controller.take_initial(300)
    allocation = {task: 22 for task in TASKS}
    priority = PLAN.partial_plugin_priority(allocation)
    controller.apply_partial_priority(version=1, observed_pilots=0,
                                      allocation=allocation, priority=priority)
    prefetched = controller.take_prefetch(660)
    before = set(controller.submitted)
    allocation[TASKS[0]], allocation[TASKS[1]] = 1, 43
    controller.apply_partial_priority(version=2, observed_pilots=100,
        allocation=allocation, priority=PLAN.partial_plugin_priority(allocation))
    assert controller.submitted == before
    assert not controller.surplus_submitted()
    assert controller.selected is None
    for key in pilots:
        controller.record_reward(key, int(key.task_id) % 2)
    assert controller.selected is not None
    assert controller.surplus_submitted() == frozenset(prefetched) - controller.selected
    assert not controller.apply_partial_priority(version=3, observed_pilots=300,
        allocation=allocation, priority=PLAN.partial_plugin_priority(allocation))


@pytest.mark.parametrize("reverse", [False, True])
def test_sync_async_final_selection_matches_under_completion_reordering(reverse) -> None:
    controllers = [PolicyController(PLAN, policy, _allocator())
                   for policy in (Policy.HBN_SYNC, Policy.HBN_ASYNC)]
    for controller in controllers:
        pilots = controller.take_initial(300)
        assert controller.take_post_barrier(1) == ()
        if controller.policy is Policy.HBN_ASYNC:
            allocation = {task: 22 for task in TASKS}
            controller.apply_partial_priority(version=1, observed_pilots=0,
                allocation=allocation, priority=PLAN.partial_plugin_priority(allocation))
            controller.take_prefetch(200)
        else:
            assert controller.take_prefetch(200) == ()
        order = tuple(reversed(pilots)) if reverse and controller.policy is Policy.HBN_ASYNC else pilots
        for key in order[:-1]:
            assert controller.record_reward(key, (int(key.task_id) + key.ordinal) % 2) is None
        key = order[-1]
        assert controller.record_reward(key, (int(key.task_id) + key.ordinal) % 2) is not None
    assert controllers[0].selected == controllers[1].selected
    assert controllers[0].allocation_result.allocation == controllers[1].allocation_result.allocation
