from __future__ import annotations

import time

from async_hbn.allocation import PartialHBNAllocator
from async_hbn.partial_priority import AsyncPartialPriorityWorker
from async_hbn.protocol import ProtocolPlan


TASKS = tuple(str(index) for index in range(60, 90))
PLAN = ProtocolPlan(TASKS, 32, 10, 660, 4, 42)


def test_background_partial_priority_is_latest_wins() -> None:
    worker = AsyncPartialPriorityWorker(
        PartialHBNAllocator(TASKS, 10, 660), PLAN
    )
    first = {task_id: (0,) for task_id in TASKS}
    second = {task_id: (0, 1) for task_id in TASKS}
    assert worker.submit(first) == 1
    assert worker.submit(second) == 2
    deadline = time.monotonic() + 2.0
    published = None
    while published is None and time.monotonic() < deadline:
        published = worker.poll()
        time.sleep(0.001)
    worker.close()
    assert published is not None
    assert published.version == 2
    assert published.allocation_result.observed_total == 60
    assert len(published.priority) == 660
    metrics = worker.metrics()
    assert metrics["partial_priority_snapshots_submitted"] == 2
    assert metrics["partial_priority_updates_published"] >= 1
