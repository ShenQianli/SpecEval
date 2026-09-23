"""Latest-wins background priority updates for partial-pilot scheduling."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .allocation import PartialAllocationResult, PartialHBNAllocator
from .identities import RequestKey
from .protocol import ProtocolPlan


@dataclass(frozen=True)
class PublishedPartialPriority:
    version: int
    allocation_result: PartialAllocationResult
    priority: tuple[RequestKey, ...]
    total_compute_ns: int


class AsyncPartialPriorityWorker:
    """Compute provisional allocations off the serialized router thread.

    Submissions replace any queued snapshot. A result is published only when
    its version is still the newest requested version, so slow stale work can
    never overwrite a priority derived from more pilot rewards.
    """

    def __init__(
        self,
        allocator: PartialHBNAllocator,
        plan: ProtocolPlan,
    ) -> None:
        self._allocator = allocator
        self._plan = plan
        self._condition = threading.Condition()
        self._pending: tuple[int, Mapping[str, tuple[int, ...]]] | None = None
        self._published: PublishedPartialPriority | None = None
        self._requested_version = 0
        self._stopping = False
        self._error: BaseException | None = None
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._published_count = 0
        self._stale_discarded = 0
        self._compute_ns = 0
        self._max_compute_ns = 0
        self._thread = threading.Thread(
            target=self._run,
            name="partial-hbn-priority",
            daemon=True,
        )
        self._thread.start()

    def submit(self, rewards: Mapping[str, Sequence[int]]) -> int:
        snapshot = {
            str(task_id): tuple(map(int, values))
            for task_id, values in rewards.items()
        }
        with self._condition:
            self._raise_if_failed()
            if self._stopping:
                raise RuntimeError("partial priority worker is stopping")
            self._requested_version += 1
            version = self._requested_version
            self._pending = (version, snapshot)
            self._submitted += 1
            self._condition.notify()
            return version

    def poll(self, *, after_version: int = 0) -> PublishedPartialPriority | None:
        with self._condition:
            self._raise_if_failed()
            if (
                self._published is None
                or self._published.version <= after_version
                or self._published.version != self._requested_version
            ):
                return None
            return self._published

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify()
        self._thread.join()
        with self._condition:
            self._raise_if_failed()

    def metrics(self) -> dict[str, int]:
        with self._condition:
            return {
                "partial_priority_snapshots_submitted": self._submitted,
                "partial_priority_computations_started": self._started,
                "partial_priority_computations_completed": self._completed,
                "partial_priority_updates_published": self._published_count,
                "partial_priority_stale_discarded": self._stale_discarded,
                "partial_priority_compute_ns": self._compute_ns,
                "partial_priority_max_compute_ns": self._max_compute_ns,
                "partial_priority_requested_version": self._requested_version,
                "partial_priority_published_version": (
                    0 if self._published is None else self._published.version
                ),
            }

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("partial priority worker failed") from self._error

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        return
                    version, rewards = self._pending
                    self._pending = None
                    self._started += 1

                started_ns = time.perf_counter_ns()
                allocation_result = self._allocator.allocate(rewards)
                priority = self._plan.partial_plugin_priority(
                    allocation_result.allocation
                )
                total_compute_ns = time.perf_counter_ns() - started_ns
                published = PublishedPartialPriority(
                    version=version,
                    allocation_result=allocation_result,
                    priority=priority,
                    total_compute_ns=total_compute_ns,
                )

                with self._condition:
                    self._completed += 1
                    self._compute_ns += total_compute_ns
                    self._max_compute_ns = max(
                        self._max_compute_ns, total_compute_ns
                    )
                    if version == self._requested_version and not self._stopping:
                        self._published = published
                        self._published_count += 1
                    else:
                        self._stale_discarded += 1
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._stopping = True
                self._condition.notify_all()
