"""Central request-level router, pilot barrier, cancellation and artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from .allocation import HBNAllocator, PartialHBNAllocator
from .config import ExperimentConfig
from .controller import BarrierResult, PolicyController
from .identities import Phase, RequestKey, physical_request_id, request_seed
from .ledger import RequestLedger, RequestRecord, RequestState
from .messages import EventType, GeneratePayload, validate_run_namespace
from .partial_priority import AsyncPartialPriorityWorker
from .profile_rewards import ProfileBernoulliRewards
from .protocol import (
    HBN_POLICIES,
    Policy,
    ProtocolPlan,
)


@dataclass(frozen=True)
class RunTiming:
    t0_ns: int
    pilot_ready_ns: int
    allocation_done_ns: int
    result_ready_ns: int
    resource_released_ns: int

    def to_seconds(self) -> dict[str, float]:
        return {
            "result_seconds": (self.result_ready_ns - self.t0_ns) / 1e9,
            "pilot_ready_seconds": (self.pilot_ready_ns - self.t0_ns) / 1e9,
            "allocation_seconds": (self.allocation_done_ns - self.pilot_ready_ns) / 1e9,
            "cleanup_seconds": (self.resource_released_ns - self.result_ready_ns) / 1e9,
            "resource_released_seconds": (self.resource_released_ns - self.t0_ns) / 1e9,
        }


@dataclass(frozen=True)
class RunOutput:
    policy: str
    run_uuid: str
    summary: Mapping[str, Any]
    requests: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    engine_samples: tuple[Mapping[str, Any], ...]
    allocation: Mapping[str, Any] | None


def _preallocation_inflight_metrics(
    samples: Iterable[Mapping[str, Any]],
    *,
    horizon_seconds: float,
    engine_ids: tuple[int, ...],
    target: int,
) -> dict[str, float | int]:
    """Integrate router-visible in-flight shortfall before allocation."""

    states = {engine_id: 0 for engine_id in engine_ids}
    previous_t = 0.0
    shortfall_area = 0.0
    target_seconds = 0.0
    ordered = sorted(samples, key=lambda row: float(row["elapsed_seconds"]))
    for row in ordered:
        current_t = min(
            max(float(row["elapsed_seconds"]), previous_t), horizon_seconds
        )
        duration = current_t - previous_t
        active = sum(states.values())
        shortfall_area += duration * max(target - active, 0)
        if active == target:
            target_seconds += duration
        previous_t = current_t
        states[int(row["engine_id"])] = int(row["active_requests"])
        if previous_t >= horizon_seconds:
            break
    if previous_t < horizon_seconds:
        duration = horizon_seconds - previous_t
        active = sum(states.values())
        shortfall_area += duration * max(target - active, 0)
        if active == target:
            target_seconds += duration
    mean_inflight = (
        target - shortfall_area / horizon_seconds if horizon_seconds else 0.0
    )
    return {
        "inflight_target": int(target),
        "pre_allocation_inflight_shortfall_request_seconds": shortfall_area,
        "pre_allocation_mean_inflight": mean_inflight,
        "pre_allocation_target_fraction": (
            target_seconds / horizon_seconds if horizon_seconds else 0.0
        ),
    }


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        tasks: Mapping[str, Mapping[str, Any]],
        pool: Any,
        validity_check: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.tasks = {str(key): dict(value) for key, value in tasks.items()}
        task_ids = tuple(self.tasks)
        if len(task_ids) != config.design.n_tasks:
            raise ValueError("task count does not match frozen design")
        self.profile_rewards = ProfileBernoulliRewards(config, task_ids=task_ids)
        self.reward_manifest = self.profile_rewards.manifest()
        self.plan = ProtocolPlan(
            task_ids=task_ids,
            budget_per_task=config.design.budget_per_task,
            pilot_per_task=config.design.pilot_per_task,
            continuation_total=config.design.continuation_total,
            speculative_depth=config.design.speculative_depth,
            master_seed=config.master_seed,
        )
        self.allocator = HBNAllocator(
            task_ids,
            config.design.pilot_per_task,
            config.design.continuation_total,
            config.design.quadrature_order,
        )
        self.partial_allocator = PartialHBNAllocator(
            task_ids,
            config.design.pilot_per_task,
            config.design.continuation_total,
            config.design.quadrature_order,
        )
        self.pool = pool
        self.validity_check = validity_check
        self.engine_ids = tuple(range(len(config.engine.gpu_ids)))
        self.dispatch_window = config.engine.max_inflight_per_engine

    def _make_ledger(self) -> RequestLedger:
        max_continuation = self.plan.max_continuations_per_task
        keys = list(self.plan.pilots()) + [
            RequestKey(task_id, Phase.CONTINUATION, ordinal)
            for task_id in self.plan.task_ids
            for ordinal in range(1, max_continuation + 1)
        ]
        records = []
        for key in keys:
            logical_id = key.logical_id(self.config.protocol_fingerprint)
            records.append(RequestRecord(
                key=key,
                logical_request_id=logical_id,
                seed=request_seed(self.config.master_seed, logical_id),
            ))
        return RequestLedger(records)

    def _payload(
        self,
        record: RequestRecord,
        run_uuid: str,
        attempt: int,
    ) -> dict[str, Any]:
        return GeneratePayload(
            run_uuid=run_uuid,
            logical_request_id=record.logical_request_id,
            physical_request_id=physical_request_id(
                run_uuid, record.logical_request_id, attempt
            ),
            task_id=record.key.task_id,
            prompt=str(self.tasks[record.key.task_id]["rendered_prompt"]),
            seed=record.seed,
            attempt=attempt,
        ).to_dict()

    def run_policy(
        self,
        policy: Policy,
        *,
        run_uuid: str,
        block_index: int,
        output_dir: str | Path | None = None,
        event_timeout: float = 3600.0,
    ) -> RunOutput:
        controller = PolicyController(
            self.plan,
            policy,
            self.allocator if policy in HBN_POLICIES else None,
        )
        partial_priority_worker = (
            AsyncPartialPriorityWorker(self.partial_allocator, self.plan)
            if policy is Policy.HBN_ASYNC
            else None
        )
        partial_prefetch_submitted = 0
        partial_priority_apply_ns = 0
        partial_priority_final_l1: int | None = None
        ledger = self._make_ledger()
        active_by_engine: dict[int, set[str]] = {
            engine_id: set() for engine_id in self.engine_ids
        }
        physical_to_key: dict[str, RequestKey] = {}
        attempt_by_key: dict[RequestKey, int] = {}
        submitted_attempts = 0
        retry_attempts = 0
        raw_events: list[dict[str, Any]] = []
        engine_samples: list[dict[str, Any]] = []
        allocation_artifact: dict[str, Any] | None = None
        def sample_engine(
            engine_id: int,
            event_type: str,
            timestamp_ns: int,
            *,
            completion_tokens_delta: int = 0,
        ) -> None:
            observed_ns = time.monotonic_ns()
            engine_samples.append({
                "timestamp_ns": int(timestamp_ns),
                "router_observed_ns": observed_ns,
                "elapsed_seconds": (observed_ns - t0_ns) / 1e9,
                "engine_id": int(engine_id),
                "event_type": str(event_type),
                "active_requests": len(active_by_engine[engine_id]),
                "completion_tokens_delta": int(completion_tokens_delta),
            })

        def submit_assignments(
            keys: Iterable[RequestKey],
            *,
            prefetched: bool,
        ) -> None:
            nonlocal submitted_attempts
            assignments: dict[int, list[dict[str, Any]]] = {
                engine_id: [] for engine_id in self.engine_ids
            }
            for key in keys:
                engine_id = min(
                    self.engine_ids,
                    key=lambda candidate: (
                        len(active_by_engine[candidate]),
                        candidate,
                    ),
                )
                if len(active_by_engine[engine_id]) >= self.dispatch_window:
                    raise RuntimeError(
                        f"engine {engine_id} dispatch window would be exceeded"
                    )
                attempt = attempt_by_key.get(key, 0)
                record = ledger[key]
                payload = self._payload(record, run_uuid, attempt)
                physical_id = str(payload["physical_request_id"])
                timestamp_ns = time.monotonic_ns()
                ledger.mark_submitted(
                    key,
                    physical_request_id=physical_id,
                    attempt=attempt,
                    engine_id=engine_id,
                    timestamp_ns=timestamp_ns,
                    prefetched=prefetched,
                )
                physical_to_key[physical_id] = key
                active_by_engine[engine_id].add(physical_id)
                assignments[engine_id].append(payload)
                submitted_attempts += 1
            for engine_id, payloads in assignments.items():
                self.pool.submit(engine_id, payloads)
                if payloads:
                    sample_engine(engine_id, "router_submit", time.monotonic_ns())

        t0_ns = time.monotonic_ns()
        if partial_priority_worker is not None:
            # Publish the prior-only allocation asynchronously.  This gives
            # the gated policy an initial, budget-sized eligible set without
            # falling back to the unbounded breadth-first candidate universe.
            partial_priority_worker.submit(controller.pilot_rewards_snapshot())

        def refill_eligible() -> None:
            available = sum(
                self.dispatch_window - len(active_by_engine[engine_id])
                for engine_id in self.engine_ids
            )
            if available <= 0:
                return
            if controller.initial_queue:
                keys = controller.take_initial(available)
            elif controller.barrier_released and controller.post_barrier_queue:
                keys = controller.take_post_barrier(available)
            else:
                return
            submit_assignments(keys, prefetched=False)

        def refill_speculation() -> None:
            if (
                policy is not Policy.HBN_ASYNC
                or controller.barrier_released
            ):
                return
            available = sum(
                self.dispatch_window - len(active_by_engine[engine_id])
                for engine_id in self.engine_ids
            )
            if available <= 0:
                return
            nonlocal partial_prefetch_submitted
            prefetch = controller.take_prefetch(available)
            if controller.partial_priority_version > 0:
                partial_prefetch_submitted += len(prefetch)
            submit_assignments(prefetch, prefetched=True)

        def apply_published_partial_priority() -> None:
            nonlocal partial_priority_apply_ns
            if partial_priority_worker is None or controller.barrier_released:
                return
            published = partial_priority_worker.poll(
                after_version=controller.partial_priority_version
            )
            if published is None:
                return
            started_ns = time.perf_counter_ns()
            controller.apply_partial_priority(
                version=published.version,
                observed_pilots=published.allocation_result.observed_total,
                allocation=published.allocation_result.allocation,
                priority=published.priority,
            )
            partial_priority_apply_ns += time.perf_counter_ns() - started_ns

        refill_eligible()
        refill_speculation()
        pilot_ready_ns: int | None = None
        allocation_done_ns: int | None = None
        result_ready_ns: int | None = None
        resource_released_ns: int | None = None

        def abort_surplus() -> None:
            if controller.selected is None:
                return
            ledger.resolve_selection(controller.selected)
            aborts: dict[int, list[str]] = {engine_id: [] for engine_id in self.engine_ids}
            for key in controller.surplus_submitted():
                record = ledger[key]
                if record.physical_request_id in active_by_engine.get(record.engine_id or 0, set()):
                    assert record.engine_id is not None and record.physical_request_id is not None
                    aborts[record.engine_id].append(record.physical_request_id)
            for engine_id, physical_ids in aborts.items():
                self.pool.abort(engine_id, physical_ids)

        def handle_barrier(released: BarrierResult, reward_ready_ns: int) -> None:
            nonlocal pilot_ready_ns, allocation_done_ns, allocation_artifact
            nonlocal partial_priority_final_l1
            pilot_ready_ns = reward_ready_ns
            allocation_done_ns = time.monotonic_ns()
            if released.allocation_result is not None:
                allocation_artifact = {
                    "allocation": dict(released.allocation_result.allocation),
                    "pilot_counts": dict(released.allocation_result.pilot_counts),
                    "posterior_variances": dict(
                        released.allocation_result.posterior_variances
                    ),
                    "compute_ns": released.allocation_result.compute_ns,
                }
                if controller.partial_priority_allocation is not None:
                    partial_priority_final_l1 = sum(
                        abs(
                            int(controller.partial_priority_allocation[task_id])
                            - int(released.allocation_result.allocation[task_id])
                        )
                        for task_id in self.plan.task_ids
                    )
            abort_surplus()

        if policy is Policy.UNIFORM_FLAT:
            allocation_done_ns = t0_ns

        while resource_released_ns is None:
            if self.validity_check is not None:
                self.validity_check()
            row = dict(self.pool.next_event(timeout=event_timeout))
            # Publish work computed while the router was blocked waiting for
            # this event before the event can enqueue a newer snapshot.
            apply_published_partial_priority()
            received_ns = time.monotonic_ns()
            event_type = str(row["type"])
            engine_id = int(row["engine_id"])
            physical_id = row.get("physical_request_id")
            sanitized = dict(row)
            if "raw_output" in sanitized:
                sanitized["raw_output_chars"] = len(str(sanitized.pop("raw_output")))
            sanitized["router_received_ns"] = received_ns

            if event_type == EventType.WORKER_ERROR.value:
                raise RuntimeError(f"worker error during run: {row}")
            if physical_id is None:
                raise RuntimeError(f"unexpected non-request event during run: {row}")
            validate_run_namespace(row, run_uuid)
            physical_id = str(physical_id)
            if physical_id not in physical_to_key:
                raise RuntimeError(f"unknown physical request event: {physical_id}")
            key = physical_to_key[physical_id]
            record = ledger[key]

            if event_type == EventType.STARTED.value:
                if record.state is RequestState.SUBMITTED:
                    ledger.mark_started(key, int(row["timestamp_ns"]))
                sanitized["active_requests_after_event"] = len(
                    active_by_engine[engine_id]
                )
                raw_events.append(sanitized)
                sample_engine(engine_id, event_type, int(row["timestamp_ns"]))
                # A queued STARTED event can cross an allocation/abort decision;
                # the fixed-ID disposition remains authoritative.
                continue

            if event_type in {
                EventType.COMPLETED.value,
                EventType.ABORTED.value,
                EventType.INFRA_ERROR.value,
            }:
                active_by_engine[engine_id].discard(physical_id)

            sanitized["active_requests_after_event"] = len(active_by_engine[engine_id])
            raw_events.append(sanitized)
            token_delta = (
                int(row.get("completion_tokens") or 0)
                if event_type == EventType.COMPLETED.value
                else int(row.get("generated_tokens_before_abort") or 0)
            )
            sample_engine(
                engine_id,
                event_type,
                int(row["timestamp_ns"]),
                completion_tokens_delta=token_delta,
            )

            if event_type == EventType.COMPLETED.value:
                # Reveal the fixed reward only after generation completes.
                reward = self.profile_rewards.reward(key)
                reward_ready = time.monotonic_ns()
                ledger.mark_completed(
                    key,
                    completed_ns=int(row["timestamp_ns"]),
                    reward_ready_ns=reward_ready,
                    reward=reward,
                    raw_output=str(row.get("raw_output") or ""),
                    prompt_tokens=int(row.get("prompt_tokens") or 0),
                    completion_tokens=int(row.get("completion_tokens") or 0),
                    finish_reason=str(row.get("finish_reason") or "stop"),
                )
                released = controller.record_reward(key, reward)
                if (
                    partial_priority_worker is not None
                    and key.phase is Phase.PILOT
                    and released is None
                ):
                    partial_priority_worker.submit(
                        controller.pilot_rewards_snapshot()
                    )
                if released is not None:
                    handle_barrier(released, reward_ready)
                if pilot_ready_ns is None and controller.all_pilot_rewards_ready:
                    pilot_ready_ns = reward_ready
                    if policy is Policy.UNIFORM_FLAT:
                        allocation_done_ns = reward_ready
                if controller.selected is not None:
                    ledger.accept_completed(controller.selected)

            elif event_type == EventType.ABORTED.value:
                ledger.mark_aborted(
                    key,
                    timestamp_ns=int(row["timestamp_ns"]),
                    prompt_tokens_before_abort=int(
                        row.get("prompt_tokens_before_abort") or 0
                    ),
                    generated_tokens_before_abort=int(
                        row.get("generated_tokens_before_abort") or 0
                    ),
                )

            elif event_type == EventType.INFRA_ERROR.value:
                ledger.mark_infra_error(
                    key, int(row["timestamp_ns"]), str(row.get("error") or "infra_error")
                )
                selected_or_unknown = (
                    controller.selected is None or key in controller.selected
                )
                attempt = attempt_by_key.get(key, 0)
                if selected_or_unknown and attempt < self.config.execution.max_infra_retries:
                    attempt_by_key[key] = attempt + 1
                    retry_attempts += 1
                    submit_assignments([key], prefetched=record.was_prefetched)
                elif selected_or_unknown:
                    raise RuntimeError(f"exhausted infra retries for {key}: {row}")

            # Replenish eligible work, then admit speculation from the latest
            # provisional allocation into remaining capacity.
            refill_eligible()
            refill_speculation()

            if result_ready_ns is None and controller.result_ready():
                result_ready_ns = time.monotonic_ns()
            if result_ready_ns is not None and not any(active_by_engine.values()):
                resource_released_ns = time.monotonic_ns()

        partial_priority_metrics = {
            "partial_priority_snapshots_submitted": 0,
            "partial_priority_computations_started": 0,
            "partial_priority_computations_completed": 0,
            "partial_priority_updates_published": 0,
            "partial_priority_stale_discarded": 0,
            "partial_priority_compute_ns": 0,
            "partial_priority_max_compute_ns": 0,
            "partial_priority_requested_version": 0,
            "partial_priority_published_version": 0,
        }
        if partial_priority_worker is not None:
            partial_priority_worker.close()
            partial_priority_metrics = partial_priority_worker.metrics()

        assert pilot_ready_ns is not None
        assert allocation_done_ns is not None
        assert result_ready_ns is not None
        assert resource_released_ns is not None
        assert controller.selected is not None
        ledger.accept_completed(controller.selected)
        accepted_keys = frozenset(
            key
            for key, record in ledger.records.items()
            if record.state is RequestState.ACCEPTED
        )
        if accepted_keys != controller.selected:
            missing = controller.selected - accepted_keys
            unexpected = accepted_keys - controller.selected
            raise AssertionError(
                "fixed-ID acceptance mismatch: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        timing = RunTiming(
            t0_ns,
            pilot_ready_ns,
            allocation_done_ns,
            result_ready_ns,
            resource_released_ns,
        )
        request_rows = tuple(
            record.to_dict()
            for record in ledger.records.values()
            if record.submitted_ns is not None
        )
        accepted = sum(row["state"] == RequestState.ACCEPTED.value for row in request_rows)
        if accepted != self.plan.accepted_total:
            raise AssertionError(f"accepted={accepted}, expected {self.plan.accepted_total}")
        completed = sum(row["raw_output"] is not None for row in request_rows)
        started = sum(row["started_ns"] is not None for row in request_rows)
        state_counts = {state.value: 0 for state in RequestState}
        for row in request_rows:
            state_counts[str(row["state"])] += 1
        token_costs = ledger.token_accounting()
        inflight_metrics = _preallocation_inflight_metrics(
            engine_samples,
            horizon_seconds=(pilot_ready_ns - t0_ns) / 1e9,
            engine_ids=self.engine_ids,
            target=len(self.engine_ids) * self.dispatch_window,
        )
        prefetch_records = [ledger[key] for key in controller.prefetched]
        prefetched_completed_before_allocation = sum(
            record.reward_ready_ns is not None
            and record.reward_ready_ns <= pilot_ready_ns
            for record in prefetch_records
        )
        prefetched_inflight_at_allocation = sum(
            record.submitted_ns is not None
            and record.submitted_ns <= pilot_ready_ns
            and (
                record.completed_ns is None
                or record.completed_ns > pilot_ready_ns
            )
            for record in prefetch_records
        )
        summary = {
            "schema_version": 1,
            "policy": policy.value,
            "run_uuid": run_uuid,
            "block_index": int(block_index),
            "protocol_fingerprint": self.config.protocol_fingerprint,
            "reward_mode": self.reward_manifest["mode"],
            "reward_fingerprint": self.reward_manifest.get("fingerprint"),
            "reward_profile_sha256": self.reward_manifest.get("profile_sha256"),
            "reward_sampler_version": self.reward_manifest.get("sampler_version"),
            "reproducibility_mode": self.config.engine.reproducibility_mode,
            "dispatch_window_per_engine": self.dispatch_window,
            "speculative_admission": (
                "partial_allocation_gated"
                if policy is Policy.HBN_ASYNC
                else "other"
            ),
            **timing.to_seconds(),
            "planned_valid": self.plan.accepted_total,
            "submitted_attempts": submitted_attempts,
            "started_logical": started,
            "completed_logical": completed,
            "actual_terminal_compute_requests": completed
            + state_counts[RequestState.ABORTED_RUNNING.value],
            "accepted_logical": accepted,
            "actual_rollouts": completed
            + state_counts[RequestState.ABORTED_RUNNING.value],
            "valid_rollouts": accepted,
            "retry_attempts": retry_attempts,
            **{f"state_{key}": value for key, value in state_counts.items()},
            **token_costs,
            **inflight_metrics,
            "prefetched_submitted": len(controller.prefetched),
            "prefetch_selected": len(controller.prefetched & set(controller.selected)),
            "prefetch_discarded": len(controller.prefetched - set(controller.selected)),
            "prefetch_max_ordinal": max(
                (key.ordinal for key in controller.prefetched), default=0
            ),
            "prefetched_completed_before_allocation": (
                prefetched_completed_before_allocation
            ),
            "prefetched_inflight_at_allocation": prefetched_inflight_at_allocation,
            **partial_priority_metrics,
            "partial_priority_updates_applied": (
                controller.partial_priority_updates_applied
            ),
            "partial_priority_apply_ns": partial_priority_apply_ns,
            "partial_priority_latest_observed_pilots": (
                controller.partial_priority_observed_pilots
            ),
            "partial_prefetch_submitted": partial_prefetch_submitted,
            "partial_priority_final_l1": partial_priority_final_l1,
        }
        if allocation_artifact is not None and controller.partial_priority_allocation:
            allocation_artifact["partial_plugin"] = {
                "admission": "partial_allocation_gated",
                "last_applied_version": controller.partial_priority_version,
                "last_observed_pilots": controller.partial_priority_observed_pilots,
                "last_provisional_allocation": dict(
                    controller.partial_priority_allocation
                ),
                "final_l1": partial_priority_final_l1,
                **partial_priority_metrics,
            }
        output = RunOutput(
            policy=policy.value,
            run_uuid=run_uuid,
            summary=summary,
            requests=request_rows,
            events=tuple(raw_events),
            engine_samples=tuple(engine_samples),
            allocation=allocation_artifact,
        )
        if output_dir is not None:
            write_run_output(output, output_dir)
        return output


def write_run_output(output: RunOutput, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(dict(output.summary), indent=2, ensure_ascii=False) + "\n"
    )
    (root / "allocation.json").write_text(
        json.dumps(output.allocation, indent=2, ensure_ascii=False) + "\n"
    )
    pd.DataFrame(output.requests).to_parquet(root / "requests.parquet", index=False)
    pd.DataFrame(output.events).to_parquet(root / "events.parquet", index=False)
    pd.DataFrame(output.engine_samples).to_parquet(
        root / "engine_samples.parquet", index=False
    )
    return root


def balanced_policy_orders(
    policies: Iterable[Policy], blocks: int, seed: int
) -> tuple[tuple[Policy, ...], ...]:
    import random

    values = list(policies)
    random.Random(seed).shuffle(values)
    return tuple(
        tuple(values[(position + block) % len(values)] for position in range(len(values)))
        for block in range(blocks)
    )
