"""Auditable request lifecycle and cost accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .identities import RequestKey


class RequestState(StrEnum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    STARTED = "started"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    CANCELLED_QUEUED = "cancelled_queued"
    ABORTED_RUNNING = "aborted_running"
    COMPLETED_DISCARDED = "completed_discarded"
    INFRA_ERROR = "infra_error"


@dataclass
class RequestRecord:
    key: RequestKey
    logical_request_id: str
    seed: int
    state: RequestState = RequestState.PLANNED
    physical_request_id: str | None = None
    attempt: int = 0
    engine_id: int | None = None
    submitted_ns: int | None = None
    started_ns: int | None = None
    completed_ns: int | None = None
    reward_ready_ns: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_before_abort: int = 0
    generated_tokens_before_abort: int = 0
    reward: int | None = None
    raw_output: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    was_prefetched: bool = False

    @property
    def completed(self) -> bool:
        return self.completed_ns is not None and self.reward_ready_ns is not None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.update({
            "task_id": self.key.task_id,
            "phase": self.key.phase.value,
            "ordinal": self.key.ordinal,
            "replicate": self.key.replicate,
            "state": self.state.value,
        })
        row.pop("key")
        return row


class RequestLedger:
    def __init__(self, records: list[RequestRecord]) -> None:
        self.records = {record.key: record for record in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate request keys")

    def __getitem__(self, key: RequestKey) -> RequestRecord:
        return self.records[key]

    def mark_submitted(
        self,
        key: RequestKey,
        *,
        physical_request_id: str,
        attempt: int,
        engine_id: int,
        timestamp_ns: int,
        prefetched: bool,
    ) -> None:
        record = self.records[key]
        if record.state not in {RequestState.PLANNED, RequestState.INFRA_ERROR}:
            raise ValueError(f"cannot submit {key} from state={record.state}")
        record.state = RequestState.SUBMITTED
        record.physical_request_id = physical_request_id
        record.attempt = int(attempt)
        record.engine_id = int(engine_id)
        record.submitted_ns = int(timestamp_ns)
        record.started_ns = None
        record.completed_ns = None
        record.reward_ready_ns = None
        record.was_prefetched = record.was_prefetched or bool(prefetched)
        record.error = None

    def mark_started(self, key: RequestKey, timestamp_ns: int) -> None:
        record = self.records[key]
        if record.state is not RequestState.SUBMITTED:
            raise ValueError(f"cannot start {key} from state={record.state}")
        record.state = RequestState.STARTED
        record.started_ns = int(timestamp_ns)

    def mark_completed(
        self,
        key: RequestKey,
        *,
        completed_ns: int,
        reward_ready_ns: int,
        reward: int,
        raw_output: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
    ) -> None:
        record = self.records[key]
        late_surplus = record.state in {
            RequestState.CANCELLED_QUEUED,
            RequestState.ABORTED_RUNNING,
        }
        if record.state not in {
            RequestState.SUBMITTED,
            RequestState.STARTED,
            RequestState.CANCELLED_QUEUED,
            RequestState.ABORTED_RUNNING,
        }:
            raise ValueError(f"cannot complete {key} from state={record.state}")
        if reward not in (0, 1):
            raise ValueError("AIME reward must be Bernoulli")
        if reward_ready_ns < completed_ns:
            raise ValueError("reward cannot be ready before generation completes")
        record.state = (
            RequestState.COMPLETED_DISCARDED if late_surplus else RequestState.COMPLETED
        )
        record.completed_ns = int(completed_ns)
        record.reward_ready_ns = int(reward_ready_ns)
        record.reward = int(reward)
        record.raw_output = str(raw_output)
        record.prompt_tokens = int(prompt_tokens)
        record.completion_tokens = int(completion_tokens)
        record.finish_reason = str(finish_reason)

    def mark_infra_error(self, key: RequestKey, timestamp_ns: int, error: str) -> None:
        record = self.records[key]
        if record.state not in {RequestState.SUBMITTED, RequestState.STARTED}:
            raise ValueError(f"cannot fail {key} from state={record.state}")
        record.state = RequestState.INFRA_ERROR
        record.completed_ns = int(timestamp_ns)
        record.error = str(error)

    def mark_aborted(
        self,
        key: RequestKey,
        *,
        timestamp_ns: int,
        prompt_tokens_before_abort: int = 0,
        generated_tokens_before_abort: int,
    ) -> None:
        record = self.records[key]
        if record.state in {RequestState.SUBMITTED, RequestState.CANCELLED_QUEUED}:
            record.state = RequestState.CANCELLED_QUEUED
        elif record.state in {RequestState.STARTED, RequestState.ABORTED_RUNNING}:
            record.state = RequestState.ABORTED_RUNNING
        else:
            raise ValueError(f"cannot abort {key} from state={record.state}")
        record.completed_ns = int(timestamp_ns)
        record.prompt_tokens_before_abort = int(prompt_tokens_before_abort)
        record.generated_tokens_before_abort = int(generated_tokens_before_abort)

    def resolve_selection(self, selected: frozenset[RequestKey]) -> None:
        for key, record in self.records.items():
            if key in selected:
                if record.state is RequestState.COMPLETED:
                    record.state = RequestState.ACCEPTED
                continue
            if record.state is RequestState.COMPLETED:
                record.state = RequestState.COMPLETED_DISCARDED
            elif record.state is RequestState.SUBMITTED:
                record.state = RequestState.CANCELLED_QUEUED
            elif record.state is RequestState.STARTED:
                record.state = RequestState.ABORTED_RUNNING

    def accept_completed(self, selected: frozenset[RequestKey]) -> None:
        for key in selected:
            record = self.records[key]
            if record.state is RequestState.COMPLETED:
                record.state = RequestState.ACCEPTED

    def counts(self) -> dict[str, int]:
        output = {state.value: 0 for state in RequestState}
        for record in self.records.values():
            output[record.state.value] += 1
        output["submitted_attempts"] = sum(
            record.submitted_ns is not None for record in self.records.values()
        )
        return output

    def token_accounting(self) -> dict[str, int]:
        accepted = [r for r in self.records.values() if r.state is RequestState.ACCEPTED]
        discarded = [
            r for r in self.records.values() if r.state is RequestState.COMPLETED_DISCARDED
        ]
        aborted = [r for r in self.records.values() if r.state is RequestState.ABORTED_RUNNING]
        return {
            "accepted_prompt_tokens": sum(r.prompt_tokens for r in accepted),
            "accepted_completion_tokens": sum(r.completion_tokens for r in accepted),
            "discarded_prompt_tokens": sum(r.prompt_tokens for r in discarded),
            "discarded_completion_tokens": sum(r.completion_tokens for r in discarded),
            "aborted_prompt_tokens": sum(r.prompt_tokens_before_abort for r in aborted),
            "aborted_generated_tokens": sum(r.generated_tokens_before_abort for r in aborted),
        }
