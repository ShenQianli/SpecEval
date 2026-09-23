"""Serializable router/worker command and event schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping


class CommandType(StrEnum):
    SUBMIT = "submit"
    ABORT = "abort"
    RESET = "reset"
    HEALTH = "health"
    SHUTDOWN = "shutdown"


class EventType(StrEnum):
    READY = "ready"
    STARTED = "started"
    COMPLETED = "completed"
    ABORTED = "aborted"
    INFRA_ERROR = "infra_error"
    RESET_COMPLETE = "reset_complete"
    HEALTHY = "healthy"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    WORKER_ERROR = "worker_error"


@dataclass(frozen=True)
class GeneratePayload:
    run_uuid: str
    logical_request_id: str
    physical_request_id: str
    task_id: str
    prompt: str
    seed: int
    attempt: int
    max_new_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def command(command_type: CommandType, **payload: Any) -> dict[str, Any]:
    return {"type": command_type.value, **payload}


def event(
    event_type: EventType,
    *,
    engine_id: int,
    timestamp_ns: int,
    **payload: Any,
) -> dict[str, Any]:
    return {
        "type": event_type.value,
        "engine_id": int(engine_id),
        "timestamp_ns": int(timestamp_ns),
        **payload,
    }


def validate_run_namespace(message: Mapping[str, Any], expected_run_uuid: str) -> None:
    actual = message.get("run_uuid")
    if actual != expected_run_uuid:
        raise ValueError(f"cross-run event: {actual!r} != {expected_run_uuid!r}")
