"""Stable logical identities and run-specific physical attempt identities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


class Phase(StrEnum):
    PILOT = "pilot"
    CONTINUATION = "continuation"


@dataclass(frozen=True, order=True)
class RequestKey:
    task_id: str
    phase: Phase
    ordinal: int
    replicate: int = 0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be nonempty")
        if self.ordinal < 1:
            raise ValueError("ordinal is one-based and must be positive")
        if self.replicate < 0:
            raise ValueError("replicate must be nonnegative")

    def logical_id(self, protocol_fingerprint: str) -> str:
        return (
            f"{protocol_fingerprint}|task={self.task_id}|phase={self.phase.value}"
            f"|ordinal={self.ordinal:04d}|replicate={self.replicate:03d}"
        )


def physical_request_id(
    run_uuid: str,
    logical_request_id: str,
    attempt: int,
) -> str:
    if not run_uuid:
        raise ValueError("run_uuid must be nonempty")
    if attempt < 0:
        raise ValueError("attempt must be nonnegative")
    return f"run={run_uuid}|attempt={attempt:02d}|{logical_request_id}"


def request_seed(master_seed: int, logical_request_id: str) -> int:
    payload = f"master_seed={int(master_seed)}|{logical_request_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def stable_task_permutation(task_ids: tuple[str, ...], master_seed: int) -> tuple[str, ...]:
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")
    return tuple(sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"prefetch-order|{master_seed}|{task_id}".encode()
        ).digest(),
    ))

