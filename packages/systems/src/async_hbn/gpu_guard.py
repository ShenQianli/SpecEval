"""Fail-closed detection of unrelated compute processes on experiment GPUs."""

from __future__ import annotations

import csv
import io
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


def _csv(command: list[str]) -> list[list[str]]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(result.stdout))
        if row
    ]


def gpu_uuids(gpu_ids: Iterable[int]) -> frozenset[str]:
    wanted = set(map(int, gpu_ids))
    rows = _csv([
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ])
    mapping = {int(index): uuid for index, uuid in rows}
    missing = wanted - set(mapping)
    if missing:
        raise RuntimeError(f"nvidia-smi did not report GPU IDs {sorted(missing)}")
    return frozenset(mapping[index] for index in wanted)


def compute_processes(target_uuids: frozenset[str]) -> list[dict[str, Any]]:
    rows = _csv([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    output = []
    for pid, uuid, process_name, used_memory in rows:
        if uuid not in target_uuids:
            continue
        output.append({
            "pid": int(pid),
            "gpu_uuid": uuid,
            "process_name": process_name,
            "used_gpu_memory_mib": int(used_memory),
        })
    return output


@dataclass(frozen=True)
class Contamination:
    detected_unix_ns: int
    unexpected_processes: tuple[dict[str, Any], ...]


class GPUProcessGuard:
    def __init__(self, gpu_ids: Iterable[int], poll_seconds: float = 10.0) -> None:
        self.target_uuids = gpu_uuids(gpu_ids)
        self.poll_seconds = float(poll_seconds)
        self.allowed_pids: frozenset[int] = frozenset()
        self.initial_processes: tuple[dict[str, Any], ...] = ()
        self.contamination: Contamination | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def require_idle(self) -> None:
        existing = compute_processes(self.target_uuids)
        if existing:
            raise RuntimeError(
                "target GPUs already have compute processes: "
                + ", ".join(f"pid={row['pid']} {row['process_name']}" for row in existing)
            )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU process guard already started")
        initial = tuple(compute_processes(self.target_uuids))
        if not initial:
            raise RuntimeError("no experiment GPU process found after worker startup")
        self.initial_processes = initial
        self.allowed_pids = frozenset(int(row["pid"]) for row in initial)
        self._thread = threading.Thread(
            target=self._monitor,
            name="async-hbn-gpu-process-guard",
            daemon=True,
        )
        self._thread.start()

    def _monitor(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                rows = compute_processes(self.target_uuids)
                unexpected = tuple(
                    row for row in rows if int(row["pid"]) not in self.allowed_pids
                )
                if unexpected:
                    self.contamination = Contamination(time.time_ns(), unexpected)
                    return
            except BaseException as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return

    def raise_if_invalid(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"GPU contamination monitor failed: {self.error}")
        if self.contamination is not None:
            raise RuntimeError(
                "unexpected GPU compute process detected: "
                + ", ".join(
                    f"pid={row['pid']} {row['process_name']}"
                    for row in self.contamination.unexpected_processes
                )
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds + 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "poll_seconds": self.poll_seconds,
            "target_gpu_uuids": sorted(self.target_uuids),
            "allowed_pids": sorted(self.allowed_pids),
            "initial_processes": list(self.initial_processes),
            "contamination": (
                None
                if self.contamination is None
                else {
                    "detected_unix_ns": self.contamination.detected_unix_ns,
                    "unexpected_processes": list(
                        self.contamination.unexpected_processes
                    ),
                }
            ),
            "monitor_error": self.error,
        }
