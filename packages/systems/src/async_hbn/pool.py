"""Persistent one-GPU worker pool and explicit group reset."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import uuid
from dataclasses import asdict
from typing import Any

from .config import ExperimentConfig
from .messages import CommandType, EventType, command
from .worker import worker_process_main


class PersistentWorkerPool:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.context = mp.get_context("spawn")
        self.event_queue = self.context.Queue()
        self.command_queues: dict[int, Any] = {}
        self.processes: dict[int, mp.Process] = {}
        model_config = {**asdict(config.model), "path": str(config.model.path)}
        engine_config = asdict(config.engine)
        sampling_config = asdict(config.sampling)
        for engine_id, gpu_id in enumerate(config.engine.gpu_ids):
            command_queue = self.context.Queue()
            process = self.context.Process(
                target=worker_process_main,
                args=(
                    engine_id,
                    gpu_id,
                    command_queue,
                    self.event_queue,
                    model_config,
                    engine_config,
                    sampling_config,
                ),
                name=f"async-hbn-engine-{engine_id}-gpu-{gpu_id}",
            )
            self.command_queues[engine_id] = command_queue
            self.processes[engine_id] = process
        self.started = False

    def _get_event(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for worker pool event")
            try:
                return dict(self.event_queue.get(timeout=min(5.0, remaining)))
            except queue.Empty:
                failed = {
                    engine_id: process.exitcode
                    for engine_id, process in self.processes.items()
                    if not process.is_alive()
                }
                if failed:
                    raise RuntimeError(f"worker processes exited: {failed}")

    def start(self, timeout: float = 1800.0) -> list[dict[str, Any]]:
        if self.started:
            raise RuntimeError("worker pool already started")
        for process in self.processes.values():
            process.start()
        ready: set[int] = set()
        events: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while len(ready) < len(self.processes):
            row = self._get_event(deadline - time.monotonic())
            events.append(row)
            if row["type"] == EventType.READY.value:
                ready.add(int(row["engine_id"]))
            elif row["type"] == EventType.WORKER_ERROR.value:
                raise RuntimeError(f"worker startup failed: {row}")
        self.started = True
        return events

    def submit(self, engine_id: int, requests: list[dict[str, Any]]) -> None:
        if requests:
            self.command_queues[engine_id].put(
                command(CommandType.SUBMIT, requests=requests)
            )

    def abort(self, engine_id: int, physical_request_ids: list[str]) -> None:
        if physical_request_ids:
            self.command_queues[engine_id].put(command(
                CommandType.ABORT,
                physical_request_ids=physical_request_ids,
            ))

    def next_event(self, timeout: float = 300.0) -> dict[str, Any]:
        return self._get_event(timeout)

    def reset(self, timeout: float = 300.0) -> dict[str, Any]:
        reset_id = uuid.uuid4().hex
        start = time.monotonic_ns()
        for queue_ in self.command_queues.values():
            queue_.put(command(CommandType.RESET, reset_id=reset_id))
        rows: list[dict[str, Any]] = []
        completed: set[int] = set()
        deadline = time.monotonic() + timeout
        while len(completed) < len(self.processes):
            row = self._get_event(deadline - time.monotonic())
            rows.append(row)
            if row["type"] == EventType.RESET_COMPLETE.value:
                if row.get("reset_id") != reset_id:
                    raise RuntimeError(f"stale reset event: {row}")
                completed.add(int(row["engine_id"]))
            elif row["type"] == EventType.WORKER_ERROR.value:
                raise RuntimeError(f"worker reset error: {row}")
            else:
                raise RuntimeError(f"request event crossed reset boundary: {row}")
        failures = [row for row in rows if not row.get("success")]
        if failures:
            raise RuntimeError(f"group reset failed: {failures}")
        return {
            "reset_id": reset_id,
            "reset_duration_ns": time.monotonic_ns() - start,
            "workers": rows,
        }

    def shutdown(self, timeout: float = 120.0) -> None:
        for queue_ in self.command_queues.values():
            queue_.put(command(CommandType.SHUTDOWN))
        deadline = time.monotonic() + timeout
        for process in self.processes.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
        self.started = False

