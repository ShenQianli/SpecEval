"""Single-GPU persistent-worker, generation, reset and abort compatibility smoke."""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .identities import Phase, RequestKey, physical_request_id, request_seed
from .messages import CommandType, EventType, GeneratePayload, command
from .tasks import load_and_render_tasks
from .worker import worker_process_main


TERMINAL_EVENTS = {
    EventType.COMPLETED.value,
    EventType.ABORTED.value,
    EventType.INFRA_ERROR.value,
}


def _get_event(event_queue: Any, process: mp.Process, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for worker event")
        try:
            return dict(event_queue.get(timeout=min(5.0, remaining)))
        except queue.Empty:
            if not process.is_alive():
                raise RuntimeError(f"worker exited with code {process.exitcode}")


def _wait_type(
    event_queue: Any,
    process: mp.Process,
    expected: set[str],
    timeout: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while True:
        event_row = _get_event(event_queue, process, deadline - time.monotonic())
        observed.append(event_row)
        if event_row["type"] in expected:
            return event_row, observed
        if event_row["type"] == EventType.WORKER_ERROR.value:
            raise RuntimeError(f"worker error: {event_row}")


def run_gpu_smoke(
    config: ExperimentConfig,
    *,
    gpu_id: int,
    max_new_tokens: int = 64,
    startup_timeout: float = 900.0,
    request_timeout: float = 300.0,
) -> dict[str, Any]:
    """Load once, generate/reset/generate/health, then explicitly shut down."""

    tasks = load_and_render_tasks(config)
    task_id = next(iter(tasks))
    key = RequestKey(task_id, Phase.PILOT, 1)
    logical_id = key.logical_id(config.protocol_fingerprint)
    seed = request_seed(config.master_seed, logical_id)

    context = mp.get_context("spawn")
    command_queue = context.Queue()
    event_queue = context.Queue()
    model_config = {
        **asdict(config.model),
        "path": str(config.model.path),
    }
    engine_config = asdict(config.engine)
    smoke_sampling = asdict(replace(config.sampling, max_new_tokens=max_new_tokens))
    process = context.Process(
        target=worker_process_main,
        args=(0, int(gpu_id), command_queue, event_queue, model_config, engine_config, smoke_sampling),
        name=f"async-hbn-smoke-gpu{gpu_id}",
    )
    process.start()
    all_events: list[dict[str, Any]] = []
    worker_pid = process.pid
    runs: list[dict[str, Any]] = []
    try:
        ready, events = _wait_type(
            event_queue, process, {EventType.READY.value}, startup_timeout
        )
        all_events.extend(events)
        all_events.append({"worker_pid": worker_pid, "ready": ready})
        for sequence in range(2):
            run_uuid = f"gpu-smoke-{sequence}-{uuid.uuid4().hex}"
            physical_id = physical_request_id(run_uuid, logical_id, 0)
            payload = GeneratePayload(
                run_uuid=run_uuid,
                logical_request_id=logical_id,
                physical_request_id=physical_id,
                task_id=task_id,
                prompt=str(tasks[task_id]["rendered_prompt"]),
                seed=seed,
                attempt=0,
            )
            command_queue.put(command(CommandType.SUBMIT, requests=[payload.to_dict()]))
            terminal, events = _wait_type(
                event_queue, process, TERMINAL_EVENTS, request_timeout
            )
            all_events.extend(events)
            if terminal["type"] != EventType.COMPLETED.value:
                raise RuntimeError(f"smoke generation failed: {terminal}")
            runs.append(terminal)
            if sequence == 0:
                reset_id = uuid.uuid4().hex
                command_queue.put(command(CommandType.RESET, reset_id=reset_id))
                reset, events = _wait_type(
                    event_queue, process, {EventType.RESET_COMPLETE.value}, request_timeout
                )
                all_events.extend(events)
                if not reset.get("success") or reset.get("active_count") != 0:
                    raise RuntimeError(f"cache reset failed: {reset}")
        health_id = uuid.uuid4().hex
        command_queue.put(command(CommandType.HEALTH, health_id=health_id))
        health, events = _wait_type(
            event_queue, process, {EventType.HEALTHY.value}, request_timeout
        )
        all_events.extend(events)
        return {
            "ok": True,
            "gpu_id": int(gpu_id),
            "worker_pid": worker_pid,
            "same_worker_across_reset": process.pid == worker_pid and process.is_alive(),
            "batch_invariant": config.engine.batch_invariant,
            "logical_request_id": logical_id,
            "seed": seed,
            "runs": runs,
            "health": health,
            "events": all_events,
        }
    finally:
        if process.is_alive():
            command_queue.put(command(CommandType.SHUTDOWN))
            process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)


def write_smoke_result(result: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return output

