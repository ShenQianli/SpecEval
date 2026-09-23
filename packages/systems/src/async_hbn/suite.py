"""Persistent-worker experiment suite and immutable run manifests."""

from __future__ import annotations

import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import (
    DesignConfig,
    ExperimentConfig,
    sha256_file,
)
from .gpu_guard import GPUProcessGuard
from .pool import PersistentWorkerPool
from .messages import EventType
from .protocol import Policy
from .runner import ExperimentRunner, balanced_policy_orders
from .tasks import load_and_render_tasks


def preflight_artifact_writer() -> None:
    """Fail before GPU startup if the Parquet runtime is incomplete."""

    import pyarrow.parquet  # noqa: F401

    buffer = io.BytesIO()
    pd.DataFrame({"preflight": [1]}).to_parquet(buffer, index=False)
    if not buffer.getvalue():
        raise RuntimeError("Parquet artifact preflight produced no bytes")


def effective_config_manifest(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize dataclass values actually used, including runtime overrides."""

    return {
        **({"equivalent_uniform": config.raw["_equivalent_uniform"]}
           if "_equivalent_uniform" in config.raw else {}),
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "master_seed": config.master_seed,
        "benchmark": {
            **asdict(config.benchmark),
            "path": str(config.benchmark.path),
        },
        "model": {
            **asdict(config.model),
            "path": str(config.model.path),
        },
        "design": asdict(config.design),
        "engine": asdict(config.engine),
        "sampling": asdict(config.sampling),
        "execution": asdict(config.execution),
        "reward": {
            **asdict(config.reward),
            "profile_path": (
                None
                if config.reward.profile_path is None
                else str(config.reward.profile_path)
            ),
        },
    }


def prewarm_model_files(model_path: Path) -> int:
    files = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("*.bin"))
    total = 0
    for path in files:
        with path.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                total += len(chunk)
    return total


def runtime_manifest(config: ExperimentConfig) -> dict[str, Any]:
    packages = {}
    for name in ("async-hbn", "numpy", "scipy", "torch", "transformers", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu_state = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except Exception as exc:
        gpu_state = [f"ERROR: {type(exc).__name__}: {exc}"]
    model_files = []
    for name in (
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        path = config.model.path / name
        if path.is_file():
            model_files.append({
                "name": name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    for path in sorted(config.model.path.glob("*.safetensors")):
        model_files.append({"name": path.name, "size": path.stat().st_size})
    return {
        "created_unix_ns": time.time_ns(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpus": gpu_state,
        "dataset": {
            "path": str(config.benchmark.path),
            "sha256": sha256_file(config.benchmark.path),
        },
        "model": {
            "path": str(config.model.path.resolve()),
            "files": model_files,
        },
        "source_config": config.raw,
        "effective_config": effective_config_manifest(config),
        "protocol_fingerprint": config.protocol_fingerprint,
        "reward": {
            **asdict(config.reward),
            "profile_path": (
                None
                if config.reward.profile_path is None
                else str(config.reward.profile_path)
            ),
        },
    }


def smoke_config(config: ExperimentConfig) -> ExperimentConfig:
    n_tasks = len(config.engine.gpu_ids)
    design = DesignConfig(
        n_tasks=n_tasks,
        budget_per_task=4,
        pilot_per_task=2,
        stage_weight=config.design.stage_weight,
        continuation_total=n_tasks * 2,
        quadrature_order=config.design.quadrature_order,
        speculative_depth=4,
    )
    return replace(
        config,
        experiment_name=config.experiment_name + "__reduced_smoke",
        design=design,
        sampling=replace(config.sampling, max_new_tokens=64),
        execution=replace(config.execution, timed_blocks=1),
    )


def common_worker_warmup(
    config: ExperimentConfig,
    tasks: dict[str, dict[str, Any]],
    pool: PersistentWorkerPool,
) -> list[dict[str, Any]]:
    task_id = next(iter(tasks))
    run_uuid = "common-warmup-" + uuid.uuid4().hex
    for engine_id in range(len(config.engine.gpu_ids)):
        logical_id = f"warmup|engine={engine_id}|task={task_id}"
        pool.submit(engine_id, [{
            "run_uuid": run_uuid,
            "logical_request_id": logical_id,
            "physical_request_id": f"{run_uuid}|{logical_id}",
            "task_id": task_id,
            "prompt": str(tasks[task_id]["rendered_prompt"]),
            "seed": config.master_seed + engine_id,
            "attempt": 0,
            "max_new_tokens": 64,
        }])
    terminal = 0
    rows: list[dict[str, Any]] = []
    while terminal < len(config.engine.gpu_ids):
        row = pool.next_event(timeout=600)
        rows.append(row)
        if row["type"] == EventType.WORKER_ERROR.value:
            raise RuntimeError(f"common warmup worker error: {row}")
        if row["type"] in {
            EventType.COMPLETED.value,
            EventType.ABORTED.value,
            EventType.INFRA_ERROR.value,
        }:
            terminal += 1
            if row["type"] != EventType.COMPLETED.value:
                raise RuntimeError(f"common warmup request failed: {row}")
    return rows


def run_suite(
    config: ExperimentConfig,
    *,
    output_root: str | Path,
    blocks: int,
    policies: Iterable[Policy] = tuple(Policy),
    reduced_smoke: bool = False,
) -> Path:
    preflight_artifact_writer()
    if reduced_smoke:
        config = smoke_config(config)
    all_tasks = load_and_render_tasks(config)
    tasks = dict(list(all_tasks.items())[: config.design.n_tasks])
    selected_policies = tuple(policies)
    if not selected_policies:
        raise ValueError("at least one policy is required")
    full_orders = balanced_policy_orders(
        selected_policies,
        blocks=max(blocks, len(selected_policies)),
        seed=config.execution.policy_order_seed,
    )
    experiment_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "__"
        + config.experiment_name
        + "__"
        + uuid.uuid4().hex[:8]
    )
    root = Path(output_root) / experiment_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = runtime_manifest(config)
    manifest["reduced_smoke"] = reduced_smoke
    manifest["requested_blocks"] = int(blocks)
    manifest["selected_policies"] = [policy.value for policy in selected_policies]
    manifest["planned_policy_orders"] = [
        [policy.value for policy in full_orders[block_index]]
        for block_index in range(blocks)
    ]
    (root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    prewarm_start = time.monotonic_ns()
    prewarmed_bytes = prewarm_model_files(config.model.path)
    prewarm_duration_ns = time.monotonic_ns() - prewarm_start
    gpu_guard = GPUProcessGuard(config.engine.gpu_ids)
    gpu_guard.require_idle()
    pool = PersistentWorkerPool(config)
    summaries: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    try:
        startup_events = pool.start()
        warmup_events = common_worker_warmup(config, tasks, pool)
        gpu_guard.start()
        startup = {
            "prewarmed_bytes": prewarmed_bytes,
            "prewarm_duration_ns": prewarm_duration_ns,
            "worker_pids": {
                str(engine_id): process.pid
                for engine_id, process in pool.processes.items()
            },
            "events": startup_events,
            "common_warmup_events": warmup_events,
        }
        (root / "startup.json").write_text(
            json.dumps(startup, indent=2, ensure_ascii=False) + "\n"
        )
        runner = ExperimentRunner(
            config,
            tasks,
            pool,
            validity_check=gpu_guard.raise_if_invalid,
        )
        for block_index in range(blocks):
            order = full_orders[block_index]
            for position, policy in enumerate(order):
                reset = pool.reset()
                reset_row = {
                    "block_index": block_index,
                    "position": position,
                    "before_policy": policy.value,
                    **reset,
                }
                reset_rows.append(reset_row)
                run_uuid = uuid.uuid4().hex
                run_dir = root / f"block={block_index:02d}" / f"policy={policy.value}"
                output = runner.run_policy(
                    policy,
                    run_uuid=run_uuid,
                    block_index=block_index,
                    output_dir=run_dir,
                )
                summary = dict(output.summary)
                summary["position"] = position
                summaries.append(summary)
                pd.DataFrame(summaries).to_csv(root / "summary_runs.csv", index=False)
                (root / "resets.json").write_text(
                    json.dumps(reset_rows, indent=2, ensure_ascii=False) + "\n"
                )
    finally:
        gpu_guard.stop()
        (root / "gpu_process_guard.json").write_text(
            json.dumps(gpu_guard.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )
        pool.shutdown()
    return root
