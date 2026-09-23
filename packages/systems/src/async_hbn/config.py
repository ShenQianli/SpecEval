"""Strict configuration loading for the frozen system experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


POLICY_NAMES = (
    "uniform_flat",
    "hbn_sync",
    "hbn_spec_fill256_partial_plugin",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkConfig:
    id: str
    path: Path
    sha256: str
    expected_tasks: int
    scorer: str
    provenance_repo: str = ""
    provenance_split: str = "test"
    system_prompt: str | None = None
    max_new_tokens: int = 16384
    input_sha256: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    family: str
    path: Path
    dtype: str
    tensor_parallel_size: int
    chat_template_kwargs: Mapping[str, Any]
    thinking_budget_tokens: int | None
    generation_prompt_suffix: str = ""


@dataclass(frozen=True)
class RewardConfig:
    """How a completed generation is converted into a Bernoulli reward."""

    mode: str = "profile_bernoulli"
    profile_path: Path | None = None
    profile_sha256: str | None = None
    tested_k: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class DesignConfig:
    n_tasks: int
    budget_per_task: int
    pilot_per_task: int
    stage_weight: float
    continuation_total: int
    quadrature_order: int
    speculative_depth: int

    @property
    def pilot_total(self) -> int:
        return self.n_tasks * self.pilot_per_task

    @property
    def accepted_total(self) -> int:
        return self.n_tasks * self.budget_per_task

    @property
    def uniform_continuations_per_task(self) -> int:
        return self.budget_per_task - self.pilot_per_task


@dataclass(frozen=True)
class EngineConfig:
    gpu_ids: tuple[int, ...]
    max_model_len: int
    max_num_seqs: int
    max_inflight_per_engine: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    reasoning_parser: str | None
    batch_invariant: bool
    reproducibility_mode: str


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int


@dataclass(frozen=True)
class ExecutionConfig:
    max_infra_retries: int
    timed_blocks: int
    policy_order_seed: int
    policies: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    source: Path
    project_root: Path
    schema_version: int
    experiment_name: str
    master_seed: int
    benchmark: BenchmarkConfig
    model: ModelConfig
    design: DesignConfig
    engine: EngineConfig
    sampling: SamplingConfig
    execution: ExecutionConfig
    raw: Mapping[str, Any]
    reward: RewardConfig = RewardConfig()

    @property
    def protocol_fingerprint(self) -> str:
        protocol = {
            "schema_version": self.schema_version,
            "benchmark": {
                "id": self.benchmark.id,
                "sha256": self.benchmark.sha256,
                "scorer": self.benchmark.scorer,
            },
            "model": {
                "alias": self.model.alias,
                "family": self.model.family,
                "chat_template_kwargs": dict(self.model.chat_template_kwargs),
                "thinking_budget_tokens": self.model.thinking_budget_tokens,
            },
            "design": vars(self.design),
            "sampling": vars(self.sampling),
            "reward": {
                **vars(self.reward),
                "profile_path": (
                    None
                    if self.reward.profile_path is None
                    else str(self.reward.profile_path)
                ),
            },
            "master_seed": self.master_seed,
        }
        return sha256_bytes(canonical_json(protocol).encode())


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text())
    return config_from_mapping(raw, source=source)


def config_from_mapping(raw: Mapping[str, Any], *, source: Path) -> ExperimentConfig:
    """Parse an in-memory configuration with the experiment validation rules."""
    source = Path(source).resolve()
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a mapping")
    root = source.parent.parent
    benchmark_raw = raw["benchmark"]
    model_raw = raw["model"]
    design_raw = raw["design"]
    engine_raw = raw["engine"]
    sampling_raw = raw["sampling"]
    execution_raw = raw["execution"]
    reward_raw = raw.get("reward", {"mode": "profile_bernoulli"})
    config = ExperimentConfig(
        source=source,
        project_root=root,
        schema_version=int(raw["schema_version"]),
        experiment_name=str(raw["experiment_name"]),
        master_seed=int(raw["master_seed"]),
        benchmark=BenchmarkConfig(
            id=str(benchmark_raw["id"]),
            path=_resolve(root, str(benchmark_raw["path"])),
            sha256=str(benchmark_raw["sha256"]),
            expected_tasks=int(benchmark_raw["expected_tasks"]),
            scorer=str(benchmark_raw["scorer"]),
            provenance_repo=str(benchmark_raw.get("provenance_repo", "")),
            provenance_split=str(benchmark_raw.get("provenance_split", "test")),
            system_prompt=(
                None
                if benchmark_raw.get("system_prompt") is None
                else str(benchmark_raw["system_prompt"])
            ),
            max_new_tokens=int(benchmark_raw.get("max_new_tokens", 16384)),
            input_sha256=benchmark_raw.get("input_sha256"),
        ),
        model=ModelConfig(
            alias=str(model_raw["alias"]),
            family=str(model_raw["family"]),
            path=_resolve(root, str(model_raw["path"])),
            dtype=str(model_raw["dtype"]),
            tensor_parallel_size=int(model_raw["tensor_parallel_size"]),
            chat_template_kwargs=dict(model_raw.get("chat_template_kwargs", {})),
            thinking_budget_tokens=(
                None
                if model_raw.get("thinking_budget_tokens") is None
                else int(model_raw["thinking_budget_tokens"])
            ),
            generation_prompt_suffix=str(
                model_raw.get("generation_prompt_suffix", "")
            ),
        ),
        design=DesignConfig(
            n_tasks=int(design_raw["n_tasks"]),
            budget_per_task=int(design_raw["budget_per_task"]),
            pilot_per_task=int(design_raw["pilot_per_task"]),
            stage_weight=float(design_raw["stage_weight"]),
            continuation_total=int(design_raw["continuation_total"]),
            quadrature_order=int(design_raw["quadrature_order"]),
            speculative_depth=int(design_raw["speculative_depth"]),
        ),
        engine=EngineConfig(
            gpu_ids=tuple(map(int, engine_raw["gpu_ids"])),
            max_model_len=int(engine_raw["max_model_len"]),
            max_num_seqs=int(engine_raw["max_num_seqs"]),
            max_inflight_per_engine=int(engine_raw["max_inflight_per_engine"]),
            gpu_memory_utilization=float(engine_raw["gpu_memory_utilization"]),
            enable_prefix_caching=bool(engine_raw["enable_prefix_caching"]),
            reasoning_parser=(
                None if engine_raw.get("reasoning_parser") is None else str(engine_raw["reasoning_parser"])
            ),
            batch_invariant=bool(engine_raw.get("batch_invariant", False)),
            reproducibility_mode=str(
                engine_raw.get("reproducibility_mode", "batch_invariant")
            ),
        ),
        sampling=SamplingConfig(
            temperature=float(sampling_raw["temperature"]),
            top_p=float(sampling_raw["top_p"]),
            top_k=int(sampling_raw["top_k"]),
            max_new_tokens=int(sampling_raw["max_new_tokens"]),
        ),
        execution=ExecutionConfig(
            max_infra_retries=int(execution_raw["max_infra_retries"]),
            timed_blocks=int(execution_raw["timed_blocks"]),
            policy_order_seed=int(execution_raw["policy_order_seed"]),
            policies=tuple(map(str, execution_raw["policies"])),
        ),
        reward=RewardConfig(
            mode=str(reward_raw.get("mode", "profile_bernoulli")),
            profile_path=(
                None
                if reward_raw.get("profile_path") is None
                else _resolve(root, str(reward_raw["profile_path"]))
            ),
            profile_sha256=(
                None
                if reward_raw.get("profile_sha256") is None
                else str(reward_raw["profile_sha256"])
            ),
            tested_k=(
                None
                if reward_raw.get("tested_k") is None
                else int(reward_raw["tested_k"])
            ),
            seed=int(reward_raw.get("seed", raw["master_seed"])),
        ),
        raw=raw,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    design = config.design
    if config.schema_version not in {1, 2}:
        raise ValueError(f"unsupported schema_version={config.schema_version}")
    if design.n_tasks <= 0 or design.pilot_per_task <= 0:
        raise ValueError("n_tasks and pilot_per_task must be positive")
    if design.pilot_per_task >= design.budget_per_task:
        raise ValueError("pilot_per_task must be smaller than budget_per_task")
    expected_continuations = design.n_tasks * (
        design.budget_per_task - design.pilot_per_task
    )
    if design.continuation_total != expected_continuations:
        raise ValueError(
            f"continuation_total={design.continuation_total}, expected {expected_continuations}"
        )
    if design.continuation_total < design.n_tasks:
        raise ValueError("positive continuation allocation is infeasible")
    if design.speculative_depth < 1:
        raise ValueError("speculative_depth must be positive")
    if tuple(config.execution.policies) != POLICY_NAMES:
        raise ValueError(f"policies must be exactly {POLICY_NAMES}")
    if not config.engine.gpu_ids or len(set(config.engine.gpu_ids)) != len(config.engine.gpu_ids):
        raise ValueError("gpu_ids must be nonempty and unique")
    if not 0 < config.engine.max_inflight_per_engine <= config.engine.max_num_seqs:
        raise ValueError(
            "max_inflight_per_engine must be in [1, max_num_seqs]"
        )
    if config.model.tensor_parallel_size != 1:
        raise ValueError("the first experiment requires one TP=1 replica per GPU")
    reward = config.reward
    if reward.mode != "profile_bernoulli":
        raise ValueError(f"unsupported reward mode {reward.mode!r}")
    if reward.mode == "profile_bernoulli":
        if reward.profile_path is None or not reward.profile_sha256:
            raise ValueError(
                "profile_bernoulli requires profile_path and profile_sha256"
            )
        if reward.tested_k is None or reward.tested_k <= 0:
            raise ValueError("profile_bernoulli requires a positive tested_k")
    expected_mode = (
        "batch_invariant" if config.engine.batch_invariant else "seeded_non_batch_invariant"
    )
    if config.engine.reproducibility_mode != expected_mode:
        raise ValueError(
            f"reproducibility_mode must be {expected_mode!r} for batch_invariant="
            f"{config.engine.batch_invariant}"
        )


def verify_static_inputs(config: ExperimentConfig) -> None:
    from .tasks import load_fixed_tasks

    load_fixed_tasks(config)
    if not config.model.path.is_dir():
        raise FileNotFoundError(config.model.path)
    from .release import read_result
    expected_model = read_result("systems/architectures.json")[config.model.alias]["config_sha256"]
    if sha256_file(config.model.path / "config.json") != expected_model:
        raise ValueError("Model configuration differs from the published architecture")
    if config.reward.mode == "profile_bernoulli":
        assert config.reward.profile_path is not None
        assert config.reward.profile_sha256 is not None
        if not config.reward.profile_path.is_file():
            raise FileNotFoundError(config.reward.profile_path)
        profile_sha256 = sha256_file(config.reward.profile_path)
        if profile_sha256 != config.reward.profile_sha256:
            raise ValueError(
                "profile SHA256 mismatch: "
                f"{profile_sha256} != {config.reward.profile_sha256}"
            )
