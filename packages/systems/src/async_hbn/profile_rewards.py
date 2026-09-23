"""Deterministic Bernoulli rewards sampled from a frozen paper profile.

The value for a logical request is fixed before execution, but callers reveal it
only after the corresponding generation completes.  This gives paired policies
common random numbers without letting scheduling order affect RNG state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig, canonical_json, sha256_bytes, sha256_file
from .identities import RequestKey


REWARD_MODE = "profile_bernoulli"
REWARD_SAMPLER_VERSION = "profile_bernoulli_v1"
_RANDOM_BITS = 128
_RANDOM_RANGE = 1 << _RANDOM_BITS


@dataclass(frozen=True)
class FrozenProfile:
    path: Path
    sha256: str
    benchmark_id: str
    model_alias: str
    tested_k: int
    task_ids: tuple[str, ...]
    pass_counts: tuple[int, ...]

    @property
    def pass_counts_by_task(self) -> dict[str, int]:
        return dict(zip(self.task_ids, self.pass_counts, strict=True))


def load_frozen_profile(config: ExperimentConfig) -> FrozenProfile:
    reward = config.reward
    if reward.mode != REWARD_MODE:
        raise ValueError(f"expected reward mode {REWARD_MODE!r}, got {reward.mode!r}")
    if reward.profile_path is None or reward.profile_sha256 is None:
        raise ValueError("profile reward configuration is incomplete")
    path = reward.profile_path
    actual_sha256 = sha256_file(path)
    if actual_sha256 != reward.profile_sha256:
        raise ValueError(
            f"profile SHA256 mismatch: {actual_sha256} != {reward.profile_sha256}"
        )
    raw: dict[str, Any] = json.loads(path.read_text())
    task_ids = tuple(map(str, raw["task_ids"]))
    pass_counts = tuple(map(int, raw["pass_counts"]))
    tested_k = int(raw["tested_k"])
    if len(task_ids) != len(pass_counts) or len(set(task_ids)) != len(task_ids):
        raise ValueError("profile task IDs/pass counts are inconsistent")
    if any(value < 0 or value > tested_k for value in pass_counts):
        raise ValueError("profile pass count lies outside [0, tested_k]")
    if str(raw["benchmark_id"]) != config.benchmark.id:
        raise ValueError("profile benchmark does not match experiment config")
    if str(raw["model_alias"]) != config.model.alias:
        raise ValueError("profile model does not match experiment config")
    if tested_k != config.reward.tested_k:
        raise ValueError(
            f"profile tested_k={tested_k}, configured {config.reward.tested_k}"
        )
    if len(task_ids) != config.design.n_tasks:
        raise ValueError(
            f"profile tasks={len(task_ids)}, design tasks={config.design.n_tasks}"
        )
    return FrozenProfile(
        path=path,
        sha256=actual_sha256,
        benchmark_id=str(raw["benchmark_id"]),
        model_alias=str(raw["model_alias"]),
        tested_k=tested_k,
        task_ids=task_ids,
        pass_counts=pass_counts,
    )


class ProfileBernoulliRewards:
    """Exact hash-to-Bernoulli sampler for one frozen task profile."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        task_ids: Iterable[str],
    ) -> None:
        self.profile = load_frozen_profile(config)
        observed_task_ids = tuple(map(str, task_ids))
        if observed_task_ids != self.profile.task_ids:
            raise ValueError(
                "rendered task order does not match frozen profile task order"
            )
        self.seed = int(config.reward.seed)
        self._counts = self.profile.pass_counts_by_task
        self.fingerprint = sha256_bytes(canonical_json({
            "version": REWARD_SAMPLER_VERSION,
            "profile_sha256": self.profile.sha256,
            "tested_k": self.profile.tested_k,
            "seed": self.seed,
        }).encode())

    def reward(self, key: RequestKey) -> int:
        count = self._counts[key.task_id]
        if count == 0:
            return 0
        if count == self.profile.tested_k:
            return 1
        payload = (
            f"{REWARD_SAMPLER_VERSION}|seed={self.seed}|"
            f"profile={self.profile.sha256}|task={key.task_id}|"
            f"phase={key.phase.value}|ordinal={key.ordinal}|"
            f"replicate={key.replicate}"
        ).encode()
        draw = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
        threshold = count * _RANDOM_RANGE // self.profile.tested_k
        return int(draw < threshold)

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": REWARD_MODE,
            "sampler_version": REWARD_SAMPLER_VERSION,
            "profile_path": str(self.profile.path),
            "profile_sha256": self.profile.sha256,
            "tested_k": self.profile.tested_k,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
        }
