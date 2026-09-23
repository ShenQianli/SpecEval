from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from async_hbn.config import RewardConfig, load_config, sha256_file
from async_hbn.identities import Phase, RequestKey
from async_hbn.profile_rewards import ProfileBernoulliRewards


ROOT = Path(__file__).resolve().parents[1]


def _profile_config(tmp_path: Path):
    base = load_config(ROOT / "tests/fixtures/single_run.yaml")
    task_ids = tuple(str(value) for value in range(60, 90))
    profile = {
        "schema_version": 1,
        "benchmark_id": "aime24",
        "model_alias": "qwen3_5_4b",
        "tested_k": 1024,
        "task_count": 30,
        "task_ids": list(task_ids),
        "pass_counts": [0, 1024] + [512] * 28,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return replace(
        base,
        reward=RewardConfig(
            mode="profile_bernoulli",
            profile_path=path,
            profile_sha256=sha256_file(path),
            tested_k=1024,
            seed=123,
        ),
    ), task_ids


def test_profile_rewards_are_fixed_by_logical_identity(tmp_path: Path) -> None:
    config, task_ids = _profile_config(tmp_path)
    first = ProfileBernoulliRewards(config, task_ids=task_ids)
    second = ProfileBernoulliRewards(config, task_ids=task_ids)
    keys = [
        RequestKey("62", Phase.PILOT, ordinal)
        for ordinal in range(1, 33)
    ]
    assert [first.reward(key) for key in keys] == [second.reward(key) for key in keys]
    assert first.reward(RequestKey("60", Phase.PILOT, 1)) == 0
    assert first.reward(RequestKey("61", Phase.CONTINUATION, 1)) == 1
    assert first.fingerprint == second.fingerprint


def test_profile_task_order_must_match_rendered_order(tmp_path: Path) -> None:
    config, task_ids = _profile_config(tmp_path)
    with pytest.raises(ValueError, match="task order"):
        ProfileBernoulliRewards(config, task_ids=reversed(task_ids))
