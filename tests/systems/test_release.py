import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from async_hbn.allocation import HBNAllocator
from async_hbn.config import canonical_json
from async_hbn.identities import Phase, RequestKey, request_seed
from async_hbn.profile_rewards import ProfileBernoulliRewards, load_frozen_profile
from async_hbn.protocol import Policy, ProtocolPlan
from async_hbn.release import load_experiment, read_result, results_dir

ROOT = Path(__file__).resolve().parents[2]


def config_file(
    tmp_path, budget=32, strategy="hbn-async", profile="aime24__model=qwen3_5_4b"
):
    spec = yaml.safe_load((ROOT / "configs/systems/example.yaml").read_text())
    spec.update(
        budget=budget, strategy=strategy, profile_data_dir=str(ROOT / "data" / "profiles")
    )
    spec["benchmark"], spec["model"] = profile.split("__model=")
    p = tmp_path / "experiment.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


@pytest.mark.parametrize("budget", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "profile_key", sorted(read_result("configurations/profiles.json"))
)
def test_published_identity_allocation_and_rewards(tmp_path, budget, profile_key):
    config, policy, _ = load_experiment(
        config_file(tmp_path, budget, profile=profile_key)
    )
    profile = load_frozen_profile(config)
    sampler = ProfileBernoulliRewards(config, task_ids=profile.task_ids)
    d = config.design
    plan = ProtocolPlan(
        profile.task_ids,
        d.budget_per_task,
        d.pilot_per_task,
        d.continuation_total,
        d.speculative_depth,
        config.master_seed,
    )
    rewards = {
        task: [
            sampler.reward(RequestKey(task, Phase.PILOT, i))
            for i in range(1, d.pilot_per_task + 1)
        ]
        for task in profile.task_ids
    }
    allocated = HBNAllocator(
        profile.task_ids, d.pilot_per_task, d.continuation_total
    ).allocate(rewards)
    keys = set(plan.pilots()) | set(
        plan.selected_continuations(Policy.HBN_SYNC, allocated.allocation)
    )
    rows = sorted(
        (
            k.logical_id(config.protocol_fingerprint),
            request_seed(config.master_seed, k.logical_id(config.protocol_fingerprint)),
            sampler.reward(k),
        )
        for k in keys
    )
    expected = read_result("validation/expected_identity.json")[
        f"{profile_key}__b={budget:03d}"
    ]
    digest = lambda v: hashlib.sha256(canonical_json(v).encode()).hexdigest()
    assert digest(rows) == expected["expected_accepted_sha256"]
    assert digest(allocated.allocation) == expected["expected_allocation_sha256"]
    assert sampler.fingerprint == expected["expected_reward_fingerprint"]


@pytest.mark.parametrize("strategy", ["uniform", "hbn-sync", "hbn-async"])
def test_single_strategy_fake_execution(tmp_path, strategy):
    from test_runner_fake_pool import FakePool
    from async_hbn.runner import ExperimentRunner

    cfg, policy, _ = load_experiment(config_file(tmp_path, 8, strategy))
    tasks = {
        t: {"task_id": t, "answer_json": 0, "scorer": "aime", "rendered_prompt": "test"}
        for t in load_frozen_profile(cfg).task_ids
    }
    result = ExperimentRunner(cfg, tasks, FakePool({t: 0 for t in tasks})).run_policy(
        Policy(policy), run_uuid="test", block_index=0, output_dir=tmp_path / strategy
    )
    assert result.summary["accepted_logical"] == 240
    assert result.summary["policy"] == policy
    if strategy == "hbn-sync":
        pilots = [r for r in result.requests if r["phase"] == "pilot"]
        continuations = [r for r in result.requests if r["phase"] == "continuation"]
        assert min(r["submitted_ns"] for r in continuations) >= max(
            r["reward_ready_ns"] for r in pilots
        )


@pytest.mark.parametrize("strategy", ["uniform", "hbn-sync", "hbn-async"])
def test_cli_runs_exactly_one_policy(tmp_path, monkeypatch, strategy):
    import sys
    from async_hbn import cli, config, suite

    calls = []
    monkeypatch.setattr(config, "verify_static_inputs", lambda cfg: None)
    monkeypatch.setattr(
        suite, "run_suite", lambda cfg, **kw: calls.append(kw) or tmp_path
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["async-hbn", "run", "--config", str(config_file(tmp_path, strategy=strategy))],
    )
    cli.main()
    assert len(calls) == 1
    assert calls[0]["blocks"] == 1 and calls[0]["reduced_smoke"] is False
    assert len(calls[0]["policies"]) == 1


def test_fixed_numerical_and_worker_components():
    for name, digest in read_result("validation/core_source_sha256.json").items():
        assert (
            hashlib.sha256((ROOT / "packages/systems/src/async_hbn" / name).read_bytes()).hexdigest()
            == digest
        )


def test_strategy_does_not_change_identity_and_unknown_settings_rejected(tmp_path):
    configs = [
        load_experiment(config_file(tmp_path, strategy=s))[0]
        for s in ("uniform", "hbn-sync", "hbn-async")
    ]
    assert len({c.protocol_fingerprint for c in configs}) == 1
    p = config_file(tmp_path)
    spec = yaml.safe_load(p.read_text())
    spec["temperature"] = 1
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        load_experiment(p)


def test_paper_tables():
    spec = importlib.util.spec_from_file_location(
        "tables", ROOT / "tools/reproduce_tables.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t = m.compute(results_dir())
    assert [round(r["useful_flops_pct"], 2) for r in t[2]] == [2.00, 3.03, 4.41, 5.62]
    assert [round(r["wasted_flops_pct"], 3) for r in t[2]] == [
        2.233,
        0.511,
        0.073,
        0.026,
    ]
    assert [round(r["async_actual_time_pct"], 2) for r in t[3]] == [
        11.15,
        2.85,
        5.49,
        6.56,
    ]
    assert [round(r["async_normalized_time_pct"], 2) for r in t[3]] == [
        8.45,
        0.14,
        1.33,
        1.38,
    ]
    assert [round(r["equivalent_uniform_time_pct"], 2) for r in t[4]] == [
        3.79,
        13.64,
        27.95,
        42.81,
    ]
    assert [round(r["delta_pp"], 2) for r in t[4]] == [-7.36, 10.79, 22.46, 36.25]
    assert [round(r['rounded_continuous_budget_pct'], 2) for r in t[4]] == [94.63, 97.66, 98.98, 99.56]
    assert [round(r['avg_continuous_equivalent_budget'], 2) for r in t[4]] == [9.37, 20.88, 47.04, 107.22]
    assert [round(r['avg_rounded_equivalent_budget'], 2) for r in t[4]] == [8.87, 20.39, 46.56, 106.75]
    assert [round(r["sync_actual_time_pct"], 2) for r in t[3]] == [
        33.73,
        16.47,
        13.38,
        11.10,
    ]
    assert [round(r["sync_normalized_time_pct"], 2) for r in t[3]] == [
        30.48,
        13.39,
        9.27,
        5.61,
    ]
    assert [round(r["actual_delta_pp"], 2) for r in t[3]] == [22.57, 13.61, 7.89, 4.54]
    assert [round(r["normalized_delta_pp"], 2) for r in t[3]] == [
        22.03,
        13.25,
        7.94,
        4.23,
    ]
    assert [round(r["actual_flops_pct"], 2) for r in t[2]] == [4.23, 3.54, 4.49, 5.64]
