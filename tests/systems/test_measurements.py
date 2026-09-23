import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from async_hbn.allocation import HBNAllocator
from async_hbn.identities import Phase, RequestKey, request_seed
from async_hbn.measurements import export_measurements, measurement
from async_hbn.profile_rewards import ProfileBernoulliRewards, load_frozen_profile
from async_hbn.protocol import Policy, ProtocolPlan
from async_hbn.release import load_experiment, read_result
from async_hbn.runner import ExperimentRunner
from async_hbn.suite import effective_config_manifest
from test_release import config_file
from test_runner_fake_pool import FakePool


def artifacts(tmp_path, strategy="hbn-async", config_path=None):
    config, policy, _ = load_experiment(config_path or config_file(tmp_path, budget=8, strategy=strategy))
    profile = load_frozen_profile(config)
    sampler = ProfileBernoulliRewards(config, task_ids=profile.task_ids)
    d = config.design
    plan = ProtocolPlan(profile.task_ids, d.budget_per_task, d.pilot_per_task,
                        d.continuation_total, d.speculative_depth, config.master_seed)
    pilot = {t: [sampler.reward(RequestKey(t, Phase.PILOT, i))
                 for i in range(1, d.pilot_per_task + 1)] for t in profile.task_ids}
    allocation = HBNAllocator(profile.task_ids, d.pilot_per_task, d.continuation_total).allocate(pilot).allocation
    keys = set(plan.pilots()) | set(plan.selected_continuations(Policy(policy), allocation))
    rows = []
    for k in sorted(keys):
        logical = k.logical_id(config.protocol_fingerprint)
        rows.append(dict(logical_request_id=logical, task_id=k.task_id,
                         seed=request_seed(config.master_seed, logical), reward=sampler.reward(k),
                         state="accepted", prompt_tokens=10, completion_tokens=2,
                         prompt_tokens_before_abort=0, generated_tokens_before_abort=0))
    if strategy == "hbn-async":
        for state in ("completed_discarded", "aborted_running", "cancelled_queued"):
            rows.append(dict(rows[0], logical_request_id=state, state=state,
                             prompt_tokens_before_abort=10, generated_tokens_before_abort=1))
    root = tmp_path / "run"
    path = root / "block=00" / f"policy={policy}"
    path.mkdir(parents=True)
    write = lambda p, value: p.write_text(json.dumps(value))
    arch = read_result("systems/architectures.json")[config.model.alias]
    write(root / "experiment_manifest.json", dict(
        effective_config=effective_config_manifest(config), reduced_smoke=False,
        dataset=dict(sha256=config.benchmark.input_sha256),
        requested_blocks=1, selected_policies=[policy], protocol_fingerprint=config.protocol_fingerprint,
        model=dict(files=[dict(name="config.json", sha256=arch["config_sha256"])])))
    write(root / "gpu_process_guard.json", dict(contamination=None, monitor_error=None))
    write(path / "summary.json", dict(policy=policy, block_index=0, run_uuid="fixture-run",
        protocol_fingerprint=config.protocol_fingerprint, reward_profile_sha256=config.reward.profile_sha256,
        reward_fingerprint=sampler.fingerprint, retry_attempts=0, accepted_logical=len(keys),
        actual_rollouts=len(keys) + (2 if strategy == "hbn-async" else 0), result_seconds=10))
    write(path / "allocation.json", dict(allocation=allocation))
    pd.DataFrame(rows).to_parquet(path / "requests.parquet", index=False)
    return root, path, arch


@pytest.mark.parametrize("strategy", ["uniform", "hbn-sync", "hbn-async"])
def test_export_raw_artifacts(tmp_path, strategy):
    root, path, arch = artifacts(tmp_path, strategy)
    row = measurement(root)
    assert row["strategy"] == strategy
    assert row["accepted_rollouts"] == 240
    assert row["extra_rollouts"] == (2 if strategy == "hbn-async" else 0)
    request_cost = lambda length, c: (2 * arch["core"] * length + 2 * arch["lm"] * c
        + 2 * arch["full_layers"] * arch["attention_width"] * length * (length + 1)) / 1e18
    assert row["useful_eflop"] == pytest.approx(240 * request_cost(12, 2))
    assert row["wasted_eflop"] == pytest.approx(
        request_cost(12, 2) + request_cost(11, 1) if strategy == "hbn-async" else 0)
    output = export_measurements([root], tmp_path / "measurements.csv")
    assert len(pd.read_csv(output)) == 1
    with pytest.raises(FileExistsError):
        export_measurements([root], output)
    with pytest.raises(ValueError, match="Duplicate"):
        export_measurements([root, root], tmp_path / "duplicate.csv")


@pytest.mark.parametrize("strategy", ["uniform", "hbn-sync", "hbn-async"])
def test_runner_outputs_export_without_reports(tmp_path, strategy):
    root, path, _ = artifacts(tmp_path, strategy)
    config, policy, _ = load_experiment(config_file(tmp_path, budget=8, strategy=strategy))
    profile = load_frozen_profile(config)
    tasks = {t: dict(task_id=t, answer_json=0, scorer="aime", rendered_prompt=f"problem-{t}")
             for t in profile.task_ids}
    output = ExperimentRunner(config, tasks, FakePool({t: 0 for t in tasks})).run_policy(
        Policy(policy), run_uuid="measurement-test", block_index=0, output_dir=path)
    row = measurement(root)
    assert row["time_seconds"] == output.summary["result_seconds"]
    assert row["accepted_rollouts"] == output.summary["accepted_logical"]


@pytest.mark.parametrize("field,value", [("retry_attempts", 1), ("accepted_logical", 239),
    ("actual_rollouts", 999), ("reward_profile_sha256", "bad"), ("reward_fingerprint", "bad"),
    ("protocol_fingerprint", "bad"), ("result_seconds", float("nan"))])
def test_invalid_summary_rejected(tmp_path, field, value):
    root, path, _ = artifacts(tmp_path)
    summary = json.loads((path / "summary.json").read_text())
    summary[field] = value
    (path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        measurement(root)


@pytest.mark.parametrize("change", ["state", "identity", "tokens", "allocation", "gpu"])
def test_invalid_artifacts_rejected(tmp_path, change):
    root, path, _ = artifacts(tmp_path)
    frame = pd.read_parquet(path / "requests.parquet")
    if change == "state":
        frame.loc[0, "state"] = "running"
    elif change == "identity":
        frame.loc[0, "seed"] += 1
    elif change == "tokens":
        frame.loc[0, "completion_tokens"] = -1
    elif change == "allocation":
        (path / "allocation.json").write_text('{"allocation": {}}')
    else:
        (root / "gpu_process_guard.json").write_text('{"contamination": {}, "monitor_error": null}')
    frame.to_parquet(path / "requests.parquet", index=False)
    with pytest.raises(ValueError):
        measurement(root)


def test_new_measurements_table_input(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("tables", root / "tools/reproduce_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.compute()
    frame = pd.read_csv(root / "results/systems/measurements.csv", float_precision="round_trip")
    frame["time_seconds"] *= 2
    path = tmp_path / "measurements.csv"
    frame.to_csv(path, index=False)
    actual = module.compute(measurements=path)
    for table in (2,3):
        for old, new in zip(original[table], actual[table]):
            assert new == pytest.approx(old)
    for old, new in zip(original[4], actual[4]):
        assert new['equivalent_uniform_time_pct'] == pytest.approx((old['equivalent_uniform_time_pct']+100)/2-100)
        assert new['async_actual_time_pct'] == pytest.approx(old['async_actual_time_pct'])
        assert new['delta_pp'] == pytest.approx(new['equivalent_uniform_time_pct']-new['async_actual_time_pct'])
        assert new['rounded_continuous_budget_pct'] == old['rounded_continuous_budget_pct']
    frame.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError):
        module.compute(measurements=path)
    replay = pd.read_csv(root / "results/replay/hbn_pair_results.csv", float_precision="round_trip")
    replay["variance_ratio"] /= 2
    replay_path = tmp_path / "replay.csv"
    replay.to_csv(replay_path, index=False)
    with pytest.raises(ValueError,match='fixed published HBN replay'):
        module.compute(replay=replay_path)


def test_measurements_cli(tmp_path, monkeypatch):
    from async_hbn.cli import main

    root, _, _ = artifacts(tmp_path)
    output = tmp_path / "cli.csv"
    monkeypatch.setattr("sys.argv", ["async-hbn", "measurements", "--runs", str(root),
                                    "--output", str(output)])
    main()
    assert pd.read_csv(output).iloc[0]["strategy"] == "hbn-async"
