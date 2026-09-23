import csv
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest
import yaml

from async_hbn.equivalent import (build_targets, load_targets, rounded_budget,
                                  derived_design, write_targets)
from async_hbn.measurements import measurement, export_measurements
from async_hbn.release import load_experiment, read_result, results_dir
from test_release import config_file
from test_measurements import artifacts


def equivalent_config(tmp_path, reference=16, profile="aime24__model=qwen3_5_4b"):
    target = load_targets(results_dir())[profile, reference]
    path = config_file(tmp_path, budget=reference, strategy="uniform", profile=profile)
    spec = yaml.safe_load(path.read_text())
    spec.update(reference_budget=reference, budget=target["budget"])
    path.write_text(yaml.safe_dump(spec))
    return path


def table_module():
    spec = importlib.util.spec_from_file_location("equivalent_tables", Path(__file__).resolve().parents[2] / "tools/reproduce_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_grid_and_exact_floor(tmp_path):
    rows = build_targets(results_dir())
    assert len(rows) == 428
    assert sum(r["budget"] == r["reference_budget"] for r in rows) == 64
    assert min(r["budget"] for r in rows) == 8 and max(r["budget"] for r in rows) == 191
    assert all(r["uniform_over_hbn_variance_ratio"] >= 1 for r in rows)
    assert rounded_budget(8, "0.5") == 16
    assert rounded_budget(8, "0.50000000000000001") == 15
    assert rounded_budget(8, "0.49999999999999999") == 16
    out = write_targets(results_dir(), tmp_path / "targets.csv")
    assert out.read_bytes() == (results_dir() / "systems/uniform_equivalent_budgets.csv").read_bytes()


@pytest.mark.parametrize("reference", [8, 16, 32, 64])
def test_config_identity_and_design(tmp_path, reference):
    old, _, _ = load_experiment(config_file(tmp_path, budget=reference, strategy="uniform"))
    config, policy, _ = load_experiment(equivalent_config(tmp_path, reference))
    assert policy == "uniform_flat"
    assert config.protocol_fingerprint != old.protocol_fingerprint
    assert config.design.pilot_per_task == old.design.pilot_per_task
    assert config.design.stage_weight == old.design.stage_weight
    assert config.design.continuation_total == config.design.n_tasks * (
        config.design.budget_per_task - config.design.pilot_per_task)
    assert config.sampling == old.sampling and config.engine == old.engine
    assert config.reward == old.reward


@pytest.mark.parametrize("change", [{"budget": 18}, {"strategy": "hbn-sync"},
                                   {"budget": 17.0}, {"reference_budget": 17}])
def test_invalid_equivalent_configuration(tmp_path, change):
    path = equivalent_config(tmp_path)
    spec = yaml.safe_load(path.read_text());spec.update(change)
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        load_experiment(path)


def test_fresh_artifact_export(tmp_path):
    path = equivalent_config(tmp_path)
    root, _, _ = artifacts(tmp_path, "uniform", config_path=path)
    row = measurement(root, equivalent=True)
    assert row["budget"] == 17 and row["reference_budget"] == 16
    assert row["accepted_rollouts"] == 510
    output = export_measurements([root], tmp_path / "measurement.csv", equivalent=True)
    assert len(pd.read_csv(output)) == 1
    with pytest.raises(ValueError, match="matching"):
        measurement(root)


def test_original_run_cannot_fill_unchanged_budget(tmp_path):
    root, _, _ = artifacts(tmp_path, "uniform")
    with pytest.raises(ValueError, match="matching"):
        measurement(root, equivalent=True)


@pytest.mark.parametrize("corruption", ["reward", "seed", "engine", "metadata", "uuid"])
def test_equivalent_artifact_validation(tmp_path, corruption):
    path = equivalent_config(tmp_path)
    root, block, _ = artifacts(tmp_path, "uniform", config_path=path)
    if corruption in {"reward", "seed"}:
        f = block / "requests.parquet";frame = pd.read_parquet(f)
        frame.loc[0, corruption] += 1;frame.to_parquet(f, index=False)
    elif corruption == "uuid":
        f = block / "summary.json";data = json.loads(f.read_text());data.pop("run_uuid")
        f.write_text(json.dumps(data))
    else:
        f = root / "experiment_manifest.json";data = json.loads(f.read_text())
        if corruption == "engine":data["effective_config"]["engine"]["max_inflight_per_engine"] = 16
        else:data["effective_config"]["equivalent_uniform"]["target_fingerprint"] = "bad"
        f.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        measurement(root, equivalent=True)


def synthetic_grid(tmp_path):
    targets = load_targets(results_dir())
    catalog = read_result("configurations/profiles.json")
    tables = table_module();values, _, _ = tables.load(results_dir())
    rows = []
    for (profile, b), target in targets.items():
        _, fingerprint, _ = derived_design(catalog[profile], target)
        # Test-only timing: 10% above the original Uniform, not experimental data.
        rows.append(dict(profile=profile, reference_budget=b, budget=target["budget"],
            model=target["model"], strategy="uniform", time_seconds=1.1 * values[profile,b,"uniform"]["time_seconds"],
            useful_eflop=1, wasted_eflop=0, extra_rollouts=0,
            accepted_rollouts=target["n_tasks"]*target["budget"],
            run_uuid=f"test-{profile}-{b}", target_fingerprint=target["target_fingerprint"],
            protocol_fingerprint=fingerprint))
    path = tmp_path / "measurements.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return path


def test_measured_table4_and_unchanged_tables23(tmp_path):
    module = table_module();path = synthetic_grid(tmp_path)
    original = module.compute();measured = module.compute(equivalent_measurements=path)
    assert measured[2] == original[2] and measured[3] == original[3]
    for row, old in zip(measured[4], original[4]):
        assert row["equivalent_uniform_time_pct"] == pytest.approx(10.)
        assert row["delta_pp"] == pytest.approx(10. - old["async_actual_time_pct"])
    with pytest.raises(FileNotFoundError):
        module.compute(equivalent_measurements=tmp_path / "missing.csv")


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "old_identity", "reuse_uuid", "budget", "target", "count", "waste", "zero_time", "nan"])
def test_measured_grid_rejections(tmp_path, corruption):
    path = synthetic_grid(tmp_path);frame = pd.read_csv(path, float_precision="round_trip")
    if corruption == "missing":frame = frame.iloc[:-1]
    elif corruption == "duplicate":frame = pd.concat([frame, frame.iloc[:1]])
    elif corruption == "old_identity":frame.loc[0, "protocol_fingerprint"] = "old"
    elif corruption == "reuse_uuid":frame.loc[0, "run_uuid"] = frame.loc[1, "run_uuid"]
    elif corruption == "budget":frame.loc[0, "budget"] += 1
    elif corruption == "target":frame.loc[0, "target_fingerprint"] = "wrong"
    elif corruption == "count":frame.loc[0, "accepted_rollouts"] += 1
    elif corruption == "waste":frame.loc[0, "wasted_eflop"] = 1
    elif corruption == "zero_time":frame.loc[0, "time_seconds"] = 0
    else:frame.loc[0, "time_seconds"] = float("nan")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        table_module().compute(equivalent_measurements=path)


def test_table4_aggregation_from_published_inputs():
    root = results_dir()
    metadata = json.loads((root / "validation/uniform_equivalent_measurements.json").read_text())
    with (root / metadata["file"]).open() as f:
        measurements = list(csv.DictReader(f))
    assert len(measurements) == metadata["observations"] == 428
    assert len({r["profile"] for r in measurements}) == metadata["profiles"] == 107
    module = table_module()
    values, _, profiles = module.load(root)
    targets = load_targets(root)
    for row in module.compute()[4]:
        b = row["budget"]
        group = [targets[p, b] for p in profiles]
        n = sum(t["n_tasks"] for t in group)
        continuous = sum(Fraction(b) / Fraction(t["hbn_variance_ratio"]) * t["n_tasks"] for t in group)
        rounded = sum(t["n_tasks"] * t["budget"] for t in group)
        assert row["avg_continuous_equivalent_budget"] == pytest.approx(float(continuous / n))
        assert row["avg_rounded_equivalent_budget"] == pytest.approx(rounded / n)
        assert row["rounded_continuous_budget_pct"] == pytest.approx(100 * float(rounded / continuous))
        original = math.fsum(values[p, b, "uniform"]["time_seconds"] for p in profiles)
        measured = math.fsum(float(r["time_seconds"]) for r in measurements if int(r["reference_budget"]) == b)
        async_time = math.fsum(values[p, b, "hbn-async"]["time_seconds"] for p in profiles)
        assert row["equivalent_uniform_time_pct"] == pytest.approx(100 * (measured / original - 1))
        assert row["async_actual_time_pct"] == pytest.approx(100 * (async_time / original - 1))
        assert row["delta_pp"] == pytest.approx(100 * (measured - async_time) / original)


def test_table4_rejects_changed_replay(tmp_path):
    published = results_dir() / "replay/hbn_pair_results.csv"
    replacement = tmp_path / "replay.csv"
    replacement.write_bytes(published.read_bytes())
    module = table_module()
    assert module.compute(replay=replacement) == module.compute()
    frame = pd.read_csv(replacement, float_precision="round_trip")
    frame.loc[0, "variance_ratio"] *= 1.001
    frame.to_csv(replacement, index=False)
    with pytest.raises(ValueError, match="fixed published HBN replay"):
        module.compute(replay=replacement)
