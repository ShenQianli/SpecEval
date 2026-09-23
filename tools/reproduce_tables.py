"""CPU-only Table 2--4 reproduction. Uses Python standard library only."""

import argparse
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/systems/src"))
from async_hbn.equivalent import measured_table4, load_targets, REPLAY_FILE

DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(results, measurements=None, replay=None):
    manifest = json.loads((results / "validation/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((results / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference fingerprint mismatch: {name}")
    source = Path(measurements) if measurements is not None else results / "systems/measurements.csv"
    rows = list(csv.DictReader(source.open()))
    replay_source = Path(replay) if replay is not None else results / "replay/hbn_pair_results.csv"
    replay = list(csv.DictReader(replay_source.open()))
    values = {}
    for r in rows:
        k = r["profile"], int(r["budget"]), r["strategy"]
        if k in values:
            raise ValueError(f"Duplicate observation: {k}")
        values[k] = {
            n: float(r[n])
            for n in (
                "time_seconds",
                "useful_eflop",
                "wasted_eflop",
                "accepted_rollouts",
                "extra_rollouts",
            )
        }
    variance = {
        (r["pair_key"], int(r["b"])): float(r["variance_ratio"]) for r in replay
    }
    if len(values) != 1284 or len(variance) != 428 or len(replay) != 428:
        raise ValueError("Expected 1284 measured runs and 428 variance ratios")
    profiles = {p for p, b in variance}
    expected = {
        (p, b, s)
        for p in profiles
        for b in (8, 16, 32, 64)
        for s in ("uniform", "hbn-sync", "hbn-async")
    }
    if (
        len(profiles) != 107
        or set(values) != expected
        or set(variance) != {(p, b) for p in profiles for b in (8, 16, 32, 64)}
    ):
        raise ValueError("Incomplete profile/budget/strategy grid")
    for p, b, s in values:
        v = values[p, b, s]
        if not all(math.isfinite(x) for x in v.values()) or any(
            x < 0 for x in v.values()
        ):
            raise ValueError("Invalid measurement")
        if min(v["time_seconds"], v["useful_eflop"], v["accepted_rollouts"]) <= 0:
            raise ValueError("Time, useful FLOPs, accepted counts must be positive")
        if v["accepted_rollouts"] != values[p, b, "uniform"]["accepted_rollouts"]:
            raise ValueError("Accepted counts differ between strategies")
        if s != "hbn-async" and (v["wasted_eflop"] or v["extra_rollouts"]):
            raise ValueError("Non-speculative strategy has waste")
    if not all(math.isfinite(v) and v > 0 for v in variance.values()):
        raise ValueError("Invalid variance ratio")
    return values, variance, sorted(profiles)


def compute(results=DEFAULT_RESULTS, measurements=None, replay=None, *, equivalent_measurements=None):
    results = Path(results)
    if equivalent_measurements is None:
        provenance = json.loads((results / 'validation/uniform_equivalent_measurements.json').read_text())
        equivalent_measurements = results / provenance['file']
        if hashlib.sha256(equivalent_measurements.read_bytes()).hexdigest() != provenance['sha256']:
            raise ValueError('Equivalent measurement fingerprint mismatch')
    if replay is not None:
        if Path(replay).read_bytes() != (Path(results) / REPLAY_FILE).read_bytes():
            raise ValueError("Measured equivalent budgets use the fixed published HBN replay")
    values, variance, profiles = load(Path(results), measurements, replay)
    tables = {2: [], 3: [], 4: []}
    for b in (8, 16, 32, 64):
        total = lambda s, field: math.fsum(values[p, b, s][field] for p in profiles)
        tu, fu = total("uniform", "time_seconds"), total("uniform", "useful_eflop")
        fs, fw = total("hbn-async", "useful_eflop"), total("hbn-async", "wasted_eflop")
        tables[2].append(
            dict(
                budget=b,
                extra_rollouts_pct=100
                * total("hbn-async", "extra_rollouts")
                / total("uniform", "accepted_rollouts"),
                useful_flops_pct=100 * (fs / fu - 1),
                wasted_flops_pct=100 * fw / fu,
                actual_flops_pct=100 * ((fs + fw) / fu - 1),
            )
        )
        row = dict(budget=b)
        for s, label in [("hbn-sync", "sync"), ("hbn-async", "async")]:
            row[label + "_actual_time_pct"] = 100 * (total(s, "time_seconds") / tu - 1)
            normalized = math.fsum(
                values[p, b, s]["time_seconds"]
                * values[p, b, "uniform"]["useful_eflop"]
                / values[p, b, s]["useful_eflop"]
                for p in profiles
            )
            row[label + "_normalized_time_pct"] = 100 * (normalized / tu - 1)
        row["actual_delta_pp"] = (
            row["sync_actual_time_pct"] - row["async_actual_time_pct"]
        )
        row["normalized_delta_pp"] = (
            row["sync_normalized_time_pct"] - row["async_normalized_time_pct"]
        )
        tables[3].append(
            {
                k: row[k]
                for k in (
                    "budget",
                    "sync_actual_time_pct",
                    "async_actual_time_pct",
                    "actual_delta_pp",
                    "sync_normalized_time_pct",
                    "async_normalized_time_pct",
                    "normalized_delta_pp",
                )
            }
        )
    targets = load_targets(results)
    for row in measured_table4(results, equivalent_measurements, values, profiles):
        b = row['budget']
        group = [targets[p, b] for p in profiles]
        actual = sum(t['n_tasks'] * t['budget'] for t in group)
        continuous = sum(t['n_tasks'] * Fraction(b) / Fraction(t['hbn_variance_ratio']) for t in group)
        tasks = sum(t['n_tasks'] for t in group)
        tables[4].append(dict(budget=b, avg_continuous_equivalent_budget=float(continuous / tasks),
                             avg_rounded_equivalent_budget=actual / tasks,
                             rounded_continuous_budget_pct=100 * float(actual / continuous),
                             **{k:v for k,v in row.items() if k != 'budget'}))
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--measurements", type=Path, help="New measured grid; published reference hashes remain checked")
    p.add_argument("--replay", type=Path, help="Fixed HBN replay defining the measured target budgets")
    p.add_argument("--equivalent-measurements", type=Path,
                   help="Use all 428 fresh rounded-equivalent Uniform measurements for Table 4")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "tables")
    args = p.parse_args()
    tables = compute(args.results_dir, args.measurements, args.replay,
                     equivalent_measurements=args.equivalent_measurements)
    for number, rows in tables.items():
        print(f"Table {number}")
        print(" | ".join(rows[0]))
        for row in rows:
            print(
                " | ".join(
                    str(v) if k == "budget" else f"{v:.6f}" for k, v in row.items()
                )
            )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for number, rows in tables.items():
            with (args.output_dir / f"table{number}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        (args.output_dir / "tables.json").write_text(
            json.dumps(tables, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
