"""Aggregate published per-profile results into the paper's Table 1 (no replay)."""

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "oracle_pct": "oracle_pair_results.csv",
    "tuned_en_pct": "en_pair_results.csv",
    "tuned_ibn_pct": "ibn_tuned_pair_results.csv",
    "hbn_pct": "hbn_pair_results.csv",
}
PAPER = {
    8: ("40.7", "94.3", "89.6", "87.2"),
    16: ("39.9", "92.6", "81.9", "80.1"),
    32: ("39.5", "91.9", "74.2", "73.0"),
    64: ("39.3", "91.5", "67.0", "66.4"),
}


def compute(replay=ROOT / "results/replay"):
    methods = {}
    expected_keys = None
    for method, filename in METHODS.items():
        with (Path(replay) / filename).open(newline="") as f:
            source = list(csv.DictReader(f))
        values = {
            (r["pair_key"], int(r["b"])): float(r["variance_ratio"]) for r in source
        }
        profiles = {p for p, _ in values}
        keys = {(p, b) for p in profiles for b in PAPER}
        if len(source) != 428 or len(profiles) != 107 or set(values) != keys:
            raise ValueError(f"{filename}: incomplete or duplicate profile/budget grid")
        if expected_keys is not None and keys != expected_keys:
            raise ValueError("Methods use different profiles")
        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("Invalid variance ratios")
        expected_keys = keys
        methods[method] = values
    rows = []
    for b in PAPER:
        row = {"budget": b}
        for method, values in methods.items():
            row[method] = (
                100
                * math.fsum(
                    v for (p, budget), v in sorted(values.items()) if budget == b
                )
                / 107
            )
        if tuple(f"{row[m]:.1f}" for m in METHODS) != PAPER[b]:
            raise ValueError(f"Paper Table 1 mismatch at budget {b}: {row}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "results/replay")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    rows = compute(args.replay_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rounded in [("table1.csv", True), ("table1_unrounded.csv", False)]:
        with (args.output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: f"{v:.1f}" if rounded and k != "budget" else v
                        for k, v in row.items()
                    }
                )
    print("Table 1: all 16 entries match the paper at one decimal percent precision.")
    print(args.output_dir / "table1.csv")


if __name__ == "__main__":
    main()
