"""Rounded-down Uniform budgets and measured-time aggregation (standard library)."""

import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

VERSION = "uniform_rounded_equivalent_v1"
TARGET_FILE = "systems/uniform_equivalent_budgets.csv"
REPLAY_FILE = "replay/hbn_pair_results.csv"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def checked_bytes(results, name):
    results = Path(results)
    manifest = json.loads((results / "validation/manifest.json").read_text())
    content = (results / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
        raise ValueError(f"Reference fingerprint mismatch: {name}")
    return content


def rounded_budget(budget, ratio):
    ratio = Fraction(str(ratio))
    if ratio <= 0:
        raise ValueError("Variance ratio must be positive")
    return int(Fraction(budget) // ratio)


def build_targets(results):
    content = checked_bytes(results, REPLAY_FILE)
    source_hash = hashlib.sha256(content).hexdigest()
    replay = list(csv.DictReader(content.decode().splitlines()))
    catalog = json.loads(checked_bytes(results, "configurations/profiles.json"))
    rows = []
    for r in replay:
        profile, b = r["pair_key"], int(r["b"])
        budget = rounded_budget(b, r["variance_ratio"])
        n = int(r["n_tasks"])
        design = catalog[profile]["budgets"][str(b)]["design"]
        if n != design["n_tasks"] or budget <= design["pilot_per_task"]:
            raise ValueError("Equivalent budget is incompatible with reference design")
        row = dict(profile=profile, model=r["model"], reference_budget=b, budget=budget,
                   n_tasks=n, hbn_variance_ratio=r["variance_ratio"],
                   continuous_equivalent_budget=float(Fraction(b) / Fraction(r["variance_ratio"])),
                   uniform_over_hbn_variance_ratio=float(Fraction(b) / (budget * Fraction(r["variance_ratio"]))),
                   replay_sha256=source_hash, protocol_version=VERSION)
        row["target_fingerprint"] = digest(row)
        rows.append(row)
    keys = {(r["profile"], r["reference_budget"]) for r in rows}
    if len(rows) != 428 or keys != {(p, b) for p in catalog for b in (8, 16, 32, 64)}:
        raise ValueError("Expected complete 107-profile target grid")
    return sorted(rows, key=lambda r: (r["profile"], r["reference_budget"]))


def write_targets(results, output):
    rows = build_targets(results)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def load_targets(results):
    rows = build_targets(results)
    saved = list(csv.DictReader(checked_bytes(results, TARGET_FILE).decode().splitlines()))
    if saved != [{k: str(v) for k, v in row.items()} for row in rows]:
        raise ValueError("Target budgets differ from fixed replay")
    return {(r["profile"], r["reference_budget"]): r for r in rows}


def derived_design(entry, target):
    b = target["reference_budget"]
    ref = entry["budgets"][str(b)]
    design = dict(ref["design"])
    design["budget_per_task"] = target["budget"]
    design["continuation_total"] = design["n_tasks"] * (target["budget"] - design["pilot_per_task"])
    metadata = dict(version=VERSION, reference_budget=b, budget=target["budget"],
                    target_fingerprint=target["target_fingerprint"])
    fingerprint = digest(dict(supplement=metadata, design=design,
                              reference_protocol_fingerprint=ref["protocol_fingerprint"]))
    return design, fingerprint, metadata


def measured_table4(results, measurements, values, profiles):
    targets = load_targets(results)
    with Path(measurements).open() as f:
        rows = list(csv.DictReader(f))
    observed, run_ids = {}, set()
    catalog = json.loads(checked_bytes(results, "configurations/profiles.json"))
    for row in rows:
        key = row["profile"], int(row["reference_budget"])
        if key not in targets or key in observed:
            raise ValueError("Unknown or duplicate equivalent observation")
        target = targets[key]
        _, fp, _ = derived_design(catalog[key[0]], target)
        run_id = row["run_uuid"]
        if not run_id or run_id in run_ids:
            raise ValueError("Missing or reused equivalent run UUID")
        run_ids.add(run_id)
        if (row["strategy"] != "uniform" or row["model"] != target["model"]
                or int(row["budget"]) != target["budget"]
                or row["target_fingerprint"] != target["target_fingerprint"]
                or row["protocol_fingerprint"] != fp):
            raise ValueError("Equivalent measurement identity mismatch")
        numeric = {k: float(row[k]) for k in ("time_seconds", "useful_eflop", "wasted_eflop",
                                             "accepted_rollouts", "extra_rollouts")}
        if (not all(math.isfinite(v) and v >= 0 for v in numeric.values())
                or numeric["time_seconds"] <= 0 or numeric["useful_eflop"] <= 0
                or numeric["accepted_rollouts"] != target["n_tasks"] * target["budget"]
                or numeric["extra_rollouts"] != 0 or numeric["wasted_eflop"] != 0):
            raise ValueError("Invalid equivalent Uniform measurement")
        observed[key] = numeric["time_seconds"]
    if set(observed) != set(targets) or set(profiles) != {p for p, _ in targets}:
        raise ValueError("Expected all 428 freshly measured equivalent settings")
    output = []
    for b in (8, 16, 32, 64):
        original = math.fsum(values[p, b, "uniform"]["time_seconds"] for p in profiles)
        uniform = math.fsum(observed[p, b] for p in profiles)
        asynchronous = math.fsum(values[p, b, "hbn-async"]["time_seconds"] for p in profiles)
        increase = 100 * (uniform / original - 1)
        actual = 100 * (asynchronous / original - 1)
        output.append(dict(budget=b, equivalent_uniform_time_pct=increase,
                           async_actual_time_pct=actual, delta_pp=increase - actual))
    return output
