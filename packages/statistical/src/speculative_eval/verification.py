"""Checks for saved designs and the parameters used in published experiments."""

import csv
import hashlib
import json
import math
from pathlib import Path


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify_designs(reference: Path, observed: Path) -> None:
    """Compare all candidate/schedule values; execution parallelism is not a parameter."""
    for method in ("en", "ibn", "hbn"):
        for kind in ("candidates", "schedule"):
            name = f"{method}_{kind}.csv"
            expected, actual = _rows(reference / name), _rows(observed / name)
            if len(expected) != len(actual):
                raise AssertionError(f"{name}: row count mismatch")
            for index, (left, right) in enumerate(zip(expected, actual)):
                if left.keys() != right.keys():
                    raise AssertionError(f"{name}: column mismatch")
                for field in left:
                    _close(right[field], left[field], f"{name}:{index + 2}:{field}")
    expected = json.loads((reference / "metadata.json").read_text())
    actual = json.loads((observed / "metadata.json").read_text())
    expected.pop("workers", None)
    actual.pop("workers", None)
    if actual != expected:
        raise AssertionError("design metadata mismatch")


def verify_published_parameters(root: Path) -> None:
    results = root / "results"
    manifest = json.loads((results / "validation/manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((results / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"published input checksum mismatch: {name}")
    schedules = {}
    for method in ("en", "ibn", "hbn"):
        def key(row):
            return (int(row["n_tasks"]), int(row["b"])) + (
                (float(row["alpha"]),) if method == "ibn" else ()
            )
        rows = _rows(results / f"design/{method}_schedule.csv")
        schedule = {key(row): row for row in rows}
        if len(schedule) != len(rows) or len(rows) != (460 if method == "ibn" else 20):
            raise AssertionError(f"{method}: invalid schedule coverage")
        schedules[method] = schedule
        filename = "ibn_tuned" if method == "ibn" else method
        replay = _rows(results / f"replay/{filename}_pair_results.csv")
        if len(replay) != 428 or len({(r['pair_key'], r['b']) for r in replay}) != 428:
            raise AssertionError(f"{method}: invalid replay coverage")
        for row in replay:
            chosen = schedule[key(row)]
            _close(row["m"], chosen["m_star"], f"{method} pilot")
            _close(row["weight"], chosen["weight_star"], f"{method} weight")
    profiles = json.loads((results / "configurations/profiles.json").read_text())
    if len(profiles) != 107:
        raise AssertionError("system profile coverage mismatch")
    for profile in profiles.values():
        if set(profile["budgets"]) != {"8", "16", "32", "64"}:
            raise AssertionError("system budget coverage mismatch")
        for entry in profile["budgets"].values():
            design = entry["design"]
            chosen = schedules["hbn"][(design["n_tasks"], design["budget_per_task"])]
            _close(design["pilot_per_task"], chosen["m_star"], "system pilot")
            _close(design["stage_weight"], chosen["weight_star"], "system weight")
