"""Offline designs used by Table 1."""

from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from threadpoolctl import threadpool_limits

from .core import (
    ALGORITHM_VERSION,
    REFERENCE_PRIOR,
    PriorSpec,
    beta_bernoulli_paths,
    exact_integer_allocation,
    exact_integer_allocation_nonnegative,
    expected_task_variance,
    independent_beta_variance_score,
    moments,
    posterior_fresh_variance,
    quadrature,
    stable_seed,
)
from .data import Profile


BUDGETS = (8, 16, 32, 64)
MAX_FEASIBLE_PILOT = max(BUDGETS) - 1
HBN_RANDOM_PREFIX_LENGTH = 32
POSITIVE_ALPHA_GRID = (
    *(index / 100.0 for index in range(1, 21)),
    0.25,
    0.50,
    1.00,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a nonempty list of uniform dictionaries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parallel_map(
    function: Callable[[dict[str, Any]], list[dict[str, object]]],
    tasks: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, object]]:
    if workers == 1:
        chunks = [function(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(function, tasks))
    return [row for chunk in chunks for row in chunk]


def _hbn_replicate(task: dict[str, Any]) -> list[dict[str, object]]:
    n_tasks = int(task["n_tasks"])
    draws = int(task["draws"])
    max_pilot = int(task["max_pilot"])
    prior = PriorSpec(**task["prior"])
    mu_grid, dispersion_grid, q_weights = quadrature(prior, 16)
    expected_v = expected_task_variance(prior, 16)
    rng = np.random.default_rng(int(task["seed"]))
    draw_mu = rng.beta(prior.mu_a, prior.mu_b, size=draws)
    draw_dispersion = rng.beta(
        prior.dispersion_a,
        prior.dispersion_b,
        size=draws,
    )
    concentration = (1.0 - draw_dispersion) / draw_dispersion
    probabilities = rng.beta(
        draw_mu[:, None] * concentration[:, None],
        (1.0 - draw_mu[:, None]) * concentration[:, None],
        size=(draws, n_tasks),
    )
    # Generate a prefix of min(max_pilot, 32) uniforms per task, then the remainder.
    # This block order defines the published Monte Carlo sample stream.
    prefix = min(max_pilot, HBN_RANDOM_PREFIX_LENGTH)
    uniform_blocks = [rng.random((draws, n_tasks, prefix))]
    if max_pilot > prefix:
        uniform_blocks.append(
            rng.random((draws, n_tasks, max_pilot - prefix))
        )
    uniforms = np.concatenate(uniform_blocks, axis=2)
    cumulative = np.cumsum(
        uniforms < probabilities[:, :, None],
        axis=2,
        dtype=np.int16,
    )
    rows: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        for pilot in range(1, max_pilot + 1):
            scores = posterior_fresh_variance(
                cumulative[:, :, pilot - 1],
                pilot,
                mu_grid,
                dispersion_grid,
                q_weights,
            )
            for budget in BUDGETS:
                if pilot >= budget:
                    continue
                allocations = exact_integer_allocation(
                    scores,
                    n_tasks * (budget - pilot),
                )
                values = np.sum(scores / allocations, axis=1)
                continuation_mass, continuation_se = moments(values)
                rows.append(
                    {
                        "replicate": int(task["replicate"]),
                        "n_tasks": n_tasks,
                        "b": budget,
                        "m": pilot,
                        "continuation_mass": continuation_mass,
                        "continuation_mass_se": continuation_se,
                        "expected_task_variance": expected_v,
                    }
                )
    return rows


def design_hbn(
    n_values: Iterable[int],
    *,
    draws: int,
    replicates: int,
    max_pilot: int,
    workers: int,
    seed: int = 20260827,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Jointly select HBN pilot size and fixed stage weight ex ante."""

    n_values = tuple(sorted(set(map(int, n_values))))
    tasks = [
        {
            "replicate": replicate,
            "n_tasks": n_tasks,
            "draws": draws,
            "max_pilot": max_pilot,
            "prior": asdict(REFERENCE_PRIOR),
            "seed": seed + 1_000_003 * replicate + 10_007 * n_tasks,
        }
        for replicate in range(replicates)
        for n_tasks in n_values
    ]
    raw = _parallel_map(_hbn_replicate, tasks, workers)
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for row in raw:
        key = (int(row["n_tasks"]), int(row["b"]), int(row["m"]))
        groups.setdefault(key, []).append(row)
    candidates: list[dict[str, object]] = []
    for (n_tasks, budget, pilot), subset in sorted(groups.items()):
        if len(subset) != replicates:
            raise ValueError(f"incomplete HBN design cell {(n_tasks, budget, pilot)}")
        expected_v = float(subset[0]["expected_task_variance"])
        continuation_mass = float(
            np.mean([float(row["continuation_mass"]) for row in subset])
        )
        continuation_se = math.sqrt(
            sum(float(row["continuation_mass_se"]) ** 2 for row in subset)
        ) / replicates
        pilot_mass = n_tasks * expected_v / pilot
        scale = budget / (n_tasks * expected_v)
        weight = continuation_mass / (pilot_mass + continuation_mass)
        risk = (
            scale
            * pilot_mass
            * continuation_mass
            / (pilot_mass + continuation_mass)
        )
        risk_derivative = (
            scale
            * pilot_mass**2
            / (pilot_mass + continuation_mass) ** 2
        )
        candidates.append(
            {
                "n_tasks": n_tasks,
                "b": budget,
                "m": pilot,
                "weight": weight,
                "prior_risk": risk,
                "prior_risk_mc_se": risk_derivative * continuation_se,
            }
        )
    schedule = _select_schedule(candidates, key_fields=("n_tasks", "b"))
    return candidates, schedule


def _ibn_replicate(task: dict[str, Any]) -> list[dict[str, object]]:
    n_tasks = int(task["n_tasks"])
    draws = int(task["draws"])
    max_pilot = int(task["max_pilot"])
    rng = np.random.default_rng(int(task["seed"]))
    uniforms = rng.random((draws, n_tasks, max_pilot))
    rows: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        for alpha in POSITIVE_ALPHA_GRID:
            expected_v = alpha / (2.0 * (2.0 * alpha + 1.0))
            cumulative = beta_bernoulli_paths(uniforms, alpha)
            for pilot in range(1, max_pilot + 1):
                scores = independent_beta_variance_score(
                    cumulative[:, :, pilot - 1],
                    pilot,
                    alpha,
                )
                for budget in BUDGETS:
                    if pilot >= budget:
                        continue
                    allocations = exact_integer_allocation(
                        scores,
                        n_tasks * (budget - pilot),
                    )
                    values = np.sum(scores / allocations, axis=1)
                    continuation_mass, continuation_se = moments(values)
                    rows.append(
                        {
                            "replicate": int(task["replicate"]),
                            "n_tasks": n_tasks,
                            "b": budget,
                            "alpha": alpha,
                            "m": pilot,
                            "continuation_mass": continuation_mass,
                            "continuation_mass_se": continuation_se,
                            "expected_task_variance": expected_v,
                        }
                    )
    return rows


def design_ibn(
    n_values: Iterable[int],
    *,
    draws: int,
    replicates: int,
    max_pilot: int,
    workers: int,
    seed: int = 20260829,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Design IBN under each alpha's matched Beta(alpha, alpha) prior."""

    n_values = tuple(sorted(set(map(int, n_values))))
    tasks = [
        {
            "replicate": replicate,
            "n_tasks": n_tasks,
            "draws": draws,
            "max_pilot": max_pilot,
            "seed": seed + 1_000_003 * replicate + 10_007 * n_tasks,
        }
        for replicate in range(replicates)
        for n_tasks in n_values
    ]
    raw = _parallel_map(_ibn_replicate, tasks, workers)
    groups: dict[
        tuple[int, int, float, int],
        list[dict[str, object]],
    ] = {}
    for row in raw:
        key = (
            int(row["n_tasks"]),
            int(row["b"]),
            float(row["alpha"]),
            int(row["m"]),
        )
        groups.setdefault(key, []).append(row)
    candidates: list[dict[str, object]] = []
    for (n_tasks, budget, alpha, pilot), subset in sorted(groups.items()):
        if len(subset) != replicates:
            raise ValueError(
                f"incomplete IBN design cell {(n_tasks, budget, alpha, pilot)}"
            )
        expected_v = float(subset[0]["expected_task_variance"])
        continuation_mass = float(
            np.mean([float(row["continuation_mass"]) for row in subset])
        )
        continuation_se = math.sqrt(
            sum(float(row["continuation_mass_se"]) ** 2 for row in subset)
        ) / replicates
        pilot_mass = n_tasks * expected_v / pilot
        scale = budget / (n_tasks * expected_v)
        weight = continuation_mass / (pilot_mass + continuation_mass)
        risk = (
            scale
            * pilot_mass
            * continuation_mass
            / (pilot_mass + continuation_mass)
        )
        risk_derivative = (
            scale
            * pilot_mass**2
            / (pilot_mass + continuation_mass) ** 2
        )
        candidates.append(
            {
                "n_tasks": n_tasks,
                "b": budget,
                "alpha": alpha,
                "m": pilot,
                "weight": weight,
                "prior_risk": risk,
                "prior_risk_mc_se": risk_derivative * continuation_se,
            }
        )
    schedule = _select_schedule(
        candidates,
        key_fields=("n_tasks", "b", "alpha"),
    )
    return candidates, schedule


def _en_profile(task: dict[str, Any]) -> list[dict[str, object]]:
    probabilities = np.asarray(task["probabilities"], dtype=np.float64)
    variances = probabilities * (1.0 - probabilities)
    variance_mass = float(np.sum(variances))
    n_tasks = probabilities.size
    draws = int(task["draws"])
    max_pilot = int(task["max_pilot"])
    rng = np.random.default_rng(
        stable_seed(
            "en-direct-grid-design",
            ALGORITHM_VERSION,
            task["pair_key"],
            draws,
        )
    )
    uniforms = rng.random((draws, n_tasks, max_pilot))
    cumulative = np.cumsum(
        uniforms < probabilities[None, :, None],
        axis=2,
        dtype=np.int16,
    )
    rows: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        for pilot in range(1, max_pilot + 1):
            scores = independent_beta_variance_score(
                cumulative[:, :, pilot - 1],
                pilot,
                alpha=0.0,
            )
            for budget in BUDGETS:
                if pilot >= budget:
                    continue
                allocations = exact_integer_allocation_nonnegative(
                    scores,
                    n_tasks * (budget - pilot),
                )
                values = (
                    np.sum(variances[None, :] / allocations, axis=1)
                    / variance_mass
                )
                coefficient, coefficient_se = moments(values)
                rows.append(
                    {
                        "pair_key": task["pair_key"],
                        "n_tasks": n_tasks,
                        "b": budget,
                        "m": pilot,
                        "continuation_coefficient": coefficient,
                        "continuation_coefficient_se": coefficient_se,
                    }
                )
    return rows


def design_en(
    profiles: list[Profile],
    *,
    draws: int,
    max_pilot: int,
    workers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Post-hoc joint EN design on the fixed evaluation profiles."""

    tasks = [
        {
            "pair_key": profile.pair_key,
            "probabilities": profile.probabilities,
            "draws": draws,
            "max_pilot": max_pilot,
        }
        for profile in profiles
    ]
    raw = _parallel_map(_en_profile, tasks, workers)
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for row in raw:
        key = (int(row["n_tasks"]), int(row["b"]), int(row["m"]))
        groups.setdefault(key, []).append(row)
    candidates: list[dict[str, object]] = []
    for (n_tasks, budget, pilot), subset in sorted(groups.items()):
        coefficient = float(
            np.mean(
                [float(row["continuation_coefficient"]) for row in subset]
            )
        )
        coefficient_se = math.sqrt(
            sum(
                float(row["continuation_coefficient_se"]) ** 2
                for row in subset
            )
        ) / len(subset)
        pilot_coefficient = 1.0 / pilot
        weight = coefficient / (pilot_coefficient + coefficient)
        risk = (
            budget
            * pilot_coefficient
            * coefficient
            / (pilot_coefficient + coefficient)
        )
        risk_derivative = (
            budget
            * pilot_coefficient**2
            / (pilot_coefficient + coefficient) ** 2
        )
        candidates.append(
            {
                "n_tasks": n_tasks,
                "b": budget,
                "m": pilot,
                "n_profiles": len(subset),
                "weight": weight,
                "empirical_risk": risk,
                "empirical_risk_mc_se": risk_derivative * coefficient_se,
            }
        )
    schedule = _select_schedule(
        candidates,
        key_fields=("n_tasks", "b"),
        risk_field="empirical_risk",
    )
    return candidates, schedule


def _select_schedule(
    candidates: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
    risk_field: str = "prior_risk",
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in candidates:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)
    schedule: list[dict[str, object]] = []
    for key, subset in sorted(groups.items()):
        winner = min(
            subset,
            key=lambda row: (float(row[risk_field]), int(row["m"])),
        )
        schedule.append(
            {
                **dict(zip(key_fields, key)),
                "m_star": int(winner["m"]),
                "weight_star": float(winner["weight"]),
                "design_risk": float(winner[risk_field]),
            }
        )
    return schedule


def reproduce_designs(
    profiles: list[Profile],
    output: Path,
    *,
    workers: int,
    en_draws: int = 8192,
    hbn_draws: int = 8192,
    hbn_replicates: int = 8,
    ibn_draws: int = 8192,
    ibn_replicates: int = 8,
    en_max_pilot: int = MAX_FEASIBLE_PILOT,
    hbn_max_pilot: int = MAX_FEASIBLE_PILOT,
    ibn_max_pilot: int = MAX_FEASIBLE_PILOT,
) -> dict[str, list[dict[str, object]]]:
    """Recompute and persist all three Table 1 designs."""

    n_values = sorted({profile.n_tasks for profile in profiles})
    output.mkdir(parents=True, exist_ok=True)
    print("designing post-hoc tuned EN", flush=True)
    en_candidates, en_schedule = design_en(
        profiles,
        draws=en_draws,
        max_pilot=en_max_pilot,
        workers=workers,
    )
    write_csv(output / "en_candidates.csv", en_candidates)
    write_csv(output / "en_schedule.csv", en_schedule)
    print("designing matched-prior tuned IBN", flush=True)
    ibn_candidates, ibn_schedule = design_ibn(
        n_values,
        draws=ibn_draws,
        replicates=ibn_replicates,
        max_pilot=ibn_max_pilot,
        workers=workers,
    )
    write_csv(output / "ibn_candidates.csv", ibn_candidates)
    write_csv(output / "ibn_schedule.csv", ibn_schedule)
    print("designing ex-ante joint HBN", flush=True)
    hbn_candidates, hbn_schedule = design_hbn(
        n_values,
        draws=hbn_draws,
        replicates=hbn_replicates,
        max_pilot=hbn_max_pilot,
        workers=workers,
    )
    write_csv(output / "hbn_candidates.csv", hbn_candidates)
    write_csv(output / "hbn_schedule.csv", hbn_schedule)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "budgets": list(BUDGETS),
                "n_values": n_values,
                "en": {
                    "draws_per_profile_candidate": en_draws,
                    "max_pilot": en_max_pilot,
                    "selection": "post-hoc on the 107 evaluation profiles",
                },
                "ibn": {
                    "draws_per_replicate": ibn_draws,
                    "replicates": ibn_replicates,
                    "max_pilot": ibn_max_pilot,
                    "positive_alpha_grid": list(POSITIVE_ALPHA_GRID),
                    "prior": "iid Beta(alpha, alpha), separately for each alpha",
                },
                "hbn": {
                    "draws_per_replicate": hbn_draws,
                    "replicates": hbn_replicates,
                    "max_pilot": hbn_max_pilot,
                    "prior": asdict(REFERENCE_PRIOR),
                },
                "workers": workers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "en": en_schedule,
        "ibn": ibn_schedule,
        "hbn": hbn_schedule,
    }
