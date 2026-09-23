"""Fixed-profile replay and aggregation for Table 1."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from .core import (
    ALGORITHM_VERSION,
    REFERENCE_PRIOR,
    exact_integer_allocation,
    exact_integer_allocation_nonnegative,
    exact_oracle_allocation,
    independent_beta_variance_score,
    moments,
    posterior_fresh_variance,
    quadrature,
    stable_seed,
    stage_weighted_ratio,
)
from .data import Profile
from .design import BUDGETS, POSITIVE_ALPHA_GRID, read_csv, write_csv


def _pair_row(
    task: dict[str, Any],
    budget: int,
    pilot: int,
    weight: float,
    values: np.ndarray,
    *,
    policy: str,
    alpha: float | None = None,
) -> dict[str, object]:
    ratio, ratio_se = moments(values)
    return {
        "pair_key": task["pair_key"],
        "benchmark": task["benchmark"],
        "model": task["model"],
        "n_tasks": len(task["probabilities"]),
        "b": budget,
        "policy": policy,
        "alpha": "" if alpha is None else f"{alpha:.2f}",
        "m": pilot,
        "weight": weight,
        "variance_ratio": ratio,
        "ratio_mc_se": ratio_se,
    }


def _replay_profile(task: dict[str, Any]) -> dict[str, list[dict[str, object]]]:
    probabilities = np.asarray(task["probabilities"], dtype=np.float64)
    variances = probabilities * (1.0 - probabilities)
    variance_mass = float(np.sum(variances))
    n_tasks = probabilities.size
    draws = int(task["draws"])
    en_design = {
        int(budget): (int(values[0]), float(values[1]))
        for budget, values in task["en_design"].items()
    }
    hbn_design = {
        int(budget): (int(values[0]), float(values[1]))
        for budget, values in task["hbn_design"].items()
    }
    ibn_design = {
        (int(budget), float(alpha)): (int(values[0]), float(values[1]))
        for budget, alpha, values in task["ibn_design"]
    }
    oracle_rows: list[dict[str, object]] = []
    en_rows: list[dict[str, object]] = []
    ibn_rows: list[dict[str, object]] = []
    hbn_rows: list[dict[str, object]] = []
    mu_grid, dispersion_grid, q_weights = quadrature(REFERENCE_PRIOR, 16)

    with threadpool_limits(limits=1):
        for budget in BUDGETS:
            oracle_allocation = exact_oracle_allocation(variances, budget)
            oracle_rows.append(
                {
                    "pair_key": task["pair_key"],
                    "benchmark": task["benchmark"],
                    "model": task["model"],
                    "n_tasks": n_tasks,
                    "b": budget,
                    "variance_ratio": float(
                        budget
                        * np.sum(variances / oracle_allocation)
                        / variance_mass
                    ),
                }
            )

            en_pilot, en_weight = en_design[budget]
            en_rng = np.random.default_rng(
                stable_seed(
                    "en-direct-independent-replay",
                    ALGORITHM_VERSION,
                    task["pair_key"],
                    budget,
                    draws,
                )
            )
            en_counts = np.sum(
                en_rng.random((draws, n_tasks, en_pilot))
                < probabilities[None, :, None],
                axis=2,
                dtype=np.int16,
            )
            en_scores = independent_beta_variance_score(
                en_counts,
                en_pilot,
                alpha=0.0,
            )
            en_allocations = exact_integer_allocation_nonnegative(
                en_scores,
                n_tasks * (budget - en_pilot),
            )
            en_rows.append(
                _pair_row(
                    task,
                    budget,
                    en_pilot,
                    en_weight,
                    stage_weighted_ratio(
                        variances,
                        en_allocations,
                        en_pilot,
                        budget,
                        en_weight,
                    ),
                    policy="tuned_en",
                )
            )

            hbn_pilot, hbn_weight = hbn_design[budget]
            hbn_rng = np.random.default_rng(
                stable_seed(
                    "bayes-reference-backtest",
                    ALGORITHM_VERSION,
                    task["pair_key"],
                    budget,
                    hbn_pilot,
                    draws,
                )
            )
            hbn_counts = hbn_rng.binomial(
                hbn_pilot,
                probabilities,
                size=(draws, n_tasks),
            )
            hbn_scores = posterior_fresh_variance(
                hbn_counts,
                hbn_pilot,
                mu_grid,
                dispersion_grid,
                q_weights,
            )
            hbn_allocations = exact_integer_allocation(
                hbn_scores,
                n_tasks * (budget - hbn_pilot),
            )
            hbn_rows.append(
                _pair_row(
                    task,
                    budget,
                    hbn_pilot,
                    hbn_weight,
                    stage_weighted_ratio(
                        variances,
                        hbn_allocations,
                        hbn_pilot,
                        budget,
                        hbn_weight,
                    ),
                    policy="hbn",
                )
            )

            max_ibn_pilot = max(
                ibn_design[budget, alpha][0]
                for alpha in POSITIVE_ALPHA_GRID
            )
            ibn_rng = np.random.default_rng(
                stable_seed(
                    "ibn-joint-policy-replay",
                    ALGORITHM_VERSION,
                    task["pair_key"],
                    budget,
                    draws,
                )
            )
            ibn_uniforms = ibn_rng.random(
                (draws, n_tasks, max_ibn_pilot)
            )
            ibn_cumulative = np.cumsum(
                ibn_uniforms < probabilities[None, :, None],
                axis=2,
                dtype=np.int16,
            )
            for alpha in POSITIVE_ALPHA_GRID:
                pilot, weight = ibn_design[budget, alpha]
                scores = independent_beta_variance_score(
                    ibn_cumulative[:, :, pilot - 1],
                    pilot,
                    alpha,
                )
                allocations = exact_integer_allocation(
                    scores,
                    n_tasks * (budget - pilot),
                )
                ibn_rows.append(
                    _pair_row(
                        task,
                        budget,
                        pilot,
                        weight,
                        stage_weighted_ratio(
                            variances,
                            allocations,
                            pilot,
                            budget,
                            weight,
                        ),
                        policy="tuned_ibn",
                        alpha=alpha,
                    )
                )

    return {
        "oracle": oracle_rows,
        "en": en_rows,
        "ibn": ibn_rows,
        "hbn": hbn_rows,
    }


def _schedule_map(
    rows: list[dict[str, object]],
) -> dict[tuple[int, int], tuple[int, float]]:
    return {
        (int(row["n_tasks"]), int(row["b"])): (
            int(row["m_star"]),
            float(row["weight_star"]),
        )
        for row in rows
    }


def _ibn_schedule_map(
    rows: list[dict[str, object]],
) -> dict[tuple[int, int, float], tuple[int, float]]:
    return {
        (
            int(row["n_tasks"]),
            int(row["b"]),
            float(row["alpha"]),
        ): (int(row["m_star"]), float(row["weight_star"]))
        for row in rows
    }


def load_schedules(
    design_dir: Path,
) -> dict[str, list[dict[str, object]]]:
    """Load the three designs persisted by reproduce_designs."""

    return {
        "en": list(read_csv(design_dir / "en_schedule.csv")),
        "ibn": list(read_csv(design_dir / "ibn_schedule.csv")),
        "hbn": list(read_csv(design_dir / "hbn_schedule.csv")),
    }


def reproduce_table(
    profiles: list[Profile],
    schedules: dict[str, list[dict[str, object]]],
    output: Path,
    *,
    draws: int,
    workers: int,
    tables_output: Path | None = None,
) -> list[dict[str, object]]:
    """Replay all Table 1 methods and write the final table."""

    en = _schedule_map(schedules["en"])
    hbn = _schedule_map(schedules["hbn"])
    ibn = _ibn_schedule_map(schedules["ibn"])
    expected_en_hbn = {
        (profile.n_tasks, budget)
        for profile in profiles
        for budget in BUDGETS
    }
    expected_ibn = {
        (profile.n_tasks, budget, alpha)
        for profile in profiles
        for budget in BUDGETS
        for alpha in POSITIVE_ALPHA_GRID
    }
    if set(en) != expected_en_hbn or set(hbn) != expected_en_hbn:
        raise ValueError("EN or HBN design is incomplete")
    if set(ibn) != expected_ibn:
        raise ValueError("IBN design is incomplete")

    tasks = [
        {
            "pair_key": profile.pair_key,
            "benchmark": profile.benchmark,
            "model": profile.model,
            "probabilities": profile.probabilities,
            "draws": draws,
            "en_design": {
                budget: en[profile.n_tasks, budget]
                for budget in BUDGETS
            },
            "hbn_design": {
                budget: hbn[profile.n_tasks, budget]
                for budget in BUDGETS
            },
            "ibn_design": [
                (
                    budget,
                    alpha,
                    ibn[profile.n_tasks, budget, alpha],
                )
                for budget in BUDGETS
                for alpha in POSITIVE_ALPHA_GRID
            ],
        }
        for profile in profiles
    ]
    if workers == 1:
        chunks = [_replay_profile(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_replay_profile, tasks))
    oracle_rows = sorted(
        (row for chunk in chunks for row in chunk["oracle"]),
        key=lambda row: (str(row["pair_key"]), int(row["b"])),
    )
    en_rows = sorted(
        (row for chunk in chunks for row in chunk["en"]),
        key=lambda row: (str(row["pair_key"]), int(row["b"])),
    )
    ibn_rows = sorted(
        (row for chunk in chunks for row in chunk["ibn"]),
        key=lambda row: (
            str(row["pair_key"]),
            int(row["b"]),
            float(row["alpha"]),
        ),
    )
    hbn_rows = sorted(
        (row for chunk in chunks for row in chunk["hbn"]),
        key=lambda row: (str(row["pair_key"]), int(row["b"])),
    )

    ibn_grid: list[dict[str, object]] = []
    for budget in BUDGETS:
        for alpha in POSITIVE_ALPHA_GRID:
            subset = [
                row
                for row in ibn_rows
                if int(row["b"]) == budget
                and float(row["alpha"]) == alpha
            ]
            ratios = np.asarray(
                [float(row["variance_ratio"]) for row in subset]
            )
            ratio_ses = np.asarray(
                [float(row["ratio_mc_se"]) for row in subset]
            )
            ibn_grid.append(
                {
                    "b": budget,
                    "alpha": f"{alpha:.2f}",
                    "mean_variance_ratio": float(np.mean(ratios)),
                    "mean_ratio_mc_se": float(
                        np.sqrt(np.sum(ratio_ses**2)) / len(subset)
                    ),
                }
            )

    table: list[dict[str, object]] = []
    tuned_ibn_rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        oracle_ratio = float(
            np.mean(
                [
                    float(row["variance_ratio"])
                    for row in oracle_rows
                    if int(row["b"]) == budget
                ]
            )
        )
        en_ratio = float(
            np.mean(
                [
                    float(row["variance_ratio"])
                    for row in en_rows
                    if int(row["b"]) == budget
                ]
            )
        )
        hbn_ratio = float(
            np.mean(
                [
                    float(row["variance_ratio"])
                    for row in hbn_rows
                    if int(row["b"]) == budget
                ]
            )
        )
        tuned = min(
            (row for row in ibn_grid if int(row["b"]) == budget),
            key=lambda row: (
                float(row["mean_variance_ratio"]),
                float(row["alpha"]),
            ),
        )
        tuned_alpha = float(tuned["alpha"])
        tuned_ibn_rows.extend(
            row
            for row in ibn_rows
            if int(row["b"]) == budget
            and float(row["alpha"]) == tuned_alpha
        )
        table.append(
            {
                "b": budget,
                "oracle": oracle_ratio,
                "tuned_en": en_ratio,
                "tuned_ibn": float(tuned["mean_variance_ratio"]),
                "tuned_alpha": tuned_alpha,
                "hbn": hbn_ratio,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "oracle_pair_results.csv", oracle_rows)
    write_csv(output / "en_pair_results.csv", en_rows)
    write_csv(output / "ibn_pair_grid.csv", ibn_rows)
    write_csv(output / "ibn_grid_summary.csv", ibn_grid)
    write_csv(output / "ibn_tuned_pair_results.csv", tuned_ibn_rows)
    write_csv(output / "hbn_pair_results.csv", hbn_rows)
    paper_table = [
        {"budget": row["b"], **{
            f"{method}_pct": 100 * float(row[method])
            for method in ("oracle", "tuned_en", "tuned_ibn", "hbn")
        }} for row in table
    ]
    tables_output = tables_output if tables_output is not None else output / "tables"
    tables_output.mkdir(parents=True, exist_ok=True)
    write_csv(tables_output / "table1_unrounded.csv", paper_table)
    write_csv(tables_output / "table1.csv", [
        {key: value if key == "budget" else f"{value:.1f}"
         for key, value in row.items()} for row in paper_table
    ])
    (output / "summary.json").write_text(
        json.dumps(
            {
                "table1": table,
                "profiles": len(profiles),
                "replay_draws_per_profile_candidate": draws,
                "selection_notes": {
                    "tuned_en": "post-hoc m,w selection on evaluation profiles",
                    "tuned_ibn": (
                        "matched-prior m,w design for each alpha; post-hoc "
                        "alpha selection on evaluation profiles"
                    ),
                    "hbn": "ex-ante m,w selection under the HBN reference prior",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return table
