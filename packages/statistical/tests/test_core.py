from __future__ import annotations

import itertools

import numpy as np

from speculative_eval.core import (
    beta_bernoulli_paths,
    exact_integer_allocation,
    exact_integer_allocation_nonnegative,
    independent_beta_variance_score,
    posterior_fresh_variance,
    quadrature,
    stage_weighted_ratio,
)
from speculative_eval.cli import build_parser
from speculative_eval.design import (
    BUDGETS,
    MAX_FEASIBLE_PILOT,
    POSITIVE_ALPHA_GRID,
    _hbn_replicate,
    _select_schedule,
)


def _brute_force(scores: np.ndarray, total: int) -> float:
    feasible = (
        np.asarray(candidate)
        for candidate in itertools.product(
            range(1, total + 1), repeat=scores.size
        )
        if sum(candidate) == total
    )
    return min(float(np.sum(scores / allocation)) for allocation in feasible)


def test_integer_allocators_match_brute_force() -> None:
    rng = np.random.default_rng(20260825)
    for n_tasks in range(2, 6):
        for total in range(n_tasks, n_tasks + 5):
            for _ in range(4):
                positive = rng.uniform(0.001, 2.0, size=n_tasks)
                allocation = exact_integer_allocation(
                    positive[None, :], total
                )[0]
                assert np.isclose(
                    np.sum(positive / allocation),
                    _brute_force(positive, total),
                )
                nonnegative = positive.copy()
                nonnegative[rng.random(n_tasks) < 0.45] = 0.0
                allocation = exact_integer_allocation_nonnegative(
                    nonnegative[None, :], total
                )[0]
                assert np.isclose(
                    np.sum(nonnegative / allocation),
                    _brute_force(nonnegative, total),
                )


def test_hbn_posterior_scores_are_positive_at_endpoints() -> None:
    counts = np.asarray([[0, 4, 0, 4], [1, 3, 2, 4]], dtype=np.int32)
    mu, dispersion, weights = quadrature(order=16)
    scores = posterior_fresh_variance(counts, 4, mu, dispersion, weights)
    assert scores.shape == counts.shape
    assert np.all(np.isfinite(scores))
    assert np.all(scores > 0.0)


def test_positive_ibn_grid_and_scores() -> None:
    assert POSITIVE_ALPHA_GRID == (
        *(index / 100.0 for index in range(1, 21)),
        0.25,
        0.50,
        1.00,
    )
    counts = np.asarray([[0, 4]], dtype=np.int32)
    for alpha in POSITIVE_ALPHA_GRID:
        scores = independent_beta_variance_score(counts, 4, alpha)
        assert np.all(scores > 0.0)


def test_default_pilot_search_covers_largest_budget() -> None:
    args = build_parser().parse_args(["reproduce"])
    assert MAX_FEASIBLE_PILOT == max(BUDGETS) - 1 == 63
    assert args.en_max_pilot == MAX_FEASIBLE_PILOT
    assert args.hbn_max_pilot == MAX_FEASIBLE_PILOT
    assert args.ibn_max_pilot == MAX_FEASIBLE_PILOT


def test_hbn_random_prefix_is_stable_above_32() -> None:
    task = {
        "n_tasks": 3,
        "draws": 8,
        "prior": {
            "mu_a": 1.0,
            "mu_b": 1.0,
            "dispersion_a": 1.0,
            "dispersion_b": 1.0,
        },
        "seed": 20260828,
        "replicate": 0,
    }
    prefix_rows = _hbn_replicate({**task, "max_pilot": 32})
    extended_rows = _hbn_replicate({**task, "max_pilot": 63})
    assert [row for row in extended_rows if int(row["m"]) <= 32] == prefix_rows
    assert any(
        int(row["b"]) == 64 and int(row["m"]) == 63
        for row in extended_rows
    )


def test_polya_paths_are_nested() -> None:
    uniforms = np.asarray(
        [[[0.25, 0.75, 0.10, 0.90], [0.75, 0.25, 0.90, 0.10]]]
    )
    paths = beta_bernoulli_paths(uniforms, alpha=0.01)
    increments = np.diff(
        np.concatenate(
            [np.zeros((1, 2, 1), dtype=np.int16), paths], axis=2
        ),
        axis=2,
    )
    assert np.all((increments == 0) | (increments == 1))
    assert np.all(paths <= np.arange(1, 5)[None, None, :])


def test_stage_weighted_ratio_recovers_uniform_allocation() -> None:
    variances = np.asarray([0.10, 0.20, 0.24])
    draws = 5
    for pilot, budget in ((3, 8), (6, 16), (10, 32), (16, 64)):
        allocations = np.full(
            (draws, variances.size),
            budget - pilot,
            dtype=np.int32,
        )
        ratios = stage_weighted_ratio(
            variances,
            allocations,
            pilot,
            budget,
            pilot / budget,
        )
        assert np.allclose(ratios, 1.0)


def test_schedule_selection_uses_risk_then_smaller_pilot() -> None:
    candidates = [
        {
            "n_tasks": 30,
            "b": 16,
            "m": 8,
            "weight": 0.4,
            "prior_risk": 0.9,
        },
        {
            "n_tasks": 30,
            "b": 16,
            "m": 7,
            "weight": 0.3,
            "prior_risk": 0.9,
        },
        {
            "n_tasks": 30,
            "b": 16,
            "m": 6,
            "weight": 0.2,
            "prior_risk": 1.0,
        },
    ]
    schedule = _select_schedule(candidates, key_fields=("n_tasks", "b"))
    assert schedule == [
        {
            "n_tasks": 30,
            "b": 16,
            "m_star": 7,
            "weight_star": 0.3,
            "design_risk": 0.9,
        }
    ]
