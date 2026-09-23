"""HBN allocation primitives for the system package's NumPy 2 environment.

This implementation is maintained separately from ``packages/statistical/src/speculative_eval/core.py``
because the statistical package pins NumPy 1.26. Tests compare the two
implementations to check numerical agreement.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import betaln, logsumexp
from scipy.stats import beta as beta_distribution


def quadrature(order: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_nodes, raw_weights = leggauss(order)
    nodes = (raw_nodes + 1.0) / 2.0
    base_weights = raw_weights / 2.0
    # The frozen reference prior is uniform on mean and dispersion.
    mu_weights = base_weights * beta_distribution.pdf(nodes, 1.0, 1.0)
    dispersion_weights = base_weights * beta_distribution.pdf(nodes, 1.0, 1.0)
    mu, dispersion = np.meshgrid(nodes, nodes, indexing="ij")
    weights = np.outer(mu_weights, dispersion_weights).ravel()
    weights /= np.sum(weights)
    return mu.ravel(), dispersion.ravel(), weights


def _count_histogram(pilot_counts: np.ndarray, pilot: int) -> np.ndarray:
    draws, n_tasks = pilot_counts.shape
    histogram = np.zeros((draws, pilot + 1), dtype=np.int16)
    rows = np.repeat(np.arange(draws), n_tasks)
    np.add.at(histogram, (rows, pilot_counts.ravel()), 1)
    return histogram


def posterior_fresh_variance(
    pilot_counts: np.ndarray,
    pilot: int,
    mu_grid: np.ndarray,
    dispersion_grid: np.ndarray,
    quadrature_weights: np.ndarray,
) -> np.ndarray:
    histogram = _count_histogram(pilot_counts, pilot)
    concentration = (1.0 - dispersion_grid) / dispersion_grid
    prior_a = mu_grid * concentration
    prior_b = (1.0 - mu_grid) * concentration
    count_grid = np.arange(pilot + 1, dtype=float)
    log_marginal = (
        betaln(
            prior_a[:, None] + count_grid[None, :],
            prior_b[:, None] + pilot - count_grid[None, :],
        )
        - betaln(prior_a, prior_b)[:, None]
    )
    log_weights = histogram @ log_marginal.T
    log_weights += np.log(quadrature_weights)[None, :]
    log_weights -= logsumexp(log_weights, axis=1, keepdims=True)
    posterior_weights = np.exp(log_weights)
    variance_by_count = (
        (prior_a[:, None] + count_grid[None, :])
        * (prior_b[:, None] + pilot - count_grid[None, :])
        / (
            (concentration[:, None] + pilot)
            * (concentration[:, None] + pilot + 1.0)
        )
    )
    scores_by_count = posterior_weights @ variance_by_count
    return np.take_along_axis(scores_by_count, pilot_counts, axis=1)


def _allocate_positive(scores: np.ndarray, total: int) -> np.ndarray:
    draws, _ = scores.shape
    lo = np.zeros(draws, dtype=np.float64)
    hi = np.max(scores, axis=1) / 2.0
    for _ in range(48):
        threshold = (lo + hi) / 2.0
        root = (
            np.sqrt(1.0 + 4.0 * scores / threshold[:, None]) - 1.0
        ) / 2.0
        allocations = np.maximum(1, np.ceil(root)).astype(np.int32)
        too_many = np.sum(allocations, axis=1) > total
        lo[too_many] = threshold[too_many]
        hi[~too_many] = threshold[~too_many]
    root = (
        np.sqrt(1.0 + 4.0 * scores / hi[:, None]) - 1.0
    ) / 2.0
    allocations = np.maximum(1, np.ceil(root)).astype(np.int32)
    excess = np.sum(allocations, axis=1) - total
    while np.any(excess > 0):
        rows = np.flatnonzero(excess > 0)
        current = allocations[rows]
        removal_cost = np.where(
            current > 1,
            scores[rows] / ((current - 1.0) * current),
            np.inf,
        )
        selected = np.argmin(removal_cost, axis=1)
        allocations[rows, selected] -= 1
        excess[rows] -= 1
    remaining = total - np.sum(allocations, axis=1)
    while np.any(remaining > 0):
        rows = np.flatnonzero(remaining > 0)
        current = allocations[rows]
        gains = scores[rows] / (current * (current + 1.0))
        selected = np.argmax(gains, axis=1)
        allocations[rows, selected] += 1
        remaining[rows] -= 1
    return allocations


def exact_integer_allocation(scores: np.ndarray, total: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    _, n_tasks = scores.shape
    if total < n_tasks:
        raise ValueError("at least one continuation rollout per task is required")
    if not np.all(np.isfinite(scores)) or not np.all(scores > 0.0):
        raise ValueError("scores must be finite and strictly positive")
    if total == n_tasks:
        return np.ones_like(scores, dtype=np.int32)
    allocations = _allocate_positive(scores, total)
    if not np.all(np.sum(allocations, axis=1) == total):
        raise AssertionError("continuation budget mismatch")
    return allocations

