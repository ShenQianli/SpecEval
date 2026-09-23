"""Statistical primitives used by the Table 1 reproduction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import betaln, logsumexp
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class PriorSpec:
    """Hyperprior for the HBN task-probability population."""

    mu_a: float = 1.0
    mu_b: float = 1.0
    dispersion_a: float = 1.0
    dispersion_b: float = 1.0


REFERENCE_PRIOR = PriorSpec()
ALGORITHM_VERSION = 1


def stable_seed(*parts: object) -> int:
    """Return a deterministic 64-bit seed."""

    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def moments(values: np.ndarray) -> tuple[float, float]:
    """Return a Monte Carlo mean and its standard error."""

    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / math.sqrt(values.size))


def quadrature(
    prior: PriorSpec = REFERENCE_PRIOR,
    order: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor-product Gauss--Legendre rule over mean and dispersion."""

    raw_nodes, raw_weights = leggauss(order)
    nodes = (raw_nodes + 1.0) / 2.0
    base_weights = raw_weights / 2.0
    mu_weights = base_weights * beta_distribution.pdf(
        nodes, prior.mu_a, prior.mu_b
    )
    dispersion_weights = base_weights * beta_distribution.pdf(
        nodes, prior.dispersion_a, prior.dispersion_b
    )
    mu, dispersion = np.meshgrid(nodes, nodes, indexing="ij")
    weights = np.outer(mu_weights, dispersion_weights).ravel()
    weights /= np.sum(weights)
    return mu.ravel(), dispersion.ravel(), weights


def expected_task_variance(
    prior: PriorSpec = REFERENCE_PRIOR,
    order: int = 16,
) -> float:
    """Return E[p(1-p)] under the HBN reference population."""

    mu, dispersion, weights = quadrature(prior, order)
    return float(np.sum(weights * mu * (1.0 - mu) * (1.0 - dispersion)))


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
    """Compute HBN E[p_i(1-p_i) | complete pilot]."""

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


def independent_beta_variance_score(
    pilot_counts: np.ndarray,
    pilot: int,
    alpha: float,
) -> np.ndarray:
    """Return IBN posterior variance, with alpha=0 denoting EN."""

    counts = np.asarray(pilot_counts, dtype=np.float64)
    if pilot < 1:
        raise ValueError("pilot must be positive")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if np.any(counts < 0.0) or np.any(counts > pilot):
        raise ValueError("pilot counts must lie between zero and pilot")
    if alpha == 0.0:
        empirical_probability = counts / float(pilot)
        return empirical_probability * (1.0 - empirical_probability)
    posterior_a = counts + alpha
    posterior_b = pilot - counts + alpha
    posterior_total = pilot + 2.0 * alpha
    return posterior_a * posterior_b / (
        posterior_total * (posterior_total + 1.0)
    )


def beta_bernoulli_paths(uniforms: np.ndarray, alpha: float) -> np.ndarray:
    """Sample nested Beta-Bernoulli counts through the Polya-urn law."""

    if alpha <= 0.0:
        raise ValueError("the matched IBN prior requires alpha > 0")
    draws, n_tasks, max_pilot = uniforms.shape
    counts = np.zeros((draws, n_tasks), dtype=np.int16)
    cumulative = np.empty_like(uniforms, dtype=np.int16)
    for index in range(max_pilot):
        predictive_probability = (counts + alpha) / (
            index + 2.0 * alpha
        )
        counts += uniforms[:, :, index] < predictive_probability
        cumulative[:, :, index] = counts
    return cumulative


def _validate_scores(scores: np.ndarray, total: int) -> tuple[int, int]:
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    draws, n_tasks = scores.shape
    if total < n_tasks:
        raise ValueError("at least one continuation rollout per task is required")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    return draws, n_tasks


def _allocate_positive(scores: np.ndarray, total: int) -> np.ndarray:
    """Exact row-wise argmin of sum_i score_i/L_i for positive scores."""

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
    """Exact positive-integer Neyman allocation for positive scores."""

    scores = np.asarray(scores, dtype=np.float64)
    _, n_tasks = _validate_scores(scores, total)
    if not np.all(scores > 0.0):
        raise ValueError("scores must be strictly positive")
    if total == n_tasks:
        return np.ones_like(scores, dtype=np.int32)
    allocations = _allocate_positive(scores, total)
    if not np.all(np.sum(allocations, axis=1) == total):
        raise AssertionError("continuation budget mismatch")
    return allocations


def exact_integer_allocation_nonnegative(
    scores: np.ndarray,
    total: int,
) -> np.ndarray:
    """Exact allocator extended to the zero scores produced by EN."""

    scores = np.asarray(scores, dtype=np.float64)
    _, n_tasks = _validate_scores(scores, total)
    if total == n_tasks:
        return np.ones_like(scores, dtype=np.int32)
    allocations = np.empty_like(scores, dtype=np.int32)
    positive_rows = np.max(scores, axis=1) > 0.0
    all_zero_rows = ~positive_rows
    if np.any(all_zero_rows):
        base, remainder = divmod(total, n_tasks)
        allocations[all_zero_rows] = base
        if remainder:
            allocations[np.ix_(all_zero_rows, np.arange(remainder))] += 1
    rows = np.flatnonzero(positive_rows)
    if rows.size:
        allocations[rows] = _allocate_positive(scores[rows], total)
    if not np.all(np.sum(allocations, axis=1) == total):
        raise AssertionError("continuation budget mismatch")
    return allocations


def exact_oracle_allocation(variances: np.ndarray, budget: int) -> np.ndarray:
    """Exact full-information integer Neyman allocation for one profile."""

    allocations = np.ones(variances.size, dtype=np.int32)
    for _ in range(variances.size * (budget - 1)):
        gains = variances / (allocations * (allocations + 1.0))
        allocations[int(np.argmax(gains))] += 1
    return allocations


def stage_weighted_ratio(
    variances: np.ndarray,
    allocations: np.ndarray,
    pilot: int,
    budget: int,
    weight: float,
) -> np.ndarray:
    """Uniform-normalized variance for a fixed two-stage weight."""

    inverse_mass = np.sum(variances[None, :] / allocations, axis=1)
    variance_mass = float(np.sum(variances))
    return (
        budget * weight**2 / pilot
        + budget * (1.0 - weight) ** 2 * inverse_mass / variance_mass
    )
