"""Rate expressions used in the theory and simulation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lpui.mechanisms import randomized_response_baseline


def privacy_signal_squared(epsilon: ArrayLike) -> NDArray[np.float64]:
    """Squared signal retained by the one-bit epsilon-LDP channel."""
    values = np.asarray(epsilon, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("epsilon must be finite and strictly positive")
    return np.tanh(values / 2.0) ** 2


@dataclass(frozen=True)
class MinimaxRate:
    """Interference and privacy components of the matched risk rate."""

    interference_term: float
    privacy_term: float

    @property
    def total(self) -> float:
        return min(1.0, self.interference_term + self.privacy_term)


def minimax_rate(
    cluster_count: int,
    cluster_size: int,
    epsilon: float,
) -> MinimaxRate:
    """Return the capped rate ``C^-1 + (Cm a_epsilon^2)^-1``.

    Design probabilities affect constants. Exposure-cell rarity is a separate privacy-overlap
    phenomenon and must not be folded into both block and message terms with one scalar.
    """
    if cluster_count <= 0 or cluster_size <= 0:
        raise ValueError("cluster_count and cluster_size must be positive")
    signal_squared = float(privacy_signal_squared(epsilon))
    sample_size = cluster_count * cluster_size
    return MinimaxRate(
        interference_term=1.0 / cluster_count,
        privacy_term=1.0 / (sample_size * signal_squared),
    )


def oracle_exposure_ht_mse(
    sample_size: int,
    exposure_probability: float,
    epsilon: float,
    target_mean: float,
) -> float:
    """Exact MSE of the one-bit HT mean in the oracle exposure subproblem.

    Exposure indicators are independent Bernoulli variables with known probability ``rho``;
    conditional outcomes are Bernoulli with mean ``target_mean``. The formula includes both
    random cell counts and randomized-response noise, so it supplies a parameter-free overlay
    for the overlap experiment rather than a curve fitted to simulated results.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0.0 < exposure_probability <= 1.0:
        raise ValueError("exposure_probability must lie in (0, 1]")
    if not 0.0 <= target_mean <= 1.0:
        raise ValueError("target_mean must lie in [0, 1]")
    signal = float(np.sqrt(privacy_signal_squared(epsilon)))
    baseline = randomized_response_baseline(epsilon)
    released_one_probability = baseline + signal * target_mean
    centered_second_moment = (
        released_one_probability * (1.0 - baseline) ** 2
        + (1.0 - released_one_probability) * baseline**2
    )
    per_user_variance = (
        exposure_probability * centered_second_moment
        - exposure_probability**2 * signal**2 * target_mean**2
    )
    normalization = (
        sample_size * exposure_probability**2 * signal**2
    )
    return float(per_user_variance / normalization)
