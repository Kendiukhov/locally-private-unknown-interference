"""Outcome-only local privacy mechanisms.

The one-bit mechanism is tailored to mean estimation for outcomes in ``[0, 1]``.
Treatments, block labels, and design probabilities are public by assumption; this module
does not claim to protect them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_epsilon(epsilon: float) -> float:
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    return epsilon


def privacy_signal(epsilon: float) -> float:
    """Return the attenuation ``tanh(epsilon / 2)`` of one-bit randomized response."""
    return float(np.tanh(_validate_epsilon(epsilon) / 2.0))


def randomized_response_baseline(epsilon: float) -> float:
    """Return ``1 / (1 + exp(epsilon))`` without overflowing at large epsilon."""
    epsilon = _validate_epsilon(epsilon)
    exp_negative_epsilon = np.exp(-epsilon)
    return float(exp_negative_epsilon / (1.0 + exp_negative_epsilon))


def one_bit_probability(outcomes: ArrayLike, epsilon: float) -> NDArray[np.float64]:
    """Probability of releasing one for each bounded outcome.

    At outcomes zero and one, the two output likelihood ratios equal ``exp(epsilon)``.
    Convexity then establishes epsilon-local differential privacy for every input in the
    interval.
    """
    values = np.asarray(outcomes, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("outcomes must be finite and lie in [0, 1]")
    epsilon = _validate_epsilon(epsilon)
    baseline = randomized_response_baseline(epsilon)
    return baseline + np.tanh(epsilon / 2.0) * values


def privatize_one_bit(
    outcomes: ArrayLike,
    epsilon: float,
    rng: np.random.Generator,
) -> NDArray[np.int8]:
    """Release one epsilon-LDP bit per bounded outcome."""
    probabilities = one_bit_probability(outcomes, epsilon)
    return rng.binomial(1, probabilities).astype(np.int8)


def debias_one_bit(releases: ArrayLike, epsilon: float) -> NDArray[np.float64]:
    """Decode one-bit releases to an unbiased, generally unbounded pseudo-outcome."""
    bits = np.asarray(releases, dtype=float)
    if np.any(~np.isfinite(bits)) or np.any((bits != 0.0) & (bits != 1.0)):
        raise ValueError("one-bit releases must contain only zero and one")
    epsilon = _validate_epsilon(epsilon)
    baseline = randomized_response_baseline(epsilon)
    return (bits - baseline) / np.tanh(epsilon / 2.0)


def privatize_laplace(
    outcomes: ArrayLike,
    epsilon: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Release bounded outcomes with the standard sensitivity-one Laplace mechanism."""
    values = np.asarray(outcomes, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("outcomes must be finite and lie in [0, 1]")
    epsilon = _validate_epsilon(epsilon)
    return values + rng.laplace(loc=0.0, scale=1.0 / epsilon, size=values.shape)
