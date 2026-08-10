"""Finite-feature outcome models motivated by the Degree of Interference.

These regressions are deliberately model-dependent. They approximate an unknown interference
response with public network-assignment summaries; they do not fit the full latent-process
model of Ohnishi, Karmakar, and Sabbaghi (2025). The main Horvitz--Thompson estimator does not
need this module for identification or validity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit

from lpui.dgp import ClusterPopulation, network_exposures
from lpui.mechanisms import randomized_response_baseline


FeatureSet = Literal["generic", "no_interference"]


@dataclass(frozen=True)
class DoISieveFit:
    coefficients: NDArray[np.float64]
    objective: float
    success: bool
    iterations: int
    gradient_norm: float


def build_features(
    population: ClusterPopulation,
    treatment: ArrayLike,
    feature_set: FeatureSet = "generic",
    *,
    own_treatment: int | None = None,
) -> NDArray[np.float64]:
    """Build public features for an observed or own-treatment-intervened assignment.

    Network exposures exclude the focal unit's treatment. Consequently, ``own_treatment``
    can set every row's focal treatment to zero or one while leaving each row's sampled peer
    assignment intact. This is the intervention needed by policy g-computation.
    """
    treatment_matrix = np.asarray(treatment, dtype=float).reshape(
        population.cluster_count, population.cluster_size
    )
    if np.any((treatment_matrix != 0.0) & (treatment_matrix != 1.0)):
        raise ValueError("treatment must be binary")
    one_hop, two_hop, _ = network_exposures(population, treatment_matrix)
    if own_treatment is not None and own_treatment not in (0, 1):
        raise ValueError("own_treatment must be zero, one, or None")
    own = treatment_matrix.ravel()
    if own_treatment is not None:
        own = np.full_like(own, float(own_treatment))
    baseline = population.baseline.ravel()
    if feature_set == "no_interference":
        return np.column_stack([np.ones_like(own), own, baseline])
    if feature_set != "generic":
        raise ValueError(f"unsupported feature_set: {feature_set}")
    one_hop = one_hop.ravel()
    two_hop = two_hop.ravel()
    return np.column_stack(
        [
            np.ones_like(own),
            own,
            baseline,
            one_hop,
            two_hop,
            one_hop**2,
            one_hop**3,
            two_hop**2,
            own * one_hop,
            own * two_hop,
            own * one_hop**2,
        ]
    )


def predict_mean(features: ArrayLike, coefficients: ArrayLike) -> NDArray[np.float64]:
    features = np.asarray(features, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    return expit(features @ coefficients)


def private_log_loss(
    coefficients: ArrayLike,
    features: ArrayLike,
    releases: ArrayLike,
    epsilon: float,
    penalty: float,
) -> float:
    """Penalized Bernoulli loss under the exact affine LDP observation model."""
    coefficients = np.asarray(coefficients, dtype=float)
    features = np.asarray(features, dtype=float)
    bits = np.asarray(releases, dtype=float)
    if features.shape != (len(bits), len(coefficients)):
        raise ValueError("features, releases, and coefficients have incompatible shapes")
    if np.any((bits != 0.0) & (bits != 1.0)):
        raise ValueError("releases must contain only zero and one")
    if epsilon <= 0.0 or penalty < 0.0:
        raise ValueError("epsilon must be positive and penalty must be nonnegative")
    attenuation = np.tanh(epsilon / 2.0)
    baseline = randomized_response_baseline(epsilon)
    private_probability = baseline + attenuation * predict_mean(features, coefficients)
    private_probability = np.clip(private_probability, 1e-12, 1.0 - 1e-12)
    negative_log_likelihood = -np.sum(
        bits * np.log(private_probability)
        + (1.0 - bits) * np.log(1.0 - private_probability)
    )
    # The intercept is not regularized; all learned interference directions are.
    return float(negative_log_likelihood + 0.5 * penalty * np.dot(coefficients[1:], coefficients[1:]))


def _private_log_loss_gradient(
    coefficients: NDArray[np.float64],
    features: NDArray[np.float64],
    bits: NDArray[np.float64],
    epsilon: float,
    penalty: float,
) -> NDArray[np.float64]:
    attenuation = np.tanh(epsilon / 2.0)
    baseline = randomized_response_baseline(epsilon)
    mean = predict_mean(features, coefficients)
    private_probability = baseline + attenuation * mean
    derivative = attenuation * mean * (1.0 - mean)
    residual_multiplier = (private_probability - bits) * derivative / (
        private_probability * (1.0 - private_probability)
    )
    gradient = features.T @ residual_multiplier
    gradient[1:] += penalty * coefficients[1:]
    return gradient


def fit_private_doi_sieve(
    features: ArrayLike,
    releases: ArrayLike,
    epsilon: float,
    penalty: float = 1e-3,
) -> DoISieveFit:
    """Fit the privatized-response finite-feature model by penalized likelihood."""
    features = np.asarray(features, dtype=float)
    bits = np.asarray(releases, dtype=float)
    if features.ndim != 2 or features.shape[0] != len(bits):
        raise ValueError("features must be a matrix with one row per release")
    initial = np.zeros(features.shape[1], dtype=float)
    optimized = minimize(
        private_log_loss,
        initial,
        args=(features, bits, epsilon, penalty),
        jac=_private_log_loss_gradient,
        method="L-BFGS-B",
        options={"maxiter": 1_000, "ftol": 1e-10, "gtol": 1e-7},
    )
    return DoISieveFit(
        coefficients=np.asarray(optimized.x, dtype=float),
        objective=float(optimized.fun),
        success=bool(optimized.success),
        iterations=int(optimized.nit),
        gradient_norm=float(np.linalg.norm(optimized.jac)),
    )


def outcome_log_loss(
    coefficients: ArrayLike,
    features: ArrayLike,
    outcomes: ArrayLike,
    penalty: float,
) -> float:
    """Penalized Bernoulli quasi-likelihood for nonprivate bounded outcomes."""
    coefficients = np.asarray(coefficients, dtype=float)
    features = np.asarray(features, dtype=float)
    values = np.asarray(outcomes, dtype=float)
    if features.shape != (len(values), len(coefficients)):
        raise ValueError("features, outcomes, and coefficients have incompatible shapes")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("outcomes must be finite and lie in [0, 1]")
    if penalty < 0.0:
        raise ValueError("penalty must be nonnegative")
    mean = np.clip(predict_mean(features, coefficients), 1e-12, 1.0 - 1e-12)
    negative_log_likelihood = -np.sum(
        values * np.log(mean) + (1.0 - values) * np.log(1.0 - mean)
    )
    return float(
        negative_log_likelihood
        + 0.5 * penalty * np.dot(coefficients[1:], coefficients[1:])
    )


def _outcome_log_loss_gradient(
    coefficients: NDArray[np.float64],
    features: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    penalty: float,
) -> NDArray[np.float64]:
    gradient = features.T @ (predict_mean(features, coefficients) - outcomes)
    gradient[1:] += penalty * coefficients[1:]
    return gradient


def fit_nonprivate_doi_sieve(
    features: ArrayLike,
    outcomes: ArrayLike,
    penalty: float = 1e-3,
) -> DoISieveFit:
    """Fit the same finite-feature outcome model to nonprivate bounded outcomes."""
    features = np.asarray(features, dtype=float)
    values = np.asarray(outcomes, dtype=float)
    if features.ndim != 2 or features.shape[0] != len(values):
        raise ValueError("features must be a matrix with one row per outcome")
    initial = np.zeros(features.shape[1], dtype=float)
    optimized = minimize(
        outcome_log_loss,
        initial,
        args=(features, values, penalty),
        jac=_outcome_log_loss_gradient,
        method="L-BFGS-B",
        options={"maxiter": 1_000, "ftol": 1e-10, "gtol": 1e-7},
    )
    return DoISieveFit(
        coefficients=np.asarray(optimized.x, dtype=float),
        objective=float(optimized.fun),
        success=bool(optimized.success),
        iterations=int(optimized.nit),
        gradient_norm=float(np.linalg.norm(optimized.jac)),
    )


def gcompute_policy_effects(
    population: ClusterPopulation,
    coefficients: ArrayLike,
    feature_set: FeatureSet,
    p_low: float,
    p_high: float,
    rng: np.random.Generator,
    draws: int = 64,
) -> dict[str, float]:
    """Integrate a fitted mean model over the two Bernoulli saturation policies.

    The Monte Carlo assignments are generated only for post-estimation integration. They do
    not consume privacy budget because the fitted coefficients are post-processing of the
    original local releases.
    """
    if not 0.0 < p_low < p_high < 1.0:
        raise ValueError("p_low and p_high must satisfy 0 < p_low < p_high < 1")
    if draws < 1:
        raise ValueError("draws must be positive")
    coefficients = np.asarray(coefficients, dtype=float)
    cell_means: dict[tuple[int, int], float] = {}
    for saturation, treatment_probability in ((0, p_low), (1, p_high)):
        prediction_sums = np.zeros(2, dtype=float)
        for _ in range(draws):
            synthetic_treatment = rng.binomial(
                1,
                treatment_probability,
                size=population.cluster_count * population.cluster_size,
            )
            for own_treatment in (0, 1):
                features = build_features(
                    population,
                    synthetic_treatment,
                    feature_set,
                    own_treatment=own_treatment,
                )
                prediction_sums[own_treatment] += predict_mean(
                    features, coefficients
                ).mean()
        for own_treatment in (0, 1):
            cell_means[saturation, own_treatment] = float(
                prediction_sums[own_treatment] / draws
            )
    return {
        "direct_low": cell_means[0, 1] - cell_means[0, 0],
        "direct_high": cell_means[1, 1] - cell_means[1, 0],
        "spillover_control": cell_means[1, 0] - cell_means[0, 0],
        "spillover_treated": cell_means[1, 1] - cell_means[0, 1],
    }
