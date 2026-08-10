"""Design-unbiased estimators and block-robust uncertainty quantification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t

from lpui.design import Assignment
from lpui.mechanisms import debias_one_bit


@dataclass(frozen=True)
class EffectEstimate:
    """Point estimate and a two-sided cluster-robust confidence interval."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float


def _decoded_release(
    releases: ArrayLike,
    release_kind: Literal["one_bit", "outcome", "laplace", "naive_bit"],
    epsilon: float | None,
) -> NDArray[np.float64]:
    values = np.asarray(releases, dtype=float)
    if release_kind == "one_bit":
        if epsilon is None:
            raise ValueError("epsilon is required for one-bit releases")
        return debias_one_bit(values, epsilon)
    if release_kind in {"outcome", "laplace", "naive_bit"}:
        if np.any(~np.isfinite(values)):
            raise ValueError("releases must be finite")
        return values
    raise ValueError(f"unsupported release_kind: {release_kind}")


def _validate_assignment(assignment: Assignment, release_count: int) -> tuple[int, int]:
    lengths = {
        len(assignment.cluster_id),
        len(assignment.saturation_arm),
        len(assignment.treatment),
        release_count,
    }
    if len(lengths) != 1:
        raise ValueError("releases and all assignment arrays must have equal length")
    if release_count == 0:
        raise ValueError("the experiment must contain observations")
    cluster_ids, cluster_sizes = np.unique(assignment.cluster_id, return_counts=True)
    if not np.array_equal(cluster_ids, np.arange(len(cluster_ids))):
        raise ValueError("cluster identifiers must be contiguous integers starting at zero")
    if np.any(cluster_sizes != cluster_sizes[0]):
        raise ValueError("this estimator currently requires equal cluster sizes")
    if np.any((assignment.saturation_arm != 0) & (assignment.saturation_arm != 1)):
        raise ValueError("saturation_arm must be binary")
    if np.any((assignment.treatment != 0) & (assignment.treatment != 1)):
        raise ValueError("treatment must be binary")
    for cluster_id in cluster_ids:
        cluster_arms = np.unique(
            assignment.saturation_arm[assignment.cluster_id == cluster_id]
        )
        if len(cluster_arms) != 1:
            raise ValueError("saturation_arm must be constant within every cluster")
    return len(cluster_ids), int(cluster_sizes[0])


def _cluster_cell_scores(
    decoded: NDArray[np.float64],
    assignment: Assignment,
    saturation: int,
    treatment: int,
    arm_probability: float,
    treatment_probability: float,
    cluster_count: int,
    cluster_size: int,
) -> NDArray[np.float64]:
    saturation_probability = arm_probability if saturation == 1 else 1.0 - arm_probability
    own_probability = treatment_probability if treatment == 1 else 1.0 - treatment_probability
    observed_cell = (assignment.saturation_arm == saturation) & (
        assignment.treatment == treatment
    )
    unit_scores = (
        observed_cell * decoded / (saturation_probability * own_probability)
    )
    return np.bincount(
        assignment.cluster_id,
        weights=unit_scores,
        minlength=cluster_count,
    ) / cluster_size


def _summarize_scores(
    scores: NDArray[np.float64], alpha: float
) -> EffectEstimate:
    cluster_count = len(scores)
    estimate = float(np.mean(scores))
    standard_error = float(np.std(scores, ddof=1) / np.sqrt(cluster_count))
    critical_value = float(t.ppf(1.0 - alpha / 2.0, df=cluster_count - 1))
    return EffectEstimate(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - critical_value * standard_error,
        ci_high=estimate + critical_value * standard_error,
    )


def estimate_policy_effects(
    releases: ArrayLike,
    assignment: Assignment,
    p_low: float,
    p_high: float,
    arm_probability: float = 0.5,
    *,
    epsilon: float | None = None,
    release_kind: Literal["one_bit", "outcome", "laplace", "naive_bit"] = "one_bit",
    alpha: float = 0.05,
) -> dict[str, EffectEstimate]:
    """Estimate saturation-specific direct and policy-spillover effects.

    The estimands marginalize over all unknown within-block interference induced by the
    randomized saturation policy. The Horvitz--Thompson cell scores remain unbiased for
    arbitrary potential-outcome surfaces inside a block.
    """
    decoded = _decoded_release(releases, release_kind, epsilon)
    cluster_count, cluster_size = _validate_assignment(assignment, len(decoded))
    if not 0.0 < p_low < p_high < 1.0:
        raise ValueError("p_low and p_high must satisfy 0 < p_low < p_high < 1")
    if not 0.0 < arm_probability < 1.0:
        raise ValueError("arm_probability must lie strictly between zero and one")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    cell_scores: dict[tuple[int, int], NDArray[np.float64]] = {}
    for saturation, treatment_probability in ((0, p_low), (1, p_high)):
        for treatment in (0, 1):
            cell_scores[saturation, treatment] = _cluster_cell_scores(
                decoded,
                assignment,
                saturation,
                treatment,
                arm_probability,
                treatment_probability,
                cluster_count,
                cluster_size,
            )

    contrast_scores = {
        "direct_low": cell_scores[0, 1] - cell_scores[0, 0],
        "direct_high": cell_scores[1, 1] - cell_scores[1, 0],
        "spillover_control": cell_scores[1, 0] - cell_scores[0, 0],
        "spillover_treated": cell_scores[1, 1] - cell_scores[0, 1],
    }
    return {name: _summarize_scores(scores, alpha) for name, scores in contrast_scores.items()}


def project_policy_effects(
    fitted: dict[str, EffectEstimate],
) -> dict[str, EffectEstimate]:
    """Project effect estimates and intervals onto their known range ``[-1, 1]``.

    Projection is appropriate for bounded-loss comparisons and cannot increase squared
    error for a true effect in ``[-1, 1]``. The unprojected Horvitz--Thompson estimate should
    still be retained when exact unbiasedness is the target property.
    """
    expected_effects = {
        "direct_low",
        "direct_high",
        "spillover_control",
        "spillover_treated",
    }
    if set(fitted) != expected_effects:
        raise ValueError("fitted must contain exactly the four policy effects")
    projected: dict[str, EffectEstimate] = {}
    for name, estimate in fitted.items():
        projected[name] = EffectEstimate(
            estimate=float(np.clip(estimate.estimate, -1.0, 1.0)),
            standard_error=estimate.standard_error,
            ci_low=float(np.clip(estimate.ci_low, -1.0, 1.0)),
            ci_high=float(np.clip(estimate.ci_high, -1.0, 1.0)),
        )
    return projected
