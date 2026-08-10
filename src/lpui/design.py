"""Two-stage randomized saturation designs for independent interference blocks."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class Assignment(NamedTuple):
    """Public randomization variables for one experiment."""

    cluster_id: NDArray[np.int64]
    saturation_arm: NDArray[np.int8]
    treatment: NDArray[np.int8]


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def draw_two_stage_assignment(
    cluster_count: int,
    cluster_size: int,
    p_low: float,
    p_high: float,
    arm_probability: float,
    rng: np.random.Generator,
) -> Assignment:
    """Randomize blocks to saturation arms, then units to treatment within each block."""
    if cluster_count < 2 or cluster_size < 2:
        raise ValueError("cluster_count and cluster_size must both be at least two")
    p_low = _validate_probability(p_low, "p_low")
    p_high = _validate_probability(p_high, "p_high")
    arm_probability = _validate_probability(arm_probability, "arm_probability")
    if p_low >= p_high:
        raise ValueError("p_low must be smaller than p_high")

    cluster_id = np.repeat(np.arange(cluster_count, dtype=np.int64), cluster_size)
    cluster_arm = rng.binomial(1, arm_probability, size=cluster_count).astype(np.int8)
    saturation_arm = np.repeat(cluster_arm, cluster_size)
    treatment_probability = np.where(saturation_arm == 1, p_high, p_low)
    treatment = rng.binomial(1, treatment_probability).astype(np.int8)
    return Assignment(cluster_id, saturation_arm, treatment)

