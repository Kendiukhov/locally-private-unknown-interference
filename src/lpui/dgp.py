"""Finite populations with unknown within-block interference surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from lpui.design import Assignment


Scenario = Literal["linear", "complex"]


@dataclass(frozen=True)
class ClusterPopulation:
    """A fixed finite population and its latent interference structure."""

    adjacency: NDArray[np.int8]
    baseline: NDArray[np.float64]
    scenario: Scenario

    @property
    def cluster_count(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def cluster_size(self) -> int:
        return int(self.adjacency.shape[1])


def _ring_with_random_chords(cluster_size: int, rng: np.random.Generator) -> NDArray[np.int8]:
    adjacency = np.zeros((cluster_size, cluster_size), dtype=np.int8)
    for index in range(cluster_size):
        for step in (1, 2):
            neighbor = (index + step) % cluster_size
            adjacency[index, neighbor] = 1
            adjacency[neighbor, index] = 1
    chord_probability = min(0.25, 2.0 / cluster_size)
    for left in range(cluster_size):
        for right in range(left + 1, cluster_size):
            if adjacency[left, right] == 0 and rng.random() < chord_probability:
                adjacency[left, right] = adjacency[right, left] = 1
    return adjacency


def make_population(
    cluster_count: int,
    cluster_size: int,
    scenario: Scenario,
    rng: np.random.Generator,
) -> ClusterPopulation:
    """Draw a finite population once; repeated experiments only rerandomize assignments."""
    if cluster_count < 1 or cluster_size < 3:
        raise ValueError("cluster_count must be positive and cluster_size must be at least three")
    if scenario not in {"linear", "complex"}:
        raise ValueError(f"unsupported scenario: {scenario}")

    if scenario == "linear":
        complete = np.ones((cluster_size, cluster_size), dtype=np.int8) - np.eye(
            cluster_size, dtype=np.int8
        )
        adjacency = np.repeat(complete[None, :, :], cluster_count, axis=0)
        baseline = rng.uniform(-0.04, 0.04, size=(cluster_count, cluster_size))
    else:
        adjacency = np.stack(
            [_ring_with_random_chords(cluster_size, rng) for _ in range(cluster_count)]
        )
        baseline = rng.normal(0.0, 0.20, size=(cluster_count, cluster_size))
    return ClusterPopulation(adjacency=adjacency, baseline=baseline, scenario=scenario)


def network_exposures(
    population: ClusterPopulation,
    treatment_by_cluster: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return one-hop, strict two-hop, and all-peer treated fractions."""
    treatment = np.asarray(treatment_by_cluster, dtype=float)
    expected_shape = (population.cluster_count, population.cluster_size)
    if treatment.shape != expected_shape:
        raise ValueError(f"treatment_by_cluster must have shape {expected_shape}")

    adjacency = population.adjacency.astype(float)
    one_hop_degree = adjacency.sum(axis=2)
    one_hop = np.einsum("cij,cj->ci", adjacency, treatment) / one_hop_degree

    path_length_two = np.einsum("cik,ckj->cij", adjacency, adjacency) > 0
    identity = np.eye(population.cluster_size, dtype=bool)[None, :, :]
    two_hop_adjacency = path_length_two & (adjacency == 0) & ~identity
    two_hop_degree = two_hop_adjacency.sum(axis=2)
    two_hop_total = np.einsum("cij,cj->ci", two_hop_adjacency, treatment)
    global_peer = (treatment.sum(axis=1, keepdims=True) - treatment) / (
        population.cluster_size - 1
    )
    two_hop = np.divide(
        two_hop_total,
        two_hop_degree,
        out=global_peer.copy(),
        where=two_hop_degree > 0,
    )
    return one_hop, two_hop, global_peer


def potential_outcomes(
    population: ClusterPopulation,
    treatment_by_cluster: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the fixed potential-outcome surface at a treatment matrix."""
    treatment = np.asarray(treatment_by_cluster, dtype=float)
    one_hop, two_hop, _ = network_exposures(population, treatment)
    if population.scenario == "linear":
        outcomes = (
            0.28
            + population.baseline
            + 0.12 * treatment
            + 0.22 * one_hop
            + 0.06 * treatment * one_hop
        )
    else:
        # This nonlinear scalar is the latent Degree of Interference in the simulation.
        latent_doi = 0.55 * one_hop + 0.25 * two_hop + 0.20 * (one_hop > 0.45)
        linear_predictor = (
            -1.15
            + population.baseline
            + 0.55 * treatment
            + 1.10 * latent_doi
            + 0.35 * treatment * (latent_doi - 0.5)
            + 0.25 * np.sin(2.0 * np.pi * one_hop)
        )
        outcomes = expit(linear_predictor)
    if np.any((outcomes < 0.0) | (outcomes > 1.0)):
        raise RuntimeError("the potential-outcome surface left the declared [0, 1] range")
    return outcomes


def observed_outcomes(
    population: ClusterPopulation,
    assignment: Assignment,
) -> NDArray[np.float64]:
    """Evaluate outcomes under the realized public treatment vector."""
    expected_count = population.cluster_count * population.cluster_size
    if len(assignment.treatment) != expected_count:
        raise ValueError("assignment size does not match the finite population")
    treatment = np.asarray(assignment.treatment, dtype=float).reshape(
        population.cluster_count, population.cluster_size
    )
    return potential_outcomes(population, treatment).ravel()


def _complex_cell_mean(
    population: ClusterPopulation,
    treatment_probability: float,
    own_treatment: int,
) -> float:
    cluster_size = population.cluster_size
    if cluster_size > 10:
        raise ValueError("exact complex-surface truth is limited to blocks of size at most ten")
    all_cluster_means = []
    for cluster_index in range(population.cluster_count):
        unit_means = []
        for unit_index in range(cluster_size):
            other_indices = [index for index in range(cluster_size) if index != unit_index]
            configurations = np.asarray(list(product((0, 1), repeat=cluster_size - 1)), dtype=float)
            assignments = np.zeros((len(configurations), cluster_size), dtype=float)
            assignments[:, other_indices] = configurations
            assignments[:, unit_index] = own_treatment
            treated_counts = configurations.sum(axis=1)
            weights = treatment_probability**treated_counts * (
                1.0 - treatment_probability
            ) ** (cluster_size - 1 - treated_counts)

            replicated_adjacency = np.repeat(
                population.adjacency[cluster_index : cluster_index + 1],
                len(assignments),
                axis=0,
            )
            replicated_baseline = np.repeat(
                population.baseline[cluster_index : cluster_index + 1],
                len(assignments),
                axis=0,
            )
            replicated_population = ClusterPopulation(
                replicated_adjacency,
                replicated_baseline,
                population.scenario,
            )
            evaluated = potential_outcomes(replicated_population, assignments)[:, unit_index]
            unit_means.append(float(np.dot(weights, evaluated)))
        all_cluster_means.append(float(np.mean(unit_means)))
    return float(np.mean(all_cluster_means))


def policy_truth(
    population: ClusterPopulation,
    p_low: float,
    p_high: float,
) -> dict[str, float]:
    """Return exact saturation-specific direct and policy-spillover estimands."""
    if not 0.0 < p_low < p_high < 1.0:
        raise ValueError("p_low and p_high must satisfy 0 < p_low < p_high < 1")
    if population.scenario == "linear":
        cell_means = {
            (saturation, own): float(
                0.28
                + population.baseline.mean()
                + 0.12 * own
                + 0.22 * probability
                + 0.06 * own * probability
            )
            for saturation, probability in ((0, p_low), (1, p_high))
            for own in (0, 1)
        }
    else:
        cell_means = {
            (saturation, own): _complex_cell_mean(population, probability, own)
            for saturation, probability in ((0, p_low), (1, p_high))
            for own in (0, 1)
        }
    return {
        "direct_low": cell_means[0, 1] - cell_means[0, 0],
        "direct_high": cell_means[1, 1] - cell_means[1, 0],
        "spillover_control": cell_means[1, 0] - cell_means[0, 0],
        "spillover_treated": cell_means[1, 1] - cell_means[0, 1],
    }

