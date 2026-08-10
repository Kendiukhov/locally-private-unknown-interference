from itertools import product

import numpy as np

from lpui.design import draw_two_stage_assignment
from lpui.design import Assignment
from lpui.estimators import estimate_policy_effects, project_policy_effects
from lpui.mechanisms import privatize_one_bit


def _simple_potential_outcomes(cluster_id, treatment):
    """A bounded surface with a known direct and policy-spillover contrast."""
    cluster_size = np.bincount(cluster_id)[0]
    treated_by_cluster = np.bincount(cluster_id, weights=treatment)
    peer_fraction = (treated_by_cluster[cluster_id] - treatment) / (cluster_size - 1)
    return 0.25 + 0.15 * treatment + 0.30 * peer_fraction


def test_policy_estimators_are_unbiased_under_arbitrary_peer_dependence():
    rng = np.random.default_rng(23)
    cluster_count = 800
    cluster_size = 6
    p_low, p_high = 0.2, 0.8
    estimates = []

    for _ in range(350):
        assignment = draw_two_stage_assignment(
            cluster_count=cluster_count,
            cluster_size=cluster_size,
            p_low=p_low,
            p_high=p_high,
            arm_probability=0.5,
            rng=rng,
        )
        outcomes = _simple_potential_outcomes(
            assignment.cluster_id,
            assignment.treatment,
        )
        fitted = estimate_policy_effects(
            outcomes,
            assignment,
            p_low=p_low,
            p_high=p_high,
            arm_probability=0.5,
            release_kind="outcome",
        )
        estimates.append([fitted[name].estimate for name in fitted])

    names = list(fitted)
    mean_estimate = dict(zip(names, np.mean(estimates, axis=0), strict=True))
    assert np.isclose(mean_estimate["direct_low"], 0.15, atol=0.015)
    assert np.isclose(mean_estimate["direct_high"], 0.15, atol=0.015)
    assert np.isclose(mean_estimate["spillover_control"], 0.18, atol=0.015)
    assert np.isclose(mean_estimate["spillover_treated"], 0.18, atol=0.015)


def test_private_estimator_returns_finite_cluster_robust_intervals():
    rng = np.random.default_rng(91)
    assignment = draw_two_stage_assignment(
        cluster_count=120,
        cluster_size=8,
        p_low=0.25,
        p_high=0.75,
        arm_probability=0.5,
        rng=rng,
    )
    outcomes = _simple_potential_outcomes(
        assignment.cluster_id,
        assignment.treatment,
    )
    releases = privatize_one_bit(outcomes, epsilon=1.0, rng=rng)
    fitted = estimate_policy_effects(
        releases,
        assignment,
        p_low=0.25,
        p_high=0.75,
        arm_probability=0.5,
        epsilon=1.0,
        release_kind="one_bit",
    )

    assert set(fitted) == {
        "direct_low",
        "direct_high",
        "spillover_control",
        "spillover_treated",
    }
    for effect in fitted.values():
        assert np.isfinite(effect.estimate)
        assert effect.standard_error > 0
        assert effect.ci_low < effect.estimate < effect.ci_high


def test_estimator_rejects_unequal_cluster_sizes():
    cluster_id = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    malformed = Assignment(
        cluster_id=cluster_id,
        saturation_arm=np.array([0, 0, 0, 1, 1], dtype=np.int8),
        treatment=np.array([0, 1, 0, 1, 0], dtype=np.int8),
    )
    with np.testing.assert_raises(ValueError):
        estimate_policy_effects(
            np.ones(5) * 0.5,
            malformed,
            p_low=0.2,
            p_high=0.8,
            arm_probability=0.5,
            release_kind="outcome",
        )


def test_estimator_rejects_arm_that_changes_within_cluster():
    malformed = Assignment(
        cluster_id=np.array([0, 0, 1, 1], dtype=np.int64),
        saturation_arm=np.array([0, 1, 1, 1], dtype=np.int8),
        treatment=np.array([0, 1, 0, 1], dtype=np.int8),
    )
    with np.testing.assert_raises_regex(ValueError, "constant within every cluster"):
        estimate_policy_effects(
            np.full(4, 0.5),
            malformed,
            p_low=0.2,
            p_high=0.8,
            release_kind="outcome",
        )


def test_projection_enforces_effect_range_and_shortens_intervals():
    rng = np.random.default_rng(92)
    assignment = draw_two_stage_assignment(30, 4, 0.2, 0.8, 0.5, rng)
    fitted = estimate_policy_effects(
        np.full(120, 20.0),
        assignment,
        p_low=0.2,
        p_high=0.8,
        release_kind="laplace",
    )
    projected = project_policy_effects(fitted)

    for name in fitted:
        assert -1.0 <= projected[name].estimate <= 1.0
        assert -1.0 <= projected[name].ci_low <= projected[name].ci_high <= 1.0
        assert projected[name].standard_error == fitted[name].standard_error


def test_ht_expectation_matches_arbitrary_lookup_table_surface_exactly():
    rng = np.random.default_rng(93)
    cluster_count, cluster_size = 2, 3
    p_low, p_high, arm_probability = 0.2, 0.8, 0.5
    # No exposure model is used: every unit has an arbitrary value at all 2^m assignments.
    potential_outcomes = rng.uniform(size=(cluster_count, cluster_size, 2**cluster_size))
    assignments = list(product((0, 1), repeat=cluster_size))

    expected_estimates = {
        name: 0.0
        for name in (
            "direct_low",
            "direct_high",
            "spillover_control",
            "spillover_treated",
        )
    }
    cluster_id = np.repeat(np.arange(cluster_count, dtype=np.int64), cluster_size)
    for arms in product((0, 1), repeat=cluster_count):
        arm_probability_mass = np.prod(
            [arm_probability if arm else 1.0 - arm_probability for arm in arms]
        )
        for first_assignment in assignments:
            for second_assignment in assignments:
                treatment_by_cluster = (first_assignment, second_assignment)
                probability_mass = arm_probability_mass
                outcomes = np.empty(cluster_count * cluster_size)
                treatment = np.empty(cluster_count * cluster_size, dtype=np.int8)
                for cluster, (arm, treatment_vector) in enumerate(
                    zip(arms, treatment_by_cluster, strict=True)
                ):
                    treatment_probability = p_high if arm else p_low
                    treated_count = sum(treatment_vector)
                    probability_mass *= treatment_probability**treated_count * (
                        1.0 - treatment_probability
                    ) ** (cluster_size - treated_count)
                    assignment_index = sum(
                        bit << index for index, bit in enumerate(treatment_vector)
                    )
                    start = cluster * cluster_size
                    stop = start + cluster_size
                    treatment[start:stop] = treatment_vector
                    outcomes[start:stop] = potential_outcomes[
                        cluster, :, assignment_index
                    ]
                public_assignment = Assignment(
                    cluster_id=cluster_id,
                    saturation_arm=np.repeat(np.asarray(arms, dtype=np.int8), cluster_size),
                    treatment=treatment,
                )
                fitted = estimate_policy_effects(
                    outcomes,
                    public_assignment,
                    p_low=p_low,
                    p_high=p_high,
                    arm_probability=arm_probability,
                    release_kind="outcome",
                )
                for name, estimate in fitted.items():
                    expected_estimates[name] += probability_mass * estimate.estimate

    cell_means: dict[tuple[int, int], float] = {}
    for saturation, treatment_probability in ((0, p_low), (1, p_high)):
        for own_treatment in (0, 1):
            unit_means = []
            for cluster in range(cluster_count):
                for unit in range(cluster_size):
                    peer_indices = [index for index in range(cluster_size) if index != unit]
                    unit_mean = 0.0
                    for peer_assignment in product((0, 1), repeat=cluster_size - 1):
                        treatment_vector = [0] * cluster_size
                        treatment_vector[unit] = own_treatment
                        for peer_index, peer_treatment in zip(
                            peer_indices, peer_assignment, strict=True
                        ):
                            treatment_vector[peer_index] = peer_treatment
                        treated_peers = sum(peer_assignment)
                        probability_mass = treatment_probability**treated_peers * (
                            1.0 - treatment_probability
                        ) ** (cluster_size - 1 - treated_peers)
                        assignment_index = sum(
                            bit << index for index, bit in enumerate(treatment_vector)
                        )
                        unit_mean += probability_mass * potential_outcomes[
                            cluster, unit, assignment_index
                        ]
                    unit_means.append(unit_mean)
            cell_means[saturation, own_treatment] = float(np.mean(unit_means))
    truth = {
        "direct_low": cell_means[0, 1] - cell_means[0, 0],
        "direct_high": cell_means[1, 1] - cell_means[1, 0],
        "spillover_control": cell_means[1, 0] - cell_means[0, 0],
        "spillover_treated": cell_means[1, 1] - cell_means[0, 1],
    }

    for name in truth:
        assert np.isclose(expected_estimates[name], truth[name], atol=2e-14)
