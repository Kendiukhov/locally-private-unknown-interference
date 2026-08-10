import numpy as np

from lpui.design import draw_two_stage_assignment
from lpui.dgp import make_population, observed_outcomes
from lpui.doi_model import (
    _private_log_loss_gradient,
    build_features,
    fit_nonprivate_doi_sieve,
    fit_private_doi_sieve,
    gcompute_policy_effects,
    private_log_loss,
    predict_mean,
)
from lpui.mechanisms import privatize_one_bit


def test_private_doi_likelihood_optimizes_the_observed_release_model():
    rng = np.random.default_rng(202)
    population = make_population(80, 8, "complex", rng)
    assignment = draw_two_stage_assignment(80, 8, 0.2, 0.8, 0.5, rng)
    outcomes = observed_outcomes(population, assignment)
    releases = privatize_one_bit(outcomes, epsilon=1.5, rng=rng)
    features = build_features(population, assignment.treatment, feature_set="generic")
    initial_loss = private_log_loss(np.zeros(features.shape[1]), features, releases, 1.5, 1e-3)
    fitted = fit_private_doi_sieve(features, releases, epsilon=1.5, penalty=1e-3)

    assert fitted.success
    assert fitted.objective < initial_loss
    predictions = predict_mean(features, fitted.coefficients)
    assert np.all((predictions > 0.0) & (predictions < 1.0))


def test_generic_feature_dictionary_strictly_contains_no_interference_dictionary():
    rng = np.random.default_rng(203)
    population = make_population(5, 8, "complex", rng)
    treatment = rng.binomial(1, 0.5, size=40)
    generic = build_features(population, treatment, feature_set="generic")
    no_interference = build_features(
        population, treatment, feature_set="no_interference"
    )

    assert generic.shape[0] == no_interference.shape[0] == 40
    assert generic.shape[1] > no_interference.shape[1]


def test_g_computation_returns_all_policy_contrasts():
    rng = np.random.default_rng(204)
    population = make_population(20, 6, "complex", rng)
    coefficient_count = build_features(
        population,
        rng.binomial(1, 0.5, size=120),
        feature_set="generic",
    ).shape[1]
    coefficients = np.zeros(coefficient_count)
    effects = gcompute_policy_effects(
        population,
        coefficients,
        feature_set="generic",
        p_low=0.2,
        p_high=0.8,
        rng=rng,
        draws=20,
    )

    assert set(effects) == {
        "direct_low",
        "direct_high",
        "spillover_control",
        "spillover_treated",
    }
    assert np.allclose(list(effects.values()), 0.0)


def test_private_likelihood_gradient_matches_finite_differences():
    rng = np.random.default_rng(205)
    features = rng.normal(size=(35, 5))
    releases = rng.binomial(1, 0.5, size=35).astype(float)
    coefficients = rng.normal(scale=0.2, size=5)
    epsilon = 0.8
    penalty = 0.07
    step = 1e-6
    numerical = np.empty_like(coefficients)
    for index in range(len(coefficients)):
        direction = np.zeros_like(coefficients)
        direction[index] = step
        numerical[index] = (
            private_log_loss(
                coefficients + direction, features, releases, epsilon, penalty
            )
            - private_log_loss(
                coefficients - direction, features, releases, epsilon, penalty
            )
        ) / (2.0 * step)
    analytic = _private_log_loss_gradient(
        coefficients, features, releases, epsilon, penalty
    )
    assert np.allclose(analytic, numerical, rtol=2e-5, atol=2e-6)


def test_nonprivate_fit_uses_the_same_bounded_mean_scale():
    rng = np.random.default_rng(206)
    features = np.column_stack([np.ones(200), rng.normal(size=200)])
    outcomes = 1.0 / (1.0 + np.exp(-(0.2 + 0.7 * features[:, 1])))
    fitted = fit_nonprivate_doi_sieve(features, outcomes, penalty=0.0)

    assert fitted.success
    assert np.allclose(fitted.coefficients, [0.2, 0.7], atol=2e-4)


def test_own_treatment_intervention_preserves_sampled_peer_exposures():
    rng = np.random.default_rng(207)
    population = make_population(4, 6, "complex", rng)
    treatment = rng.binomial(1, 0.5, size=24)
    control = build_features(
        population, treatment, "generic", own_treatment=0
    )
    treated = build_features(
        population, treatment, "generic", own_treatment=1
    )

    assert np.all(control[:, 1] == 0.0)
    assert np.all(treated[:, 1] == 1.0)
    assert np.allclose(control[:, 2:8], treated[:, 2:8])
