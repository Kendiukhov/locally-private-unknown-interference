import numpy as np

from lpui.design import draw_two_stage_assignment
from lpui.dgp import make_population, observed_outcomes, policy_truth


def test_linear_population_has_analytic_policy_effects():
    rng = np.random.default_rng(18)
    population = make_population(12, 6, "linear", rng)
    truth = policy_truth(population, p_low=0.2, p_high=0.8)

    assert np.isclose(truth["direct_low"], 0.12 + 0.06 * 0.2)
    assert np.isclose(truth["direct_high"], 0.12 + 0.06 * 0.8)
    assert np.isclose(truth["spillover_control"], 0.22 * 0.6)
    assert np.isclose(truth["spillover_treated"], 0.28 * 0.6)


def test_complex_population_outcomes_are_bounded_and_reproducible():
    population_rng = np.random.default_rng(31)
    assignment_rng = np.random.default_rng(32)
    population = make_population(20, 8, "complex", population_rng)
    assignment = draw_two_stage_assignment(20, 8, 0.25, 0.75, 0.5, assignment_rng)
    first = observed_outcomes(population, assignment)
    second = observed_outcomes(population, assignment)

    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert np.std(first) > 0.05


def test_complex_truth_is_computed_exactly_for_small_blocks():
    rng = np.random.default_rng(49)
    population = make_population(4, 5, "complex", rng)
    truth = policy_truth(population, p_low=0.3, p_high=0.7)

    assert set(truth) == {
        "direct_low",
        "direct_high",
        "spillover_control",
        "spillover_treated",
    }
    assert all(np.isfinite(value) for value in truth.values())
    assert truth["spillover_control"] > 0

