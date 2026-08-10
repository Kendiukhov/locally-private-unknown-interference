import pandas as pd

from lpui.dgp import make_population, policy_truth
from lpui.simulation import (
    run_exposure_overlap_monte_carlo,
    run_design_monte_carlo,
    run_doi_monte_carlo,
    summarize_exposure_overlap,
    summarize_monte_carlo,
    summarize_point_monte_carlo,
)


def test_small_monte_carlo_is_reproducible_and_well_formed():
    import numpy as np

    population = make_population(20, 5, "linear", np.random.default_rng(4))
    truth = policy_truth(population, 0.2, 0.8)
    first = run_design_monte_carlo(
        population,
        truth,
        p_low=0.2,
        p_high=0.8,
        epsilons=[0.5, 1.0],
        replications=4,
        seed=77,
    )
    second = run_design_monte_carlo(
        population,
        truth,
        p_low=0.2,
        p_high=0.8,
        epsilons=[0.5, 1.0],
        replications=4,
        seed=77,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4 * 2 * 4 * 4  # replications x epsilon x methods x effects
    assert first["estimate"].notna().all()
    assert set(first["method"]) == {"nonprivate", "one_bit", "laplace", "naive_bit"}


def test_monte_carlo_summary_reports_uncertainty_diagnostics():
    import numpy as np

    population = make_population(20, 5, "linear", np.random.default_rng(5))
    truth = policy_truth(population, 0.2, 0.8)
    raw = run_design_monte_carlo(
        population,
        truth,
        p_low=0.2,
        p_high=0.8,
        epsilons=[1.0],
        replications=5,
        seed=78,
    )
    summary = summarize_monte_carlo(raw)

    assert {
        "bias",
        "rmse",
        "projected_rmse",
        "coverage",
        "coverage_low",
        "coverage_high",
        "average_standard_error",
        "empirical_standard_deviation",
        "standard_error_ratio",
        "average_interval_length",
        "outside_parameter_space",
        "mcse_rmse",
    }.issubset(summary.columns)
    assert len(summary) == 4 * 4


def test_small_doi_monte_carlo_separates_design_and_model_based_estimators():
    import numpy as np

    population = make_population(25, 5, "complex", np.random.default_rng(6))
    truth = policy_truth(population, 0.2, 0.8)
    raw = run_doi_monte_carlo(
        population,
        truth,
        p_low=0.2,
        p_high=0.8,
        epsilons=[1.0],
        replications=2,
        seed=79,
        integration_draws=3,
    )
    summary = summarize_point_monte_carlo(raw)

    assert set(raw["method"]) == {
        "design_ht",
        "doi_nonprivate_generic",
        "doi_private_generic",
        "doi_private_no_interference",
    }
    assert len(raw) == 2 * 4 * 4
    assert len(summary) == 4 * 4
    assert {
        "fit_success_rate",
        "average_iterations",
        "maximum_gradient_norm",
        "average_coefficient_norm",
    }.issubset(summary.columns)


def test_exact_exposure_experiment_records_exponential_overlap_loss():
    raw = run_exposure_overlap_monte_carlo(
        sample_size=2_000,
        degrees=[0, 3],
        peer_treatment_probability=0.5,
        target_mean=0.6,
        epsilons=[1.0],
        replications=20,
        seed=80,
    )
    summary = summarize_exposure_overlap(raw)

    assert len(raw) == 2 * 1 * 20 * 2
    assert len(summary) == 2 * 1 * 2
    assert set(summary["exposure_probability"]) == {1.0, 0.125}
