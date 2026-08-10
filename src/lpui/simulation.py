"""Deterministic Monte Carlo orchestration and uncertainty summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from lpui.design import draw_two_stage_assignment
from lpui.dgp import ClusterPopulation, observed_outcomes
from lpui.doi_model import (
    build_features,
    fit_nonprivate_doi_sieve,
    fit_private_doi_sieve,
    gcompute_policy_effects,
)
from lpui.estimators import EffectEstimate, estimate_policy_effects
from lpui.mechanisms import (
    privatize_laplace,
    privatize_one_bit,
    randomized_response_baseline,
)


def _append_effect_rows(
    rows: list[dict[str, float | int | str | bool]],
    fitted: dict[str, EffectEstimate],
    truth: dict[str, float],
    population: ClusterPopulation,
    epsilon: float,
    replicate: int,
    method: str,
) -> None:
    for effect_name, estimate in fitted.items():
        target = truth[effect_name]
        rows.append(
            {
                "scenario": population.scenario,
                "cluster_count": population.cluster_count,
                "cluster_size": population.cluster_size,
                "sample_size": population.cluster_count * population.cluster_size,
                "epsilon": float(epsilon),
                "replicate": replicate,
                "method": method,
                "effect": effect_name,
                "truth": target,
                "estimate": estimate.estimate,
                "standard_error": estimate.standard_error,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "error": estimate.estimate - target,
                "covered": estimate.ci_low <= target <= estimate.ci_high,
            }
        )


def run_design_monte_carlo(
    population: ClusterPopulation,
    truth: dict[str, float],
    *,
    p_low: float,
    p_high: float,
    epsilons: Iterable[float],
    replications: int,
    seed: int,
    arm_probability: float = 0.5,
) -> pd.DataFrame:
    """Evaluate private estimators, privacy baselines, and a nonprivate benchmark."""
    epsilons = [float(epsilon) for epsilon in epsilons]
    if replications < 1 or not epsilons:
        raise ValueError("replications and epsilons must both be nonempty")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    for replicate in range(replications):
        assignment = draw_two_stage_assignment(
            population.cluster_count,
            population.cluster_size,
            p_low,
            p_high,
            arm_probability,
            rng,
        )
        outcomes = observed_outcomes(population, assignment)
        nonprivate = estimate_policy_effects(
            outcomes,
            assignment,
            p_low,
            p_high,
            arm_probability,
            release_kind="outcome",
        )
        for epsilon in epsilons:
            releases = privatize_one_bit(outcomes, epsilon, rng)
            one_bit = estimate_policy_effects(
                releases,
                assignment,
                p_low,
                p_high,
                arm_probability,
                epsilon=epsilon,
                release_kind="one_bit",
            )
            naive_bit = estimate_policy_effects(
                releases,
                assignment,
                p_low,
                p_high,
                arm_probability,
                release_kind="naive_bit",
            )
            laplace_releases = privatize_laplace(outcomes, epsilon, rng)
            laplace = estimate_policy_effects(
                laplace_releases,
                assignment,
                p_low,
                p_high,
                arm_probability,
                release_kind="laplace",
            )
            for method, fitted in (
                ("nonprivate", nonprivate),
                ("one_bit", one_bit),
                ("laplace", laplace),
                ("naive_bit", naive_bit),
            ):
                _append_effect_rows(
                    rows,
                    fitted,
                    truth,
                    population,
                    epsilon,
                    replicate,
                    method,
                )
    return pd.DataFrame(rows)


def summarize_monte_carlo(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute point-risk and interval diagnostics with Monte Carlo uncertainty."""
    required = {
        "scenario",
        "cluster_count",
        "cluster_size",
        "sample_size",
        "epsilon",
        "method",
        "effect",
        "truth",
        "estimate",
        "error",
        "standard_error",
        "ci_low",
        "ci_high",
        "covered",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw results are missing columns: {sorted(missing)}")

    grouping = [
        "scenario",
        "cluster_count",
        "cluster_size",
        "sample_size",
        "epsilon",
        "method",
        "effect",
    ]

    def summarize_group(group: pd.DataFrame) -> pd.Series:
        errors = group["error"].to_numpy(dtype=float)
        squared_errors = errors**2
        rmse = float(np.sqrt(np.mean(squared_errors)))
        truth = group["truth"].to_numpy(dtype=float)
        estimates = group["estimate"].to_numpy(dtype=float)
        projected_errors = np.clip(estimates, -1.0, 1.0) - truth
        projected_squared_errors = projected_errors**2
        projected_rmse = float(np.sqrt(np.mean(projected_squared_errors)))
        replication_count = len(group)
        if replication_count > 1 and rmse > 0.0:
            mcse_rmse = float(
                np.std(squared_errors, ddof=1)
                / np.sqrt(replication_count)
                / (2.0 * rmse)
            )
        else:
            mcse_rmse = 0.0
        if replication_count > 1 and projected_rmse > 0.0:
            mcse_projected_rmse = float(
                np.std(projected_squared_errors, ddof=1)
                / np.sqrt(replication_count)
                / (2.0 * projected_rmse)
            )
        else:
            mcse_projected_rmse = 0.0
        coverage = float(group["covered"].mean())
        z_score = 1.959963984540054
        wilson_denominator = 1.0 + z_score**2 / replication_count
        wilson_center = (
            coverage + z_score**2 / (2.0 * replication_count)
        ) / wilson_denominator
        wilson_half_width = (
            z_score
            * np.sqrt(
                coverage * (1.0 - coverage) / replication_count
                + z_score**2 / (4.0 * replication_count**2)
            )
            / wilson_denominator
        )
        empirical_sd = float(np.std(estimates, ddof=1)) if replication_count > 1 else 0.0
        average_standard_error = float(group["standard_error"].mean())
        interval_lengths = (
            group["ci_high"].to_numpy(dtype=float)
            - group["ci_low"].to_numpy(dtype=float)
        )
        clipped_interval_lengths = (
            np.clip(group["ci_high"].to_numpy(dtype=float), -1.0, 1.0)
            - np.clip(group["ci_low"].to_numpy(dtype=float), -1.0, 1.0)
        )
        return pd.Series(
            {
                "replications": replication_count,
                "bias": float(np.mean(errors)),
                "mcse_bias": float(np.std(errors, ddof=1) / np.sqrt(replication_count))
                if replication_count > 1
                else 0.0,
                "rmse": rmse,
                "mcse_rmse": mcse_rmse,
                "projected_rmse": projected_rmse,
                "mcse_projected_rmse": mcse_projected_rmse,
                "outside_parameter_space": float(np.mean(np.abs(estimates) > 1.0)),
                "empirical_standard_deviation": empirical_sd,
                "average_standard_error": average_standard_error,
                "standard_error_ratio": average_standard_error / empirical_sd
                if empirical_sd > 0.0
                else np.nan,
                "coverage": coverage,
                "mcse_coverage": float(
                    np.sqrt(coverage * (1.0 - coverage) / replication_count)
                ),
                "coverage_low": wilson_center - wilson_half_width,
                "coverage_high": wilson_center + wilson_half_width,
                "average_interval_length": float(np.mean(interval_lengths)),
                "average_projected_interval_length": float(
                    np.mean(clipped_interval_lengths)
                ),
            }
        )

    return raw.groupby(grouping, observed=True, sort=True).apply(
        summarize_group, include_groups=False
    ).reset_index()


def run_doi_monte_carlo(
    population: ClusterPopulation,
    truth: dict[str, float],
    *,
    p_low: float,
    p_high: float,
    epsilons: Iterable[float],
    replications: int,
    seed: int,
    arm_probability: float = 0.5,
    penalty: float = 0.05,
    integration_draws: int = 32,
) -> pd.DataFrame:
    """Compare design HT with explicitly model-dependent DoI-feature regressions."""
    epsilons = [float(epsilon) for epsilon in epsilons]
    if replications < 1 or not epsilons:
        raise ValueError("replications and epsilons must both be nonempty")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    for replicate in range(replications):
        assignment = draw_two_stage_assignment(
            population.cluster_count,
            population.cluster_size,
            p_low,
            p_high,
            arm_probability,
            rng,
        )
        outcomes = observed_outcomes(population, assignment)
        integration_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        generic_features = build_features(
            population, assignment.treatment, "generic"
        )
        nonprivate_fit = fit_nonprivate_doi_sieve(
            generic_features,
            outcomes,
            penalty=penalty,
        )
        nonprivate_estimates = gcompute_policy_effects(
            population,
            nonprivate_fit.coefficients,
            "generic",
            p_low,
            p_high,
            np.random.default_rng(integration_seed),
            draws=integration_draws,
        )
        for epsilon in epsilons:
            releases = privatize_one_bit(outcomes, epsilon, rng)
            design_fit = estimate_policy_effects(
                releases,
                assignment,
                p_low,
                p_high,
                arm_probability,
                epsilon=epsilon,
                release_kind="one_bit",
            )
            for effect_name, estimate in design_fit.items():
                rows.append(
                    {
                        "scenario": population.scenario,
                        "cluster_count": population.cluster_count,
                        "cluster_size": population.cluster_size,
                        "sample_size": population.cluster_count * population.cluster_size,
                        "epsilon": epsilon,
                        "replicate": replicate,
                        "method": "design_ht",
                        "effect": effect_name,
                        "truth": truth[effect_name],
                        "estimate": estimate.estimate,
                        "error": estimate.estimate - truth[effect_name],
                        "fit_success": True,
                        "iterations": 0,
                        "gradient_norm": 0.0,
                        "coefficient_norm": 0.0,
                    }
                )

            for effect_name, estimate in nonprivate_estimates.items():
                rows.append(
                    {
                        "scenario": population.scenario,
                        "cluster_count": population.cluster_count,
                        "cluster_size": population.cluster_size,
                        "sample_size": population.cluster_count * population.cluster_size,
                        "epsilon": epsilon,
                        "replicate": replicate,
                        "method": "doi_nonprivate_generic",
                        "effect": effect_name,
                        "truth": truth[effect_name],
                        "estimate": estimate,
                        "error": estimate - truth[effect_name],
                        "fit_success": nonprivate_fit.success,
                        "iterations": nonprivate_fit.iterations,
                        "gradient_norm": nonprivate_fit.gradient_norm,
                        "coefficient_norm": float(
                            np.linalg.norm(nonprivate_fit.coefficients)
                        ),
                    }
                )

            for feature_set, method in (
                ("generic", "doi_private_generic"),
                ("no_interference", "doi_private_no_interference"),
            ):
                features = build_features(
                    population, assignment.treatment, feature_set
                )
                fitted = fit_private_doi_sieve(
                    features,
                    releases,
                    epsilon=epsilon,
                    penalty=penalty,
                )
                estimates = gcompute_policy_effects(
                    population,
                    fitted.coefficients,
                    feature_set,
                    p_low,
                    p_high,
                    np.random.default_rng(integration_seed),
                    draws=integration_draws,
                )
                for effect_name, estimate in estimates.items():
                    rows.append(
                        {
                            "scenario": population.scenario,
                            "cluster_count": population.cluster_count,
                            "cluster_size": population.cluster_size,
                            "sample_size": population.cluster_count * population.cluster_size,
                            "epsilon": epsilon,
                            "replicate": replicate,
                            "method": method,
                            "effect": effect_name,
                            "truth": truth[effect_name],
                            "estimate": estimate,
                            "error": estimate - truth[effect_name],
                            "fit_success": fitted.success,
                            "iterations": fitted.iterations,
                            "gradient_norm": fitted.gradient_norm,
                            "coefficient_norm": float(
                                np.linalg.norm(fitted.coefficients)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_point_monte_carlo(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize point-estimation experiments that do not share an interval method."""
    grouping = [
        "scenario",
        "cluster_count",
        "cluster_size",
        "sample_size",
        "epsilon",
        "method",
        "effect",
    ]

    def summarize_group(group: pd.DataFrame) -> pd.Series:
        errors = group["error"].to_numpy(dtype=float)
        squared_errors = errors**2
        rmse = float(np.sqrt(np.mean(squared_errors)))
        replication_count = len(group)
        return pd.Series(
            {
                "replications": replication_count,
                "bias": float(np.mean(errors)),
                "mcse_bias": float(
                    np.std(errors, ddof=1) / np.sqrt(replication_count)
                )
                if replication_count > 1
                else 0.0,
                "rmse": rmse,
                "mcse_rmse": float(
                    np.std(squared_errors, ddof=1)
                    / np.sqrt(replication_count)
                    / (2.0 * rmse)
                )
                if replication_count > 1 and rmse > 0.0
                else 0.0,
                "fit_success_rate": float(group["fit_success"].mean()),
                "average_iterations": float(group["iterations"].mean()),
                "maximum_gradient_norm": float(group["gradient_norm"].max()),
                "average_coefficient_norm": float(group["coefficient_norm"].mean()),
            }
        )

    return raw.groupby(grouping, observed=True, sort=True).apply(
        summarize_group, include_groups=False
    ).reset_index()


def run_exposure_overlap_monte_carlo(
    *,
    sample_size: int,
    degrees: Iterable[int],
    peer_treatment_probability: float,
    target_mean: float,
    epsilons: Iterable[float],
    replications: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate an exact-exposure mean as its assignment probability decays with degree."""
    if sample_size < 1 or replications < 1:
        raise ValueError("sample_size and replications must be positive")
    if not 0.0 < peer_treatment_probability < 1.0:
        raise ValueError("peer_treatment_probability must lie in (0, 1)")
    if not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for degree in degrees:
        if degree < 0:
            raise ValueError("degrees must be nonnegative")
        exposure_probability = peer_treatment_probability**degree
        for epsilon in epsilons:
            attenuation = np.tanh(float(epsilon) / 2.0)
            randomized_response_flip = randomized_response_baseline(float(epsilon))
            for replicate in range(replications):
                exposed_count = int(rng.binomial(sample_size, exposure_probability))
                if exposed_count == 0:
                    nonprivate_estimate = one_bit_estimate = 0.0
                else:
                    outcome_ones = int(rng.binomial(exposed_count, target_mean))
                    released_ones = int(
                        rng.binomial(outcome_ones, 1.0 - randomized_response_flip)
                        + rng.binomial(
                            exposed_count - outcome_ones,
                            randomized_response_flip,
                        )
                    )
                    # Horvitz--Thompson normalization keeps the cell-probability cost explicit.
                    normalization = sample_size * exposure_probability
                    nonprivate_estimate = outcome_ones / normalization
                    one_bit_estimate = (
                        released_ones - randomized_response_flip * exposed_count
                    ) / (normalization * attenuation)
                for method, estimate in (
                    ("nonprivate", nonprivate_estimate),
                    ("one_bit", one_bit_estimate),
                ):
                    error = estimate - target_mean
                    rows.append(
                        {
                            "sample_size": sample_size,
                            "degree": degree,
                            "exposure_probability": exposure_probability,
                            "epsilon": float(epsilon),
                            "replicate": replicate,
                            "method": method,
                            "truth": target_mean,
                            "estimate": estimate,
                            "error": error,
                            "squared_error": error**2,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_exposure_overlap(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize the oracle exact-exposure experiment and its Monte Carlo error."""
    grouping = [
        "sample_size",
        "degree",
        "exposure_probability",
        "epsilon",
        "method",
    ]

    def summarize_group(group: pd.DataFrame) -> pd.Series:
        errors = group["error"].to_numpy(dtype=float)
        squared_errors = errors**2
        mse = float(np.mean(squared_errors))
        return pd.Series(
            {
                "replications": len(group),
                "truth": float(group["truth"].iloc[0]),
                "bias": float(np.mean(errors)),
                "mse": mse,
                "mcse_mse": float(np.std(squared_errors, ddof=1) / np.sqrt(len(group))),
            }
        )

    return raw.groupby(grouping, observed=True, sort=True).apply(
        summarize_group, include_groups=False
    ).reset_index()
