#!/usr/bin/env python3
"""Run only the model-based DoI-sieve sensitivity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lpui.dgp import make_population, policy_truth
from lpui.simulation import run_doi_monte_carlo, summarize_point_monte_carlo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    settings = config["doi_sieve"]
    design = config["design"]
    population = make_population(
        settings["cluster_count"],
        settings["cluster_size"],
        settings["scenario"],
        np.random.default_rng(settings["population_seed"]),
    )
    truth = policy_truth(population, design["p_low"], design["p_high"])
    raw = run_doi_monte_carlo(
        population,
        truth,
        p_low=design["p_low"],
        p_high=design["p_high"],
        arm_probability=design["arm_probability"],
        epsilons=settings["epsilons"],
        replications=settings["replications"],
        seed=settings["experiment_seed"],
        penalty=settings["penalty"],
        integration_draws=settings["integration_draws"],
    )
    raw_path = args.output / "raw" / "doi_sieve.csv"
    summary_path = args.output / "summary" / "doi_sieve.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False, float_format="%.10g")
    summarize_point_monte_carlo(raw).to_csv(summary_path, index=False, float_format="%.10g")
    print(f"wrote {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()

