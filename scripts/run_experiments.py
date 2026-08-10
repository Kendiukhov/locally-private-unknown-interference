#!/usr/bin/env python3
"""Run every experiment reported in the paper from one versioned configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lpui.dgp import make_population, policy_truth
from lpui.simulation import (
    run_design_monte_carlo,
    run_doi_monte_carlo,
    run_exposure_overlap_monte_carlo,
    summarize_exposure_overlap,
    summarize_monte_carlo,
    summarize_point_monte_carlo,
)
from lpui.reproducibility import write_reproducibility_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smoke-test grid with few replications and integration draws.",
    )
    parser.add_argument("--skip-doi", action="store_true", help="Skip the model-based study.")
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")
    print(f"wrote {path} ({len(frame):,} rows)", flush=True)


def _effective_configuration(config: dict[str, Any], quick: bool) -> dict[str, Any]:
    effective = json.loads(json.dumps(config))
    if quick:
        for name in ("benchmark", "sample_size_scaling", "interference_scaling"):
            effective[name]["replications"] = 8
        effective["doi_sieve"]["replications"] = 2
        effective["doi_sieve"]["integration_draws"] = 3
        effective["exposure_overlap"]["sample_size"] = 2_000
        effective["exposure_overlap"]["replications"] = 20
    return effective


def _run_design_study(
    study: dict[str, Any],
    design: dict[str, float],
    *,
    cluster_count: int,
    cluster_size: int,
    population_seed: int,
    experiment_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = make_population(
        cluster_count,
        cluster_size,
        study["scenario"],
        np.random.default_rng(population_seed),
    )
    truth = policy_truth(population, design["p_low"], design["p_high"])
    raw = run_design_monte_carlo(
        population,
        truth,
        p_low=design["p_low"],
        p_high=design["p_high"],
        arm_probability=design["arm_probability"],
        epsilons=study["epsilons"],
        replications=study["replications"],
        seed=experiment_seed,
    )
    return raw, summarize_monte_carlo(raw)


def main() -> None:
    args = _parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = _effective_configuration(json.load(stream), args.quick)
    output = args.output
    design = config["design"]

    benchmark = config["benchmark"]
    print("running nonlinear benchmark", flush=True)
    raw, summary = _run_design_study(
        benchmark,
        design,
        cluster_count=benchmark["cluster_count"],
        cluster_size=benchmark["cluster_size"],
        population_seed=benchmark["population_seed"],
        experiment_seed=benchmark["experiment_seed"],
    )
    _write_csv(raw, output / "raw" / "benchmark.csv")
    _write_csv(summary, output / "summary" / "benchmark.csv")

    scaling = config["sample_size_scaling"]
    scaling_raw = []
    for index, cluster_count in enumerate(scaling["cluster_counts"]):
        print(f"running sample-size study: C={cluster_count}", flush=True)
        raw, _ = _run_design_study(
            scaling,
            design,
            cluster_count=cluster_count,
            cluster_size=scaling["cluster_size"],
            population_seed=scaling["population_seed"] + index,
            experiment_seed=scaling["experiment_seed"] + index,
        )
        scaling_raw.append(raw)
    raw = pd.concat(scaling_raw, ignore_index=True)
    _write_csv(raw, output / "raw" / "sample_size_scaling.csv")
    _write_csv(
        summarize_monte_carlo(raw), output / "summary" / "sample_size_scaling.csv"
    )

    interference = config["interference_scaling"]
    interference_raw = []
    for index, cluster_size in enumerate(interference["cluster_sizes"]):
        cluster_count, remainder = divmod(interference["sample_size"], cluster_size)
        if remainder:
            raise ValueError("every cluster size must divide the fixed sample size")
        print(f"running interference study: C={cluster_count}, m={cluster_size}", flush=True)
        raw, _ = _run_design_study(
            interference,
            design,
            cluster_count=cluster_count,
            cluster_size=cluster_size,
            population_seed=interference["population_seed"] + index,
            experiment_seed=interference["experiment_seed"] + index,
        )
        interference_raw.append(raw)
    raw = pd.concat(interference_raw, ignore_index=True)
    _write_csv(raw, output / "raw" / "interference_scaling.csv")
    _write_csv(
        summarize_monte_carlo(raw), output / "summary" / "interference_scaling.csv"
    )

    if not args.skip_doi:
        doi = config["doi_sieve"]
        print("running model-based DoI-feature study", flush=True)
        population = make_population(
            doi["cluster_count"],
            doi["cluster_size"],
            doi["scenario"],
            np.random.default_rng(doi["population_seed"]),
        )
        truth = policy_truth(population, design["p_low"], design["p_high"])
        raw = run_doi_monte_carlo(
            population,
            truth,
            p_low=design["p_low"],
            p_high=design["p_high"],
            arm_probability=design["arm_probability"],
            epsilons=doi["epsilons"],
            replications=doi["replications"],
            seed=doi["experiment_seed"],
            penalty=doi["penalty"],
            integration_draws=doi["integration_draws"],
        )
        _write_csv(raw, output / "raw" / "doi_sieve.csv")
        _write_csv(
            summarize_point_monte_carlo(raw), output / "summary" / "doi_sieve.csv"
        )

    overlap = config["exposure_overlap"]
    print("running oracle exact-exposure study", flush=True)
    raw = run_exposure_overlap_monte_carlo(
        sample_size=overlap["sample_size"],
        degrees=overlap["degrees"],
        peer_treatment_probability=overlap["peer_treatment_probability"],
        target_mean=overlap["target_mean"],
        epsilons=overlap["epsilons"],
        replications=overlap["replications"],
        seed=overlap["experiment_seed"],
    )
    _write_csv(raw, output / "raw" / "exposure_overlap.csv")
    _write_csv(
        summarize_exposure_overlap(raw),
        output / "summary" / "exposure_overlap.csv",
    )

    metadata_path, checksum_path = write_reproducibility_files(
        output=output,
        project_root=Path(__file__).resolve().parents[1],
        config_path=args.config,
        configuration=config,
        quick=args.quick,
        skip_doi=args.skip_doi,
    )
    print(f"wrote {metadata_path} and {checksum_path}", flush=True)


if __name__ == "__main__":
    main()
