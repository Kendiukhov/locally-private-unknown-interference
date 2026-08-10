#!/usr/bin/env python3
"""Run the oracle exact-exposure overlap study from the main experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lpui.simulation import run_exposure_overlap_monte_carlo, summarize_exposure_overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        settings = json.load(stream)["exposure_overlap"]
    raw = run_exposure_overlap_monte_carlo(
        sample_size=settings["sample_size"],
        degrees=settings["degrees"],
        peer_treatment_probability=settings["peer_treatment_probability"],
        target_mean=settings["target_mean"],
        epsilons=settings["epsilons"],
        replications=settings["replications"],
        seed=settings["experiment_seed"],
    )
    raw_path = args.output / "raw" / "exposure_overlap.csv"
    summary_path = args.output / "summary" / "exposure_overlap.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False, float_format="%.10g")
    summarize_exposure_overlap(raw).to_csv(summary_path, index=False, float_format="%.10g")
    print(f"wrote {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
