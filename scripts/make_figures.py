#!/usr/bin/env python3
"""Create all paper figures from stored Monte Carlo summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lpui.theory import oracle_exposure_ht_mse


COLORS = {
    "one_bit": "#0072B2",
    "laplace": "#D55E00",
    "naive_bit": "#999999",
    "nonprivate": "#009E73",
    "design_ht": "#0072B2",
    "doi_nonprivate_generic": "#333333",
    "doi_private_generic": "#009E73",
    "doi_private_no_interference": "#CC79A7",
}
LABELS = {
    "one_bit": "one-bit randomized response",
    "laplace": "Laplace LDP",
    "naive_bit": "raw private bit (not decoded)",
    "nonprivate": "nonprivate",
    "design_ht": "design-based LDP",
    "doi_nonprivate_generic": "generic features, nonprivate",
    "doi_private_generic": "generic features, LDP",
    "doi_private_no_interference": "no-interference model, LDP",
}
MARKERS = {
    "one_bit": "o",
    "laplace": "s",
    "naive_bit": "^",
    "nonprivate": "D",
    "design_ht": "o",
    "doi_nonprivate_generic": "D",
    "doi_private_generic": "s",
    "doi_private_no_interference": "^",
}
EFFECT_LABELS = {
    "direct_high": "Direct effect under high saturation",
    "spillover_control": "Spillover effect on controls",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/summary"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    return parser.parse_args()


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{suffix}")
    plt.close(fig)
    print(f"wrote {output / (stem + '.pdf')}")


def plot_privacy_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15), sharex="col")
    effects = ("direct_high", "spillover_control")
    methods = ("one_bit", "laplace", "naive_bit", "nonprivate")
    for column, effect in enumerate(effects):
        selected = summary[summary["effect"] == effect]
        for method in methods:
            values = selected[selected["method"] == method].sort_values("epsilon")
            axes[0, column].errorbar(
                values["epsilon"],
                values["rmse"],
                yerr=1.96 * values["mcse_rmse"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4,
                linewidth=1.4,
                capsize=2,
                label=LABELS[method],
            )
            coverage_lower = values["coverage"] - values.get(
                "coverage_low", values["coverage"]
            )
            coverage_upper = values.get(
                "coverage_high", values["coverage"]
            ) - values["coverage"]
            axes[1, column].errorbar(
                values["epsilon"],
                values["coverage"],
                yerr=np.vstack([coverage_lower, coverage_upper]),
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4,
                linewidth=1.4,
                capsize=2,
                label=LABELS[method],
            )
        axes[0, column].set_title(EFFECT_LABELS[effect])
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("RMSE" if column == 0 else "")
        axes[1, column].axhline(0.95, color="#333333", linestyle="--", linewidth=0.9)
        axes[1, column].set_ylim(0.0, 1.01)
        axes[1, column].set_ylabel("95% CI coverage" if column == 0 else "")
        axes[1, column].set_xlabel(r"Privacy budget $\epsilon$")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks(sorted(selected["epsilon"].unique()))
        axes[1, column].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.88, hspace=0.18, wspace=0.22)
    _save(fig, output, "privacy_tradeoff")


def plot_rate_scaling(
    sample_summary: pd.DataFrame,
    interference_summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.45))
    effect = "spillover_control"
    selected = sample_summary[
        (sample_summary["effect"] == effect) & (sample_summary["method"] == "one_bit")
    ]
    for epsilon, values in selected.groupby("epsilon", sort=True):
        values = values.sort_values("sample_size")
        signal_squared = np.tanh(float(epsilon) / 2.0) ** 2
        rate = 1.0 / values["cluster_count"] + 1.0 / (
            values["sample_size"] * signal_squared
        )
        normalized_mse = values["rmse"] ** 2 / rate
        normalized_mse_error = (
            2.0 * values["rmse"] * 1.96 * values["mcse_rmse"] / rate
        )
        axes[0].errorbar(
            values["sample_size"],
            normalized_mse,
            yerr=normalized_mse_error,
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2,
            label=rf"$\epsilon={epsilon:g}$",
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Total users $N$ (block size fixed)")
    axes[0].set_ylabel(r"MSE / $\{C^{-1}+(N a_\epsilon^2)^{-1}\}$")
    axes[0].set_title("Risk normalizes by the additive rate")
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.25, 0.995),
        bbox_transform=fig.transFigure,
        ncol=3,
    )

    chosen = interference_summary[
        (interference_summary["effect"] == effect)
        & (interference_summary["method"].isin(["one_bit", "nonprivate"]))
    ]
    for (method, epsilon), values in chosen.groupby(["method", "epsilon"], sort=True):
        if method == "nonprivate" and epsilon != chosen["epsilon"].max():
            continue
        values = values.sort_values("cluster_size")
        label = "nonprivate" if method == "nonprivate" else rf"one-bit, $\epsilon={epsilon:g}$"
        axes[1].errorbar(
            values["cluster_size"],
            values["rmse"],
            yerr=1.96 * values["mcse_rmse"],
            marker=MARKERS[method],
            markersize=4,
            linewidth=1.4,
            capsize=2,
            label=label,
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("Users per interference block $m$ ($N=1600$)")
    axes[1].set_ylabel("RMSE of spillover estimate")
    axes[1].set_title("Fewer randomized blocks raise the error floor")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.75, 0.995),
        bbox_transform=fig.transFigure,
        ncol=2,
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.70, bottom=0.18, wspace=0.35)
    _save(fig, output, "rate_scaling")


def plot_doi_sieve(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.65), sharex="col")
    methods = (
        "design_ht",
        "doi_nonprivate_generic",
        "doi_private_generic",
        "doi_private_no_interference",
    )
    for column, effect in enumerate(("direct_high", "spillover_control")):
        selected = summary[summary["effect"] == effect]
        for method in methods:
            values = selected[selected["method"] == method].sort_values("epsilon")
            axes[0, column].errorbar(
                values["epsilon"],
                values["rmse"],
                yerr=1.96 * values["mcse_rmse"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4,
                linewidth=1.4,
                capsize=2,
                label=LABELS[method],
            )
            axes[1, column].errorbar(
                values["epsilon"],
                values["bias"],
                yerr=1.96 * values["mcse_bias"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4,
                linewidth=1.4,
                capsize=2,
                label=LABELS[method],
            )
        axes[0, column].set_title(EFFECT_LABELS[effect])
        axes[0, column].set_ylabel("RMSE" if column == 0 else "")
        axes[1, column].axhline(0.0, color="#333333", linewidth=0.8)
        axes[1, column].set_ylabel("Bias" if column == 0 else "")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks(sorted(selected["epsilon"].unique()))
        axes[1, column].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        axes[1, column].set_xlabel(r"Privacy budget $\epsilon$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.82, hspace=0.18, wspace=0.22)
    _save(fig, output, "doi_sieve")


def plot_exposure_overlap(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.30))
    one_bit = summary[summary["method"] == "one_bit"]
    for color_index, (epsilon, values) in enumerate(
        one_bit.groupby("epsilon", sort=True)
    ):
        values = values.sort_values("degree")
        color = f"C{color_index}"
        axes[0].errorbar(
            values["degree"],
            values["mse"],
            yerr=1.96 * values["mcse_mse"],
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2,
            color=color,
            label=rf"one-bit, $\epsilon={epsilon:g}$",
        )
        analytic_mse = np.array(
            [
                oracle_exposure_ht_mse(
                    int(row.sample_size),
                    float(row.exposure_probability),
                    float(epsilon),
                    float(row.truth),
                )
                for row in values.itertuples()
            ]
        )
        axes[0].plot(
            values["degree"],
            analytic_mse,
            color=color,
            linestyle="--",
            linewidth=1.0,
        )
        normalized = (
            values["sample_size"]
            * values["exposure_probability"]
            * np.tanh(float(epsilon) / 2.0) ** 2
            * values["mse"]
        )
        normalized_error = (
            values["sample_size"]
            * values["exposure_probability"]
            * np.tanh(float(epsilon) / 2.0) ** 2
            * 1.96
            * values["mcse_mse"]
        )
        axes[1].errorbar(
            values["degree"],
            normalized,
            yerr=normalized_error,
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2,
            color=color,
            label=rf"$\epsilon={epsilon:g}$",
        )
        axes[1].plot(
            values["degree"],
            values["sample_size"]
            * values["exposure_probability"]
            * np.tanh(float(epsilon) / 2.0) ** 2
            * analytic_mse,
            color=color,
            linestyle="--",
            linewidth=1.0,
        )
    nonprivate = summary[
        (summary["method"] == "nonprivate")
        & (summary["epsilon"] == summary["epsilon"].max())
    ].sort_values("degree")
    axes[0].errorbar(
        nonprivate["degree"],
        nonprivate["mse"],
        yerr=1.96 * nonprivate["mcse_mse"],
        color="#333333",
        marker="D",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label="nonprivate",
    )
    axes[0].set_yscale("log", base=2)
    axes[0].set_xlabel("Peers in exact exposure $d$")
    axes[0].set_ylabel("MSE of exposure-cell mean")
    axes[0].set_title(r"Raw risk grows as $2^d$")
    axes[1].set_xlabel("Peers in exact exposure $d$")
    axes[1].set_ylabel(r"$N\rho a_\epsilon^2\,\mathrm{MSE}$")
    axes[1].set_title("Privacy--overlap normalization collapses")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.76, bottom=0.18, wspace=0.30)
    _save(fig, output, "exposure_overlap")


def main() -> None:
    args = _parse_args()
    _style()
    benchmark = pd.read_csv(args.results / "benchmark.csv")
    sample = pd.read_csv(args.results / "sample_size_scaling.csv")
    interference = pd.read_csv(args.results / "interference_scaling.csv")
    plot_privacy_tradeoff(benchmark, args.output)
    plot_rate_scaling(sample, interference, args.output)
    doi_path = args.results / "doi_sieve.csv"
    if doi_path.exists():
        plot_doi_sieve(pd.read_csv(doi_path), args.output)
    overlap_path = args.results / "exposure_overlap.csv"
    if overlap_path.exists():
        plot_exposure_overlap(pd.read_csv(overlap_path), args.output)


if __name__ == "__main__":
    main()
