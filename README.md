# Locally private causal inference with unknown within-block interference

This anonymous research artifact studies randomized experiments in which outcomes are locally
private and treatment may interfere arbitrarily within known independent blocks. It accompanies
the manuscript in [`output/pdf/locally_private_unknown_interference.pdf`](output/pdf/locally_private_unknown_interference.pdf).

The project makes three deliberately separate contributions:

1. A one-bit outcome mechanism and block-level Horvitz--Thompson estimator for
   saturation-specific direct effects and stochastic-policy spillover effects.
2. A matched bounded minimax rate that separates independent-block replication from
   person-level privacy, plus an oracle lower bound for rare exposure cells.
3. An exact privatized-bit observation layer for latent Degree-of-Interference (DoI) models,
   with a model-dependent finite-feature sensitivity analysis.

The primary estimator is design-based. The DoI-feature regressions are point-estimation
ablations and do not supply identification or causal confidence intervals.

## Scope and privacy contract

| Item | Assumption |
|---|---|
| Randomization | Bernoulli block saturation, then Bernoulli treatment within block |
| Interference | Arbitrary within known blocks; no cross-block interference |
| Private field | Each bounded outcome in `[0, 1]` |
| Public fields | Treatment, saturation arm, block membership, covariates, network summaries |
| Local protocol | One noninteractive epsilon-LDP bit per participant |
| Main targets | Two saturation-specific direct effects and two policy-spillover effects |

Outcome-only LDP limits leakage from a participant's own message. It is not network or group
privacy and does not prevent inference from correlated neighbors' messages.

## Quick start

Python 3.11 is the reference environment. The package supports Python 3.10 or newer.

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
make quick
~~~

`make quick` runs a deterministic smoke-test grid and writes only under
`tmp/quick_artifact`. It does not overwrite the full reported results.

## Reproduce the paper

The full Monte Carlo grid is computationally heavier:

~~~bash
make reproduce
make verify
~~~

`make reproduce` runs every reported experiment from
[`configs/experiments.json`](configs/experiments.json), regenerates all figures and the LaTeX
table, compiles the manuscript, and writes result metadata and SHA-256 checksums. A TeX
installation with `latexmk` is required for the manuscript target.

To reuse the checked-in raw results without rerunning Monte Carlo:

~~~bash
make figures
make table
make paper
PYTHONPATH=src python scripts/finalize_results.py
make verify
~~~

All random seeds are explicit. Optimizer status, gradient norms, and coefficient norms are
stored for the model-based experiment. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the
artifact contract and expected result sizes.

## Repository map

- `paper/` -- anonymous LaTeX manuscript and generated table.
- `src/lpui/` -- privacy mechanisms, randomization, estimators, DGPs, theory diagnostics, and
  DoI-feature likelihood code.
- `tests/` -- unit, exact-enumeration, gradient, and end-to-end Monte Carlo tests.
- `scripts/` -- experiment, figure, table, provenance, and verification entry points.
- `configs/` -- the single versioned experiment grid.
- `results/raw/` and `results/summary/` -- reported Monte Carlo outputs.
- `figures/` -- publication PDF and PNG figures.
- `references/verified_references.bib` -- 56 verified bibliography entries.
- `research_notes/literature_verification.md` -- source-by-source web verification ledger.

## Core API

~~~python
from lpui import estimate_policy_effects, privatize_one_bit, project_policy_effects

released_bits = privatize_one_bit(outcomes, epsilon=1.0, rng=rng)
unprojected = estimate_policy_effects(
    released_bits,
    assignment,
    p_low=0.2,
    p_high=0.8,
    epsilon=1.0,
    release_kind="one_bit",
)
bounded_loss_estimates = project_policy_effects(unprojected)
~~~

The unprojected estimator is exactly unbiased. Projection to `[-1, 1]` is for bounded
squared-error comparisons and sacrifices exact unbiasedness.

## Citation and license

The repository is anonymous for review. Citation metadata are in
[`CITATION.cff`](CITATION.cff). Code and manuscript sources are released under the MIT License.
