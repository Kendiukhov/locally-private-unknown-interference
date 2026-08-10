# Reproducibility contract

## Reference environment

The reported artifact was produced with Python 3.11.9 and the exact packages in
`requirements-lock.txt`. Runtime versions and hashes of executable source files are recorded in
`results/run_metadata.json`. `results/checksums.json` covers exactly the ten reported raw and
summary CSV files and excludes filesystem metadata such as AppleDouble files.

Matplotlib PDF bytes and final optimizer iterates can vary across operating systems or library
builds. Numerical result CSVs are deterministic under the reference environment and explicit
seeds in `configs/experiments.json`.

## Expected result grid

| Study | Raw rows | Summary rows |
|---|---:|---:|
| Nonlinear privacy benchmark | 64,000 | 80 |
| Sample-size scaling | 76,800 | 192 |
| Fixed-N block-size scaling | 76,800 | 192 |
| Oracle exposure overlap | 72,000 | 36 |
| DoI-feature sensitivity | 7,200 | 48 |

The DoI study has four methods: design HT, generic features with private bits, the same generic
features with raw outcomes, and a private no-interference regression. It reports point error and
optimizer diagnostics, not causal interval coverage.

## Commands

Run the complete experiment and manuscript pipeline:

~~~bash
make reproduce
~~~

Run tests only:

~~~bash
python -m pytest -q
~~~

Rebuild presentation artifacts from stored summaries:

~~~bash
make figures
make table
make paper
~~~

Finalize and verify stored outputs:

~~~bash
PYTHONPATH=src python scripts/finalize_results.py
PYTHONPATH=src python scripts/verify_artifact.py
~~~

The final verification checks every stored CSV digest and requires at least 30 unique citations
in the compiled manuscript auxiliary file. The manuscript currently cites 49 verified sources;
the bibliography contains 56 verified entries.

## Independent checks in the test suite

- exact endpoint likelihood ratios and decoder unbiasedness for the one-bit channel;
- numerical stability at very large privacy budgets;
- exact HT expectations for an arbitrary lookup-table interference surface;
- malformed two-stage assignment rejection;
- finite-difference verification of the private-likelihood gradient;
- exact moment calculation for the oracle exposure-cell HT risk;
- bounded projection and capped minimax-rate checks;
- deterministic end-to-end Monte Carlo smoke tests.

## Reference verification

Every bibliography record was checked on 2026-08-10 against a DOI endpoint or an official
JMLR/PMLR page. The verification URL and relevance of all 56 records appear in
`research_notes/literature_verification.md`. Downloaded third-party PDFs used during development
are intentionally excluded from the release.
