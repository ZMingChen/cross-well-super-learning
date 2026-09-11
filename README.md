# Grouped and Class-Calibrated Super Learning for Cross-Well Lithology Classification

This repository contains the migrated manuscript algorithm and non-geological synthetic demonstrations. It is not ready for journal submission until the clean-environment and public-release checks below are completed.

## What is included

- `src/cageo_repro/step21_pipeline.py`: eight learners, grouped nested validation, OOF predictions, constrained Super Learner weights, classwise alternatives, stacking, and class-offset calibration.
- `src/cageo_repro/final_application.py`: final OOF fitting, method selection, confidence analysis, and candidate annotation.
- `src/cageo_repro/las_preprocessing.py`, `feature_engineering.py`, `label_intervals.py`: LAS ingestion, 1 m resampling, multi-window features, and interval label matching.
- `src/cageo_repro/failure_analysis.py`: class imbalance, interval thickness, redundancy, and model-diversity diagnostics.
- `src/cageo_repro/demo_pipeline.py`: smaller three-model smoke test.
- `examples/synthetic_well_logs.csv`: synthetic, non-geological data with the same column roles used by the demonstration.
- `scripts/run_demo.py`: one-command entry point that writes metrics and predictions to `demo_output/`.
- `scripts/repository_audit.py`: release gate checking required documentation, executable entry points, and accidental sensitive-file inclusion.
- `requirements.txt`: exact direct pins for the small demo.
- `requirements-paper.txt`: versions recovered from the Python 3.10 environment used for the complete experiment.
- `tests/test_demo.py`: regression checks for deterministic output, training-only imputation, class alignment, and invalid inputs.
- `.github/workflows/demo.yml`: Windows/Linux demo checks on push or pull request after the repository is uploaded. Remote execution has not been verified.

The confidential LAS files, manual interpretation workbooks, well identifiers, raw depths, and point-level predictions are **not** included. They remain under the data owner's access restrictions.

## Quick start

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_demo.py
python scripts/repository_audit.py
python scripts/release_preflight.py
```

Expected demo outputs are `demo_output/metrics.json` and `demo_output/predictions.csv`. The demo uses synthetic wells only; its numerical results are not the manuscript's reported results.

## Reproducing the manuscript analysis

The full analysis requires the data owner's confidential well-log and manual-label files. An authorized researcher should:

1. obtain written permission for the data;
2. place only the authorized, local input files in a private data directory;
3. map those files to the schema documented in `docs/data_schema.md`;
4. run the analysis entry point supplied by the authors from a pinned environment;
5. retain raw data and point-level outputs locally and publish only aggregate metrics and non-sensitive figures.

No result from the synthetic demo should be presented as an independent geological validation.

## Run the paper algorithm

Install `requirements-paper.txt`, generate the synthetic full-pipeline input, then run:

```bash
python scripts/generate_paper_synthetic.py
python scripts/run_paper.py --input examples/synthetic_manuscript_input.csv --features examples/synthetic_feature_list.csv --output full_demo_output --outer-repeats 1 --outer-folds 3 --inner-folds 2 --max-outer 1 --fast --jobs 1
```

For the controlled paper run, replace the two example paths with authorized local files and use `--outer-repeats 2 --outer-folds 5 --inner-folds 3`. The output directory receives fold metrics, class metrics, OOF-fitted weights and biases, diversity statistics, predictions, audits, confidence intervals and the model-comparison figure.

The default `--imputation strict` refits medians within every inner training fold. Use `--imputation historical` only to reproduce the July 2026 record, which calculated outer-training medians once before inner OOF splitting. That behavior can make inner risk estimates optimistic, although held-out outer wells remain excluded.

## Data and code availability statement

The code is licensed under MIT (LICENSE). Public access has not been verified. Only synthetic data are included. Confidential field data and point-level manuscript outputs require the data owner's permission for redistribution.

## Reproducibility record

- Small demo: Python 3.12 on Windows.
- Full algorithm smoke test: Python 3.10 on Windows using the recovered eight-learner environment.
- Original result record: 42,033 human points, 93 wells, 165 features, 2 x 5 grouped outer folds, 3 grouped inner folds; CalibratedSuperLearner mean Macro-F1 0.682742.
- Random seed used by the demo: 42
- Primary validation unit in the manuscript: well (no samples from a test well enter its training fold)
- Manuscript metrics are based on the authors' controlled-data run, not on the synthetic demo.

## Citation

Use the manuscript citation after publication. Until acceptance, cite the repository URL and the exact release tag/commit supplied by the authors.

## Release checklist

Before submission, repeat the strict full-data experiment, compare it with the historical record, verify a clean installation and CI, confirm the license owner, verify public anonymous access, and update the Computer Code Availability section. Archive the exact release with a DOI if used in the manuscript.

On Windows, run `powershell -File .\run_local_demo.ps1` from this directory after installing the dependencies. It checks dependency consistency, runs regression tests, executes the demo, and audits repository files. The launcher also works when called by absolute path from another directory. This checks the local demonstration only, not submission readiness.

Missing feature values are imputed separately inside each training fold. Probability columns are aligned to the fixed class order (0, 3, 4). Macro-F1 includes all three classes, even if a held-out fold lacks a class. Inputs require at least three wells and at least two classes in each training fold. All-missing training features cause an explicit error.
