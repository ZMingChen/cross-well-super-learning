# Local verification: 2026-09-11

Scope: synthetic demonstration only, not manuscript reproduction or submission approval.

- Windows launcher invoked successfully from the parent directory.
- Existing Python 3.12 virtual environment: pip check passed for the demo dependency set.
- Regression tests: 11 passed; pytest cache disabled; CSV validation uses in-memory input.
- Demo: 18 synthetic rows, 3 wells, mean macro-F1 1.0 and balanced accuracy 1.0. These are not manuscript results.
- Full-algorithm smoke test: completed one outer fold on synthetic data with all eight manuscript learners (LogisticRegression, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM, CatBoost, and MLP); this is not a manuscript result.
- Repository heuristic audit passed; this is not proof that all confidential content is absent.
- Python compilation and git diff whitespace checks passed.
- GitHub Actions configuration added for Python 3.12 on Windows and Linux; not remotely executed.
- The recovered Python 3.10 environment contains the pinned eight-learner stack listed in `requirements-paper.txt`. The restricted package index cannot install XGBoost into the current Python 3.12 environment, so a clean full-pipeline installation was not claimed.
- No dependency vulnerability audit, lint, type checking, or coverage measurement performed in this update.

Corrections: fold-local median imputation, explicit class probability alignment, rejection of fractional class labels, and validation of insufficient groups/classes or entirely missing training features.

Remaining: controlled-data manuscript reproduction, preprocessing and figure/table reproduction, author confirmation of license ownership, public repository URL and release verification.

Three temporary pytest-cache-files directories were left by an initial permission-related test failure. Cleanup was blocked by the approval service; their contents were not inspected. They are ignored by Git. The final tests do not use these directories.
