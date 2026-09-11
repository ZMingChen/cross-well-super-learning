# Data schema for an authorized full-data run

The public demo uses only synthetic data. The controlled full-data run must use an approved local copy and must not commit it to Git.

| Field | Type | Meaning | Release status |
|---|---|---|---|
| `well` | string | Group identifier used for well-disjoint validation | confidential; never publish |
| `DEPTH` | float | Depth coordinate in metres | confidential; publish only if owner approves |
| `final_code` | integer | Verified class code (0, 3, or 4) | confidential; synthetic substitute is public |
| predictors | float | Every column listed in the separate `feature` CSV | share only after owner approval |
| `final_label_source` | string | Optional; only `human` rows are retained | confidential; never publish |

The manuscript run used 165 selected predictors. `feature_engineering.py` documents the original curve, derivative, rolling-window, resistivity-ratio and interaction transformations. Selection depends on missingness in the authorized dataset. No column containing source paths, well identifiers, raw depths, manual annotations, or point-level predictions may be released without written authorization.
