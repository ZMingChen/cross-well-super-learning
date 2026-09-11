import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROLLING_WINDOWS = [3, 5, 9]
MAIN_CURVES = ["GR", "AC", "CAL", "SP", "RILD", "RILM", "RFOC"]
OPTIONAL_CURVES = ["CNL", "DEN"]
MISSING_RATE_KEEP = float(os.environ.get("FEATURE_MISSING_RATE_KEEP", "0.40"))


def clean_curve_values(data):
    work = data.copy()
    numeric_cols = [c for c in work.columns if c not in {"well", "source_file"}]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work.loc[work[col] <= -900, col] = np.nan

    for col in ["AC", "CAL", "RILD", "RILM", "RFOC", "DEN"]:
        if col in work.columns:
            work.loc[work[col] <= 0, col] = np.nan
    if "GR" in work.columns:
        work.loc[work["GR"] <= 0, "GR"] = np.nan
    if "AC" in work.columns:
        work.loc[work["AC"] < 20, "AC"] = np.nan
    if "CAL" in work.columns:
        work.loc[work["CAL"] < 4, "CAL"] = np.nan
    if "SP" in work.columns:
        work.loc[work["SP"].abs() < 1e-6, "SP"] = np.nan
    for col in ["RILD", "RILM", "RFOC"]:
        if col in work.columns:
            work.loc[work[col] < 0.02, col] = np.nan
    if "DEN" in work.columns:
        work.loc[work["DEN"] < 1.0, "DEN"] = np.nan
    if "CNL" in work.columns:
        work.loc[work["CNL"] < -5, "CNL"] = np.nan
    return work


def rolling_slope(values, depths, window):
    half = window // 2
    y1 = values.shift(-half)
    y0 = values.shift(half)
    x1 = depths.shift(-half)
    x0 = depths.shift(half)
    denom = (x1 - x0).replace(0, np.nan)
    return (y1 - y0) / denom


def add_curve_features(part, curve):
    values = pd.to_numeric(part[curve], errors="coerce")
    depths = pd.to_numeric(part["DEPTH"], errors="coerce")
    out = {}
    out[curve] = values
    out[f"missing_{curve}"] = values.isna().astype(float)
    out[f"{curve}_diff1"] = values.diff() / depths.diff().replace(0, np.nan)
    out[f"{curve}_absdiff1"] = out[f"{curve}_diff1"].abs()
    for window in ROLLING_WINDOWS:
        min_periods = max(2, window // 2)
        roll = values.rolling(window=window, center=True, min_periods=min_periods)
        out[f"{curve}_mean_{window}m"] = roll.mean()
        out[f"{curve}_std_{window}m"] = roll.std()
        out[f"{curve}_range_{window}m"] = roll.max() - roll.min()
        out[f"{curve}_slope_{window}m"] = rolling_slope(values, depths, window)
        out[f"{curve}_roughness_{window}m"] = out[f"{curve}_absdiff1"].rolling(
            window=window, center=True, min_periods=min_periods
        ).mean()
    return out


def add_resistivity_features(features):
    eps = 1e-6
    for curve in ["RILD", "RILM", "RFOC"]:
        if curve in features:
            values = features[curve]
            features[f"log10_{curve}"] = np.log10(values.where(values > 0))
    if "RILD" in features and "RILM" in features:
        features["log10_RILD_RILM_ratio"] = np.log10(
            features["RILD"].clip(lower=eps) / features["RILM"].clip(lower=eps)
        )
    if "RILD" in features and "RFOC" in features:
        features["log10_RILD_RFOC_ratio"] = np.log10(
            features["RILD"].clip(lower=eps) / features["RFOC"].clip(lower=eps)
        )
    if "RILM" in features and "RFOC" in features:
        features["log10_RILM_RFOC_ratio"] = np.log10(
            features["RILM"].clip(lower=eps) / features["RFOC"].clip(lower=eps)
        )


def add_combination_features(features):
    if "GR" in features and "AC" in features:
        features["GR_AC_product"] = features["GR"] * features["AC"]
    if "GR" in features and "CNL" in features:
        features["GR_CNL_product"] = features["GR"] * features["CNL"]
    if "AC" in features and "CNL" in features:
        features["AC_CNL_product"] = features["AC"] * features["CNL"]
    if "CAL" in features and "GR" in features:
        features["CAL_GR_product"] = features["CAL"] * features["GR"]
    if "GR_mean_9m" in features and "GR_std_9m" in features:
        denom = features["GR_std_9m"].replace(0, np.nan)
        features["GR_local_z_9m"] = (features["GR"] - features["GR_mean_9m"]) / denom
    if "GR_range_9m" in features and "GR_mean_9m" in features:
        features["GR_serration_index_9m"] = features["GR_range_9m"] / features["GR_mean_9m"].abs().replace(0, np.nan)


def make_features_for_well(part, curve_cols):
    part = part.sort_values("DEPTH").reset_index(drop=True)
    features = {
        "well": part["well"],
        "DEPTH": part["DEPTH"],
        "source_file": part["source_file"],
    }
    for curve in curve_cols:
        features.update(add_curve_features(part, curve))
    add_resistivity_features(features)
    add_combination_features(features)
    return pd.DataFrame(features)


def build_feature_table(data):
    available = [c for c in MAIN_CURVES + OPTIONAL_CURVES if c in data.columns]
    parts = []
    for _, part in data.groupby("well", sort=True):
        parts.append(make_features_for_well(part, available))
    features = pd.concat(parts, ignore_index=True, sort=False)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, available


def choose_feature_columns(features):
    meta_cols = {"well", "DEPTH", "source_file"}
    numeric_cols = [c for c in features.columns if c not in meta_cols]
    missing = features[numeric_cols].isna().mean().sort_values()
    keep = [c for c in numeric_cols if missing[c] <= MISSING_RATE_KEEP]
    return keep, missing


def make_cluster_ready(features, feature_cols):
    work = features[["well", "DEPTH", "source_file"] + feature_cols].copy()
    medians = work[feature_cols].median(numeric_only=True)
    work[feature_cols] = work[feature_cols].fillna(medians)
    return work, medians


def main():
    parser = argparse.ArgumentParser(description="Construct multi-window log features.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    data = clean_curve_values(data)
    data = data.dropna(subset=["well", "DEPTH"]).sort_values(["well", "DEPTH"]).reset_index(drop=True)

    features, available_curves = build_feature_table(data)
    feature_cols, missing = choose_feature_columns(features)
    cluster_ready, medians = make_cluster_ready(features, feature_cols)

    features_path = out_dir / "electrofacies_features_1m.csv"
    cluster_path = out_dir / "electrofacies_clustering_features_1m.csv"
    feature_cols_path = out_dir / "feature_columns_for_clustering.csv"
    summary_stats_path = out_dir / "feature_missing_summary.csv"

    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    cluster_ready.to_csv(cluster_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(feature_cols_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "feature": missing.index,
            "missing_rate": missing.values,
            "selected_for_clustering": [name in feature_cols for name in missing.index],
            "median_impute_value": [float(medians.get(name, np.nan)) if name in medians.index else np.nan for name in missing.index],
        }
    ).to_csv(summary_stats_path, index=False, encoding="utf-8-sig")

    summary = {
        "input": str(input_path),
        "rows": int(len(features)),
        "available_curves": available_curves,
        "rolling_windows_m": ROLLING_WINDOWS,
        "feature_columns_total": int(len([c for c in features.columns if c not in {"well", "DEPTH", "source_file"}])),
        "feature_columns_selected_for_clustering": int(len(feature_cols)),
        "missing_rate_keep_threshold": MISSING_RATE_KEEP,
        "outputs": {
            "electrofacies_features_1m": str(features_path),
            "electrofacies_clustering_features_1m": str(cluster_path),
            "feature_columns_for_clustering": str(feature_cols_path),
            "feature_missing_summary": str(summary_stats_path),
        },
    }
    (out_dir / "step2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
