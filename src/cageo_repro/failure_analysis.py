import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CLASS_CODES = [0, 3, 4]
CLASS_NAMES = {0: "fine sandstone", 3: "mudstone", 4: "silty mudstone"}


def consecutive_segments(data):
    rows = []
    for well, group in data.sort_values(["well", "DEPTH"]).groupby("well", sort=False):
        depth = pd.to_numeric(group["DEPTH"], errors="coerce").to_numpy(dtype=float)
        code = group["final_code"].to_numpy(dtype=int)
        if not len(group):
            continue
        depth_gap = np.r_[True, (~np.isfinite(np.diff(depth))) | (np.diff(depth) > 1.6)]
        class_change = np.r_[True, code[1:] != code[:-1]]
        segment_id = np.cumsum(depth_gap | class_change)
        work = pd.DataFrame({"segment_id": segment_id, "depth": depth, "code": code})
        for _, segment in work.groupby("segment_id", sort=False):
            rows.append(
                {
                    "well": well,
                    "class_code": int(segment["code"].iloc[0]),
                    "points": len(segment),
                    "thickness_m": float(segment["depth"].max() - segment["depth"].min() + 1.0),
                }
            )
    return pd.DataFrame(rows)


def segment_summary(segments):
    rows = []
    for code, group in segments.groupby("class_code"):
        rows.append(
            {
                "class_code": int(code),
                "class_name": CLASS_NAMES[int(code)],
                "segments": len(group),
                "median_thickness_m": group["thickness_m"].median(),
                "q25_thickness_m": group["thickness_m"].quantile(0.25),
                "q75_thickness_m": group["thickness_m"].quantile(0.75),
                "thin_le_2m_fraction": float((group["thickness_m"] <= 2.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_redundancy(data, features):
    sample = data.sample(n=min(8000, len(data)), random_state=42)
    x = sample[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    variable = x.std(axis=0) > 1e-10
    x = x.loc[:, variable]
    corr = x.corr().abs().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(random_state=42).fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return {
        "sample_rows": len(sample),
        "nonconstant_features": x.shape[1],
        "median_absolute_feature_correlation": float(np.nanmedian(upper)),
        "fraction_abs_correlation_ge_0_90": float(np.nanmean(upper >= 0.90)),
        "pca_components_for_90pct_variance": int(np.searchsorted(cumulative, 0.90) + 1),
        "pca_components_for_95pct_variance": int(np.searchsorted(cumulative, 0.95) + 1),
    }


def class_and_well_summary(data):
    rows = []
    for code in CLASS_CODES:
        subset = data[data["final_code"].eq(code)]
        rows.append(
            {
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "points": len(subset),
                "fraction": len(subset) / len(data),
                "wells": int(subset["well"].nunique()),
            }
        )
    per_well = (
        data.groupby(["well", "final_code"]).size().unstack(fill_value=0).reindex(columns=CLASS_CODES, fill_value=0)
    )
    per_well_fraction = per_well.div(per_well.sum(axis=1), axis=0)
    well_stats = []
    for code in CLASS_CODES:
        well_stats.append(
            {
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "well_fraction_median": per_well_fraction[code].median(),
                "well_fraction_q25": per_well_fraction[code].quantile(0.25),
                "well_fraction_q75": per_well_fraction[code].quantile(0.75),
                "well_fraction_max": per_well_fraction[code].max(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(well_stats)


def model_diagnostics(result_dir):
    diagnostics = {}
    diversity_path = Path(result_dir) / "step21_base_learner_diversity.csv"
    summary_path = Path(result_dir) / "step21_model_summary.csv"
    if diversity_path.exists():
        diversity = pd.read_csv(diversity_path)
        diagnostics.update(
            {
                "mean_base_probability_correlation": float(diversity["probability_correlation"].mean()),
                "median_base_probability_correlation": float(diversity["probability_correlation"].median()),
                "mean_base_prediction_disagreement": float(diversity["prediction_disagreement"].mean()),
                "median_base_prediction_disagreement": float(diversity["prediction_disagreement"].median()),
            }
        )
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        diagnostics["completed_outer_folds"] = int(summary["n_outer_folds"].max())
        diagnostics["top_models"] = summary.head(8)[["model", "macro_f1_mean", "macro_f1_std"]].to_dict(
            orient="records"
        )
    return diagnostics


def main():
    import argparse
    from .step21_pipeline import load_human_data
    parser = argparse.ArgumentParser(description="Compute class, thickness and feature redundancy diagnostics.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, features = load_human_data(args.input, args.features)
    args.output.mkdir(parents=True, exist_ok=True)
    classes, wells = class_and_well_summary(data)
    segments = consecutive_segments(data)
    classes.to_csv(args.output / "class_imbalance_summary.csv", index=False)
    wells.to_csv(args.output / "well_distribution_shift_summary.csv", index=False)
    segment_summary(segments).to_csv(args.output / "segment_thickness_summary.csv", index=False)
    payload = {"feature_redundancy": feature_redundancy(data, features),
               "model_diversity": model_diagnostics(args.results)}
    (args.output / "failure_mode_statistics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
