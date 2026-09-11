import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


from . import step21_pipeline as sl


def make_key(frame):
    return frame["well"].astype(str) + "|" + pd.to_numeric(frame["DEPTH"], errors="coerce").round(3).map(
        lambda value: f"{value:.3f}"
    )


def extract_unlabeled_features(features, target_path, full_feature_path):
    target = pd.read_csv(
        target_path,
        usecols=["well", "DEPTH", "human_code"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    unlabeled = target[pd.to_numeric(target["human_code"], errors="coerce").isna()].copy()
    unlabeled["join_key"] = make_key(unlabeled)
    unlabeled = unlabeled.drop_duplicates("join_key")
    wanted = set(unlabeled["join_key"])
    chunks = []
    usecols = ["well", "DEPTH"] + features
    for chunk in pd.read_csv(
        full_feature_path,
        usecols=usecols,
        encoding="utf-8-sig",
        low_memory=False,
        chunksize=30000,
    ):
        chunk["join_key"] = make_key(chunk)
        selected = chunk[chunk["join_key"].isin(wanted)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise RuntimeError("No unlabeled target rows matched the full feature table.")
    feature_rows = pd.concat(chunks, ignore_index=True).drop_duplicates("join_key")
    merged = unlabeled[["join_key", "well", "DEPTH"]].merge(
        feature_rows.drop(columns=["well", "DEPTH"]), on="join_key", how="left", validate="one_to_one"
    )
    missing = merged[features].isna().all(axis=1)
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} unlabeled target rows could not be matched to features.")
    return merged


def choose_final_method(summary):
    candidates = [
        "CalibratedSuperLearner",
        "CBClasswiseSL",
        "MacroRiskSL",
        "ClasswiseSL",
        "BalancedConvexSL",
        "ConvexSL",
        "SoftVoting",
    ]
    eligible = summary[summary["model"].isin(candidates)].copy()
    if eligible.empty:
        return "CalibratedSuperLearner"
    return str(eligible.sort_values("macro_f1_mean", ascending=False).iloc[0]["model"])


def confidence_table(y, proba):
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    class_labels = np.arange(proba.shape[1])
    rows = []
    for threshold in np.arange(0.50, 0.991, 0.01):
        selected = confidence >= threshold
        selected_counts = np.bincount(y[selected], minlength=len(class_labels)) if selected.any() else np.zeros(len(class_labels), dtype=int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "points": int(selected.sum()),
                "coverage": float(selected.mean()),
                "accuracy": float(accuracy_score(y[selected], pred[selected])) if selected.any() else math.nan,
                "macro_f1": (
                    float(
                        f1_score(
                            y[selected],
                            pred[selected],
                            labels=class_labels,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if selected.any()
                    else math.nan
                ),
                "classes_present": int(np.count_nonzero(selected_counts)),
                "minimum_class_points": int(selected_counts.min()) if selected.any() else 0,
            }
        )
    return pd.DataFrame(rows)


def choose_confidence_threshold(table):
    acceptable = table[(table["accuracy"] >= 0.95) & (table["coverage"] >= 0.10)]
    if acceptable.empty:
        return 0.90
    return float(acceptable.sort_values("threshold").iloc[0]["threshold"])


def build_intervals(points):
    work = points.sort_values(["well", "DEPTH"]).copy()
    same_well = work["well"].astype(str).eq(work["well"].astype(str).shift())
    depth_gap = pd.to_numeric(work["DEPTH"], errors="coerce").diff().abs()
    same_code = work["pred_code"].eq(work["pred_code"].shift())
    same_priority = work["review_priority"].eq(work["review_priority"].shift())
    new_interval = ~(same_well & same_code & same_priority & depth_gap.le(1.6))
    work["interval_id"] = new_interval.cumsum()
    intervals = (
        work.groupby("interval_id", as_index=False)
        .agg(
            well=("well", "first"),
            top_depth=("DEPTH", "min"),
            bottom_depth=("DEPTH", "max"),
            pred_code=("pred_code", "first"),
            pred_lithology=("pred_lithology", "first"),
            points=("DEPTH", "size"),
            mean_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
            mean_model_agreement=("model_agreement", "mean"),
            review_priority=("review_priority", "first"),
        )
        .sort_values(["well", "top_depth"])
    )
    return intervals


def main():
    parser = argparse.ArgumentParser(description="Fit the manuscript ensemble and annotate unlabelled intervals.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--target-labels", type=Path, required=True)
    parser.add_argument("--full-features", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--imputation", choices=["strict", "historical"], default="strict")
    args = parser.parse_args()
    OUT_DIR = args.output.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    human, features = sl.load_human_data(args.input, args.features)
    unlabeled = extract_unlabeled_features(features, args.target_labels, args.full_features)
    x_train, x_unlabeled = sl.prepare_matrix(human, unlabeled, features)
    y_train = sl.encode_labels(human["final_code"])
    groups = human["well"].astype(str).to_numpy()
    inner, inner_seed = sl.build_inner_splits(y_train, groups, args.inner_folds, 2026)
    specs = sl.model_library(2026, fast=args.fast, jobs=args.jobs)
    oof_blocks = []
    unlabeled_blocks = []
    names = []

    for model_idx, spec in enumerate(specs):
        oof = np.zeros((len(human), len(sl.CLASS_CODES)), dtype=float)
        for inner_id, (fit_idx, valid_idx) in enumerate(inner, start=1):
            fit_x, valid_x = sl.inner_matrices(human, features, x_train, fit_idx, valid_idx, args.imputation)
            model = sl.fit_model(spec, fit_x, y_train[fit_idx], 2026 + model_idx * 1000 + inner_id)
            oof[valid_idx] = sl.aligned_proba(model, valid_x)
        model = sl.fit_model(spec, x_train, y_train, 2026 + model_idx * 1000 + 999)
        unlabeled_proba = sl.aligned_proba(model, x_unlabeled)
        oof_blocks.append(oof)
        unlabeled_blocks.append(unlabeled_proba)
        names.append(spec.name)
        print(f"[step21-final] fitted {spec.name}", flush=True)

    oof_stack = np.stack(oof_blocks, axis=1)
    unlabeled_stack = np.stack(unlabeled_blocks, axis=1)
    outputs = {name: unlabeled_stack[:, idx, :] for idx, name in enumerate(names)}
    oof_outputs = {name: oof_stack[:, idx, :] for idx, name in enumerate(names)}
    outputs["SoftVoting"] = np.mean(unlabeled_stack, axis=1)
    oof_outputs["SoftVoting"] = np.mean(oof_stack, axis=1)

    convex_w, _, _ = sl.global_convex_weights(oof_stack, y_train, balanced=False)
    outputs["ConvexSL"] = sl.mix_global(unlabeled_stack, convex_w)
    oof_outputs["ConvexSL"] = sl.mix_global(oof_stack, convex_w)
    convex_bias, _ = sl.fit_class_bias(y_train, oof_outputs["ConvexSL"])
    outputs["CalibratedSuperLearner"] = sl.apply_class_bias(outputs["ConvexSL"], convex_bias)
    oof_outputs["CalibratedSuperLearner"] = sl.apply_class_bias(oof_outputs["ConvexSL"], convex_bias)

    balanced_w, _, _ = sl.global_convex_weights(oof_stack, y_train, balanced=True)
    outputs["BalancedConvexSL"] = sl.mix_global(unlabeled_stack, balanced_w)
    oof_outputs["BalancedConvexSL"] = sl.mix_global(oof_stack, balanced_w)

    class_w, _, _ = sl.classwise_convex_weights(oof_stack, y_train)
    classwise_unlabeled = sl.mix_classwise(unlabeled_stack, class_w)
    classwise_oof = sl.mix_classwise(oof_stack, class_w)
    outputs["ClasswiseSL"] = classwise_unlabeled
    oof_outputs["ClasswiseSL"] = classwise_oof
    class_bias, _ = sl.fit_class_bias(y_train, classwise_oof)
    outputs["CBClasswiseSL"] = sl.apply_class_bias(classwise_unlabeled, class_bias)
    oof_outputs["CBClasswiseSL"] = sl.apply_class_bias(classwise_oof, class_bias)

    macro_w, macro_bias, _, _ = sl.macro_risk_weights(oof_stack, y_train, convex_w)
    outputs["MacroRiskSL"] = sl.apply_class_bias(sl.mix_global(unlabeled_stack, macro_w), macro_bias)
    oof_outputs["MacroRiskSL"] = sl.apply_class_bias(sl.mix_global(oof_stack, macro_w), macro_bias)

    summary = pd.read_csv(args.summary)
    selected_method = choose_final_method(summary)
    proba = outputs[selected_method]
    oof_proba = oof_outputs[selected_method]
    coverage = confidence_table(y_train, oof_proba)
    threshold = choose_confidence_threshold(coverage)

    prediction = unlabeled[["well", "DEPTH"]].copy()
    encoded = np.argmax(proba, axis=1)
    prediction["pred_code"] = [sl.CLASS_CODES[idx] for idx in encoded]
    prediction["pred_lithology"] = prediction["pred_code"].map(sl.CLASS_NAMES)
    prediction["confidence"] = np.max(proba, axis=1)
    prediction["entropy"] = -np.sum(proba * np.log(np.clip(proba, sl.EPS, 1.0)), axis=1)
    base_pred = np.argmax(unlabeled_stack, axis=2)
    prediction["model_agreement"] = [np.bincount(row, minlength=len(sl.CLASS_CODES)).max() / len(names) for row in base_pred]
    for idx, code in enumerate(sl.CLASS_CODES):
        prediction[f"prob_code_{code}"] = proba[:, idx]
    prediction["review_priority"] = "routine_review"
    active = (
        (prediction["confidence"] < 0.70)
        | (prediction["model_agreement"] < 0.75)
        | prediction["pred_code"].eq(4)
    )
    quick = (prediction["confidence"] >= threshold) & (prediction["model_agreement"] >= 0.75) & ~active
    prediction.loc[active, "review_priority"] = "active_review"
    prediction.loc[quick, "review_priority"] = "quick_confirm_candidate"
    intervals = build_intervals(prediction)

    prediction.to_csv(OUT_DIR / "step21_unlabeled_intelligent_annotation_points.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(OUT_DIR / "step21_unlabeled_intelligent_annotation_intervals.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT_DIR / "step21_oof_confidence_coverage.csv", index=False, encoding="utf-8-sig")
    weights = pd.DataFrame(
        {
            "model": names,
            "convex_weight": convex_w,
            "balanced_convex_weight": balanced_w,
            "macro_risk_weight": macro_w,
            "classwise_weight_code_0": class_w[0],
            "classwise_weight_code_3": class_w[1],
            "classwise_weight_code_4": class_w[2],
        }
    )
    weights.to_csv(OUT_DIR / "step21_final_meta_weights.csv", index=False, encoding="utf-8-sig")
    status = (
        prediction.groupby(["review_priority", "pred_code", "pred_lithology"], as_index=False)
        .agg(points=("DEPTH", "size"), wells=("well", "nunique"), mean_confidence=("confidence", "mean"))
    )
    status.to_csv(OUT_DIR / "step21_annotation_status_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "selected_method": selected_method,
        "selection_basis": "highest mean Macro-F1 among prespecified ensemble candidates in repeated grouped outer validation",
        "human_training_points": len(human),
        "human_training_wells": int(human["well"].nunique()),
        "unlabeled_target_points": len(prediction),
        "unlabeled_target_wells": int(prediction["well"].nunique()),
        "inner_group_folds": len(inner),
        "inner_seed": inner_seed,
        "oof_confidence_threshold_for_95pct_accuracy": threshold,
        "review_priority_counts": prediction["review_priority"].value_counts().to_dict(),
        "predicted_class_counts": prediction["pred_lithology"].value_counts().to_dict(),
        "guardrail": "Predictions for unlabeled points are candidate annotations, not independent geological truth.",
    }
    (OUT_DIR / "step21_final_application_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
