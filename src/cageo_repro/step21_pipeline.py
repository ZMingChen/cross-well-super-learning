import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t, wilcoxon
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


CLASS_CODES = [0, 3, 4]
CLASS_NAMES = {0: "fine sandstone", 3: "mudstone", 4: "silty mudstone"}
EPS = 1e-8


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    weight_power: float
    balanced_resample: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Re-audit and rebuild the grouped Super Learner experiment.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with well, DEPTH, final_code and predictors.")
    parser.add_argument("--features", type=Path, required=True, help="CSV containing a feature column.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated results.")
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-outer", type=int, default=0, help="Run only the first N outer folds; 0 runs all.")
    parser.add_argument("--force", action="store_true", help="Recompute completed outer folds.")
    parser.add_argument("--fast", action="store_true", help="Use smaller estimators for a direction check.")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel workers for supported learners; use 1 if restricted.")
    parser.add_argument("--imputation", choices=["strict", "historical"], default="strict")
    return parser.parse_args()


def model_library(seed, fast=False, require_all=True, jobs=-1):
    trees = 70 if fast else 180
    boosts = 80 if fast else 180
    mlp_iter = 120 if fast else 260
    specs = [
        ModelSpec(
            "LogisticRegression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=700, solver="lbfgs", random_state=seed),
            ),
            0.5,
        ),
        ModelSpec(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=trees,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=jobs,
                random_state=seed,
            ),
            0.0,
        ),
        ModelSpec(
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=trees,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=jobs,
                random_state=seed,
            ),
            0.0,
        ),
        ModelSpec(
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                max_iter=boosts,
                learning_rate=0.05,
                l2_regularization=0.1,
                random_state=seed,
            ),
            0.5,
        ),
    ]
    optional = {"xgboost": XGBClassifier, "lightgbm": LGBMClassifier, "catboost": CatBoostClassifier}
    missing = [name for name, estimator in optional.items() if estimator is None]
    if require_all and missing:
        raise RuntimeError(
            "The manuscript model library requires eight learners. Missing packages: " + ", ".join(missing)
        )
    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                "XGBoost",
                XGBClassifier(
                    n_estimators=boosts,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=3,
                    tree_method="hist",
                    n_jobs=jobs,
                    random_state=seed,
                ),
                0.5,
            )
        )
    if LGBMClassifier is not None:
        specs.append(
            ModelSpec(
                "LightGBM",
                LGBMClassifier(
                    n_estimators=boosts,
                    num_leaves=31,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="multiclass",
                    n_jobs=jobs,
                    random_state=seed,
                    verbosity=-1,
                ),
                0.5,
            )
        )
    if CatBoostClassifier is not None:
        specs.append(
            ModelSpec(
                "CatBoost",
                CatBoostClassifier(
                    iterations=boosts,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="MultiClass",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=jobs,
                ),
                0.5,
            )
        )
    specs.append(
        ModelSpec(
            "MLP",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(96, 48, 24),
                    alpha=1e-4,
                    early_stopping=True,
                    validation_fraction=0.1,
                    max_iter=mlp_iter,
                    random_state=seed,
                ),
            ),
            0.5,
            balanced_resample=True,
        )
    )
    return specs


def load_human_data(input_path, feature_path):
    usecols = pd.read_csv(input_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    feature_table = pd.read_csv(feature_path, encoding="utf-8-sig")
    if "feature" not in feature_table:
        raise ValueError("Feature-list CSV must contain a 'feature' column.")
    feature_names = feature_table["feature"].astype(str).tolist()
    required = ["well", "DEPTH", "final_code"]
    missing = sorted(set(required).difference(usecols))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    missing_features = sorted(set(feature_names).difference(usecols))
    if missing_features:
        raise ValueError(f"Input CSV is missing listed features: {missing_features}")
    optional = [column for column in ("final_lithology", "final_label_source") if column in usecols]
    selected = required + optional + feature_names
    data = pd.read_csv(input_path, usecols=selected, encoding="utf-8-sig", low_memory=False)
    if "final_label_source" in data:
        data = data[data["final_label_source"].astype(str).eq("human")].copy()
    codes = pd.to_numeric(data["final_code"], errors="raise")
    if codes.isna().any() or not np.isfinite(codes).all() or not (codes == np.floor(codes)).all():
        raise ValueError("Class codes must be finite integers")
    data["final_code"] = codes.astype(int)
    data = data[data["final_code"].isin(CLASS_CODES)].reset_index(drop=True)
    if data.empty:
        raise ValueError("No human-labelled rows with class codes 0, 3, or 4 were found.")
    if set(data["final_code"].unique()) != set(CLASS_CODES):
        raise ValueError("The analysis requires all three class codes: 0, 3, and 4.")
    if data["well"].isna().any():
        raise ValueError("Well group identifiers must not be missing.")
    return data, feature_names


def encode_labels(series):
    mapping = {code: idx for idx, code in enumerate(CLASS_CODES)}
    return series.map(mapping).to_numpy(dtype=int)


def prepare_matrix(train_df, test_df, features):
    train_x = train_df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = train_x.median(numeric_only=True)
    train_x = train_x.fillna(medians).fillna(0.0)
    test_x = test_df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    test_x = test_x.fillna(medians).fillna(0.0)
    return train_x.to_numpy(dtype=np.float32), test_x.to_numpy(dtype=np.float32)


def inner_matrices(frame, features, prepared, fit_idx, valid_idx, mode):
    if mode == "strict":
        return prepare_matrix(frame.iloc[fit_idx], frame.iloc[valid_idx], features)
    if mode == "historical":
        return prepared[fit_idx], prepared[valid_idx]
    raise ValueError("Unknown imputation mode")


def class_sample_weights(y, power):
    if power <= 0:
        return np.ones(len(y), dtype=float)
    counts = np.bincount(y, minlength=len(CLASS_CODES)).astype(float)
    class_w = (len(y) / (len(CLASS_CODES) * np.maximum(counts, 1.0))) ** power
    sample_w = class_w[y]
    return sample_w / np.mean(sample_w)


def balanced_resample(x, y, power, seed):
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=len(CLASS_CODES)).astype(float)
    target = np.sqrt(np.maximum(counts, 1.0) * np.max(counts)) if power == 0.5 else np.repeat(np.max(counts), len(counts))
    target = np.clip(np.rint(target).astype(int), 250, 16000)
    picked = []
    for cls, n_target in enumerate(target):
        idx = np.flatnonzero(y == cls)
        picked.append(rng.choice(idx, size=n_target, replace=n_target > len(idx)))
    selected = np.concatenate(picked)
    rng.shuffle(selected)
    return x[selected], y[selected]


def fit_model(spec, x, y, seed):
    model = clone(spec.estimator)
    if spec.balanced_resample:
        fit_x, fit_y = balanced_resample(x, y, spec.weight_power, seed)
        model.fit(fit_x, fit_y)
    else:
        sample_weight = class_sample_weights(y, spec.weight_power)
        if hasattr(model, "named_steps"):
            final_step = list(model.named_steps)[-1]
            model.fit(x, y, **{f"{final_step}__sample_weight": sample_weight})
        else:
            model.fit(x, y, sample_weight=sample_weight)
    return model


def aligned_proba(model, x):
    raw = np.asarray(model.predict_proba(x), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps[list(model.named_steps)[-1]], "classes_", None)
    if classes is None:
        return np.clip(raw, EPS, 1.0)
    out = np.full((len(x), len(CLASS_CODES)), EPS, dtype=float)
    for source_idx, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, source_idx]
    out /= out.sum(axis=1, keepdims=True)
    return np.clip(out, EPS, 1.0)


def split_has_all_classes(y, train_idx, test_idx):
    expected = set(range(len(CLASS_CODES)))
    return set(np.unique(y[train_idx])) == expected and set(np.unique(y[test_idx])) == expected


def build_outer_splits(data, repeats, n_splits):
    y = encode_labels(data["final_code"])
    groups = data["well"].astype(str).to_numpy()
    outer = []
    repeat_seeds = [42 + 40 * i for i in range(repeats)]
    for repeat_id, requested_seed in enumerate(repeat_seeds, start=1):
        accepted = None
        for seed in range(requested_seed, requested_seed + 200):
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            candidate = list(splitter.split(np.zeros(len(y)), y, groups))
            if all(split_has_all_classes(y, tr, te) for tr, te in candidate):
                accepted = (seed, candidate)
                break
        if accepted is None:
            raise RuntimeError(f"Could not form a complete stratified grouped split for repeat {repeat_id}.")
        actual_seed, candidate = accepted
        for fold_id, (train_idx, test_idx) in enumerate(candidate, start=1):
            outer.append(
                {
                    "outer_id": len(outer) + 1,
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "requested_seed": requested_seed,
                    "actual_seed": actual_seed,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    return outer


def build_inner_splits(y, groups, n_splits, seed):
    n_splits = min(n_splits, len(np.unique(groups)))
    for candidate_seed in range(seed, seed + 200):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=candidate_seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        if all(set(np.unique(y[tr])) == set(range(len(CLASS_CODES))) for tr, _ in splits):
            return splits, candidate_seed
    raise RuntimeError("Could not form inner grouped folds with all classes in every training partition.")


def weighted_log_loss(y, proba, sample_weight=None):
    p = np.clip(proba[np.arange(len(y)), y], EPS, 1.0)
    loss = -np.log(p)
    if sample_weight is None:
        return float(np.mean(loss))
    return float(np.average(loss, weights=sample_weight))


def global_convex_weights(oof, y, balanced=False):
    n_models = oof.shape[1]
    sample_weight = class_sample_weights(y, 1.0) if balanced else None

    def objective(weights):
        proba = np.einsum("imk,m->ik", oof, weights)
        return weighted_log_loss(y, proba, sample_weight) + 1e-5 * float(np.sum(weights**2))

    result = minimize(
        objective,
        np.repeat(1.0 / n_models, n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 600, "ftol": 1e-11},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        warnings.warn(f"Global convex optimization did not fully converge: {result.message}")
    weights = np.clip(result.x, 0.0, None)
    return weights / np.sum(weights), float(result.fun), bool(result.success)


def classwise_convex_weights(oof, y):
    n_models = oof.shape[1]
    n_classes = oof.shape[2]
    weights = np.zeros((n_classes, n_models), dtype=float)
    losses = []
    successes = []
    for cls in range(n_classes):
        binary_y = (y == cls).astype(int)
        positive = max(int(binary_y.sum()), 1)
        negative = max(int((1 - binary_y).sum()), 1)
        sample_weight = np.where(binary_y == 1, len(y) / (2 * positive), len(y) / (2 * negative))

        def objective(w):
            p = np.clip(np.einsum("im,m->i", oof[:, :, cls], w), EPS, 1.0 - EPS)
            loss = -(binary_y * np.log(p) + (1 - binary_y) * np.log(1 - p))
            return float(np.average(loss, weights=sample_weight) + 1e-5 * np.sum(w**2))

        result = minimize(
            objective,
            np.repeat(1.0 / n_models, n_models),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 600, "ftol": 1e-11},
        )
        w = np.clip(result.x, 0.0, None)
        weights[cls] = w / np.sum(w)
        losses.append(float(result.fun))
        successes.append(bool(result.success))
    return weights, losses, successes


def mix_global(probability_stack, weights):
    mixed = np.einsum("imk,m->ik", probability_stack, weights)
    mixed = np.clip(mixed, EPS, None)
    return mixed / mixed.sum(axis=1, keepdims=True)


def mix_classwise(probability_stack, weights):
    scores = np.einsum("imk,km->ik", probability_stack, weights)
    scores = np.clip(scores, EPS, None)
    return scores / scores.sum(axis=1, keepdims=True)


def soft_macro_f1(y, proba):
    one_hot = np.eye(len(CLASS_CODES), dtype=float)[y]
    true_positive = np.sum(one_hot * proba, axis=0)
    false_positive = np.sum((1.0 - one_hot) * proba, axis=0)
    false_negative = np.sum(one_hot * (1.0 - proba), axis=0)
    f1 = 2.0 * true_positive / np.maximum(2.0 * true_positive + false_positive + false_negative, EPS)
    return float(np.mean(f1))


def macro_risk_weights(oof, y, initial_weights):
    n_models = oof.shape[1]
    initial = np.concatenate([initial_weights, np.zeros(2, dtype=float)])

    def objective(params):
        weights = params[:n_models]
        bias = np.array([params[n_models], 0.0, params[n_models + 1]])
        proba = apply_class_bias(mix_global(oof, weights), bias)
        balanced_nll = weighted_log_loss(y, proba, class_sample_weights(y, 1.0))
        return (
            1.0
            - soft_macro_f1(y, proba)
            + 0.025 * balanced_nll
            + 1e-5 * float(np.sum(weights**2))
            + 2e-4 * float(np.sum(bias**2))
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models + [(-1.25, 1.25), (-1.75, 1.75)],
        constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x[:n_models]) - 1.0)}],
        options={"maxiter": 800, "ftol": 1e-11},
    )
    weights = np.clip(result.x[:n_models], 0.0, None)
    weights /= np.sum(weights)
    bias = np.array([result.x[n_models], 0.0, result.x[n_models + 1]])
    return weights, bias, float(result.fun), bool(result.success)


def fit_class_bias(y, proba):
    grid_major = np.linspace(-0.45, 0.45, 19)
    grid_minor = np.linspace(-1.0, 1.0, 41)
    logp = np.log(np.clip(proba, EPS, 1.0))
    best = (f1_score(y, np.argmax(logp, axis=1), average="macro"), np.zeros(3))
    for bias0 in grid_major:
        for bias2 in grid_minor:
            bias = np.array([bias0, 0.0, bias2])
            pred = np.argmax(logp + bias, axis=1)
            score = f1_score(y, pred, average="macro") - 2e-4 * float(np.sum(bias**2))
            if score > best[0]:
                best = (score, bias.copy())
    return best[1], float(best[0])


def apply_class_bias(proba, bias):
    scores = np.exp(np.log(np.clip(proba, EPS, 1.0)) + np.asarray(bias))
    return scores / scores.sum(axis=1, keepdims=True)


def stacked_logistic(oof, test_stack, y, balanced=False, seed=42):
    x_meta = oof.reshape(len(oof), -1)
    x_test = test_stack.reshape(len(test_stack), -1)
    meta = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced" if balanced else None,
        random_state=seed,
    )
    meta.fit(x_meta, y)
    return aligned_proba(meta, x_test), meta


def metric_row(outer_info, model, y_true, proba):
    pred = np.argmax(proba, axis=1)
    return {
        "outer_id": outer_info["outer_id"],
        "repeat_id": outer_info["repeat_id"],
        "fold_id": outer_info["fold_id"],
        "model": model,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "weighted_f1": f1_score(y_true, pred, average="weighted"),
        "kappa": cohen_kappa_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=np.arange(len(CLASS_CODES))),
    }


def class_metric_rows(outer_info, model, y_true, proba):
    pred = np.argmax(proba, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(len(CLASS_CODES)), zero_division=0
    )
    rows = []
    for idx, code in enumerate(CLASS_CODES):
        rows.append(
            {
                "outer_id": outer_info["outer_id"],
                "repeat_id": outer_info["repeat_id"],
                "fold_id": outer_info["fold_id"],
                "model": model,
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
                "support": int(support[idx]),
            }
        )
    return rows


def fit_outer_fold(data, features, outer_info, inner_folds, fast, jobs=-1, imputation="strict"):
    train_df = data.iloc[outer_info["train_idx"]].copy()
    test_df = data.iloc[outer_info["test_idx"]].copy()
    x_train, x_test = prepare_matrix(train_df, test_df, features)
    y_train = encode_labels(train_df["final_code"])
    y_test = encode_labels(test_df["final_code"])
    train_groups = train_df["well"].astype(str).to_numpy()
    seed = outer_info["actual_seed"] + outer_info["fold_id"] * 101
    inner, inner_seed = build_inner_splits(y_train, train_groups, inner_folds, seed)
    specs = model_library(seed, fast=fast, jobs=jobs)
    oof_blocks = []
    test_blocks = []
    fitted_names = []
    fit_log = []
    started = time.time()

    for model_idx, spec in enumerate(specs):
        model_started = time.time()
        oof = np.zeros((len(train_df), len(CLASS_CODES)), dtype=float)
        for inner_id, (fit_idx, valid_idx) in enumerate(inner, start=1):
            fit_x, valid_x = inner_matrices(train_df, features, x_train, fit_idx, valid_idx, imputation)
            model = fit_model(spec, fit_x, y_train[fit_idx], seed + model_idx * 1000 + inner_id)
            oof[valid_idx] = aligned_proba(model, valid_x)
        final_model = fit_model(spec, x_train, y_train, seed + model_idx * 1000 + 999)
        test_proba = aligned_proba(final_model, x_test)
        oof_blocks.append(oof)
        test_blocks.append(test_proba)
        fitted_names.append(spec.name)
        fit_log.append(
            {
                "model": spec.name,
                "weight_power": spec.weight_power,
                "inner_oof_log_loss": weighted_log_loss(y_train, oof),
                "inner_oof_balanced_log_loss": weighted_log_loss(y_train, oof, class_sample_weights(y_train, 1.0)),
                "fit_seconds": time.time() - model_started,
            }
        )
        print(
            f"[step21] outer={outer_info['outer_id']} model={spec.name} "
            f"oof_macro_f1={f1_score(y_train, np.argmax(oof, axis=1), average='macro'):.4f} "
            f"seconds={fit_log[-1]['fit_seconds']:.1f}",
            flush=True,
        )

    oof_stack = np.stack(oof_blocks, axis=1)
    test_stack = np.stack(test_blocks, axis=1)
    outputs = {name: test_stack[:, idx, :] for idx, name in enumerate(fitted_names)}
    oof_outputs = {name: oof_stack[:, idx, :] for idx, name in enumerate(fitted_names)}
    outputs["SoftVoting"] = np.mean(test_stack, axis=1)
    oof_outputs["SoftVoting"] = np.mean(oof_stack, axis=1)

    individual_losses = np.array([weighted_log_loss(y_train, oof_stack[:, idx, :]) for idx in range(len(fitted_names))])
    legacy_weights = np.exp(-individual_losses)
    legacy_weights /= legacy_weights.sum()
    outputs["LegacyRiskWeighted"] = mix_global(test_stack, legacy_weights)
    oof_outputs["LegacyRiskWeighted"] = mix_global(oof_stack, legacy_weights)

    standard_w, standard_obj, standard_ok = global_convex_weights(oof_stack, y_train, balanced=False)
    outputs["ConvexSL"] = mix_global(test_stack, standard_w)
    oof_outputs["ConvexSL"] = mix_global(oof_stack, standard_w)

    balanced_w, balanced_obj, balanced_ok = global_convex_weights(oof_stack, y_train, balanced=True)
    outputs["BalancedConvexSL"] = mix_global(test_stack, balanced_w)
    oof_outputs["BalancedConvexSL"] = mix_global(oof_stack, balanced_w)

    class_w, class_obj, class_ok = classwise_convex_weights(oof_stack, y_train)
    outputs["ClasswiseSL"] = mix_classwise(test_stack, class_w)
    oof_outputs["ClasswiseSL"] = mix_classwise(oof_stack, class_w)
    class_bias, bias_oof_score = fit_class_bias(y_train, oof_outputs["ClasswiseSL"])
    outputs["CBClasswiseSL"] = apply_class_bias(outputs["ClasswiseSL"], class_bias)
    oof_outputs["CBClasswiseSL"] = apply_class_bias(oof_outputs["ClasswiseSL"], class_bias)

    macro_w, macro_bias, macro_obj, macro_ok = macro_risk_weights(oof_stack, y_train, standard_w)
    outputs["MacroRiskSL"] = apply_class_bias(mix_global(test_stack, macro_w), macro_bias)
    oof_outputs["MacroRiskSL"] = apply_class_bias(mix_global(oof_stack, macro_w), macro_bias)

    bias_candidates = fitted_names + ["SoftVoting", "ConvexSL", "BalancedConvexSL"]
    bias_records = {}
    for candidate in bias_candidates:
        calibrated_bias, calibrated_oof_score = fit_class_bias(y_train, oof_outputs[candidate])
        calibrated_name = "CalibratedSuperLearner" if candidate == "ConvexSL" else f"{candidate}+OOFBias"
        outputs[calibrated_name] = apply_class_bias(outputs[candidate], calibrated_bias)
        oof_outputs[calibrated_name] = apply_class_bias(oof_outputs[candidate], calibrated_bias)
        bias_records[candidate] = {
            "bias": calibrated_bias.tolist(),
            "oof_macro_f1": calibrated_oof_score,
        }

    outputs["StackedLogistic"], _ = stacked_logistic(oof_stack, test_stack, y_train, balanced=False, seed=seed)
    outputs["BalancedStacking"], _ = stacked_logistic(oof_stack, test_stack, y_train, balanced=True, seed=seed)

    metrics = []
    class_metrics = []
    for name, proba in outputs.items():
        metrics.append(metric_row(outer_info, name, y_test, proba))
        class_metrics.extend(class_metric_rows(outer_info, name, y_test, proba))

    prediction = test_df[["well", "DEPTH", "final_code"]].copy()
    prediction.insert(0, "outer_id", outer_info["outer_id"])
    prediction["true_encoded"] = y_test
    final_p = outputs["CalibratedSuperLearner"]
    prediction["pred_encoded"] = np.argmax(final_p, axis=1)
    prediction["pred_code"] = [CLASS_CODES[idx] for idx in prediction["pred_encoded"]]
    prediction["confidence"] = np.max(final_p, axis=1)
    for idx, code in enumerate(CLASS_CODES):
        prediction[f"prob_code_{code}"] = final_p[:, idx]

    disagreement = []
    for i in range(len(fitted_names)):
        for j in range(i + 1, len(fitted_names)):
            pred_i = np.argmax(oof_stack[:, i, :], axis=1)
            pred_j = np.argmax(oof_stack[:, j, :], axis=1)
            disagreement.append(
                {
                    "outer_id": outer_info["outer_id"],
                    "model_a": fitted_names[i],
                    "model_b": fitted_names[j],
                    "probability_correlation": float(
                        np.corrcoef(oof_stack[:, i, :].ravel(), oof_stack[:, j, :].ravel())[0, 1]
                    ),
                    "prediction_disagreement": float(np.mean(pred_i != pred_j)),
                }
            )

    weight_rows = []
    for model_idx, name in enumerate(fitted_names):
        row = {
            "outer_id": outer_info["outer_id"],
            "model": name,
            "legacy_weight": legacy_weights[model_idx],
            "convex_weight": standard_w[model_idx],
            "balanced_convex_weight": balanced_w[model_idx],
            "macro_risk_weight": macro_w[model_idx],
        }
        for class_idx, code in enumerate(CLASS_CODES):
            row[f"classwise_weight_code_{code}"] = class_w[class_idx, model_idx]
        weight_rows.append(row)

    audit = {
        "outer_id": outer_info["outer_id"],
        "repeat_id": outer_info["repeat_id"],
        "fold_id": outer_info["fold_id"],
        "actual_seed": outer_info["actual_seed"],
        "inner_seed": inner_seed,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_wells": int(train_df["well"].nunique()),
        "test_wells": int(test_df["well"].nunique()),
        "test_class_counts": {str(code): int((test_df["final_code"] == code).sum()) for code in CLASS_CODES},
        "standard_objective": standard_obj,
        "standard_converged": standard_ok,
        "balanced_objective": balanced_obj,
        "balanced_converged": balanced_ok,
        "classwise_objectives": class_obj,
        "classwise_converged": class_ok,
        "class_bias": class_bias.tolist(),
        "class_bias_oof_macro_f1": bias_oof_score,
        "macro_risk_bias": macro_bias.tolist(),
        "macro_risk_objective": macro_obj,
        "macro_risk_converged": macro_ok,
        "oof_bias_calibration": bias_records,
        "fit_seconds": time.time() - started,
        "fit_log": fit_log,
    }
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(class_metrics),
        pd.DataFrame(weight_rows),
        pd.DataFrame(disagreement),
        prediction,
        audit,
    )


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, math.nan, math.nan, math.nan
    std = float(np.std(values, ddof=1))
    half = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def summarize_outputs(out_dir):
    metric_files = sorted(out_dir.glob("fold_??_metrics.csv"))
    if not metric_files:
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    class_metrics = pd.concat([pd.read_csv(path) for path in sorted(out_dir.glob("fold_*_class_metrics.csv"))], ignore_index=True)
    weights = pd.concat([pd.read_csv(path) for path in sorted(out_dir.glob("fold_*_weights.csv"))], ignore_index=True)
    diversity = pd.concat([pd.read_csv(path) for path in sorted(out_dir.glob("fold_*_diversity.csv"))], ignore_index=True)
    predictions = pd.concat([pd.read_csv(path) for path in sorted(out_dir.glob("fold_*_predictions.csv"))], ignore_index=True)

    metrics.to_csv(out_dir / "step21_metrics_all_outer_folds.csv", index=False, encoding="utf-8-sig")
    class_metrics.to_csv(out_dir / "step21_class_metrics_all_outer_folds.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(out_dir / "step21_meta_weights_all_outer_folds.csv", index=False, encoding="utf-8-sig")
    diversity.to_csv(out_dir / "step21_base_learner_diversity.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "step21_calibrated_super_learner_predictions.csv", index=False, encoding="utf-8-sig")

    rows = []
    for model, group in metrics.groupby("model", sort=False):
        row = {"model": model, "n_outer_folds": len(group)}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa", "log_loss"]:
            mean, std, lower, upper = confidence_interval(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(out_dir / "step21_model_summary.csv", index=False, encoding="utf-8-sig")

    class_summary = (
        class_metrics.groupby(["model", "class_code", "class_name"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            support_total=("support", "sum"),
        )
    )
    class_summary.to_csv(out_dir / "step21_class_summary.csv", index=False, encoding="utf-8-sig")

    ensemble_candidates = [
        "ConvexSL",
        "BalancedConvexSL",
        "ClasswiseSL",
        "CBClasswiseSL",
        "CalibratedSuperLearner",
        "MacroRiskSL",
    ]
    eligible_ensembles = summary[summary["model"].isin(ensemble_candidates)]
    final_name = str(eligible_ensembles.sort_values("macro_f1_mean", ascending=False).iloc[0]["model"])
    base_names = [
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "MLP",
        "SoftVoting",
        "StackedLogistic",
    ]
    base_names += [f"{name}+OOFBias" for name in base_names if name not in {"StackedLogistic"}]
    paired = []
    pivot = metrics.pivot(index="outer_id", columns="model", values="macro_f1")
    if final_name in pivot:
        for comparator in [name for name in base_names if name in pivot]:
            valid = pivot[[final_name, comparator]].dropna()
            delta = valid[final_name] - valid[comparator]
            mean, std, low, high = confidence_interval(delta)
            if len(delta) >= 2 and np.any(np.abs(delta.to_numpy()) > 0):
                wilcoxon_p = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)
                corrected_se = float(np.sqrt((1.0 / len(delta) + 0.25) * np.var(delta, ddof=1)))
                corrected_t = float(mean / corrected_se) if corrected_se > 0 else math.nan
                corrected_p = float(2.0 * t.sf(abs(corrected_t), len(delta) - 1)) if np.isfinite(corrected_t) else math.nan
                corrected_half = float(t.ppf(0.975, len(delta) - 1) * corrected_se)
            else:
                wilcoxon_p = 1.0
                corrected_t = math.nan
                corrected_p = math.nan
                corrected_half = math.nan
            paired.append(
                {
                    "target": final_name,
                    "comparator": comparator,
                    "n": len(delta),
                    "mean_macro_f1_delta": mean,
                    "std_delta": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "win_rate": float(np.mean(delta > 0)),
                    "wilcoxon_p_exploratory": wilcoxon_p,
                    "nadeau_bengio_corrected_t": corrected_t,
                    "nadeau_bengio_corrected_p": corrected_p,
                    "corrected_ci95_low": mean - corrected_half if np.isfinite(corrected_half) else math.nan,
                    "corrected_ci95_high": mean + corrected_half if np.isfinite(corrected_half) else math.nan,
                }
            )
    pd.DataFrame(paired).to_csv(out_dir / "step21_paired_macro_f1_differences.csv", index=False, encoding="utf-8-sig")

    if plt is not None:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plot_models = summary.head(12).sort_values("macro_f1_mean")
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.barh(plot_models["model"], plot_models["macro_f1_mean"], color="#247BA0")
        ax.errorbar(
            plot_models["macro_f1_mean"],
            np.arange(len(plot_models)),
            xerr=plot_models["macro_f1_std"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_xlabel("Macro-F1 (mean +/- SD across grouped outer folds)")
        ax.set_xlim(
            max(0.0, plot_models["macro_f1_mean"].min() - 0.08),
            min(1.0, plot_models["macro_f1_mean"].max() + 0.05),
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "figure_21_model_macro_f1_reaudit.png", dpi=300)
        plt.close(fig)

    dataset_counts = pd.read_csv(out_dir / "step21_dataset_audit.csv")
    payload = {
        "status": "completed_or_partial",
        "completed_outer_folds": int(metrics["outer_id"].nunique()),
        "models": summary["model"].tolist(),
        "dataset": dataset_counts.to_dict(orient="records"),
        "best_macro_f1_model": str(summary.iloc[0]["model"]),
        "best_macro_f1_mean": float(summary.iloc[0]["macro_f1_mean"]),
        "selected_primary_ensemble": final_name,
        "final_method_macro_f1_mean": float(summary.loc[summary["model"].eq(final_name), "macro_f1_mean"].iloc[0]),
        "mean_base_probability_correlation": float(diversity["probability_correlation"].mean()),
        "mean_base_prediction_disagreement": float(diversity["prediction_disagreement"].mean()),
    }
    (out_dir / "step21_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data, features = load_human_data(args.input, args.features)
    model_library(42, fast=args.fast, require_all=True, jobs=args.jobs)
    audit_rows = []
    for code in CLASS_CODES:
        subset = data[data["final_code"].eq(code)]
        audit_rows.append(
            {
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "points": len(subset),
                "fraction": len(subset) / len(data),
                "wells": int(subset["well"].nunique()),
            }
        )
    pd.DataFrame(audit_rows).to_csv(out_dir / "step21_dataset_audit.csv", index=False, encoding="utf-8-sig")
    outer_splits = build_outer_splits(data, args.outer_repeats, args.outer_folds)
    if args.max_outer:
        outer_splits = outer_splits[: args.max_outer]
    config = {
        "input": str(args.input),
        "feature_list": str(args.features),
        "human_rows": len(data),
        "wells": int(data["well"].nunique()),
        "features": len(features),
        "outer_repeats": args.outer_repeats,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "max_outer": args.max_outer,
        "fast": args.fast,
        "jobs": args.jobs,
        "imputation": args.imputation,
        "class_codes": CLASS_CODES,
        "method_guardrail": "All meta-weights and class biases are fitted on grouped OOF predictions from outer-training wells only.",
    }
    (out_dir / "step21_run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step21] rows={len(data)} wells={data['well'].nunique()} features={len(features)}", flush=True)

    for outer_info in outer_splits:
        prefix = out_dir / f"fold_{outer_info['outer_id']:02d}"
        metrics_path = Path(str(prefix) + "_metrics.csv")
        if metrics_path.exists() and not args.force:
            print(f"[step21] skip completed outer={outer_info['outer_id']}", flush=True)
            continue
        print(
            f"[step21] start outer={outer_info['outer_id']} repeat={outer_info['repeat_id']} "
            f"fold={outer_info['fold_id']}",
            flush=True,
        )
        metrics, class_metrics, weights, diversity, predictions, audit = fit_outer_fold(
            data, features, outer_info, args.inner_folds, args.fast, args.jobs, args.imputation
        )
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        class_metrics.to_csv(Path(str(prefix) + "_class_metrics.csv"), index=False, encoding="utf-8-sig")
        weights.to_csv(Path(str(prefix) + "_weights.csv"), index=False, encoding="utf-8-sig")
        diversity.to_csv(Path(str(prefix) + "_diversity.csv"), index=False, encoding="utf-8-sig")
        predictions.to_csv(Path(str(prefix) + "_predictions.csv"), index=False, encoding="utf-8-sig")
        Path(str(prefix) + "_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        summarize_outputs(out_dir)
        best = metrics.sort_values("macro_f1", ascending=False).iloc[0]
        print(
            f"[step21] finish outer={outer_info['outer_id']} best={best['model']} "
            f"macro_f1={best['macro_f1']:.4f} seconds={audit['fit_seconds']:.1f}",
            flush=True,
        )
    summarize_outputs(out_dir)
    print(f"[step21] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
