from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_CODES = (0, 3, 4)


def load_synthetic_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"well", "depth", "gr", "ac", "resistivity", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    if not set(frame["label"]).issubset(CLASS_CODES):
        raise ValueError("Synthetic labels must be one of 0, 3, or 4")
    return frame.sort_values(["well", "depth"]).reset_index(drop=True)


def _models(seed: int) -> dict[str, object]:
    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=seed)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
    }


def grouped_demo(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = ["gr", "ac", "resistivity"]
    if frame.empty or frame["well"].isna().any() or frame["well"].nunique() < 3:
        raise ValueError("At least three non-missing well groups are required")
    if not frame["label"].isin(CLASS_CODES).all():
        raise ValueError("Labels must be one of 0, 3, or 4")
    x = frame[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    y = frame["label"].map({code: i for i, code in enumerate(CLASS_CODES)}).to_numpy()
    groups = frame["well"].to_numpy()
    models = _models(seed)
    oof = {name: np.zeros((len(frame), len(CLASS_CODES)), dtype=float) for name in models}
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=3).split(x, y, groups), start=1):
        if len(np.unique(y[train_idx])) < 2:
            raise ValueError(f"Fold {fold}: training wells need at least two classes")
        if x.iloc[train_idx].isna().all().any():
            raise ValueError(f"Fold {fold}: a feature is entirely missing in training wells")
        for name, estimator in models.items():
            fitted = make_pipeline(SimpleImputer(strategy="median"), clone(estimator))
            fitted.fit(x.iloc[train_idx], y[train_idx])
            # Align columns even when a training fold lacks one of the three classes.
            probability = fitted.predict_proba(x.iloc[test_idx])
            oof[name][np.ix_(test_idx, fitted.classes_.astype(int))] = probability
        averaged = np.mean([oof[name][test_idx] for name in models], axis=0)
        pred = np.argmax(averaged, axis=1)
        fold_rows.append(
            {
                "fold": fold,
                "macro_f1": f1_score(y[test_idx], pred, labels=np.arange(3), average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
                "test_wells": int(frame.iloc[test_idx]["well"].nunique()),
            }
        )
    averaged_oof = np.mean(list(oof.values()), axis=0)
    predictions = frame[["well", "depth"]].copy()
    predictions["predicted_code"] = [CLASS_CODES[i] for i in np.argmax(averaged_oof, axis=1)]
    predictions["confidence"] = np.max(averaged_oof, axis=1)
    return pd.DataFrame(fold_rows), predictions
