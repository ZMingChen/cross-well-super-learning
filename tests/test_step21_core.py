import numpy as np
import pandas as pd
import pytest

from cageo_repro import step21_pipeline as sl


def probability_stack():
    first = np.array([[0.80, 0.15, 0.05], [0.15, 0.75, 0.10], [0.10, 0.20, 0.70]])
    second = np.array([[0.60, 0.25, 0.15], [0.25, 0.55, 0.20], [0.20, 0.25, 0.55]])
    return np.stack([first, second], axis=1)


def test_convex_weights_and_probability_mixing():
    y = np.array([0, 1, 2])
    stack = probability_stack()
    weights, _, converged = sl.global_convex_weights(stack, y)
    mixed = sl.mix_global(stack, weights)
    assert converged
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(mixed.sum(axis=1), 1.0)


def test_classwise_weights_and_bias_are_valid():
    y = np.array([0, 1, 2])
    stack = probability_stack()
    weights, _, converged = sl.classwise_convex_weights(stack, y)
    mixed = sl.mix_classwise(stack, weights)
    bias, _ = sl.fit_class_bias(y, mixed)
    calibrated = sl.apply_class_bias(mixed, bias)
    assert all(converged)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_train_only_imputation():
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
    test = pd.DataFrame({"a": [100.0, np.nan], "b": [100.0, np.nan]})
    train_x, test_x = sl.prepare_matrix(train, test, ["a", "b"])
    assert train_x[1, 0] == pytest.approx(2.0)
    assert test_x[1].tolist() == pytest.approx([2.0, 3.0])


def test_missing_optional_learners_never_silently_shrinks_library(monkeypatch):
    monkeypatch.setattr(sl, "XGBClassifier", None)
    with pytest.raises(RuntimeError, match="eight learners"):
        sl.model_library(42, fast=True)
