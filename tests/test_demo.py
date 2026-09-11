from pathlib import Path
from io import StringIO

import numpy as np
import pytest
from sklearn.impute import SimpleImputer

from cageo_repro.demo_pipeline import grouped_demo, load_synthetic_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frame():
    return load_synthetic_csv(ROOT / "examples" / "synthetic_well_logs.csv")


def test_reproducible_predictions(frame):
    metrics, predictions = grouped_demo(frame)
    repeated_metrics, repeated_predictions = grouped_demo(frame)
    assert metrics.equals(repeated_metrics)
    assert predictions.equals(repeated_predictions)
    assert len(metrics) == 3
    assert len(predictions) == len(frame)
    assert predictions["confidence"].between(0, 1).all()
    assert predictions["predicted_code"].isin([0, 3, 4]).all()


def test_imputer_only_sees_training_wells(frame, monkeypatch):
    seen = []
    original = SimpleImputer.fit

    def record_fit(self, x, y=None):
        seen.append(set(frame.loc[x.index, "well"]))
        return original(self, x, y)

    monkeypatch.setattr(SimpleImputer, "fit", record_fit)
    frame.loc[0, "gr"] = np.nan
    grouped_demo(frame)
    assert len(seen) == 9
    assert all(len(wells) == 2 for wells in seen)


def test_missing_training_class_is_aligned(frame):
    frame.loc[frame["well"] != "SYN_A", "label"] = frame.loc[
        frame["well"] != "SYN_A", "label"
    ].replace(4, 3)
    _, predictions = grouped_demo(frame)
    assert predictions["confidence"].between(0, 1).all()
    assert not (predictions.loc[frame["well"] == "SYN_A", "predicted_code"] == 4).any()


def test_too_few_wells(frame):
    with pytest.raises(ValueError, match="three"):
        grouped_demo(frame[frame["well"] != "SYN_A"])


def test_single_training_class(frame):
    frame["label"] = 0
    with pytest.raises(ValueError, match="two classes"):
        grouped_demo(frame)


def test_entirely_missing_feature(frame):
    frame["gr"] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        grouped_demo(frame)


def test_fractional_label_rejected(frame):
    frame["label"] = 3.5
    with pytest.raises(ValueError, match="labels"):
        load_synthetic_csv(StringIO(frame.to_csv(index=False)))
