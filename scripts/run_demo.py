from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cageo_repro.demo_pipeline import grouped_demo, load_synthetic_csv


def main() -> None:
    frame = load_synthetic_csv(ROOT / "examples" / "synthetic_well_logs.csv")
    metrics, predictions = grouped_demo(frame)
    out = ROOT / "demo_output"
    out.mkdir(exist_ok=True)
    metrics.to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    summary = {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "mean_macro_f1": float(metrics["macro_f1"].mean()),
        "mean_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
        "note": "Synthetic demonstration only; not manuscript results.",
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
