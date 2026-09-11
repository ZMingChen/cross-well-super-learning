"""Generate non-geological grouped data for exercising the complete paper pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(2026)
    rows = []
    class_centres = {0: (-1.2, 0.5), 3: (1.0, -0.8), 4: (0.1, 1.5)}
    labels = np.tile([0] * 26 + [3] * 26 + [4] * 8, 12)
    for well_index in range(12):
        well_shift = rng.normal(0, 0.12, size=2)
        well_labels = labels[well_index * 60 : (well_index + 1) * 60].copy()
        rng.shuffle(well_labels)
        for depth_index, code in enumerate(well_labels):
            centre = np.asarray(class_centres[int(code)]) + well_shift
            feature_1, feature_2 = rng.normal(centre, [0.55, 0.5])
            rows.append(
                {
                    "well": f"SYN_{well_index + 1:02d}",
                    "DEPTH": 1000.0 + depth_index,
                    "final_code": int(code),
                    "final_lithology": {0: "fine sandstone", 3: "mudstone", 4: "silty mudstone"}[int(code)],
                    "final_label_source": "human",
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "feature_interaction": feature_1 * feature_2 + rng.normal(0, 0.1),
                }
            )
    output = ROOT / "examples" / "synthetic_manuscript_input.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame({"feature": ["feature_1", "feature_2", "feature_interaction"]}).to_csv(
        ROOT / "examples" / "synthetic_feature_list.csv", index=False
    )
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
