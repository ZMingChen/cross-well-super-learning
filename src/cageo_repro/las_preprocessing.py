import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_STEP_M = float(os.environ.get("SAMPLE_STEP_M", "1.0"))


def norm_well_name(name):
    value = str(name).strip()
    value = re.sub(r"\.(las|txt)$", "", value, flags=re.I)
    return value.upper()


def parse_las(path):
    curves = []
    data_lines = []
    in_curve = False
    in_ascii = False
    null_value = -999.25
    wrap = ""
    well_name = path.stem

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()

            if lower.startswith("wrap"):
                parts = re.split(r"\s+", line.replace(":", " "))
                for part in parts:
                    if part.upper() in {"YES", "NO"}:
                        wrap = part.upper()
                        break
            if lower.startswith("null"):
                for token in re.split(r"\s+", line.replace(":", " ")):
                    try:
                        null_value = float(token)
                        break
                    except Exception:
                        pass
            if lower.startswith("well"):
                body = line.split(":", 1)[0]
                if "." in body:
                    tokens = body.split(".", 1)[1].strip().split()
                    if tokens:
                        well_name = tokens[0]
            if lower.startswith("~curve"):
                in_curve = True
                in_ascii = False
                continue
            if lower.startswith("~a"):
                in_ascii = True
                in_curve = False
                continue
            if line.startswith("~") and not lower.startswith("~a"):
                in_curve = False
                continue

            if in_curve and ":" in line:
                left = line.split(":", 1)[0].strip()
                curve = left.split(".", 1)[0].strip().split()[0]
                if curve:
                    curves.append(curve.upper())
            elif in_ascii and not line.startswith("#"):
                data_lines.append(line)

    n_curves = len(curves)
    tokens = []
    for line in data_lines:
        for token in line.split():
            try:
                tokens.append(float(token))
            except Exception:
                tokens.append(np.nan)

    parsed = []
    if n_curves > 0:
        usable = len(tokens) - (len(tokens) % n_curves)
        parsed = [tokens[i : i + n_curves] for i in range(0, usable, n_curves)]

    if not parsed or not curves:
        return pd.DataFrame(), {
            "well": norm_well_name(path.stem),
            "file": path.name,
            "wrap": wrap,
            "curves_declared": n_curves,
            "raw_rows": 0,
            "parse_status": "empty_or_unparsed",
        }

    df = pd.DataFrame(parsed, columns=curves).replace(null_value, np.nan)
    if "DEPT" in df.columns and "DEPTH" not in df.columns:
        df.rename(columns={"DEPT": "DEPTH"}, inplace=True)
    if "DEPT.M" in df.columns and "DEPTH" not in df.columns:
        df.rename(columns={"DEPT.M": "DEPTH"}, inplace=True)
    if "DEPTH" not in df.columns:
        df.rename(columns={df.columns[0]: "DEPTH"}, inplace=True)

    df["well"] = norm_well_name(path.stem)
    df["source_file"] = path.name
    meta = {
        "well": norm_well_name(path.stem),
        "header_well": norm_well_name(well_name),
        "file": path.name,
        "wrap": wrap,
        "curves_declared": n_curves,
        "raw_rows": int(len(df)),
        "parse_status": "ok",
    }
    return df, meta


def resample_1m(df):
    if df.empty:
        return df
    work = df.copy()
    work = work.sort_values("DEPTH")
    work["_depth_bin"] = np.floor(work["DEPTH"] / SAMPLE_STEP_M).astype(int)
    sampled = work.groupby("_depth_bin", as_index=False).first().drop(columns=["_depth_bin"])
    return sampled.sort_values(["well", "DEPTH"]).reset_index(drop=True)


def curve_inventory(raw_df, sampled_df, meta):
    curve_cols = [
        c
        for c in raw_df.columns
        if c not in {"well", "source_file"} and c != "DEPTH"
    ]
    rows = []
    for curve in sorted(curve_cols):
        raw_non_null = int(raw_df[curve].notna().sum())
        sampled_non_null = int(sampled_df[curve].notna().sum()) if curve in sampled_df.columns else 0
        rows.append(
            {
                "well": meta["well"],
                "file": meta["file"],
                "curve": curve,
                "present": 1,
                "raw_non_null_points": raw_non_null,
                "sampled_non_null_points": sampled_non_null,
                "sampled_non_null_rate": sampled_non_null / max(len(sampled_df), 1),
            }
        )
    return rows


def well_summary(raw_df, sampled_df, meta, path):
    if raw_df.empty:
        depth_min = np.nan
        depth_max = np.nan
        sampled_rows = 0
    else:
        depth_min = float(raw_df["DEPTH"].min())
        depth_max = float(raw_df["DEPTH"].max())
        sampled_rows = int(len(sampled_df))
    return {
        **meta,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "sample_step_m": SAMPLE_STEP_M,
        "sampled_rows": sampled_rows,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_span_m": depth_max - depth_min if np.isfinite(depth_min) and np.isfinite(depth_max) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse LAS into one-metre bins.")
    parser.add_argument("--las-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    las_dir = args.las_dir
    if not las_dir.exists():
        raise FileNotFoundError(las_dir)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sampled = []
    well_rows = []
    inventory_rows = []
    for las_path in sorted(las_dir.glob("*.las")):
        raw_df, meta = parse_las(las_path)
        sampled = resample_1m(raw_df) if not raw_df.empty else raw_df
        if not sampled.empty:
            all_sampled.append(sampled)
        well_rows.append(well_summary(raw_df, sampled, meta, las_path))
        if not raw_df.empty:
            inventory_rows.extend(curve_inventory(raw_df, sampled, meta))

    if all_sampled:
        dataset = pd.concat(all_sampled, ignore_index=True, sort=False)
        first_cols = [c for c in ["well", "DEPTH", "source_file"] if c in dataset.columns]
        other_cols = sorted([c for c in dataset.columns if c not in first_cols])
        dataset = dataset[first_cols + other_cols]
    else:
        dataset = pd.DataFrame()

    if dataset.empty:
        raise ValueError("No usable LAS data found")
    well_df = pd.DataFrame(well_rows).sort_values("well")
    inventory_df = pd.DataFrame(inventory_rows)
    if not inventory_df.empty:
        curve_summary = (
            inventory_df.groupby("curve")
            .agg(
                well_count=("well", "nunique"),
                sampled_non_null_points=("sampled_non_null_points", "sum"),
                mean_sampled_non_null_rate=("sampled_non_null_rate", "mean"),
            )
            .reset_index()
            .sort_values(["well_count", "sampled_non_null_points", "curve"], ascending=[False, False, True])
        )
    else:
        curve_summary = pd.DataFrame(columns=["curve", "well_count", "sampled_non_null_points", "mean_sampled_non_null_rate"])

    dataset_path = out_dir / "log_dataset_1m.csv"
    well_path = out_dir / "well_depth_summary.csv"
    inventory_path = out_dir / "curve_inventory_by_well.csv"
    summary_path = out_dir / "curve_coverage_summary.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    well_df.to_csv(well_path, index=False, encoding="utf-8-sig")
    inventory_df.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    curve_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    core_candidates = ["GR", "AC", "CNL", "CAL", "SP", "RILD", "RILM", "RFOC", "DEN"]
    core_available = [c for c in core_candidates if c in dataset.columns]
    core_path = out_dir / "log_dataset_1m_core_curves.csv"
    dataset[[c for c in ["well", "DEPTH", "source_file"] if c in dataset.columns] + core_available].to_csv(
        core_path, index=False, encoding="utf-8-sig"
    )

    summary = {
        "las_dir": str(las_dir),
        "las_file_count": int(len(list(las_dir.glob("*.las")))),
        "parsed_well_count": int(well_df[well_df["parse_status"].eq("ok")]["well"].nunique()) if not well_df.empty else 0,
        "sample_step_m": SAMPLE_STEP_M,
        "total_sampled_rows": int(len(dataset)),
        "total_columns": int(len(dataset.columns)),
        "depth_min": float(dataset["DEPTH"].min()) if not dataset.empty else None,
        "depth_max": float(dataset["DEPTH"].max()) if not dataset.empty else None,
        "core_available_curves": core_available,
        "outputs": {
            "log_dataset_1m": str(dataset_path),
            "log_dataset_1m_core_curves": str(core_path),
            "well_depth_summary": str(well_path),
            "curve_inventory_by_well": str(inventory_path),
            "curve_coverage_summary": str(summary_path),
        },
    }
    (out_dir / "step1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
