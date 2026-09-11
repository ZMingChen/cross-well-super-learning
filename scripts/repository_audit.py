from __future__ import annotations

import ast
import json
import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "docs" / "data_schema.md",
    ROOT / "docs" / "release_checklist.md",
    ROOT / "examples" / "synthetic_well_logs.csv",
    ROOT / "src" / "cageo_repro" / "demo_pipeline.py",
    ROOT / "scripts" / "run_demo.py",
    ROOT / "scripts" / "release_preflight.py",
    ROOT / "CODE_AVAILABILITY.md",
    ROOT / ".gitignore",
    ROOT / "requirements-paper.txt",
    ROOT / "scripts" / "run_paper.py",
    ROOT / "scripts" / "run_annotation.py",
    ROOT / "src" / "cageo_repro" / "step21_pipeline.py",
    ROOT / "docs" / "provenance.md",
]
FORBIDDEN_SUFFIXES = {
    ".las", ".xlsx", ".xls", ".sqlite", ".db", ".tif", ".tiff", ".parquet",
    ".feather", ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5",
    ".pt", ".pth", ".onnx",
}
SENSITIVE_TERMS = re.compile(
    r"(?:C:\\Users\\|AppData|Documents\\|source_file|well_name|manual_label|"
    r"manual_interpret|point[_ -]?level|point[_ -]?prediction|raw[_ -]?depth|"
    r"well[_ -]?(?:list|identifier))",
    re.I,
)
IGNORED_TOP_LEVEL = {".git", ".venv", ".venv-paper", "demo_output", ".pytest_cache", "__pycache__"}
PUBLIC_CSV = {"synthetic_well_logs.csv", "synthetic_manuscript_input.csv", "synthetic_feature_list.csv"}
FORBIDDEN_NAME = re.compile(
    r"(?:manual|private|confidential|raw[_ -]?depth|well[_ -]?(?:list|identifier)|"
    r"point[_ -]?(?:level|prediction)|prediction[s]?)",
    re.I,
)


def release_files():
    for directory, directories, files in os.walk(ROOT):
        directories[:] = [name for name in directories if name not in IGNORED_TOP_LEVEL
                          and not name.startswith(("pytest-cache-files-", "full_demo_output", "paper_output"))]
        for name in files:
            yield Path(directory) / name


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")
    if not (ROOT / "LICENSE").exists() and not (ROOT / "LICENSE.txt").exists():
        errors.append("missing real LICENSE file (authors must choose the license)")
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    for phrase in ("Quick start", "Data and code availability", "confidential", "synthetic"):
        if phrase.lower() not in readme.lower():
            errors.append(f"README missing required topic: {phrase}")
    for path in release_files():
        relative = path.relative_to(ROOT)
        if not path.is_file() or relative.parts[0] in IGNORED_TOP_LEVEL or "__pycache__" in relative.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden data-like file tracked: {path.relative_to(ROOT)}")
        if FORBIDDEN_NAME.search(path.stem) and path.name not in {"repository_audit.py"}:
            errors.append(f"sensitive-looking filename: {path.relative_to(ROOT)}")
        if path.suffix.lower() == ".csv":
            if relative.parent != Path("examples") or path.name not in PUBLIC_CSV:
                errors.append(f"unapproved data file: {relative}")
            elif path.name != "synthetic_feature_list.csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows or any(not row.get("well", "").startswith("SYN_") for row in rows):
                    errors.append(f"non-synthetic well identifier: {relative}")
        if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if path.suffix in {".md", ".txt"}:
                # Policy text may intentionally name prohibited artifacts; only
                # absolute local paths are sensitive in documentation.
                pattern = re.compile(r"(?:[A-Z]:[\\/]|AppData)", re.I)
            else:
                # Source code may legitimately use schema column names such as
                # ``well_name`` and ``source_file``; reject only machine-local
                # absolute paths here.
                pattern = re.compile(r"(?:[A-Z]:[\\/]|AppData)", re.I)
            if pattern.search(text) and path.name not in {"repository_audit.py"}:
                errors.append(f"possible local/sensitive path in: {path.relative_to(ROOT)}")
    try:
        tracked = os.popen("git ls-files").read().splitlines()
        for name in tracked:
            tracked_path = ROOT / name
            if tracked_path.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"forbidden tracked file: {name}")
            if FORBIDDEN_NAME.search(tracked_path.stem) and tracked_path.name not in {"repository_audit.py"}:
                errors.append(f"sensitive-looking tracked filename: {name}")
    except Exception as exc:
        errors.append(f"could not inspect tracked files: {exc}")
    try:
        ast.parse((ROOT / "src" / "cageo_repro" / "demo_pipeline.py").read_text(encoding="utf-8"))
        ast.parse((ROOT / "scripts" / "run_demo.py").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Python syntax check failed: {exc}")
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "required_files": len(REQUIRED)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
