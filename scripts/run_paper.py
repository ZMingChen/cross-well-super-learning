"""Portable entry point for the migrated Step 21 experiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cageo_repro.step21_pipeline import main

if __name__ == "__main__":
    main()
