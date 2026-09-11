"""Portable entry point for final fitting and candidate annotation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cageo_repro.final_application import main

if __name__ == "__main__":
    main()
