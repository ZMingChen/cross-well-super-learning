import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cageo_repro.failure_analysis import main

if __name__ == "__main__":
    main()
