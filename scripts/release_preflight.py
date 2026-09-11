from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "-m", "pip", "check"])
    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    run([sys.executable, "scripts/run_demo.py"])
    run([sys.executable, "scripts/repository_audit.py"])
    required_outputs = [ROOT / "demo_output" / "metrics.json", ROOT / "demo_output" / "predictions.csv"]
    missing = [str(path.relative_to(ROOT)) for path in required_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"Demo did not produce expected output(s): {missing}")
    print("Local demo checks passed. Journal submission readiness is NOT verified.")


if __name__ == "__main__":
    main()
