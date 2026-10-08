#!/usr/bin/env python3
"""One-command real-data benchmark; commands run in separate import/config scopes."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/current")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json"
    )
    args = parser.parse_args()
    for script, flags in (
        ("validate_meva.py", []),
        ("process_meva.py", ["--run-dir", str(args.run_dir)]),
        ("evaluate_meva.py", ["--run-dir", str(args.run_dir), "--verify-api"]),
        ("review_meva.py", ["--run-dir", str(args.run_dir)]),
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / script),
                "--manifest",
                str(args.manifest),
                *flags,
            ],
            cwd=ROOT,
        )
        if result.returncode:
            sys.exit(result.returncode)
