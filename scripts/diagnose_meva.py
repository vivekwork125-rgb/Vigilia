#!/usr/bin/env python3
"""Separate perception, temporal-event and search failures on a completed run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from meva.dataset import DEFAULT_MANIFEST, ROOT, load_manifest
from meva.diagnostics import analyze
from meva.processing import configure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/current")
    args = parser.parse_args()
    try:
        result = analyze(load_manifest(args.manifest), configure(args.run_dir))
        print(json.dumps(result["per_type"], indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")
