#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from meva.dataset import DEFAULT_MANIFEST, load_manifest, validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fail on missing/corrupt MEVA data or malformed KPF annotations"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(load_manifest(args.manifest)), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
