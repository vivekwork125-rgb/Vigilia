#!/usr/bin/env python3
"""Probe raw YOLO detection coverage independently of ByteTrack and events."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from meva.dataset import DEFAULT_MANIFEST, ROOT, load_manifest
from meva.perception_probe import probe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/current")
    args = parser.parse_args()
    try:
        print(json.dumps(probe(load_manifest(args.manifest), args.run_dir)["summary"], indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")
