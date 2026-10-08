#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from meva.dataset import DEFAULT_MANIFEST, ROOT, load_manifest
from meva.processing import configure, process_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index all eight real sources with YOLO + ByteTrack in an isolated benchmark DB"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/baseline")
    args = parser.parse_args()
    try:
        process_manifest(load_manifest(args.manifest), configure(args.run_dir))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")
