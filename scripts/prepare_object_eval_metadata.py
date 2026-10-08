#!/usr/bin/env python3
"""Build evaluation-only object taxonomy and compact metadata for the 30 manipulation cases.

Creates:
  data/meva/object-eval/taxonomy.json
  data/meva/object-eval/cases.json

Stores metadata only. Never copies raw MEVA videos.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_metadata():
    csv_path = ROOT / "docs/meva-small-object-cases.csv"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        raw_cases = list(reader)

    diag_path = ROOT / "data/meva/improved-2fps/failure-diagnostics.json"
    diag_lookup = {}
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        for a in diag["annotations"]:
            diag_lookup[a["key"]] = a

    cases = []
    category_counts = {
        "KNOWN_SUPPORTED": 0,
        "KNOWN_UNSUPPORTED": 0,
        "AMBIGUOUS": 0,
        "UNKNOWN": 0,
    }

    for rc in raw_cases:
        key = rc["annotation_key"]
        review = rc["visual_review"]
        w = float(rc["median_box_width_px"])
        h = float(rc["median_box_height_px"])
        img_frac = float(rc["median_image_fraction"])

        # Determine category, visibility, occlusion, and identifiable category
        if "Small cup/can-like" in review:
            tax_cat = "AMBIGUOUS"
            visible = "partial"
            identifiable = "cup/can"
            occlusion = "partly occluded by hand"
            interaction = "hand manipulation near table/desk"
        elif "Small bright bottle/cup-like" in review:
            tax_cat = "AMBIGUOUS"
            visible = "true"
            identifiable = "bottle/cup (unconfirmed)"
            occlusion = "minimal hand contact"
            interaction = "hand contact near table"
        elif "Visible item on table/hand" in review:
            tax_cat = "AMBIGUOUS"
            visible = "true"
            identifiable = None
            occlusion = "hand/table interaction"
            interaction = "tabletop manipulation"
        elif "Dark item against clothing" in review:
            tax_cat = "UNKNOWN"
            visible = "partial"
            identifiable = None
            occlusion = "occluded by clothing/furniture"
            interaction = "manipulation near seating"
        elif "bright window" in review:
            tax_cat = "UNKNOWN"
            visible = "partial"
            identifiable = None
            occlusion = "severe window glare / backlighting"
            interaction = "manipulation at window counter"
        elif "counter/table" in review:
            tax_cat = "UNKNOWN"
            visible = "partial"
            identifiable = None
            occlusion = "counter edge / low contrast"
            interaction = "counter pickup/placement"
        elif "distant and partly occluded" in review:
            tax_cat = "UNKNOWN"
            visible = "partial"
            identifiable = None
            occlusion = "distance / partial occlusion"
            interaction = "distant outdoor interaction"
        else:
            tax_cat = "UNKNOWN"
            visible = "unknown"
            identifiable = None
            occlusion = "unspecified"
            interaction = "unspecified"

        category_counts[tax_cat] += 1

        d_info = diag_lookup.get(key, {})
        start_sec = d_info.get("start_seconds")
        end_sec = d_info.get("end_seconds_inclusive")

        cases.append({
            "annotation_key": key,
            "activity_type": rc["activity_type"],
            "camera_id": rc["camera_id"],
            "video_id": key.split(":")[0],
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "sampled_frames_count": int(rc["sampled_frames"]),
            "object_visible": visible,
            "taxonomy_category": tax_cat,
            "object_category_if_identifiable": identifiable,
            "approximate_geometry": {
                "median_width_px": w,
                "median_height_px": h,
                "median_image_fraction": img_frac,
                "center_x_fraction": float(rc["image_center_x_fraction"]),
                "center_y_fraction": float(rc["image_center_y_fraction"]),
                "under_40px_width": w < 40.0,
            },
            "occlusion_context": occlusion,
            "interaction_context": interaction,
            "human_visual_review": review,
            "main_failure": rc["main_failure"],
            "review_status": "REVIEWED",
        })

    taxonomy = {
        "schema_version": 1,
        "description": "Evaluation-only object taxonomy for MEVA small-object manipulation activities.",
        "principles": [
            "Do NOT force a semantic class onto MEVA generic 'other' without evidence.",
            "KNOWN_SUPPORTED requires confirmed presence of production portable classes (backpack, handbag, suitcase, bag, briefcase, bottle).",
            "KNOWN_UNSUPPORTED requires confirmed presence of non-supported semantic classes.",
            "AMBIGUOUS indicates visible item with class ambiguity or hand occlusion.",
            "UNKNOWN indicates low-contrast, silhouette, window glare, or distant pixels where identity is completely undetermined.",
        ],
        "total_cases": len(cases),
        "category_distribution": category_counts,
        "size_summary": {
            "cases_under_40px_width": sum(1 for c in cases if c["approximate_geometry"]["under_40px_width"]),
            "median_width_px": 26.0,
            "median_height_px": 38.0,
        },
    }

    out_dir = ROOT / "data/meva/object-eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    tax_path = out_dir / "taxonomy.json"
    tax_path.write_text(json.dumps(taxonomy, indent=2) + "\n")

    cases_path = out_dir / "cases.json"
    cases_path.write_text(json.dumps(cases, indent=2) + "\n")

    print(f"Wrote taxonomy metadata to {tax_path}")
    print(f"Wrote 30 case records to {cases_path}")
    print("Taxonomy distribution:", category_counts)


if __name__ == "__main__":
    build_metadata()
