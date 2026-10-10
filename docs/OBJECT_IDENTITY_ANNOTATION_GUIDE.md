# MEVA object-identity pilot annotation guide

This is a **human-authored, evaluation-only** dataset. Start the local source-frame review tool from the normal VIGILIA repository folder:

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
.venv/bin/python scripts/review_object_identity.py --port 8765
```

Open `http://127.0.0.1:8765/`. The UI displays only decoded source video and human annotations. It does not request, prefill, or display model detections. The 15 clip definitions and source SHA-256 references are in [`clips.json`](../datasets/meva/object-identity/clips.json); reviews are saved atomically in [`annotations.jsonl`](../datasets/meva/object-identity/annotations.jsonl). Raw videos stay under `MEVA_ROOT`, outside Git. The MEVA source license is CC-BY-4.0; VIGILIA does not own the footage.

## Review workflow

1. Select a clip. Read its video ID, camera ID, frame index and timestamp before drawing. The pilot samples ten consecutive points at 2 FPS in each five-second clip. Use previous/next to inspect motion and identity continuity.
2. Draw the tightest box around the **visible pixels** of an eligible object on the original decoded frame. The UI stores original-frame `xyxy` coordinates even when the canvas is scaled in the browser. Do not extrapolate a hidden body or paste a detector's box.
3. Select an existing human object ID only when the same physical object is visually supported by appearance, location **and** continuous trajectory. Use **Create new ID** for a different object. IDs are scoped to one clip; do not link a G421 object to G424 or across separated clips.
4. Choose the narrowest defensible class: `vehicle`, `bag`, `bottle`, `cup`, `phone`, `other_visible_object`, or `unknown_object`. `person` is available as context but is excluded from the object-detector comparison. If a tiny item could be a cup or phone, use `unknown_object`; an annotation label is not evidence of semantic identity.
5. Mark `visible` when the object is unobstructed, `partially_visible` for some visible pixels, `occluded` for a heavily obscured but still boxable region, `uncertain` for questionable object presence, and `not_visible` when the tracked object is absent from the image. **Never draw a box for `not_visible`.** Record occlusion as `none`, `partial`, `heavy`, or `uncertain` consistently. A tiny object is not automatically invisible: inspect source pixels and use a note when the box is uncertain.
6. When an object disappears and later reappears, reuse its ID only with distinguishing evidence and a `reidentification_note`; otherwise create a new uncertain ID. Use **Uncertain new ID** for an unresolved physical identity. `UNRESOLVED-*` IDs cannot be reused across frames, preventing a false persistent identity. Similar color or detector class alone is insufficient.
7. Enter a note explaining small/ambiguous objects, partial occlusion, similar neighboring objects, and any class uncertainty. Correct or delete an object with the frame's annotation controls. Save the frame before navigating. An interrupted save leaves the prior valid JSONL snapshot intact; reopen the same clip/frame to resume.
8. Keep `review_state=partial` until **all eligible countable objects** in that frame have been independently considered. Eligible objects are people as optional context and all visually distinguishable vehicles, bags, bottles, cups, phones, or other discrete movable objects. Do not annotate fixed architecture, benches, wall signs, shadows, detector artifacts, or repeated background texture. A `complete` empty frame requires the exact note `EMPTY_SCENE_CONFIRMED`; this prevents accidental empty ground truth. Current seed reviews are explicitly **partial target-only reviews**.

The review should include ordinary motion, stationary objects, occlusion and ambiguous cases. Do not restrict annotation to MEVA pickup/placement intervals. Do not use MEVA activity boxes or model predictions to draw or correct a human ground-truth box. The selected clips are separated in time, but all 15 are currently `pilot_review_only`; there is **no held-out test split**. Before tuning a detector, create a separate recording-level evaluation set. Adjacent frames from one clip must never be split between tuning and held-out evaluation.

## Quality control and disagreements

Run the validator after each review session:

```bash
.venv/bin/python scripts/validate_object_identity.py --allow-incomplete
```

The command checks clip provenance, source hashes, camera mapping, sampling grid, box bounds, timestamp, visibility/box consistency, duplicate IDs, unresolved-ID reuse, class changes, and reappearance notes. It writes an ignored local validation report under `data/meva/object-identity/`. Without `--allow-incomplete`, it fails until all 150 frames are marked complete. **Schema-valid partial records are not an evaluation-ready benchmark.**

A second reviewer should independently annotate a subset of complete clips **without seeing the first reviewer's boxes**, then compare boxes, object IDs, visibility and re-identification notes. Resolve disagreements by returning to the decoded source sequence. Preserve both original reviews and the adjudication log before calculating agreement. No independent second review has occurred for the initial 30 target-only frame records; do not report inter-annotator agreement or claim a double-reviewed subset.

Once every selected frame has an exhaustive, validated review, use:

```bash
.venv/bin/python scripts/validate_object_identity.py
.venv/bin/python scripts/evaluate_object_identity_detection.py
.venv/bin/python scripts/evaluate_object_identity_tracking.py
```

Before completion, the scripts require `--allow-partial-diagnostic` and report only **reviewed target-box coverage** and target-track continuity. They withhold detection precision/recall, false-positive rates, HOTA and IDF1. Matching is one-to-one, class-compatible, and requires source-frame box IoU ≥0.5. `bag` accepts COCO backpack/handbag/suitcase; `vehicle` accepts car/truck/bus/motorcycle; `unknown_object` is class-agnostic **within evaluated mobile-object detector classes**, not a claim that the model recognized a category. Person-triggered crops use runtime person boxes only; they never use human object coordinates to choose a crop.
