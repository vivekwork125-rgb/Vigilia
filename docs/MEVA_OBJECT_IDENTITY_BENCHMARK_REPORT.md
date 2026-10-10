# MEVA manually reviewed object-identity pilot

## A. Executive summary

This commit establishes the **annotation workflow and a partial, source-reviewed pilot**, not a complete object-identity benchmark. Fifteen five-second clips from three MEVA cameras are defined at 2 FPS (150 planned source frames). A local source-frame UI, versioned schema, atomic JSONL saves, validator, one-to-one detector matcher and guarded tracking diagnostic are implemented. Thirty consecutive sampled frames—ten from each of three clips—were visually reviewed for **one target object per frame**, yielding 30 human boxes and three persistent human identities. Source-frame box overlays were visually checked. All 30 records remain `partial` because other eligible objects were not exhaustively annotated. **Zero of 150 frames are complete; formal detector precision, recall, false-positive rate, HOTA and IDF1 are not reported.** No second reviewer has independently labeled any frame.

The target-only diagnostic found YOLO11n full-frame overlaps for 20/30 selected boxes and runtime-person-crop overlaps for 19/30, at one-to-one IoU ≥0.5. These are **not detector recall estimates**. The matches differ by camera and the crop's target matches use more provisional track IDs. Production remains unchanged.

## B–C. Camera and clip selection

The same eight-video [MEVA manifest](../datasets/meva/manifest.json) was inspected for actual source paths, SHA-256, dimensions, 30 FPS metadata and annotation mapping. We visually reviewed source frames from G421, G331, G424, G436 and G638 before choosing G421's indoor lounge (small objects and occlusion), G331's bus waiting room (nearby luggage and ordinary behavior), and G424's outdoor parking view (moving and stationary vehicles at several distances). Selection used decoded footage, **not** detector success or MEVA activity labels. G424's 0–5 s frames were visually green/corrupted, so that candidate clip was explicitly excluded from this manual pilot and replaced by 30–35 s. The source video and previous full-video benchmarks remain unchanged.

Each clip is 5 s and contains ten samples separated by 15 of the 30 source frames. The [clip manifest](../datasets/meva/object-identity/clips.json) preserves source reference/hash, camera, frame indices, sample rate and selection rationale. Preparation decoded the first and last frame of every clip; the three partially annotated clips were inspected frame by frame.

| Camera | Scene | Five clip intervals (s; ten frames each) | Planned frames | Target-reviewed frames |
|---|---|---|---:|---:|
| G421 | School lounge | 0–5, 60–65, 120–125, 180–185, 240–245 | 50 | 10 |
| G331 | Bus waiting room | 0–5, 60–65, 120–125, 180–185, 240–245 | 50 | 10 |
| G424 | School parking | 30–35, 60–65, 120–125, 180–185, 240–245 | 50 | 10 |

The 75 seconds of clips contain ordinary activity, not just manipulation. The selected material includes small/occluded items, but those difficult objects have **not yet been exhaustively annotated**. At 2 FPS, fast motion or a brief occlusion can occur between samples; a higher-rate subset may eventually be needed.

## D. Annotation protocol and tool

[`scripts/review_object_identity.py`](../scripts/review_object_identity.py) serves a loopback-only [canvas interface](../scripts/object_identity_review.html). It displays original source frames, video/camera/clip/frame/time, existing human annotations and previous/next controls. A reviewer can draw/correct/delete a box, select or create an object ID, mark visibility/occlusion/identity uncertainty, save and resume. It shows **no model proposals**. The first 30 records were manually transcribed from source-frame grid/contact sheets and checked again as box overlays; neither detector boxes nor MEVA KPF geometry generated them.

Schema version `1.0` records source video/camera/hash, clip, original frame, `frame/FPS` time, review state, and object box/class/visibility/occlusion/human ID/confidence/notes. `not_visible` requires a null box. `UNRESOLVED-*` identities cannot be reused across frames. Reuse after a frame gap requires a re-identification note; visually similar objects are not assumed identical. IDs are clip-scoped, with no cross-camera linkage. The [annotation guide](OBJECT_IDENTITY_ANNOTATION_GUIDE.md) defines small-object, occlusion, disappearance, uncertainty and disagreement rules.

The three followed targets are a floor backpack in G421, a rolling suitcase partly hidden by a bench in G331, and a dark SUV moving continuously leftward in G424. One human ID was assigned within each ten-frame sequence. Fixed scene fixtures are excluded. Person boxes are permitted as context but excluded from object-detector scoring.

## E–F. Statistics and quality control

| Measure | Current status |
|---|---:|
| Cameras / clips | 3 / 15 |
| Planned frames / duration | 150 / 75 s |
| Saved target-reviewed frames / fully reviewed frames | 30 / **0** |
| Human object boxes / persistent IDs | 30 / 3 |
| Visible / partially visible instances | 20 / 10 |
| Occluded / not visible / uncertain instances | 0 / 0 / 0 |
| Uncertain identity instances | 0 |
| Box width <40 / 40–159 / ≥160 source px | 0 / 10 / 20 |
| Independently double-reviewed frames | **0** |
| Schema/consistency validation errors | **0** |

[`scripts/validate_object_identity.py`](../scripts/validate_object_identity.py) checks the clip source/video/camera/hash, sample grid, frame and timestamp, box bounds, visibility/box consistency, duplicate IDs, unresolved-ID reuse, class changes and reappearance notes. The ignored local `data/meva/object-identity/validation-report.json` records `valid_schema=true`, `complete_pilot=false`, 30 saved partial frames and zero errors. Validation without `--allow-incomplete` **fails**, as intended. The overlay check is one reviewer's QA, not independent double annotation or inter-annotator agreement. The remaining 120 frames are unreviewed; the initial 30 also need exhaustive inventory before `complete` status. A second reviewer should independently label a subset before formal scoring.

## G. Split and leakage controls

All clips are `pilot_review_only`; there is **no held-out tuning/test split**. A future split must keep clips from the same continuous recording together, and never divide adjacent event frames between tuning and testing. Three cameras provide little generalization evidence; G421 and G424 share the same school recording period. The UI reads only human records and decoded source video, not detector caches, production observations, MEVA activity labels or KPF geometry. Prior YOLO full-video caches were closed before these human labels were made. Post-hoc matching reads human boxes afterward. Runtime person crops use production person tracks only. Raw videos remain external to Git.

## H–J. Guarded detection, tracking and crop comparison

[`scripts/evaluate_object_identity_detection.py`](../scripts/evaluate_object_identity_detection.py) matches boxes globally one-to-one at IoU ≥0.5 with class compatibility (`bag` → backpack/handbag/suitcase; `vehicle` → car/truck/bus/motorcycle; unknown is class-agnostic within evaluated mobile-object classes). It suppresses same-class duplicate boxes and retains raw counts. It **refuses precision/recall** until the pilot is complete. The following run used `--allow-partial-diagnostic` and means only that a compatible model box overlaps a manually selected target. Unlabeled other objects prohibit any false-positive denominator. Both models used frozen YOLO11n 640/.30 detections cached over the complete eight videos; the crop configuration uses 2.0× runtime-person interaction regions, never human object boxes.

| Configuration | G421 backpack | G331 suitcase | G424 SUV | Target-box overlaps | Formal precision / recall |
|---|---:|---:|---:|---:|---|
| Full frame | 10/10 | 0/10 | 10/10 | 20/30 | **Not estimable** |
| Runtime-person crop | 9/10 | 10/10 | 0/10 | 19/30 | **Not estimable** |

On these 30 frames, the full-frame cache has 169 raw evaluated-class boxes and 160 after deduplication (9 suppressed); the crop has 88 and 80 (8 suppressed). Those include unreviewed objects and are **not false-detection counts**. The partial suitcase is recovered only by the crop. The moving SUV is missed by the crop because this camera lacks usable runtime person ROIs in the selected clip. Thus the two strategies have complementary target coverage, not a demonstrated precision trade-off. Earlier full-eight-video processing wall times were 229.447 s full frame and 225.124 s for person crops on this host; these are historical cache measurements, **not** a new 150-frame timing study.

[`scripts/evaluate_object_identity_tracking.py`](../scripts/evaluate_object_identity_tracking.py) reports only reviewed-target continuity. Full-frame matched G421 and G424 each use one provisional ID across ten matched sampled frames; G331 has no matched target. The crop matched G331 in ten frames with **two** provisional IDs and one change among matched pairs; G421 in nine frames with **two** IDs and two changes; G424 is unmatched. These are provisional-linker changes on three reviewed targets, **not a global identity-switch rate**. HOTA, IDF1, tracking precision/recall, and occlusion-reappearance performance remain uncomputed.

## K–L. Limits and next decision

The dataset is **not yet sufficient for a dedicated object-tracking improvement experiment**. It proves that human-reviewed, source-linked consecutive-frame targets and guarded metrics are feasible. It does not establish detector precision, false association rate or camera generalization with only three target IDs, zero exhaustive frames, no small-object identity sequence and no second reviewer. The next work is manual: inventory the remaining 120 frames, finish the first 30, retain difficult small/occluded/uncertain objects, independently double-review a subset, and adjudicate before formal scoring. No production detector, tracker, event rule, evidence schema, pickup/placement classifier or model configuration changed.

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
.venv/bin/python scripts/prepare_object_identity_pilot.py
.venv/bin/python scripts/review_object_identity.py --port 8765
.venv/bin/python scripts/validate_object_identity.py --allow-incomplete
.venv/bin/python scripts/evaluate_object_identity_detection.py --allow-partial-diagnostic
.venv/bin/python scripts/evaluate_object_identity_tracking.py --allow-partial-diagnostic
```

Generated validation/evaluation JSON stays ignored under `data/meva/object-identity/`; only small human annotation records, clip metadata, tools, tests and docs are committed. Historical reports and MEVA footage were not overwritten.
