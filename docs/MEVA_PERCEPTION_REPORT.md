# VIGILIA — controlled MEVA perception experiment

This is an evaluation of detector and tracking coverage on the eight selected real MEVA videos. It keeps commit `6f7da5fe64ec1c4241ccb6ab80b652c50f55eaa3` as the frozen full-run baseline. MEVA activity labels and actor boxes are used **only after inference** in `backend/meva` and diagnostic scripts. Production event rules, sampling, retrieval, and the independent DIRECT matching rule were not changed for the experiment. These 89 labels were visible during diagnostic selection, so this is not a held-out generalization claim.

## Frozen configuration and benchmark

The dataset has eight videos, 2,400.833 seconds, 72,025 source frames and 294 activity annotations: 89 DIRECT, 26 APPROXIMATE, 179 UNSUPPORTED. The frozen configuration is YOLO11n (`models/yolo11n.pt`, SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`), ByteTrack's installed `bytetrack.yaml` defaults, 2 FPS source-frame sampling, image size 640, confidence 0.30, and the temporal rules in commit `6f7da5f`. The tracker has `track_high_thresh=0.25`, `track_low_thresh=0.1`, `new_track_thresh=0.25`, `track_buffer=30`, `match_thresh=0.8`, and score fusion enabled. No event threshold or mapping was retuned.

The frozen full run had 89 DIRECT GT, 17 predictions, 8 matches, 47.06% precision, 8.99% recall, 52.94% unmatched-prediction fraction, and 0.5976 mean temporal IoU on matches. Vehicle starts were 3/32 matched, stops 5/27, pickups 0/15, placements 0/15. Its raw actor coverage was 16/32 starts and 17/27 stops, while stored matching tracks covered 12/32 and 13/27. No manipulation object was matched by the raw detector in its activity interval. The full run generated 2,443 predictions with 2,443 valid source-evidence chains. The immutable local run remains `data/meva/improved-2fps/`.

## Fixed-frame detector comparison

The diagnostic subset is **all 89 DIRECT annotations**, rather than a selected set of successes. It contains 382 unique exact 2 FPS source frames across all eight cameras, including all 15 pickups and 15 placements. Each configuration predicts from the same original pixels. Annotation boxes are compared afterward at IoU ≥ 0.1; a single overlapping frame counts as raw actor coverage, not an event or verified identity. Vehicle classes are car/truck/bus/motorcycle. MEVA's `other` label is not an object class, so the table distinguishes any non-person/non-vehicle box overlap from a class usable by VIGILIA's unchanged portable-object rule. The complete per-actor hits, confidence, IoU, sample runs, box sizes, and nearest misses are in the ignored local JSON results and reproducible with `scripts/experiment_meva_detector.py`.

| Model | Size | Confidence | Vehicle start | Vehicle stop | Vehicles ≥2 hits | Object overlap | Production-supported object | Detections/frame | Inference seconds* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 640 | .20 | 19/32 | 20/27 | 35/59 | 0/30 | 0/30 | 10.24 | 8.51 |
| YOLO11n | 640 | .25 | 18/32 | 19/27 | 33/59 | 0/30 | 0/30 | 8.57 | 9.11 |
| **YOLO11n baseline** | **640** | **.30** | **16/32** | **17/27** | **31/59** | **0/30** | **0/30** | **7.46** | **8.52** |
| YOLO11n | 640 | .35 | 15/32 | 17/27 | 27/59 | 0/30 | 0/30 | 6.51 | 10.02 |
| YOLO11n | 640 | .40 | 14/32 | 14/27 | 25/59 | 0/30 | 0/30 | 5.50 | 8.77 |
| YOLO11n | 960 | .30 | 27/32 | 22/27 | 43/59 | 0/30 | 0/30 | 10.01 | 16.14 |
| YOLO11n | 1280 | .30 | 27/32 | 24/27 | 49/59 | 1/30† | 0/30 | 12.00 | 28.35 |
| YOLO11s | 640 | .30 | 26/32 | 25/27 | 46/59 | 0/30 | 0/30 | 9.91 | 15.04 |

*Inference wall time is for 382 frames on this Mac, excludes video seeks and CSV/report generation, and is not a hardware-generalized speed claim. Different runs show normal timing noise. YOLO11s weights were obtained from the [official Ultralytics release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt), SHA-256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` (18.4 MB). [Ultralytics documents YOLO11s as a compatible detection model](https://docs.ultralytics.com/models/yolo11/).

†The only 1280 overlap was a `laptop` box crossing annotation G421:22's handheld item (IoU 0.2211). It is neither a validated identification of that item nor a class used by pickup/placement extraction. There are **zero production-supported object detections in every full-frame configuration**. Twenty-one of the 30 annotated object boxes have median width below 40 source pixels; 28/30 annotations are from G421, many showing repeated small handheld items in a staged setting. [The 30-row object review](meva-perception-objects.csv) records each source box, visual review, full-frame hit and crop-probe hit. Some items are visually ambiguous, and MEVA's generic `other` type does not establish a COCO class. The [26-row missed-vehicle review](meva-perception-vehicle-misses.csv) records size, image location, recovery in each setting, and a conservative single-frame visual attribution. Several G328/G336/G639 misses are distant small cars; G436 and G638 cases are occluded by foliage or structure; G424 has green-tinted corrupt source frames; the G421 window cases have no discernible vehicle in the reviewed frame. These review tags do not imply a validated visibility label for every frame in an interval.

The activity annotations do not exhaustively label all objects in each frame. The machine-readable CSV reports unmatched detector boxes per frame only as a workload/quality diagnostic; that quantity is **not** a validated false-detection rate. The full benchmark's unmatched-event fraction is separately reported under the unchanged independent matching rule.

## Optional person-centered object probe

An evaluation-only prototype starts with full-frame YOLO11n 640/.30 **person detections**, expands each detected person's box, and runs YOLO on each resulting crop. MEVA geometry is consulted only after all crops and predictions are complete. On 49 unique object-activity frames it examined 400 person crops in 17.139 seconds, finding broad overlaps for 15/30 object annotations. Twelve annotations had a cup or cell-phone overlap; the remaining overlapping classes included chair and skateboard, which are not safe object identifications. No overlap was in VIGILIA's current portable set (`backpack`, `handbag`, `suitcase`, `bag`, `briefcase`, `bottle`). The crop experiment was **not** fed into production tracking or event generation and earns zero benchmark credit. It shows that full-frame scale is a real obstacle, while ontology, association, continuity, false associations, and computational cost remain unresolved.

## Full unchanged-pipeline result and decision

YOLO11s 640/.30 was run through the full eight videos with production ByteTrack and unchanged event logic. Raw vehicle coverage rose from 33/59 to 51/59; corresponding stored-track coverage rose from 25/59 to 42/59 (starts 21/32, stops 21/27). Five raw-detected start actors and four raw-detected stop actors still lacked matching stored tracks. The independent DIRECT benchmark produced 89 GT, 15 predictions, 8 matches: 53.33% precision, **8.99% recall**, 46.67% unmatched predictions, and 0.6840 mean temporal IoU. Starts remained 3/32 matched and stops 5/27; pickup and placement remained 0/15 each. It generated 3,053 events, all with valid evidence chains and passing event/evidence/graph API checks. The extra tracked vehicles mainly moved cases into the **event-extraction failure** stage, rather than improving final matches; starts had 16 such misses and stops 15 in the full diagnostic. This is evidence against adopting YOLO11s solely for its larger raw-coverage number.

YOLO11n 960/.30 processed the same eight full videos, reaching 49/59 raw vehicle annotations and 42/59 stored matching tracks (starts 24/32, stops 18/27). Three raw-detected start actors and four raw-detected stop actors still lacked a matching stored track. The unchanged event logic produced six same-actor start hypotheses and six stop hypotheses among those tracked labels, yielding one more independently matched start than baseline. No manipulation event was produced.

| Full-run measure | Frozen YOLO11n 640/.30 | YOLO11s 640/.30 | YOLO11n 960/.30 |
|---|---:|---:|---:|
| DIRECT GT / predictions / matches | 89 / 17 / 8 | 89 / 15 / 8 | 89 / 21 / 9 |
| Precision / recall | 47.06% / 8.99% | 53.33% / 8.99% | 42.86% / 10.11% |
| Unmatched prediction fraction | 52.94% | 46.67% | 57.14% |
| Mean temporal IoU on matches | 0.5976 | 0.6840 | 0.6246 |
| Vehicle starts matched / GT | 3 / 32 | 3 / 32 | 4 / 32 |
| Vehicle stops matched / GT | 5 / 27 | 5 / 27 | 5 / 27 |
| Pickups / placements matched | 0 / 15; 0 / 15 | 0 / 15; 0 / 15 | 0 / 15; 0 / 15 |
| Raw vehicle actor coverage | 33 / 59 | 51 / 59 | 49 / 59 |
| Stored vehicle track coverage | 25 / 59 | 42 / 59 | 42 / 59 |
| Total source-linked predictions | 2,443 | 3,053 | 3,145 |
| Valid evidence chains | 2,443 / 2,443 | 3,053 / 3,053 | 3,145 / 3,145 |
| Eight-video processing wall time* | 167.728 s | 267.381 s | 287.651 s |
| Detector + tracker wall time* | 102.857 s | 200.267 s | 215.618 s |
| Mean search latency* | 78.694 ms | 107.586 ms | 112.789 ms |

*Host-specific observations on the same footage and 4,805 sample frames; not hardware-generalized speed claims. Decode, detector/tracker, event extraction, and indexing are separately recorded in each `processing-results.json`. Search latency includes different indexes/result counts, so it is not solely a detector-cost measure. YOLO11s has 9,458,752 parameters and a 19,313,732-byte weight file versus YOLO11n's 2,624,080 parameters and 5,613,764-byte file. Neither full candidate improves pickup or placement recall. The 960 run gains **one** match while adding three unmatched DIRECT predictions and about 71% processing wall time. YOLO11s gains no DIRECT match while adding about 59% processing wall time. Mean IoU is calculated over different matched sets; its increase does not imply that every shared event localized better.

The retrieval layer did not turn better raw coverage into a clear search gain. DIRECT Precision@1 was 0.1667 frozen and 0.1111 for each candidate. Precision@5 was 0.0778, 0.0889, 0.0889; Recall@5 was 0.0376, 0.0438, 0.0561; MRR was 0.1667, 0.1500, 0.1667. These values are kept separate from detection metrics. Both candidate runs passed all event, evidence, and scoped-graph API checks, decoded eight source thumbnails, and had zero search-integrity failures.

The production configuration **remains YOLO11n 640/.30 with ByteTrack at 2 FPS**. The raw-coverage gains are substantial, but they do not translate into meaningful event recall; 960 adds a single start match at lower precision and much higher cost. A ByteTrack parameter sweep is not supported as the next experiment: stored coverage rises by 17 actors with YOLO11s, median longest matching vehicle segments remain about five sampled frames, and most additional tracked labels still fail during event extraction. On YOLO11s, only five of 21 tracked starts and six of 21 tracked stops generate a same-actor hypothesis in the diagnostic window. On YOLO11n 960, the counts are six of 24 and six of 18. Fragmentation is present but secondary: the YOLO11s diagnostics find multiple matching fragments for four of 21 tracked starts and two of 21 tracked stops. No cross-camera identity or identity-switch rate is claimed.

The highest-leverage next work is **two separate controlled changes**, with new independent evaluation before adoption: (1) investigate why tracked vehicle motion rarely becomes a correctly localized start/stop under the existing temporal-state definitions, and (2) build a selective, class-aware small-object ROI detector with stable object tracks, then determine whether cup/phone-like MEVA targets should map to a supported manipulation ontology. The current all-person crop prototype is too expensive and produces some unrelated class overlaps; simply adding `cup` or `cell phone` to the event rule would be ungrounded without association and temporal validation. The 179 UNSUPPORTED MEVA activities remain excluded from accuracy credit; all 30 DIRECT manipulation labels remain in the denominator as misses. MEVA actor IDs and activity times never enter runtime generation.

## Reproduce the comparison

The source videos and annotation repository stay outside Git. From the repository root, set `MEVA_ROOT="$HOME/VIGILIA_DATA"` and install the documented backend/model dependencies. The fixed subset is derived from `data/meva/improved-2fps/failure-diagnostics.json`; it is never passed to production inference. To reproduce an individual raw run and the combined CSV tables:

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
.venv/bin/python scripts/experiment_meva_detector.py \
  --model models/yolo11n.pt --size 640 --confidence 0.30 \
  --output data/meva/perception-experiment/yolo11n-640-c030.json
.venv/bin/python scripts/summarize_meva_perception.py \
  --run-dir data/meva/perception-experiment
```

Repeat the first command with the remaining model/size/confidence rows in the table and a distinct output filename before running the summarizer. The stronger weight can be downloaded by Ultralytics with `.venv/bin/python -c 'from ultralytics import YOLO; YOLO("models/yolo11s.pt")'`; verify its SHA-256 above. The optional ROI probe is:

```bash
.venv/bin/python scripts/probe_meva_person_roi.py \
  --model models/yolo11n.pt --size 640 --confidence 0.30 \
  --output data/meva/perception-experiment/person-roi-yolo11n-640-c030.json
```

The exact full-run commands used in this comparison, each with an isolated fresh run directory, are:

```bash
export PERCEPTION_BACKEND=yolo SAMPLE_FPS=2 DETECTION_CONFIDENCE=0.30
export ENABLE_CLIP=false ENABLE_OCR=false
YOLO_MODEL=models/yolo11s.pt DETECTION_SIZE=640 \
  .venv/bin/python scripts/run_meva.py --run-dir data/meva/perception-yolo11s-640-c030
YOLO_MODEL=models/yolo11n.pt DETECTION_SIZE=960 \
  .venv/bin/python scripts/run_meva.py --run-dir data/meva/perception-yolo11n-960-c030
```

`run_meva.py` validates all files and hashes, processes the real videos, independently scores DIRECT and APPROXIMATE activities, checks the evidence APIs, and writes complete GT/prediction/error and detector-coverage artifacts. The 179 UNSUPPORTED activities remain in dataset statistics and receive no accuracy credit. Generated AVIs, model weights, databases, and full local benchmark artifacts are ignored by Git.

## Validation and boundaries

The full backend suite passed **68 tests**; evaluation-tool Ruff critical/import checks and Python compilation passed. The frontend TypeScript check and optimized production build passed. All eight manifest videos, hashes, annotation YAML files, and geometry files validated in each full run. The YOLO11s run passed 3,053 event/evidence/scoped-graph endpoint checks and the 960 run passed 3,145, with eight decodable source thumbnails each and no evidence-chain or search-integrity failures. The recorded production-code SHA-256 is identical in baseline and both candidates: `9e12fd1dda5307cd2ea37a783824b095ec038db43aca52e138ac25126906e8a4`. No production setting, event rule, benchmark mapping, DIRECT denominator, or annotation was changed.

This experiment does **not** establish general-world activity accuracy. The subset was selected from known MEVA labels, camera scenes differ greatly, 28/30 manipulation annotations are from one camera, generic `other` objects lack a verified class label, and some source frames are occluded or corrupted. A raw one-frame overlap is weaker than a stable object track, and an event match under the independent rule is weaker than manual activity adjudication. Cross-camera identity was not evaluated. These limits are why the default production detector remains unchanged despite materially better raw vehicle coverage.
