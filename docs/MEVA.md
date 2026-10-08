# Real MEVA benchmark

VIGILIA evaluates eight **real** videos from MEVA `drops-123-r13`, starting around 2018-03-15 15:55. The committed [manifest](../datasets/meva/manifest.json) identifies every source and annotation by portable path, metadata and SHA-256. The selected recordings total **2,400.833 seconds**, at measured 30 FPS, and contain **294 activity annotations**. This is a selected evaluation subset, not a training split or a claim about the entire MEVA dataset.

MEVA — Multiview Extended Video with Activities — is provided by Kitware Inc. and IARPA under **CC-BY-4.0**. VIGILIA does not own the footage. See the [official dataset](https://mevadata.org/), [annotation repository](https://gitlab.kitware.com/meva/meva-data-repo), its LICENSE, activity definitions, and KPF specification. The annotation revision and file hashes are recorded in the manifest. The small detector-box regression excerpt retains attribution; raw footage is never committed.

## Data configuration

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
# Optional overrides:
# export MEVA_VIDEO_ROOT="$MEVA_ROOT/videos/meva"
# export MEVA_ANNOTATION_ROOT="$MEVA_ROOT/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware"
```

Expected videos are under `$MEVA_VIDEO_ROOT/2018-03-15-1555/`. Annotation files are discovered below `$MEVA_ANNOTATION_ROOT` only when preparing the manifest; subsequent runs use exact recorded relative paths. No machine-specific absolute dataset paths are committed. Runtime DB rows reference the external videos in place; the integration never copies the large AVI files into the repository.

## Reproduce from already downloaded data

Install the core requirements and YOLO dependencies into the existing virtual environment, and provide `models/yolo11n.pt`. For example:

```bash
.venv/bin/python -m pip install -r backend/requirements.txt 'ultralytics>=8.3,<9' 'lap>=0.5'
export PERCEPTION_BACKEND=yolo
export YOLO_MODEL=models/yolo11n.pt
export SAMPLE_FPS=2
export DETECTION_SIZE=640
export DETECTION_CONFIDENCE=0.3
export ENABLE_CLIP=false ENABLE_OCR=false
.venv/bin/python scripts/run_meva.py --run-dir data/meva/reproduction
```

Run from the VIGILIA root. Choose a new directory when production code/model/configuration changes. Resuming a run with a different fingerprint fails. Completed source videos may be resumed under the same configuration. The benchmark refuses HOG, missing local weights, missing sources, changed hashes, failed processing, incomplete decoding, broken evidence or wrong-type search results. Weak accuracy does **not** fail the integration gate.

The equivalent separate commands are:

```bash
.venv/bin/python scripts/validate_meva.py
.venv/bin/python scripts/process_meva.py --run-dir data/meva/reproduction
.venv/bin/python scripts/evaluate_meva.py --run-dir data/meva/reproduction --tolerance 2 --min-iou 0.1 --min-spatial-iou 0.1 --verify-api
.venv/bin/python scripts/review_meva.py --run-dir data/meva/reproduction
```

`scripts/prepare_meva.py` regenerates the manifest and fully validates it when intentionally adopting a different dataset checkout. Ordinary reproduction validates the committed manifest; it does not silently replace recorded hashes.

Each run writes ignored local artifacts:

- `vigilia.db`, `run-config.json`, `processing-results.json`
- `benchmark-results.json`, `benchmark-results.csv`
- `predictions.json`, `ground-truth.json`
- `error-analysis.json`, `error-analysis.csv` — every unmatched prediction/label, plus nearest counterpart when available; nearest is diagnostic, not a match
- `error-contact-sheet.jpg` and source-frame reference JSON — a diverse sample of errors, not every error
- `.api-token` — generated local API credential, mode 0600; never committed or included in reports

The JSON records detector/tracker, weight hash, production code hash and diff against the recorded Git revision, package versions, sampling/size/confidence, unattended interval, measured processing times, and query latency. Detector and tracker wall time is combined because `YOLO.track` exposes no separate tracker measurement here. Decoded processing FPS and sampled inference throughput are reported separately. No investigation-time savings or controlled speedup is claimed.

## Isolation and timing

`backend/meva` is an external data/evaluation package. `backend/app` never imports it, enforced by a regression test. The processing controller loads **only video/camera metadata**, inserts no annotation events/actors/objects, and calls `app.processing.process`. Ground truth is read later by the evaluator, in a separate process. Evaluation cannot create production events or identities.

KPF `act.act2`, `id2`, actor `id1`, and original `tsr0` intervals are retained. KPF intervals are zero-based and inclusive (the official converter iterates through `end + 1`). Reports retain `start_frame`, `end_frame`, their actual-FPS seconds, and a separate end-exclusive boundary. Source clock timezone is unspecified, so recording times are preserved without inventing a timezone. Camera starts differ by up to seven seconds; local activity intervals remain relative to the correct source.

Validation checks opening and decoding first/last frames, positive finite metadata, recorded hashes, all activity/type/geometry YAML, valid ranges, actor references, duplicate IDs, duplicate geometry actor/frame records and box validity. Full pipeline runs decode all source frames and require decoded counts to equal the manifest.

## Matching and metrics

Primary scores include **DIRECT** activities only. **APPROXIMATE** mappings have their own scores; **UNSUPPORTED** labels are counted but never credited as successful detections. See [every mapping](MEVA_MAPPING.md).

Predictions match labels using the same camera/video/type, compatible runtime entity classes, and either temporal IoU at least 0.1 **or both interval endpoints within two seconds**. Independently annotated actor/object boxes must overlap runtime observation boxes at actual sampled frames by IoU at least 0.1. Pickup/placement require both person and object overlap. No annotation ID is compared to any generated event, track or entity ID. Maximum-cardinality one-to-one assignment prevents duplicates receiving multiple credits. Thresholds are explicit CLI options and are recorded, not adapted per camera or difficult activity.

Detection reports GT, prediction, match, FP/FN counts, precision/recall, temporal IoU and mean absolute start/end error **on matched pairs**. The reported operational `false_positive_rate` is unmatched predictions divided by predictions, technically **false discovery rate**. A true-negative-based false-positive rate is unmeasured because annotation coverage does not supply exhaustive negative intervals.

Search runs all 13 requested query forms on all eight cameras (104 queries). P@1, P@5, Recall@5 and MRR are macro averages over camera/activity queries with mapped positive labels; P@5 has a fixed denominator of five. Empty-label mapped queries contribute to a separate negative-query result rate. Unsupported queries receive no accuracy score. MEVA has no corresponding `approached_object` or visible-camera `unattended` label, so relevance accuracy for those searches remains **unmeasured**. Intentional `person_abandons_package` is incompatible with a visible stationary-object hypothesis and remains unsupported.

## Evidence and search gate

Every generated event is checked against its evidence, primary observation, supporting tracks/entities, source video/camera, frame/time range, recording clock and source-derived evidence hash. API verification exercises every event, evidence and event-scoped graph endpoint, and decodes a source thumbnail for each camera. Representative thumbnails and playback are also reviewed in the browser. Search integrity checks exact requested event type, camera, source evidence and ranking explanations; an irrelevant but semantically similar observation fails.

`GET /graph?event_id=...` includes the requested event even when it falls outside the overview graph's first 200 events. Proximity relations remain **INFERRED**; cross-camera identities are not joined using MEVA actor IDs. Single-camera detection is evaluated separately from cross-camera correlation, which remains unvalidated.

## Baseline and failure interpretation

[Baseline summary](meva-baseline-results.json) preserves the initial production-rule state, including pre-existing pickup fixes. It was recorded before semantic corrections. The baseline had 89 directly mapped labels, 290 predictions, 7 matches, 283 unmatched predictions and 82 missed labels. Pickup and placement each had zero predictions. All 1,469 generated events had valid evidence; 191 search hits violated the requested type. Source-linked errors are retained locally under `data/meva/baseline`.

Corrections separate stationary presence from a moving-to-stationary transition, require prior sustained states, split long detection gaps, compare coupling vectors in the same coordinate scale, abstain on ambiguous person association, require prior coupling and sustained separation for placement, and constrain supported/unsupported action search. Regressions include measured MEVA vehicle boxes spanning a 10.5-second detection gap. Tests do not inject labels into runtime.

G424 frame 300 (10 seconds) produces a green-tinted image in random-access and sequential OpenCV decoding and in independent FFmpeg extraction. It is retained and exposed in error analysis. This is a source/decoding-content anomaly, not a missing file or a reason to remove the camera. Small manipulated objects, occlusion, track fragmentation, detector jitter, unmodeled activity semantics and temporal localization remain major limitations. Weak results are retained. Final measured results and verification are in [the engineering report](MEVA_REPORT.md) and [result summary](meva-results.json).
