# MEVA vehicle state reasoning experiment

This is a diagnostic of the frozen `ba838cf` VIGILIA configuration: YOLO11n, 640-pixel inference, confidence 0.30, 2 FPS and ByteTrack. The independent MEVA matching rule, 59 vehicle labels (32 starts, 27 stops), source tracks and runtime event code were held fixed. No annotation field enters the candidate algorithms. The 89-label DIRECT denominator remains unchanged for the full benchmark.

## Track evidence and failure attribution

Run `./scripts/analyze_meva_vehicle_states.py` to regenerate [the per-track CSV](meva-vehicle-state-analysis.csv) from the completed eight-video SQLite artifacts. It contains 129 annotation–track rows across the baseline and two diagnostic detector runs. The local `data/meva/vehicle-state-traces.json` has every source frame, box, box height/center, raw and three-edge median normalized speed, acceleration/deceleration, gap, and estimated state. Estimated states are **not** MEVA ground truth. Historical observations persist only a *track-mean* detector confidence; the trace marks this explicitly rather than inventing per-frame scores.

The existing rule requires at least seven samples, a stationary interval of at least 1.5 seconds, active normalized speed of at least 0.28 box heights/second, and corroborating displacement. It uses later samples to verify a transition and reports only the observed transition interval. A gap over 1.5 seconds breaks a continuous track in normal processing.

The table counts **track–annotation pairs**, not distinct GT activities. One activity can overlap multiple unrelated or fragmented tracks, and only a match whose evidence observation belongs to that exact track counts as `MATCHED`. The named failure reason is a diagnostic first failing condition, not a human action adjudication.

| Diagnostic reason | YOLO11n 640 baseline | YOLO11s 640 | YOLO11n 960 |
|---|---:|---:|---:|
| Matched event's evidence track | 8 | 8 | 9 |
| No stationary state preceding start | 5 | 9 | 8 |
| No sustained motion after start | 2 | 7 | 3 |
| No stationary state after stop | 7 | 10 | 9 |
| Motion too slow or unobserved | 1 | 4 | 3 |
| Insufficient track duration | 0 | 4 | 10 |
| Candidate outside annotated window | 4 | 4 | 4 |
| Candidate near window but failed boundary/spatial assignment | 3 | 3 | 4 |

This explains the extra detector tracks: many begin after the stationary phase, end before a verified stop, remain too short, or follow a nearby but different visible object. At 2 FPS, a short transition can fall between samples. A larger model creates more tracks but cannot restore a missing pre-transition state on the same stable identity. The trace also exposes box jitter and perspective-dependent scale, though a definitive perspective correction would need camera geometry that is not available here.

Representative source-derived cases (seconds are video-relative):

| Case | Measured evidence | Interpretation |
|---|---|---|
| G424 annotation 4, start | Baseline track `TRK-B25700125A1A`, 8.5–17.0 s, observed stationary history 2.833 s before GT start 14.333 s; candidate 14.5–16.0 s | Correctly matched transition. |
| G424 annotations 5 stop / 6 start | Baseline track `TRK-2FD50FE4D693`, 17.0–24.5 s, median normalized speed 1.0963; no stationary time after stop or before next start | Immediate annotated stop/start is not established as a full halt by this track; the rule abstains. |
| G336 annotation 15, start | YOLO11s tracks split at 86.0/88.5 s around GT start 85.867 s; earlier fragment has stationary evidence and later fragment has motion | Neither fragment alone establishes a continuous stationary-to-moving transition; joining them would require justified identity continuity. |
| G336 annotation 9, start | YOLO11n 960 actor-overlapping track 124.0–125.0 s, only three samples against 122.067–125.7 s GT | Insufficient duration for a start claim. |
| G424 annotation 10, start | Baseline track has start hypotheses 26.5–31.5 and 30.0–31.5 s, GT 35.033–38.267 s | Rule generates a transition, but it is too early; changing the matching window would conceal a timing error. |

## Offline alternative formulations

`./scripts/compare_meva_vehicle_rules.py` evaluates three predeclared geometry-only alternatives on **all** stored vehicle tracks from the frozen run. It reuses the production independent camera/type/temporal/spatial assignment (2 s tolerance, minimum temporal IoU 0.1, minimum spatial IoU 0.1). Its `current` row reproduces the exact frozen 17 predictions and 8 matches, a useful sanity check. These are offline proposals, not new VIGILIA events or a new production benchmark.

| Formulation | Vehicle predictions | Matches / 59 GT | Start matches / 32 | Stop matches / 27 | Precision | Mean matched IoU | Mean start/end error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current state machine | 17 | 8 | 3 | 5 | 47.06% | 0.5976 | 0.854 / 0.596 s |
| Velocity transition | 10 | 7 | 2 | 5 | 70.00% | 0.5293 | 0.767 / 0.614 s |
| Change-point median | 15 | 8 | 3 | 5 | 53.33% | 0.5097 | 0.658 / 0.875 s |
| Three-window persistence | 15 | 8 | 3 | 5 | 53.33% | 0.5097 | 0.658 / 0.875 s |

The velocity formulation improves precision by making fewer claims, but **loses a correct start**. The other two do not improve matched recall and localize matched intervals less accurately. Their higher precision comes from fewer predictions, not better coverage. The observed evidence therefore does **not** justify replacing the production vehicle rule. There is no before/after production gain to claim: vehicle starts remain 3/32 and stops 5/27, with 8/59 vehicle matches. The full DIRECT benchmark remains 8/89, precision 47.06%, recall 8.99%, mean IoU 0.5976. All 179 UNSUPPORTED activities remain outside accuracy credit.

A fresh full eight-video run in ignored `data/meva/vehicle-object-verification/` independently reproduced 17 DIRECT predictions, eight matches, nine unmatched predictions, 81 missed labels, mean start/end error 0.854/0.596 seconds, and 2,443 total source-linked predictions. All 2,443 evidence chains and corresponding event/evidence/scoped-graph API checks passed; eight source thumbnails decoded and search-integrity failures were zero. It decoded 72,025 frames, sampled 4,805, and spent 160.057 seconds processing on this host, of which 97.068 seconds were detector/tracker work. Mean search latency was 77.900 ms. These timings are host-specific and the rule experiment did not change production compute cost.

| Full eight-video DIRECT measure | Frozen `ba838cf` | Fresh unchanged-production verification |
|---|---:|---:|
| GT / predictions / matches | 89 / 17 / 8 | 89 / 17 / 8 |
| Precision / recall | 47.06% / 8.99% | 47.06% / 8.99% |
| Mean matched temporal IoU | 0.5976 | 0.5976 |
| Vehicle start / stop matches | 3 / 5 | 3 / 5 |
| False discovery rate | 52.94% | 52.94% |
| Valid evidence chains | 2,443 / 2,443 | 2,443 / 2,443 |

`tests/test_meva_selective_diagnostics.py` protects the diagnostic abstention boundary on parked jitter and missing/ambiguous object observations. Existing `tests/test_temporal_intelligence.py` continues to protect the production real-MEVA deceleration, short gap, parked jitter, stop/start and source-evidence behavior. No temporal rule, detector setting, tracker parameter, benchmark matching definition, or production evidence chain changed in this experiment.

Final validation: **72 backend tests passed**; Python compilation, frontend TypeScript checking and the optimized production build passed. Ruff passed on every added Python file. A whole-repository Ruff run still reports 55 pre-existing findings in unchanged files, so the repository-wide style check is not clean; none originates in this change.

The highest-leverage vehicle work is a **verified short-gap identity continuity test** on fragmented same-camera tracks, with negative examples for two nearby vehicles. Simply relaxing stationary or motion thresholds would turn unobserved states into asserted events. Such work requires a held-out evaluation set or visual identity adjudication before production adoption.

With the external MEVA files configured as in [MEVA.md](MEVA.md), reproduce the fixed baseline and offline comparison without the historical candidate run directories:

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
export PERCEPTION_BACKEND=yolo YOLO_MODEL=models/yolo11n.pt
export SAMPLE_FPS=2 DETECTION_SIZE=640 DETECTION_CONFIDENCE=0.30
export ENABLE_CLIP=false ENABLE_OCR=false
.venv/bin/python scripts/run_meva.py --run-dir data/meva/reasoning-reproduction
.venv/bin/python scripts/analyze_meva_vehicle_states.py \
  --runs data/meva/reasoning-reproduction \
  --csv docs/meva-vehicle-state-analysis.csv \
  --traces data/meva/vehicle-state-traces.json
.venv/bin/python scripts/compare_meva_vehicle_rules.py \
  --run-dir data/meva/reasoning-reproduction
```
