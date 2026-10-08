# VIGILIA — MEVA failure analysis and accuracy pass

The subsequent [secondary manipulation confirmation gate experiment](MEVA_MANIPULATION_CONFIRMATION_REPORT.md), [hand-object interaction experiment](MEVA_HAND_CONTACT_REPORT.md), [vehicle continuity experiment](MEVA_VEHICLE_CONTINUITY_REPORT.md), [vehicle state reasoning experiment](MEVA_VEHICLE_REASONING_REPORT.md), [selective small-object pipeline](MEVA_MANIPULATION_PIPELINE_REPORT.md), and [selective small-object experiment](MEVA_SMALL_OBJECT_REPORT.md) analyze the frozen baseline without changing production perception, temporal rules, or independent matching. Their diagnostic counts are separate from this historical full-benchmark result.

The vehicle continuity experiments found that bridging same-camera track gaps across 0.5s–3.0s produced zero false merges but did not improve downstream event matches (starts remained 3/32, stops 5/27), as vehicle event failures stem from sub-1.5s non-stationary decelerations/accelerations or raw detection absences rather than track fragmentation. The full-chain manipulation pipeline established an evaluation taxonomy (16 AMBIGUOUS, 14 UNKNOWN, 0 KNOWN_SUPPORTED) and evaluated 9 runtime ROI configurations, verifying that portable object detection remains 0/30 across all tested configurations. The hand-object interaction experiment evaluated class-agnostic contact progression on runtime person crops, capturing 60.0%–80.0% of pickup/placement intervals, but demonstrated that everyday human hand postures generate high false alarm rates on negative controls without a secondary confirmation gate. The secondary manipulation confirmation gate experiment evaluated five independent visual/temporal signals (localized motion, relative motion coupling, appearance change, region persistence, separation), demonstrating that while placement precision reaches up to 50.0%–54.5% and relative motion coupling suppresses 80.0% of negative control false alarms, sub-40px pickup objects lack sufficient pixel delta in standoff surveillance, so production remains strictly frozen at commit 89dc216.

This report compares the frozen commit 7ccb717 run with a new full eight-video run on the same MEVA sources, YOLO11n weights, ByteTrack, 2 FPS sampling, 640-pixel detector size, 0.3 confidence threshold and unchanged independent matching rule. This selected set is not a held-out generalization test. Detection accuracy remains poor despite a meaningful reduction in unsupported motion hypotheses.

## Dataset and reproducibility

The eight real MEVA drops-123-r13 videos total 2,400.833 seconds (72,025 source frames). Their 294 activity annotations cover 24 observed activity types: 89 DIRECT, 26 APPROXIMATE, and 179 UNSUPPORTED. Source paths, SHA-256 hashes, measured FPS, frame counts, camera clocks and annotation revision are in the [portable manifest](../datasets/meva/manifest.json). The [data guide](MEVA.md) documents installation and configuration. MEVA footage and annotations are attributed to Kitware/IARPA under CC-BY-4.0; the large AVIs remain external.

The detector is YOLO11n (model SHA-256 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1) with ByteTrack. The main comparison uses 2 FPS. The improved run decoded all 72,025 source frames and sampled 4,805 frames in 167.728 seconds on this host. This is an uncontrolled wall-time observation, not a speedup claim. Its mean of 104 search timings was 78.694 ms; detector and tracker time remains measured together.

## Frozen baseline versus improved full run

| Measure | 7ccb717 baseline | Improved 2 FPS |
|---|---:|---:|
| DIRECT ground truth | 89 | 89 |
| DIRECT predictions | 88 | 17 |
| DIRECT matches | 2 | 8 |
| False positives / false negatives | 86 / 87 | 9 / 81 |
| Precision / recall | 2.27% / 2.25% | 47.06% / 8.99% |
| Unmatched-prediction rate | 97.73% | 52.94% |
| Mean temporal IoU on matches | 0.3383 | 0.5976 |
| Mean start / end error on matches | 0.867 / 1.317 s | 0.854 / 0.596 s |
| Search Precision@1 / Precision@5 | 0 / 0.0111 | 0.1667 / 0.0778 |
| Search Recall@5 / MRR | 0.0043 / 0.0278 | 0.0376 / 0.1667 |
| All source-linked predictions | 2,464 | 2,443 |

“False positive” means a prediction unmatched under the independent annotation, class, geometry and timing rule; the dataset does not establish exhaustive true negatives. The reported rate is therefore false discovery rate, not a true-negative-based FPR. Fewer predictions explain much of the precision gain. Eight matches out of 89 labels still means 91% of directly mapped activities were missed. Search improved primarily because correct vehicle events exist and there are fewer incorrect vehicle events to outrank them. One of eight matched events was still absent from the top five for its query.

The pre-improvement summary is preserved in [meva-pre-improvement-results.json](meva-pre-improvement-results.json); the [new machine-readable summary](meva-results.json) and ignored local full benchmark preserve every camera/type breakdown and error record. The older [initial-rule baseline](meva-baseline-results.json) predates commit 7ccb717 and is not the baseline for this comparison.

## Per-event diagnosis

Raw coverage independently reruns YOLO.predict on 382 unique exact production sample frames, with the recorded weights, size and confidence. Annotation geometry is compared only after inference; labels never enter the model. Tracked coverage requires a class-compatible stored YOLO/ByteTrack observation whose measured box overlaps the annotated actor at an exact source sample frame by IoU at least 0.1. Neither coverage measure depends on event classification. “Rule success” asks whether a same-actor, same-type event appears in the annotation window ±2 seconds; final matches additionally obey the unchanged independent temporal and spatial assignment rule. Fragment counts count overlapping local track segments and do not prove an identity switch.

| MEVA activity → VIGILIA event | GT | Pred | Matched | Precision | Recall | Raw actor coverage | Tracked coverage | Rule success given tracking | Mean IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vehicle_starts → started_moving | 32 | 11 | 3 | 27.27% | 9.38% | 16/32 | 12/32 | 5/12 | 0.4739 |
| vehicle_stops → stopped | 27 | 6 | 5 | 83.33% | 18.52% | 17/27 | 13/27 | 6/13 | 0.6719 |
| person_picks_up_object → picked_up | 15 | 0 | 0 | — | 0% | 0/15 paired | 0/15 paired | — | — |
| person_puts_down_object → placed_object | 15 | 0 | 0 | — | 0% | 0/15 paired | 0/15 paired | — | — |

For starts, median matched start/end errors are 0.833/0.667 seconds; for stops, 1.067/0.067 seconds. The local `data/meva/improved-2fps/detector-coverage.json` stores every raw class, confidence, frame and spatial overlap. The adjacent `failure-diagnostics.csv` and `.json` give every annotation's frame, source clock, actor boxes/track fragments, stage, event ID and search rank. Local `error-analysis.csv` retains every unmatched prediction/label and nearest counterpart, and `error-contact-sheet.jpg` supports human review.

The dominant failure is raw detection coverage: YOLO misses 16/32 start vehicles and 10/27 stop vehicles, then four raw-detected examples of each fail to become matching stored tracks. Two of those raw-only examples have a single hit; others have two or three low-confidence hits (roughly 0.31–0.43), still insufficient to assert continuous motion. Pickup and placement objects have **zero raw detections** in their annotated intervals; raw person coverage is 14/15 and 15/15, respectively. The stored tracker overlaps one pickup object but no placement objects and forms no complete person-object pair. Those manipulation cases are **unsupported by current perception on this selected footage**, as a diagnostic qualification. They remain in the unchanged primary DIRECT denominator with zero credit; no difficult labels were removed. Among tracked vehicles, 7/12 starts and 7/13 stops still lack a final independent match, due to rule abstention or localization/association.

Track fragmentation is measurable in the diagnostic file, but is not the largest aggregate explanation: 3/12 visible start vehicles and 2/13 visible stop vehicles overlap more than one track, while most missed vehicle actors have no raw detection at all. The longest matched vehicle segment has a median of five sampled frames, with median track confidence 0.68 for starts and 0.66 for stops. Position, normalized displacement/speed, acceleration/deceleration, gaps and stationary spans are recorded per matching track; overlapping fragments are not asserted to be verified ID switches. Some annotated stop/start intervals describe deceleration followed immediately by acceleration without a sustained stationary state. VIGILIA abstains rather than assert a stop from merely reduced speed. Tiny manipulated objects, occlusion, camera perspective, YOLO11n class coverage and 2 FPS transitions remain limits.

Two of the nine unmatched direct predictions start at G424 frame 300, where the source decodes as a green-tinted image in both OpenCV and FFmpeg; they are retained, not filtered away. Unmatched events elsewhere may be real but unannotated activity or model errors. The error sheet is for review, not a substitute for adjudication.

## Annotation and temporal sanity check

The KPF parser reads original zero-based inclusive frame endpoints and uses each video's measured 30 FPS; an end-exclusive boundary is separately retained at (last frame + 1)/FPS. The official MEVA converter also iterates through end + 1. Recording clocks are preserved as dataset-local, with timezone unspecified; no UTC or date offset is applied. Manual source-frame checks found:

| Source example | Raw inclusive frames | Relative seconds | Source clock | Runtime geometry |
|---|---|---|---|---|
| G424 vehicle start, annotation 4 | 430–500 | 14.333–16.667 | 15:55:14.333–15:55:16.667 | Car overlaps at frames 435, 450, 465, 480, 495 |
| G424 vehicle stop, annotation 11 | 4968–5040 | 165.600–168.000 | 15:57:45.600–15:57:48.000 | Car overlaps at five sampled frames |
| G421 pickup, annotation 4 | 606–635 | 20.200–21.167 | 15:55:20.200–15:55:21.167 | Person overlaps at 615/630; object has no compatible YOLO box |

These examples and parser tests support frame/time alignment. They do not prove every human activity label is a perfect geometric stationary/moving transition.

## What changed and why

Vehicle start/stop logic now uses median speed over adjacent tracker edges, normalized by observed box height, plus separate sustained stationary and moving states. A stop's reported interval covers the observed deceleration leading into verified stillness, rather than the later stationary plateau. A start requires visible stationary history followed by sustained displacement. Real detector-box regressions cover a decelerating MEVA car, a start after a short detection gap, and parked-car jitter; additional tests cover stop-then-start, object pass-bys, manipulation ambiguity and evidence. These changes are dataset-independent production rules; no MEVA activity labels, IDs or timings enter runtime.

The processing sampler now chooses source frames on the requested wall-clock grid. Previously, a nominal 4 FPS run on 30 FPS footage sampled every eighth frame, yielding 3.75 FPS. Evidence frame derivation now rounds to the nearest decoded frame; the first higher-rate diagnostic exposed floating-point truncation that produced HTTP-200 evidence records with no source box. The exact-rate rerun passed all evidence API checks.

A controlled diagnostic on the same G421/G328 sources compared the existing 2 FPS observations with an exact 4 FPS rerun under the new rules. G421 tracked pickup objects remained 1/14 and placement objects 0/14; no complete manipulation pair appeared. G328 vehicle track coverage was 7/14 starts at both rates and 7/12 versus 8/12 stops. At 2 FPS, those two cameras had 5/54 direct matches (two starts, three stops); 4 FPS had 4/54 (one start, three stops). Higher sampling did not improve benchmark matches on this subset. The 4 FPS pass sampled 2,402 frames and took 68.428 seconds; the 2 FPS pass on the same cameras sampled 1,202 frames and took 43.024 seconds. Production remains at 2 FPS.

Retrieval rules were not retuned. The stage analysis shows seven of eight correctly matched events in the top five. Unsupported action queries still return no action hits, and no search result had the wrong event type, camera, source evidence or missing explanation signal.

## Unsupported and remaining limits

The selected subset has 179 UNSUPPORTED labels across 18 activity types: person_carries_heavy_object, person_closes_vehicle_door, person_embraces_person, person_enters_vehicle, person_interacts_with_laptop, person_opens_facility_door, person_opens_vehicle_door, person_reads_document, person_rides_bicycle, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, vehicle_makes_u_turn, vehicle_reverses, vehicle_turns_left and vehicle_turns_right. None receives detection credit. The 26 structure-entry/exit labels have only APPROXIMATE appearance/disappearance proxies and are scored separately: three matches, 638 predictions, and 635 unmatched predictions. Approached-object and unattended relevance lack compatible MEVA labels in this subset. Cross-camera identity remains unvalidated; MEVA actor IDs never become VIGILIA identities.

The full improved run validated all eight source hashes and annotation/geometry files, processed every declared source frame, and retained **2,443/2,443 valid event → evidence → observation → track → video chains**. All 2,443 event, evidence and scoped graph endpoints passed, with eight decodable source thumbnails. These are integrity checks, not activity-accuracy evidence.

The final backend suite passed **66 tests**; Ruff critical/import checks, Python compilation, frontend TypeScript checking and the Next.js production build also passed. The complete one-command runner was repeated against the finished isolated run directory and successfully resumed the eight sources, re-evaluated them, generated stage and raw-detector diagnostics, and refreshed the source-frame review sheet.

Reproduce with the documented dependencies and model, from the VIGILIA root:

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
export PERCEPTION_BACKEND=yolo YOLO_MODEL=models/yolo11n.pt
export SAMPLE_FPS=2 DETECTION_SIZE=640 DETECTION_CONFIDENCE=0.3
export ENABLE_CLIP=false ENABLE_OCR=false
.venv/bin/python scripts/run_meva.py --run-dir data/meva/reproduction
```

The runner validates the dataset, processes footage, independently scores it, writes failure-stage diagnostics and source-frame review artifacts. Choose a fresh run directory when production code or configuration changes; raw videos are never copied into Git.
