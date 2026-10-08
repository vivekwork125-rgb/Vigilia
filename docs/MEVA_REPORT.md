# VIGILIA MEVA engineering report

The eight selected MEVA sources and their 294 activity annotations validated successfully. YOLO11n + ByteTrack processed all **2,400.833 seconds** (40m 0.8s) of source video, with no silently skipped video. The final run is reproducible from the [portable manifest](../datasets/meva/manifest.json) and [documented commands](MEVA.md). Its activity detection is weak; the scores below are reported without removing difficult cases.

## Dataset and pipeline

| Item | Measured state |
|---|---:|
| Videos / cameras | 8 / 8 |
| Full source duration | 2,400.833 s |
| MEVA activities / distinct selected types | 294 / 24 |
| Directly mapped / approximate / unsupported activity instances | 89 / 26 / 179 |
| Detector / tracker | YOLO11n / ByteTrack |
| Model SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Sample rate / image size / confidence | 2 FPS / 640 / 0.3 |
| Detector device | CPU, verified when loading the local weights |
| Unattended minimum | 5 s; behavior remains an uncertain visible-camera hypothesis |
| Decoded frames / sampled frames | 72,025 / 4,805 |
| Processing wall time | 182.229 s across all eight videos |
| Decoded throughput / sampled throughput | 395.24 / 26.37 frames/s |
| Mean search latency across 104 measured camera/query checks | 78.16 ms |

Detector and tracker time is measured together inside `YOLO.track`; the exact separate tracking cost is unavailable. Indexing and event extraction times are reported separately for each video in the local `processing-results.json`. No controlled speed comparison or investigation-time reduction was measured. The specific browser query reviewed on G328 took 1.56 s, illustrating latency variance beyond the macro mean.

## Independent benchmark

Metrics use one-to-one camera, type, time, class and annotated/runtime actor-box compatibility. Directly mapped activities are the primary score. Approximate scene entry/exit proxies are deliberately separate.

| Detection measure | Direct | Approximate |
|---|---:|---:|
| Ground truth | 89 | 26 |
| Predictions | 88 | 638 |
| Matched | 2 | 3 |
| False positives / false negatives | 86 / 87 | 635 / 23 |
| Precision / recall | 2.27% / 2.25% | 0.47% / 11.54% |
| Unmatched-prediction rate (false discovery rate) | 97.73% | 99.53% |
| Mean temporal IoU on matches | 0.3383 | 0.0000 |
| Mean start / end error on matches | 0.867 / 1.317 s | 0.811 / 1.678 s |

“False positive” means an unmatched prediction under the independent labels and matching rule. It does not prove the action never occurred. A true-negative-based false-positive rate is unavailable. Zero temporal IoU for approximate matches means both endpoints fell within the two-second tolerance while the intervals did not overlap; this underscores the weakness of the proxy.

| Direct event type | GT | Predicted | Matched |
|---|---:|---:|---:|
| `picked_up` | 15 | 0 | 0 |
| `placed_object` | 15 | 0 | 0 |
| `started_moving` | 32 | 38 | 2 |
| `stopped` | 27 | 50 | 0 |

Vehicle starts are the strongest directly mapped type, but only two of 32 labels matched. Pickup, placement and stops matched none. The selected MEVA footage includes 18 other unsupported activity types covering 179 annotations, including phone use, conversation, bicycling, turns and reversals. Their labels receive no detection credit. See [all explicit mappings](MEVA_MAPPING.md) and the [machine-readable breakdown by camera/activity/event](meva-results.json).

| Search metric over 18 positive direct camera/query cases | Value |
|---|---:|
| Precision@1 | 0.0000 |
| Precision@5 | 0.0111 |
| Recall@5 | 0.0043 |
| MRR | 0.0278 |
| Negative query result rate, 14 direct cases | 0.0000 |

Approximate entry/exit retrieval had zero P@1, P@5, R@5 and MRR and an 80% result rate on five negative cases. The 13 query forms were run on all eight cameras. Unsupported action queries produced no misleading action hits; all 104 query checks had zero wrong-type/camera/evidence/explanation integrity failures. Relevant ranked retrieval remains very weak. `approached_object` and `unattended` have no compatible MEVA activity labels in this subset, so their search accuracy is unmeasured.

## Baseline, corrections and error analysis

The initial baseline was preserved **before** the temporal-rule and search corrections: 7/89 direct matches, 290 direct predictions, 283 unmatched predictions, 82 missed labels, and **191 wrong-type search hits**. The stricter rules cut unmatched direct predictions to 86 and wrong-type search hits to zero, while reducing matched labels to two. They did not improve recall. The [baseline summary](meva-baseline-results.json) preserves those measurements, model/configuration, initial production diff and all activity breakdowns.

The major observed failures are tiny or missed manipulated objects, vehicle tracking gaps and fragmentation, inconsistent action localization, structure entry/exit represented only by weak appearance proxies, and an anomalous green-tinted segment in G424. Frame 300 of G424 is green-tinted when decoded sequentially with OpenCV, through random access, and independently by FFmpeg; the camera was retained. Full local `error-analysis.csv` and `.json` include every FP/FN with source video/camera, time/frame, track/evidence references and nearest label/prediction diagnostics. `error-contact-sheet.jpg` shows a diverse, source-derived sample for human review.

Changes to motion rules require a prior sustained state, split gaps longer than 1.5 seconds, separate stationary presence from an observed stop, compare person/object vectors on one scale, abstain when association is ambiguous, require prior coupled motion for placement, and preserve uncertainty for unattended hypotheses. The real G328 car browser inspection displayed an exact source seek at nine seconds (frame 270), track timeline, the supporting bounding box and source SHA-256. A standalone regression contains seven measured MEVA detector boxes across a 10.5-second gap; additional synthetic deterministic tests cover duplicate person associations, opposite motion, jitter and false placement.

## Validation and traceability

- All eight videos opened, their hashes and metadata matched the manifest, all corresponding annotation/type/geometry YAML parsed, and intervals/IDs/frame ranges validated. All 72,025 declared frames were decoded by the processing runs.
- All **55 backend tests** passed. Python imports/compilation, focused Ruff checks, frontend TypeScript checking and Next.js production build passed.
- **2,464/2,464 predictions (100%)** passed event → evidence → observation → track → source checks, including frame/time consistency and recomputed evidence signatures. The API returned all 2,464 event and evidence records and all 2,464 scoped event graphs. One source thumbnail per video decoded successfully.
- Browser search returned only `started_moving` results for “vehicle started” on G328. Opening one result showed video playback at the expected source second with an overlay, timeline, provenance and ranking signals. Unsupported vehicle reversal returned no action hit and is explicitly labeled unsupported in the interface.
- Cross-camera identity was **not evaluated**; the application does not import MEVA actor IDs. Appearance associations remain correlations rather than verified identities.

Reproduce with `MEVA_ROOT=~/VIGILIA_DATA .venv/bin/python scripts/run_meva.py --run-dir data/meva/reproduction` after installing the documented dependencies/weights. The committed manifest is `datasets/meva/manifest.json`; the local full benchmark, CSV, error records and contact sheet are under ignored `data/meva/current/`. MEVA remains external under `~/VIGILIA_DATA` and is attributed under CC-BY-4.0.
