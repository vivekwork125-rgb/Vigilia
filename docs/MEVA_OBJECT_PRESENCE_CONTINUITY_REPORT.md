# MEVA runtime-only object presence and continuity

## A. Executive summary and decision

**Decision F — evaluation is inconclusive for reliable physical-object continuity. Production change: NO.** This evaluation-only experiment processed all eight selected MEVA videos at 2 FPS, generated detections before reading activity annotations, and compared three isolated configurations. Full-frame YOLO11n and YOLO11s had **0/30** class-agnostic box overlaps with the annotated manipulation objects at IoU ≥0.3. A person-triggered interaction crop produced overlaps in **12/30** activity annotations, but each overlap had at most **one annotated sampled frame on any single provisional track**. These responses do not establish that the same physical object persisted through a pickup or placement. They are concentrated in one camera, are class-ambiguous, and do not support a production manipulation event.

The data does show a useful detector contrast: on annotated vehicle-control frames, full-frame YOLO11s overlaps 703/1,159 frames versus 451/1,159 for YOLO11n. This is **annotated-box frame coverage**, not detector recall over an exhaustive object inventory, and is separate from manipulation recognition. The crop configuration is triggered by frozen runtime person tracks and therefore has almost no coverage in cameras without such tracks. Neither a larger full-frame detector nor these crops demonstrate robust object identity or person-object possession.

## B. Historical production baseline

At commit `9fadd3b`, production used YOLO11n, 640 input pixels, confidence 0.30, ByteTrack and 2 FPS. The historical eight-video DIRECT benchmark had 89 ground-truth activities, 17 predictions, 8 temporal matches, 47.06% precision, 8.99% recall and mean matched temporal IoU 0.5976. It matched 3/32 vehicle starts, 5/27 vehicle stops, 0/15 pickups and 0/15 placements; 2,443/2,443 production predictions passed the evidence-chain validation. **None of these are new event-benchmark measurements.** The [full-video manipulation-confirmation audit](MEVA_MANIPULATION_CONFIRMATION_REPORT.md) reported only 2/15 pickup and 3/15 placement candidate matches after its best recall-preserving gate, with 5.0% and 8.8% candidate precision. This experiment creates no production events and does not rerun that event benchmark.

## C. Dataset and reviewed subset

The [manifest](../datasets/meva/manifest.json) identifies source videos, source SHA-256, camera, measured 30 FPS, dimensions and annotation files. All eight source videos and matching annotations were resolved under `MEVA_ROOT`; the recordings span **2,400.833 s** and **72,025 source frames**. The 294 activity annotations include 15 pickups and 15 placements. The manipulation cases are a selected development set, not a held-out or exhaustive object-detection dataset.

| Video (MEVA camera) | Scene | Duration (s) | Activity annotations | Pickup / placement |
|---|---|---:|---:|---:|
| `2018-03-15.15-55-00.16-00-00.school.G421.r13.avi` | school | 300.033 | 61 | 14 / 14 |
| `2018-03-15.15-55-00.16-00-00.school.G336.r13.avi` | school | 300.233 | 35 | 0 / 0 |
| `2018-03-15.15-55-00.16-00-00.school.G424.r13.avi` | school | 300.233 | 43 | 0 / 0 |
| `2018-03-15.15-55-01.16-00-01.school.G328.r13.avi` | school | 300.067 | 55 | 0 / 0 |
| `2018-03-15.15-55-01.16-00-01.school.G639.r13.avi` | school | 300.000 | 25 | 0 / 0 |
| `2018-03-15.15-55-00.16-00-00.school.G638.r13.avi` | school | 300.000 | 17 | 1 / 0 |
| `2018-03-15.15-55-00.16-00-00.bus.G331.r13.avi` | bus | 300.267 | 26 | 0 / 1 |
| `2018-03-15.15-55-07.16-00-07.hospital.G436.r13.avi` | hospital | 300.000 | 32 | 0 / 0 |

The separate [visual-review record](../datasets/meva/object-visibility-review.json) retains all 30 cases and records approximate source-pixel size, an appearance note, review provenance and uncertainty. It is a **single selected source-frame** review based on the existing contact sheets and human notes, not verification of visibility across consecutive 2 FPS samples. The reviewer categorized 1 as clearly visible, 13 partially occluded, 2 small/potentially resolvable, 6 severely occluded or blurred and 8 unknown. The one clearly visible case has an ambiguous semantic class. Twenty-one of 30 have an annotated width below 40 source pixels, but width alone was not used to declare resolvability. Occlusion, blur, contrast, motion and temporal visibility are not completely adjudicated; unverified fields remain unknown rather than inferred from the MEVA label. No case was removed from the 30-case total.

## D–E. Runtime method and leakage controls

[`scripts/experiment_meva_object_presence.py`](../scripts/experiment_meva_object_presence.py) decodes each source video on the fixed 2 FPS grid (**4,805 sampled frames**) and runs the selected YOLO model on either the full frame or a crop centered on an already stored runtime person box. The crop uses the existing `person_roi(..., strategy="interaction", factor=2.0)` geometry from [`backend/meva/selective_objects.py`](../backend/meva/selective_objects.py). The person boxes come from the frozen `improved-2fps/vigilia.db`, not MEVA actors. A missing runtime person box yields no crop; that is a coverage failure, not a skipped ground-truth case. The full-frame runs process every sampled frame regardless of annotations. Per-video JSON caches permit resumption after a failed video without rerunning completed videos, and capture model/source hashes, class, confidence, box, camera, source frame and seconds. Crops are mapped back to source coordinates.

[`backend/meva/object_presence.py`](../backend/meva/object_presence.py) forms **evaluation-only provisional tracks** by suppressing same-class crop duplicates and greedily assigning at most one detection per track and one track per detection. Links require same video/class, compatible size, bounded normalized displacement and a gap ≤1.1 s. Near-tied crossing links abstain and start a new track. A track becomes “persistent” after two detections; this describes a linked box sequence, **not verified physical identity**. The evaluator saves each track's detection IDs, frames, timestamps, class, source SHA-256 and association method in its ignored `evaluation.json`. The frozen production ByteTrack observations remain separate; no production tracker setting or database schema changes. The existing production run previously had no complete person-plus-portable-object pair on these 30 manipulation intervals.

Only [`scripts/evaluate_meva_object_presence.py`](../scripts/evaluate_meva_object_presence.py) opens the MEVA activity and geometry files and the separate visibility review. It does so **after all eight runtime video caches exist**. It scores annotated manipulation-object boxes at the exact sampled source frame by class-agnostic IoU ≥0.1 (diagnostic) and ≥0.3 (stricter); it scores known-type bicycle and vehicle actors at IoU ≥0.3 with compatible detector classes. One-to-one frame assignment is provided for unambiguous box matching, but the descriptive per-actor coverage below takes each actor's best compatible box; crowded frames can therefore share a hit. No event candidate is produced, so event-level one-to-one temporal precision/recall is not applicable. Neither annotation times, actor IDs, GT boxes, GT crops, visual-review labels nor matched predictions are read by the detector or object association code. The same 30 annotations remain in the post-hoc denominator for every configuration.

## F. Detection results

All models used local weights and Ultralytics 8.4.174, Torch 2.14.1, 640 input pixels, confidence 0.30 and the same 2 FPS grid. Inference ran on CPU on this host; MPS was available but not selected. YOLO11n weights SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`; YOLO11s: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`. All COCO detector classes were retained; the manipulation comparison is class-agnostic because MEVA `other` does not identify a reliable COCO category.

| Evaluation-only configuration | Inference calls | All detections | Manipulation boxes overlapped ≥0.3 | Bicycle-control frames overlapped | Vehicle-control frames overlapped |
|---|---:|---:|---:|---:|---:|
| YOLO11n full frame | 4,805 | 29,071 | 0/30 cases | 5/255 | 451/1,159 |
| YOLO11s full frame | 4,805 | 41,236 | 0/30 cases | 40/255 | 703/1,159 |
| YOLO11n runtime-person interaction crop, 2.0× | 7,330 crops | 21,134 | 12/30 cases | 0/255 | 4/1,159 |

The crop's control coverage is expected to be poor because it only inspects regions around stored people. The 12 manipulation overlaps occur **only in G421**: six pickup annotations and six placement annotations, each on 1–2 annotated sampled frames. Detected labels at their best overlaps are mostly `cup`, with one `wine glass` and one `cell phone`; these class outputs are not validated semantic identities. The 12 correspond to 12 MEVA actor IDs but do not demonstrate 12 distinct physical objects: several are repeated interactions in one scene, and the annotation geometry is sparse. The G331 clearly visible but class-ambiguous item had no overlapping detection; the G638 case had none. A larger full-frame model improves known vehicle coverage but not the target manipulation-object overlap.

| Visual-review stratum | Cases | Full-frame YOLO11n / YOLO11s overlap | Person-crop overlap | Crop cases touching a persistent provisional track |
|---|---:|---:|---:|---:|
| Clearly visible, class ambiguous | 1 | 0 / 0 | 0 | 0 |
| Partially occluded | 13 | 0 / 0 | 11 | 9 |
| Small, potentially resolvable | 2 | 0 / 0 | 1 | 1 |
| Severely occluded or blurred | 6 | 0 / 0 | 0 | 0 |
| Unknown | 8 | 0 / 0 | 0 | 0 |

For the 21 sub-40px cases, the crop overlaps 12; for nine cases at least 40px wide it overlaps zero. This does not imply that smaller objects are intrinsically easier: the small hits cluster in one camera and the larger cases differ in scene, occlusion and person-track availability. Across all cases there are only **52 GT object-geometry frames on the 2 FPS grid**. MEVA geometry here is not an exhaustive frame-by-frame inventory, so **object detection precision, formal recall and false-detection rate are not estimable**. The table reports annotated-box overlap/coverage only. “Missed visible objects” is at least the one reviewed clearly visible case in all three configurations, subject to its single-frame review and ambiguous class.

## G–H. Tracking and person association

| Configuration | Nonperson detections after same-class overlap deduplication | Tentative / persistent provisional tracks | Median persistent duration | Persistent tracks with ≥1 sampled gap | Ambiguous-predecessor track starts | Persistent tracks uniquely near a runtime person |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n full frame | 20,423 | 2,051 / 650 | 2.0 s | 248 | 913 | 130 |
| YOLO11s full frame | 30,195 | 3,550 / 1,038 | 2.0 s | 406 | 1,780 | 191 |
| YOLO11n interaction crop | 16,299 | 5,002 / 1,768 | 1.5 s | 858 | 2,135 | 864 |

These are **provisional tracklet counts** from one transparent association baseline, not ByteTrack MOT scores. A “gap” means at least one missing 0.5 s sample within a linked tracklet. “Ambiguous predecessor” means the association rule deliberately refused a near-tied link; it is not a measured identity switch. The detector class, box sequence, physical-object identity and semantic identity are four different claims. The available sparse GT does not establish true identity switches, false track associations, continuity after occlusion or reappearance, or object-track precision. The crop produces more short tracklets partly because overlapping person crops and class changes fragment proposals.

For manipulation annotations, full-frame configurations have **0/30** cases touching any matched object track. The crop has **10/30** cases touching a track that is persistent somewhere in the video, but **0/30** have two annotated sampled frames matched to the **same** provisional track. Thus the crop does not establish target-object continuity through an interaction. The evaluator also applies the existing runtime-only proximity association to persistent tracks: it finds 130, 191 and 864 uniquely near a person across all video tracks in the three runs. Those are counts of a spatial candidate, not verified object-person associations, motion coupling, carrying or possession. No association precision or recall is supportable from this GT. Because the target object's identity is not established, the analysis stops here; no pickup or placement candidate is generated and event precision/recall/IoU are **not applicable**.

Known-type controls show that full-frame YOLO11n overlaps 75/144 annotated vehicle actors and YOLO11s 124/144 at least once, but 20 and 27 of those actors respectively touch multiple provisional tracklets. These counts show fragmentation may occur; they do not certify ID switches. Bicycle control coverage is 3/9 actors for both full-frame models, with 5 versus 40 matched annotated sample frames. The crop is not a valid general vehicle/bicycle detector because it depends on person ROIs.

## J. Runtime and source evidence

| Configuration | Video-processing wall time (host) | Sum of model inference times | Approx. sampled frames / wall-second | Cached artifact disk |
|---|---:|---:|---:|---:|
| YOLO11n full frame | 229.447 s | 98.228 s | 20.9 | 11 MiB |
| YOLO11s full frame | 355.488 s | 216.170 s | 13.5 | 16 MiB |
| YOLO11n interaction crop | 225.124 s | 173.322 s | 21.3* | 10 MiB |

Model object-load times were 0.022, 0.029 and 0.026 s; initialization/warmup, decoding, crop extraction, Python and I/O are included in wall time. Simple offline association took about 0.2–0.3 s per full eight-video evaluation; post-hoc annotation evaluation took approximately 12 s and is **not runtime inference**. `*`The crop run records all 4,805 sample indices but only issues 7,330 crop inferences where runtime person tracks exist, so its frame/wall-second value is not comparable with a full-frame processing FPS. Peak memory was not measured. These are host-specific timings, not hardware-generalized throughput claims. One ROI run first failed inside native inference under default threading; resuming with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` completed all eight videos. The cache and resume design retained previously finished videos.

All cached detections in all three runs were audited for unique detection ID, source video ID/camera, source-video SHA-256, sampled frame within source bounds and exact frame/FPS timestamp: **29,071/29,071**, **41,236/41,236** and **21,134/21,134** records were internally valid, respectively. The evaluator's experimental tracks reference those detection IDs and source frames. They are **not persisted production observations/events** and have no event/evidence API IDs. The historical 2,443/2,443 production evidence-chain result was not remeasured or changed by this experiment.

## K–L. Failure analysis and limitations

1. **Detector/ROI:** Both full-frame models miss all 30 manipulation-object overlaps at this threshold and size; person crops recover limited box responses in one camera. This implicates full-frame scale/context and the COCO class taxonomy, but the one clearly visible case still fails, so neither “all objects are unresolvable” nor “crops solve detection” is justified.
2. **Visibility and annotations:** Most reviewed manipulation objects are small, occluded, blurred or visually uncertain. Review covers only one selected frame per case; no consecutive-frame visibility ground truth exists. Only one case was categorized clearly visible. Sparse geometry provides 52 sampled-frame boxes across 30 activities. Thus failure cannot be cleanly apportioned among invisibility, detector miss, semantic mismatch and sampling loss.
3. **Tracking:** No target manipulation case has repeated annotated-frame evidence on the same provisional track. Class changes (`cup`/`wine glass`/`cell phone`), crop duplicates and overlapping people make identity fragile. The sparse annotations do not permit measured ID-switch or false-association rates.
4. **Person association:** Proximity counts are unvalidated. The frozen person-tracking DB has no sampled person observations in several cameras, so ROI coverage is camera-dependent. No carrying, contact or release claim follows from the current evidence.
5. **Sampling and controls:** A 2 FPS grid may miss a brief exposure or occlusion transition; this experiment did not compare higher FPS. Vehicle/bicycle controls confirm that the detector can match known-type boxes, but do not prove manipulation-object recall. GT actor geometry and human review are not exhaustive, so no trustworthy detector precision/FPR denominator exists.

The result is specific to these eight videos, their current annotations, these weights/configurations and this provisional linker. It is not a held-out accuracy estimate or a general claim about surveillance footage. No manipulation event score was produced; the historical production scores remain unchanged.

## M. Production recommendation and reproduction

**Decision F — evaluation is inconclusive. Keep production unchanged.** The highest-leverage next step is an independently reviewed, frame-sequence object-visibility and identity set spanning **multiple cameras**, with object boxes/classes and persistent physical-object IDs across consecutive 2 FPS frames plus confirmed nonobject regions. That would permit real detection precision/recall, tracking continuity and false-association measurements. Then compare a runtime person crop and full-frame detector on that held-out sequence set before considering any event rule. Until a physical object track is independently supported, VIGILIA should abstain from object-specific manipulation assertions.

From `/Users/yanalavivekreddy/VIGILIA` with the existing environment, local model weights, external data and frozen `data/meva/improved-2fps/vigilia.db`:

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/experiment_meva_object_presence.py --model models/yolo11n.pt --run-dir data/meva/object-presence-yolo11n-640-c030
.venv/bin/python scripts/experiment_meva_object_presence.py --model models/yolo11s.pt --run-dir data/meva/object-presence-yolo11s-640-c030
.venv/bin/python scripts/experiment_meva_object_presence.py --model models/yolo11n.pt --roi-strategy interaction --roi-factor 2.0 --run-dir data/meva/object-presence-yolo11n-interaction2-640-c030
for run in object-presence-yolo11n-640-c030 object-presence-yolo11s-640-c030 object-presence-yolo11n-interaction2-640-c030; do
  .venv/bin/python scripts/evaluate_meva_object_presence.py --run-dir "data/meva/$run" --output "data/meva/$run/evaluation.json" --csv "data/meva/$run/cases.csv"
done
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests
.venv/bin/ruff check --no-cache backend/meva/object_presence.py scripts/experiment_meva_object_presence.py scripts/evaluate_meva_object_presence.py tests/test_meva_object_presence.py
PYTHONPYCACHEPREFIX=/tmp/vigilia-object-pycache .venv/bin/python -m compileall -q backend tests scripts
cd frontend && npm run typecheck && NEXT_TELEMETRY_DISABLED=1 npm run build
```

The three ignored run directories each contain resumable per-video detections, configuration, `cases.csv` and `evaluation.json` with per-case and per-track provenance. No raw MEVA video or generated detection cache is committed. The 9 new focused regression tests cover missing/duplicate boxes, gaps, ambiguity, source time, deterministic linking, one-to-one frame assignment and annotation-free extraction/resumption. The full backend suite passes **136 tests**; scoped Ruff, Python compilation, frontend typecheck and production build pass. No production source, model, event rule, tracker, database schema or configuration file was modified.
