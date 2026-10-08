# MEVA selective small-object experiment

This experiment holds VIGILIA's production perception at **YOLO11n, 640 pixels, confidence 0.30, 2 FPS, ByteTrack**. It does not add cup or phone to the production pickup/placement ontology and does not emit runtime manipulation events. The experiment is deliberately isolated under `backend/meva/` and `scripts/`; MEVA annotations select diagnostic frames and score outputs *after* inference. Candidate ROIs, detections, tracklets and person associations use only stored VIGILIA person observations and decoded source pixels.

## What the labels and images actually show

The 30 DIRECT manipulation labels are 15 pickups and 15 placements. Their annotated physical entities are MEVA `other`, not verified bottle/backpack/handbag/suitcase classes. In the [visual-review ontology table](meva-small-object-cases.csv), 13 items look cup/can-like near the hand, five are dark against clothing or furniture, five are small items against a bright window, three have low contrast near a counter, two resemble bottle/cup-like items but are not certain, and two more are visually ambiguous or partly occluded. These descriptions are visual judgments, **not** semantic class ground truth. Twenty-one of 30 median boxes are under 40 source pixels wide, and 28 of 30 cases occur in camera G421. The frozen full-frame YOLO11n run has zero usable production-portable-object overlaps on all 30.

The prior broad all-person ROI probe found a detector-class overlap for 15/30 annotation boxes, including 11 `cup`, one `cell phone`, two `chair`, and one `skateboard` best overlap. A class-box overlap can be a coincident object, false label or part of the person. No overlap established object identity, persistence or a pickup/placement. The generic MEVA `other` label cannot be converted automatically to a COCO class; ambiguous items remain `AMBIGUOUS_OBJECT` for event purposes.

## Selective pipeline and measurement

`scripts/experiment_meva_selective_objects.py` reads VIGILIA's stored person tracks. At each of the fixed benchmark sample frames, a person box triggers one crop. Three geometry-only ROI families are tested: expanded person, lower-body region, and wider interaction space, each at 1.25×, 1.5× and 2.0×. YOLO11n runs at 640/.30 on each crop, predicts a source-relative box and class, and duplicate crop detections are suppressed before adjacent-sample class-consistent tracklets are formed. The tracklet linker tolerates one short missing sample but never joins a gap over one second. A candidate person association requires repeated proximity to one track and abstains on near ties. The analyzer also computes relative velocity, motion residual, approach, separation and persistence for audit. This is **preliminary spatial association**, not proof of physical possession or coupled motion.

The [case table](meva-small-object-cases.csv) lists each annotation's approximate pixel size, visual uncertainty and which strategy produced a raw overlap, a provisional object tracklet and a candidate person association. The local ignored JSON files under `data/meva/selective-objects/` preserve measured counts and per-case keys. The diagnostic uses 49 unique annotated source frames; it does not represent continuous processing of all 40 minutes.

| ROI strategy | Factor | Crops / 49 frames | Broad raw overlap / 30 | Broad provisional track / 30 | Candidate association / 30 | Portable raw / 30 | Portable track / 30 | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Expanded | 1.25 | 359 | 8 | 0 | 0 | 1 | 0 | 14.557 s |
| Expanded | 1.50 | 359 | 10 | 0 | 0 | 0 | 0 | 14.517 s |
| Expanded | 2.00 | 359 | 10 | 3 | 3 | 0 | 0 | 14.782 s |
| Lower body | 1.25 | 359 | 2 | 0 | 0 | 0 | 0 | 17.289 s |
| Lower body | 1.50 | 359 | 5 | 1 | 1 | 0 | 0 | 15.738 s |
| Lower body | 2.00 | 359 | 11 | 2 | 1 | 1 | 0 | 15.508 s |
| Wider interaction | 1.25 | 359 | 7 | 0 | 0 | 0 | 0 | 15.890 s |
| Wider interaction | 1.50 | 359 | 10 | 0 | 0 | 1 | 0 | 16.264 s |
| Wider interaction | 2.00 | 359 | 12 | 5 | 3 | 0 | 0 | 16.272 s |

The denominator above is the same 30 manipulation annotations for every row; none was removed for difficulty. “Broad” accepts any non-person/non-vehicle detector class and is diagnostic only. “Portable” requires VIGILIA's existing conservative portable class set. MEVA has no reliable semantic class ground truth for most `other` objects, so this cannot establish detector class accuracy or a true false-positive rate. A two-detection class/spatial tracklet is a *candidate* identity; it is not yet verified physical-object continuity across a manipulation transition.

Across all nine configurations, the union is 15/30 with a broad raw overlap, 6/30 with a broad provisional tracklet, and 4/30 with a candidate association to an annotation-overlapping person track. No configuration has a portable-class tracklet overlapping a manipulation object. The strongest broad row, wider interaction 2.0×, has 12 raw overlaps, five provisional tracks and three candidate associations, but **zero** portable tracks. Thus the first stage improves over full-frame zero coverage only for broad, semantically uncertain classes; it does not make the manipulation chain usable. The 359 crops per 49 frames are about 7.3 crops per frame. The best broad row costs 16.272 seconds on this host, about 0.332 seconds per selected source frame, before any production integration or full-video processing.

Because many activity intervals contain only one or two production samples, the strongest ROI was also tested on a ±2-second evaluation context around each label. This adds only 2 FPS source-grid frames; the crops still derive entirely from stored VIGILIA person tracks. YOLO11n 640/.30 processed 225 unique frames and 1,644 person crops in 50.801 seconds. Broad overlap stayed 12/30, while provisional track coverage rose from 5/30 to 12/30 and candidate association from 3/30 to 8/30. **Portable-class raw, track and association coverage remained 0/30.** The extra context exposes more class-consistent broad detections, but does not make an annotated manipulation object usable by the current production event ontology. It also raises crop cost to roughly 0.226 seconds per selected frame on this host; timings vary with scene occupancy and decode/cache effects.

The same 225 frames and ROIs were then run with locally available YOLO11s at 640/.30. Broad raw/track coverage was 14/30 and 13/30, but candidate person association fell to 5/30 and portable-class coverage stayed **0/30 at every stage**. Wall time rose to 101.323 seconds, about twice the YOLO11n crop run on this host. A larger detector did not solve object class/identity grounding and is not selected for production.

| Wider 2.0× ROI, ±2 s context | Broad raw | Broad provisional track | Candidate association | Portable raw / track / association | Wall time |
|---|---:|---:|---:|---:|---:|
| YOLO11n 640/.30 | 12/30 | 12/30 | 8/30 | 0 / 0 / 0 | 50.801 s |
| YOLO11s 640/.30 | 14/30 | 13/30 | 5/30 | 0 / 0 / 0 | 101.323 s |
| YOLO11n 960/.30 | 9/30 | 3/30 | 1/30 | 0 / 0 / 0 | 102.725 s |

The 960-pixel crop run is worse at every measured coverage stage than the 640-pixel YOLO11n crop run and costs about twice as much on this host. These are 225 GT-selected diagnostic frames, not an unbiased full-video speed benchmark. None of the three candidates justifies changing production perception or adding a pickup/placement event path.

The chain required for a production claim remains: portable or explicitly unknown object detected → stable identity before and after transition → uniquely associated person → stationary-to-coupled movement or coupled-to-stationary separation → source-linked event. The current diagnostic does not establish that chain on a MEVA pickup or placement. The production event rule therefore continues to **abstain**: pickup 0/15 and placement 0/15, with no new event or invented evidence. A crop detector hit alone earns no event credit.

The unchanged-production full eight-video rerun passed 2,443/2,443 event → evidence → observation/track → video checks and all 2,443 scoped-graph API checks. The object probe is evaluation-only, so its provisional detections do not create events or evidence IDs. This preserves source grounding rather than claiming a pickup from an unlinked crop.

Compute cost is host-specific and includes source decode and ROI inference over either 49 interval frames or 225 context frames. One crop is run for each stored person observation at those frames, so this is selective relative to a full-frame small-object second pass, but remains expensive in crowded views. It is not an estimate for deployment over all sampled frames. The best next experiment is to obtain object-specific temporal labels for a small adjudicated subset, then test a lightweight small-object detector with real object continuity and motion coupling under a compute budget. A new semantic class or specialized model should enter production only after that complete chain improves independent pickup/placement matches.

Reproduce the nine isolated configurations from the repository root (after setting `MEVA_ROOT` as documented in [MEVA.md](MEVA.md)):

```bash
for strategy in expanded lower interaction; do
  for factor in 1.25 1.5 2.0; do
    .venv/bin/python scripts/experiment_meva_selective_objects.py \
      --strategy "$strategy" --factor "$factor" \
      --output "data/meva/selective-objects/${strategy}-${factor}.json"
  done
done
.venv/bin/python scripts/summarize_meva_selective_objects.py

# Compare the strongest ROI with temporal context; outputs remain ignored local artifacts.
.venv/bin/python scripts/experiment_meva_selective_objects.py \
  --strategy interaction --factor 2.0 --context-seconds 2 \
  --model models/yolo11n.pt --size 640 \
  --output data/meva/selective-object-candidates/yolo11n-640-context2.json
.venv/bin/python scripts/experiment_meva_selective_objects.py \
  --strategy interaction --factor 2.0 --context-seconds 2 \
  --model models/yolo11s.pt --size 640 \
  --output data/meva/selective-object-candidates/yolo11s-640-context2.json
.venv/bin/python scripts/experiment_meva_selective_objects.py \
  --strategy interaction --factor 2.0 --context-seconds 2 \
  --model models/yolo11n.pt --size 960 \
  --output data/meva/selective-object-candidates/yolo11n-960-context2.json
```

The regression checks in `tests/test_meva_selective_diagnostics.py` cover ROI clipping, duplicate crop suppression, short-gap versus long-gap track linking, class changes, single-hit abstention and ambiguous two-person proximity. No raw video, model weights, benchmark SQLite database or full diagnostic JSON is committed.
