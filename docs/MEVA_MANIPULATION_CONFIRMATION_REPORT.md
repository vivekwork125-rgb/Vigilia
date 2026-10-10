# MEVA secondary manipulation confirmation: full-video audit

## Decision

**D — No meaningful confirmation of physical manipulation. Production change: NO.** A secondary gate removes some interaction-like candidates, but the remaining candidates are not trustworthy manipulation evidence. In the leakage-controlled, full-video evaluation, the strongest recall-preserving two-signal gate retained only **2/15 pickup** and **3/15 placement** temporal-and-person matches, with **5.0% pickup precision** and **8.8% placement precision** among all emitted candidates. No object identity, actual contact, possession, or release was established. These are candidate-level diagnostic numbers, not production event accuracy.

## Frozen production and previous exploratory baseline

Production remains YOLO11n, 640 pixels, confidence 0.30, ByteTrack, 2 FPS. The frozen eight-video DIRECT benchmark has 89 GT events, 17 predictions, 8 matches, 47.06% precision, 8.99% recall, mean temporal IoU 0.5976, 0/15 pickups, 0/15 placements, and 2,443/2,443 valid production evidence chains. No production code, event rule, model, tracking parameter, or independent benchmark definition changed here.

The `89dc216` hand/interaction-region experiment reported 9/15 pickup and 9/15 placement case matches (60% recall), with 34.62% and 47.37% precision respectively. The subsequent `cd7b7d4` confirmation pilot reported similar results and a best negative-control rejection rate of 80%. **Those pilot accuracy numbers are not valid full-video estimates.** Their scripts selected inference frames using GT activity times plus context and reset the state machine at each labeled window. The `cd7b7d4` pilot also called a case matched if any candidate overlapped its GT interval or merely started within two seconds, without one-to-one matching or actor-box compatibility; its “precision” divided case-level matches by a mix of candidates and negative controls. The pilot did not inject labels into the pose model or feature formulas, but its candidate selection and scoring violated the requested no-leakage evaluation protocol. Its generated results remain historical exploratory artifacts only.

## Corrected experiment and leakage boundary

[`scripts/experiment_meva_confirmation_unbiased.py`](../scripts/experiment_meva_confirmation_unbiased.py) has two separate commands. `extract` reads the manifest, the frozen `improved-2fps/vigilia.db` person tracks, the source videos, and YOLO11n-pose weights. It runs over **every sampled frame with a stored person track in all eight videos**. It does not open the MEVA activity annotations or `object-eval/cases.json`. Runtime person boxes generate the interaction ROIs. The 1.0-second persistence candidate generator and confirmation feature implementation from `89dc216`/`cd7b7d4` are reused unchanged.

Only after the extraction artifact is closed does `evaluate` read the 30 selected MEVA manipulation annotations and actor geometry. It assigns candidates one-to-one by same video, pickup/placement type, temporal IoU ≥0.1 **or both endpoints within ±2 seconds**, and annotated-person/runtime-person box IoU ≥0.1 at a common source frame. The gate never sees the labels. The same 155 candidates and feature vectors are used for all ablations and thresholds. The ten negative-control windows are inspected only after extraction; none overlaps an annotated pickup/placement interval. Because no object box or object track is established, matching a candidate to an annotation is an **upper bound on manipulation recognition**, not proof of object handling.

The eight videos yielded 1,624 sampled frames with stored people, 7,330 pose crops, and 155 upstream candidates (92 pickup-like, 63 placement-like). Some videos had no stored person observations, which is a perception/track limitation rather than an excluded GT case. All 15 pickup and 15 placement annotations remain in the denominator.

## Confirmation features and what they actually measure

| Feature | Measurement | Limitation |
|---|---|---|
| Localized motion | Pixel difference in a hand-centered interaction box relative to body pixel difference | Camera/body motion and shadows can dominate; not object motion. |
| Relative motion | Smoothness of **absolute wrist displacement** across observations | Despite the name, current code does not subtract body/region velocity or establish object coupling. |
| Appearance change | Pre/post pixel difference at the last hand-centered region | A person leaving, shadow, or lighting shift can look like a changed object. |
| Region persistence | IoU of successive wrist-centered interaction boxes | Coherent hand ROIs do not imply a persistent physical object. |
| Separation | Post-interval wrist displacement from its final position, normalized by person height | A hand withdrawing does not establish release or an object left behind. |

The scorer is a transparent weighted sum: `best_two = 0.4 × appearance + 0.6 × separation`; `best_three = 0.25 × localized motion + 0.35 × appearance + 0.4 × separation`; `all = 0.20 × localized motion + 0.15 × relative motion + 0.25 × appearance + 0.15 × persistence + 0.25 × separation`. No learned model or MEVA-trained threshold is used. Features and source observations are retained per candidate in the ignored local artifact.

## Full-video ablation at confirmation threshold 0.45

`P` is one-to-one matches divided by all emitted candidates of that type across eight videos; `R` is matches divided by 15 GT events. Negative counts are confirmed candidates in ten verified non-manipulation windows. These are false candidate counts, **not** a false-positive rate with an assumed true-negative denominator.

| Gate | Pickup cand. | Pickup matches | Pickup P / R | Placement cand. | Placement matches | Placement P / R | Negative pickup / placement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Upstream pass-through | 92 | 2 | 2.2% / 13.3% | 63 | 3 | 4.8% / 20.0% | 4 / 2 |
| Localized motion | 43 | 2 | 4.7% / 13.3% | 29 | 0 | 0% / 0% | 3 / 2 |
| Relative motion | 37 | 1 | 2.7% / 6.7% | 20 | 2 | 10.0% / 13.3% | 1 / 0 |
| Appearance change | 34 | 1 | 2.9% / 6.7% | 24 | 2 | 8.3% / 13.3% | 3 / 2 |
| Separation | 42 | 2 | 4.8% / 13.3% | 34 | 3 | 8.8% / 20.0% | 2 / 1 |
| Appearance + separation | 40 | 2 | 5.0% / 13.3% | 34 | 3 | 8.8% / 20.0% | 3 / 1 |
| Motion + appearance + separation | 42 | 2 | 4.8% / 13.3% | 35 | 3 | 8.6% / 20.0% | 4 / 2 |
| All five | 46 | 2 | 4.3% / 13.3% | 37 | 3 | 8.1% / 20.0% | 3 / 2 |

The best recall-preserving *exploratory* choice is appearance + separation at 0.45; it reduces all-video candidates from 155 to 74, but 69/74 remain unmatched to these 30 GT events. Relative-motion-only rejects 5/6 negative-window candidates, but loses one of two pickup matches and one of three placement matches. No tested gate makes the candidates defensible as physical manipulation evidence.

## Threshold trade-off for all five signals

| Threshold | Pickup candidates / matches | Placement candidates / matches | Negative pickup / placement |
|---:|---:|---:|---:|
| 0.35 | 66 / 2 | 50 / 3 | 4 / 2 |
| 0.40 | 58 / 2 | 44 / 3 | 3 / 2 |
| 0.45 | 46 / 2 | 37 / 3 | 3 / 2 |
| 0.50 | 40 / 2 | 33 / 3 | 3 / 2 |
| 0.55 | 34 / 2 | 24 / 2 | 2 / 1 |
| 0.60 | 23 / 2 | 17 / 2 | 1 / 1 |

Tighter filtering improves the candidate fraction only by discarding many unmatched hypotheses; it does not recover missed real interactions. The local `confirmation-unbiased-results.json` includes sweeps for the two- and three-signal combinations and strict ±1/±2/±3-second endpoint diagnostics.

## Pickup, placement, and resolution

The full-video upstream produces only 2/15 pickup and 3/15 placement matches under one-to-one temporal/person matching. The appearance + separation gate retains those five matches. Its matched mean temporal IoU is 0.4439 for pickups and 0.4976 for placements. Of the 11 GT pickups with sub-40px object width, only one matches; one of four larger-object pickups matches. Of the ten sub-40px placements, two match; one of five larger placements matches. This tiny sample does **not** support a reliable resolution-specific effect: both groups are weak. The original GT-centered pilot overstated detectability in both groups.

The ten negative windows emit six upstream candidates (four pickup-like, two placement-like). Appearance + separation keeps four (three pickup-like, one placement-like), rejecting 2/6. Relative motion keeps one (one pickup-like), rejecting 5/6, but also sacrifices true matches. These controls cover only ten short windows and should not be generalized into a world-wide false-positive rate.

## Runtime and evidence integrity

On this host, full-video extraction took **289.97 seconds** for 7,330 pose crops. Feature computation alone took **0.549 seconds** for 155 candidates, about **3.54 ms/candidate**; this excludes video seek/decode and pose inference. Total incremental confirmation I/O cost and peak memory were not separately measured, so no memory or generalized latency claim is made. The evaluator's post-hoc annotation parsing and matching are outside runtime inference.

Every extracted candidate records video ID, camera, source SHA-256, track ID, source start/end frames and seconds, person boxes, supporting observations, and the five feature values. These are **evaluation candidates**, not persisted application events: they have no event/evidence IDs or API evidence chain. The frozen production run previously measured 2,443/2,443 valid evidence chains; this experiment does not change or newly remeasure that production result. An explanatory claim may say *“pickup-like hand interaction with local appearance change and wrist separation”*; it cannot say an object was grasped, carried, or placed.

## Main failure modes and recommendation

1. GT-windowed pilot selection produced optimistic counts; full-video inference exposes many ordinary hand actions.
2. The upstream wrist/interaction-region state machine emits candidates without detecting a physical object.
3. Each secondary feature largely measures hand or person behavior, not independent object continuity. Shadows, occlusion, and camera motion confound local pixels.
4. Small and ambiguous objects remain unresolved at the source resolution. Even larger-object cases are weak in this full-video evaluation.
5. More restrictive thresholds reduce candidate volume but do not create missing object evidence.

**Do not integrate the gate into production.** The highest-leverage next step is a controlled object-presence and continuity study on cameras/clips where an object is visibly resolvable, with runtime-only crops and a held-out evaluation set. Until an object can be independently observed and associated across the transition, VIGILIA should abstain from pickup/placement assertions.

## Reproduction

From `/Users/yanalavivekreddy/VIGILIA` with the already-downloaded MEVA data and frozen processed DB:

```bash
export MEVA_ROOT="$HOME/VIGILIA_DATA"
.venv/bin/python scripts/experiment_meva_confirmation_unbiased.py extract
.venv/bin/python scripts/experiment_meva_confirmation_unbiased.py evaluate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests
.venv/bin/ruff check --no-cache scripts/experiment_meva_confirmation_unbiased.py tests/test_meva_confirmation_unbiased.py backend/meva/manipulation_confirmation.py tests/test_manipulation_confirmation.py
PYTHONPYCACHEPREFIX=/tmp/vigilia-pycache .venv/bin/python -m compileall -q backend tests scripts
```

The two generated JSON files live in ignored `data/meva/`: `confirmation-full-video-candidates.json` and `confirmation-unbiased-results.json`. Raw footage and generated artifacts are not committed. `tests/test_meva_confirmation_unbiased.py` protects actor-grounded, one-to-one, endpoint-tolerant matching. The 30 labels are a selected development set, not a held-out accuracy estimate. No tuning result here should be marketed as real-world activity-recognition accuracy.
