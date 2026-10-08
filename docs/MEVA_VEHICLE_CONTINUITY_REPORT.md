# MEVA vehicle track continuity experiment

This experiment evaluates **same-camera vehicle track continuity** under VIGILIA's frozen baseline configuration: YOLO11n, 640-pixel inference, confidence 0.30, 2 FPS sampling, and ByteTrack. The independent MEVA benchmark matching rules, 59 vehicle ground truth activities (32 starts, 27 stops), and downstream temporal event extraction rules were held strictly constant. Ground-truth annotations are never used to form candidate track pairs.

## Motivation and problem diagnosis

Previous controlled experiments demonstrated that increasing detector resolution (YOLO11n 960/1280) or capacity (YOLO11s) significantly increased vehicle track count but failed to improve downstream event matches (starts remained 3/32, stops 5/27). Detailed track-level failure attribution revealed that tracked vehicles failed primarily during **event extraction**, including:
1. Missing stationary history before a moving track.
2. Track fragments splitting vehicle history.
3. Decelerations not reaching sustained stillness ($\ge 1.5$ seconds).
4. Short track segments ending before verified halts.

This experiment investigates whether short detection gaps between fragmented same-camera tracks can be safely bridged using runtime-derived kinematic and geometric evidence alone, and whether the resulting continuous tracks improve downstream vehicle start/stop recognition.

## Same-camera continuity formulation

The continuity layer operates strictly as an inference/correlation layer across tracks generated within the **same camera and video**. Never performs cross-camera identity matching.

### 1. Candidate pair generation
Given predecessor Track A ending at $(t_A, B_A)$ and successor Track B starting at $(t_B, B_B)$:
* Same video and camera: `video_id(A) == video_id(B)` and `camera_id(A) == camera_id(B)`.
* Temporal order and proximity: $0 < \Delta t = t_B - t_A \le \text{max\_gap}$.
* Compatible vehicle class: $C_A, C_B \in \{\text{car}, \text{truck}, \text{bus}, \text{motorcycle}\}$.

### 2. Measurable continuity features
* **Temporal**: Gap duration $\Delta t$.
* **Spatial & Kinematics**:
  * Characteristic scale $S = \max(1.0, (H_A + H_B) / 2.0)$.
  * Normalized displacement: $D_{\text{norm}} = \| C_B - C_A \|_2 / S$.
  * Normalized gap speed: $V_{\text{gap}} = D_{\text{norm}} / \Delta t$.
  * Forward predicted center: $C_{\text{pred}, A} = C_A + V_A \cdot \Delta t$; prediction error $E_A = \| C_B - C_{\text{pred}, A} \|_2 / S$.
  * Backward predicted center: $C_{\text{pred}, B} = C_B - V_B \cdot \Delta t$; prediction error $E_B = \| C_{\text{pred}, B} - C_A \|_2 / S$.
  * Direction consistency: cosine similarity $\cos(V_A, V_B)$ when both tracks exhibit active motion.
* **Geometry**:
  * Width ratio: $\min(W_A, W_B) / \max(W_A, W_B)$.
  * Height ratio: $\min(H_A, H_B) / \max(H_A, H_B)$.
  * Aspect ratio ratio: $\min(AR_A, AR_B) / \max(AR_A, AR_B)$.
* **Class**: Exact match vs compatible vehicle category.

### 3. Rejection of unsafe merges (Safety gates)
A candidate merge is rejected if:
* **Geometry change**: Width ratio, height ratio, or aspect ratio ratio $< 0.50$ (abrupt scale change).
* **Implausible kinematics**: Gap speed $V_{\text{gap}} > 2.5$ box heights/second (teleportation across scene).
* **Opposing direction**: Both tracks moving fast ($> 4$ px/s) with $\cos(V_A, V_B) < -0.3$ (instantaneous U-turn during gap).
* **Competition / Ambiguity**: Track A has multiple successor candidates or Track B has multiple predecessor candidates within an ambiguity margin of 0.15 score.
* **Crossing tracks**: Another vehicle track occupies the spatial gap corridor during the gap interval.

When evidence is ambiguous or uncertain, the system strictly **abstains**.

## Experimental evaluation across gap durations

The experiment was run across all 279 vehicle observations in the completed eight-video MEVA dataset, evaluating gap thresholds of 0.5s, 1.0s, 1.5s, 2.0s, and 3.0s using `./scripts/experiment_meva_vehicle_continuity.py`.

Ground-truth actor overlaps were checked *only afterward* to classify candidate merges as:
* **CORRECT**: Both tracks overlap the same MEVA ground-truth vehicle actor.
* **INCORRECT**: Tracks overlap different MEVA ground-truth vehicle actors (identity switch).
* **UNANNOTATED**: Tracks belong to unannotated vehicles (e.g. background parked cars).

Continuous tracks were then evaluated with the unchanged production vehicle event state machine and independent MEVA matching.

### Results summary

| Configuration | Merges | Correct merges | Incorrect merges | Unannotated merges | Total vehicle predictions | DIRECT vehicle matches | Starts matched / 32 | Stops matched / 27 | Precision | Recall (veh) | Mean temporal IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (frozen)** | **0** | **0** | **0** | **0** | **17** | **8** | **3** | **5** | **47.06%** | **13.56%** | **0.5976** |
| Gap 0.5s | 16 | 0 | 0 | 16 | 17 | 8 | 3 | 5 | 47.06% | 13.56% | 0.5976 |
| Gap 1.0s | 21 | 1 | 0 | 20 | 18 | 8 | 3 | 5 | 44.44% | 13.56% | 0.5976 |
| Gap 1.5s | 21 | 1 | 0 | 20 | 18 | 8 | 3 | 5 | 44.44% | 13.56% | 0.5976 |
| Gap 2.0s | 25 | 1 | 0 | 24 | 18 | 8 | 3 | 5 | 44.44% | 13.56% | 0.5976 |
| Gap 3.0s | 27 | 1 | 0 | 26 | 18 | 8 | 3 | 5 | 44.44% | 13.56% | 0.5976 |

*(Note: Baseline full DIRECT denominator includes 30 manipulation cases (0/30) yielding 8/89 matches, 8.99% overall recall. The vehicle-only denominator above is 8/59 = 13.56% recall).*

## Key findings and failure analysis

1. **Zero false merges**: Across all tested gap thresholds up to 3.0 seconds, the continuity scorer produced **zero incorrect identity merges** (0/27). Unsafe merge rejection successfully prevented false merges between distinct vehicles and crossing vehicles.
2. **Correct merge identification**: At gaps $\ge 1.0\text{s}$, a valid track split was correctly bridged on G436 (Actor 37: `TRK-78A21532635B` and `TRK-E18E29F6FB6B`, gap 1.0s, continuity score 0.8459). However, Actor 37 performed a `vehicle_makes_u_turn`, which is an UNSUPPORTED activity in MEVA (not a start or stop), so no start/stop match could result.
3. **Why event matches did not increase**:
   * Out of 32 vehicle starts: 20 failed at **raw perception** (0 YOLO detections in the activity window), 3 matched, 2 had temporal/spatial boundary offsets, and 7 failed during event extraction.
   * Out of 27 vehicle stops: 14 failed at **raw perception**, 5 matched, 1 had temporal offset, and 7 failed during event extraction.
   * Inspection of the 7 event extraction start failures and 7 stop failures revealed that each was **already tracked continuously by a single track** (e.g. `TRK-2FD50FE4D693`, `TRK-2DDA9BB96FBC`, `TRK-8DEF795AF69A`). They failed event extraction because the vehicles exhibited non-stationary slowdowns ($< 1.5$ seconds stillness or continuous rolling) rather than fragmented tracks across a stationary/moving boundary.
4. **Precision impact**: Bridging background parking tracks at gap $\ge 1.0\text{s}$ formed a long continuous track that triggered 1 additional unannotated vehicle transition hypothesis, slightly lowering precision from 47.06% to 44.44%.

## Production decision

Because same-camera vehicle track continuity produced:
* **Zero recall gain** on ground-truth vehicle starts (remained 3/32) and stops (remained 5/27),
* A slight precision penalty (47.06% $\to$ 44.44%) due to unannotated background track merging,

**Continuity is NOT promoted to the production configuration.**
In accordance with engineering guidelines, the production pipeline remains frozen at commit `585a9ce`. The continuity module remains preserved as an offline inference evaluation asset with complete test coverage in `tests/test_meva_vehicle_continuity.py`.

## Regression test coverage

Ten regression tests in `tests/test_meva_vehicle_continuity.py` protect the continuity scorer and safety gates:
1. `test_1_same_vehicle_short_gap`: Valid 1.0s gap merges with high score.
2. `test_2_same_vehicle_longer_gap`: 5.0s gap exceeds max_gap and is rejected.
3. `test_3_two_nearby_vehicles`: Parallel vehicles in separate lanes are not merged.
4. `test_4_crossing_vehicles`: Third vehicle crossing the gap corridor blocks merge.
5. `test_5_competing_candidate_fragments`: Ambiguous competing successors abstain.
6. `test_6_incompatible_classes`: Incompatible object classes are rejected.
7. `test_7_inconsistent_velocity`: Opposing velocity vectors are rejected.
8. `test_8_vehicle_entering_leaving_frame`: Teleportation across screen is rejected.
9. `test_9_stationary_vehicle`: Parked jitter across short gap merges safely.
10. `test_10_ambiguous_gap`: Ambiguous competing predecessors abstain.
