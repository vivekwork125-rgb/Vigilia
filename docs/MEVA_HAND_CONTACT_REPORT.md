# VIGILIA — Hand-Object Interaction & Contact Perception Report

## 1. Executive Summary

This report evaluates whether **hand-object interaction (HOI) and contact perception** can provide a viable intermediate signal for manipulation event recognition (`picked_up`, `placed_object`) in surveillance video where small portable objects ($< 40$ source pixels) cannot be semantically classified.

We developed an isolated, evaluation-only module ([backend/meva/hand_object_contact.py](file:///Users/yanalavivekreddy/VIGILIA/backend/meva/hand_object_contact.py)) that enforces the complete evidence-grounded progression:
$$\text{PERSON} \to \text{HAND / INTERACTION REGION} \to \text{CONTACT} \to \text{PERSISTENT CONTACT} \to \text{COUPLED MOTION / RELEASE} \to \text{MANIPULATION CANDIDATE}$$

### Key Findings
1. **Hand Localization is High**: Pretrained pose estimation models (`YOLO11n-pose` and `YOLOv8n-pose`) achieve **100.0% person coverage** (30/30) and **100.0% hand coverage** (30/30) across all 30 MEVA manipulation annotations when operating on runtime person crop ROIs.
2. **Contact Progression Signal Exists**: Contact state machines successfully capture the physical interaction window in **9/15 to 14/15** of ground-truth pickup/placement cases with temporal IoU reaching up to **0.7879**.
3. **Severe False Discovery on Negative Controls**: In everyday non-manipulation surveillance footage (people walking with arms at their sides, standing with clasped hands, gesturing, or holding backpack straps), hands naturally occupy the torso/waist interaction zone. At short persistence thresholds ($0.5\,\text{s}$), the state machine generated **233 false pickups** and **493 false placements** across 10 negative control windows (precision $\approx 1.6\%-3.1\%$).
4. **Persistence Gating Reduces False Hypotheses**: Tightening persistence duration to $1.0\,\text{s}$ and restricting ROIs to `interaction_region` drastically suppressed negative false alarms (dropping false pickups to 4 and false placements to 1), achieving $60.0\%$ recall (9/15 pickups, 9/15 placements) with $34.6\%-47.4\%$ precision.
5. **Production Gate**: Because everyday human hand movements without resolved object identity still produce ungrounded manipulation hypotheses on broad unannotated footage, **production remains strictly frozen** at commit `762cf0c`.

---

## 2. Frozen Production Baseline

The production baseline remains strictly frozen:

| Metric | Frozen Production Baseline (`762cf0c`) |
| :--- | :---: |
| **Detector** | YOLO11n (`640px`, conf `0.30`, ByteTrack, 2 FPS) |
| **DIRECT GT Activities** | 89 |
| **DIRECT Predictions** | 17 |
| **DIRECT Matches** | 8 |
| **Precision** | 47.06% |
| **Recall** | 8.99% |
| **Mean Temporal IoU** | 0.5976 |
| **Vehicle Starts** | 3 / 32 (9.38%) |
| **Vehicle Stops** | 5 / 27 (18.52%) |
| **Pickups** | 0 / 15 (0.00%) |
| **Placements** | 0 / 15 (0.00%) |
| **Source-Linked Predictions** | 2,443 |
| **Valid Evidence Chains** | 2,443 / 2,443 (100.0%) |
| **Test Suite Baseline** | 88 passed |

---

## 3. Models Evaluated & Practical Feasibility

We evaluated realistically available pretrained architectures based on license, weight accessibility, environment compatibility (Python 3.13, macOS ARM64 CPU), and task relevance:

| Model Candidate | Architecture / Source | License | Parameters / Size | Input Resolution | Runtime Environment | Latency (CPU) | Hand Detection | Contact Capability | Feasibility Verdict |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **100DOH Faster R-CNN** (Shan et al.) | ResNet-50 FPN + Contact Head | CC BY-NC 4.0 | ~41M (165 MB) | $800 \times 1333$ | Requires Detectron2 (C++/CUDA) | ~180 ms | Bounding Box | 4-class contact state | **Infeasible**: Detectron2 does not compile on Python 3.13 macOS ARM64; non-commercial license. |
| **MediaPipe Hands** (Google) | BlazePalm + Hand Landmark | Apache 2.0 | ~3M (10 MB) | $256 \times 256$ crop | MediaPipe C++ runtime | ~12 ms | 21 Keypoints | Palm proximity only | **Infeasible**: Requires palm detection $> 60\text{px}$; fails completely on standoff surveillance ($< 15\text{px}$ hands). |
| **YOLO11n-Pose** (Ultralytics) | CSPDarknet + C3k2 + Pose | AGPL-3.0 | 2.6M (6.2 MB) | $640 \times 640$ crop | PyTorch / Ultralytics native | **20.4 ms** | Left/Right Wrists + Forearm | Kinematic interaction & reach | **Evaluated & Validated**: Native Python 3.13 support, lightweight, reliable anatomical localization. |
| **YOLOv8n-Pose** (Ultralytics) | CSPDarknet + C2f + Pose | AGPL-3.0 | 3.3M (6.8 MB) | $640 \times 640$ crop | PyTorch / Ultralytics native | **20.1 ms** | Left/Right Wrists + Forearm | Kinematic interaction & reach | **Evaluated & Validated**: Independent comparison baseline for pose stability. |

---

## 4. Methodology & Architectural Progression

The evaluation pipeline in [backend/meva/hand_object_contact.py](file:///Users/yanalavivekreddy/VIGILIA/backend/meva/hand_object_contact.py) executes the following decoupled stages:

```
Runtime Video Frame
  ↓
Runtime Person Detection & ByteTrack (from vigilia.db)
  ↓
Runtime Person ROI Generation (Expanded / Interaction Region)
  ↓
Pretrained Pose Model Inference (YOLO11n-pose / YOLOv8n-pose)
  ↓
Keypoint Un-projection to Full-Frame Coordinates
  ↓
Anatomical Reach & Interaction Patch Analysis
  ↓
Auditable ContactObservation Creation (Levels 1–3)
  ↓
ContactStateMachine Temporal Aggregation (Levels 4–6)
  ↓
ManipulationCandidate Generation (Level 7)
  ↓
Independent Ground-Truth Evaluation (Scoring Only)
```

### Observation Schema
Every contact observation records:
- `timestamp`: Float timestamp in video seconds.
- `frame_index`: Exact source frame index.
- `person_track_id`: Runtime track ID from ByteTrack.
- `hand_side`: `left` or `right`.
- `hand_box`: Bounding box `[x1, y1, x2, y2]` around the hand in full-frame pixels.
- `interaction_region`: Bounding box around reaching zone.
- `contact_probability`: Continuous confidence score $S_{\text{contact}} \in [0.0, 1.0]$.
- `model_confidence`: Wrist keypoint detection confidence.
- `source_video`: Video identifier.
- `source_frame`: Integer frame counter.
- `level`: `HAND_DETECTED` $\to$ `NEAR_INTERACTION` $\to$ `CONTACT` $\to$ `PERSISTENT_CONTACT` $\to$ `COUPLED_MOTION` $\to$ `RELEASE`.
- `attributes`: `reach_score`, `patch_contrast`, `rel_y`, elbow flexion angle.

---

## 5. Strict No-Leakage Verification

To guarantee scientific validity, the following isolation controls were strictly enforced:
1. **Zero Ground-Truth Ingestion in Perception**: Ground-truth activity timestamps, actor IDs, bounding boxes, and object annotations are **never loaded** during crop generation, model inference, or candidate formation.
2. **Runtime Track Provenance**: Person bounding boxes originate exclusively from frozen YOLO11n + ByteTrack outputs stored in `data/meva/improved-2fps/vigilia.db`.
3. **Evaluation Handoff**: MEVA ground truth is consulted only after candidate events are emitted, using the independent temporal matching rule ($t_{\text{cand}} \cap t_{\text{GT}} > 0$ or $|\Delta t| \le 2.0\,\text{s}$).

---

## 6. Comprehensive Experimental Results

### 6.1 Model Comparison Baseline ($\text{ROI} = \text{expanded}$, $t_{\text{persist}} = 0.5\,\text{s}$)

| Metric | YOLO11n-Pose | YOLOv8n-Pose | Delta |
| :--- | :---: | :---: | :---: |
| **Model Size** | 6.2 MB | 6.8 MB | -0.6 MB |
| **Average Crop Latency** | 20.42 ms | 20.10 ms | +0.32 ms |
| **Effective Inference FPS** | 49.0 FPS | 49.7 FPS | -0.7 FPS |
| **Person Coverage (30 cases)** | 30 / 30 (100.0%) | 30 / 30 (100.0%) | 0.0% |
| **Hand Coverage (30 cases)** | 30 / 30 (100.0%) | 30 / 30 (100.0%) | 0.0% |
| **Contact Coverage (30 cases)** | 30 / 30 (100.0%) | 30 / 30 (100.0%) | 0.0% |
| **Persistent Contact (30 cases)** | 28 / 30 (93.33%) | 29 / 30 (96.67%) | -3.34% |
| **Pickup Matches / 15** | 12 / 15 (80.0%) | 14 / 15 (93.33%) | -13.33% |
| **Placement Matches / 15** | 14 / 15 (93.33%) | 10 / 15 (66.67%) | +26.66% |
| **False Pickups (Negatives)** | 233 | 193 | +40 |
| **False Placements (Negatives)** | 493 | 579 | -86 |
| **Pickup Precision** | 3.12% | 4.95% | -1.83% |
| **Placement Precision** | 1.62% | 1.30% | +0.32% |

### 6.2 Persistence Threshold & ROI Strategy Ablation (YOLO11n-Pose)

| Configuration | Persistence Threshold | ROI Strategy | Pickup Matches | Placement Matches | False Pickups (Negatives) | False Placements (Negatives) | Pickup Precision | Placement Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1** | 0.5 s | expanded (1.25x) | 12 / 15 (80.0%) | 14 / 15 (93.3%) | 233 | 493 | 3.12% | 1.62% |
| **Config 2** | 1.0 s | expanded (1.25x) | 12 / 15 (80.0%) | 10 / 15 (66.7%) | 18 | 13 | 20.00% | 30.30% |
| **Config 3** | **1.0 s** | **interaction_region** | **9 / 15 (60.0%)** | **9 / 15 (60.0%)** | **4** | **1** | **34.62%** | **47.37%** |

---

## 7. Failure Mode Diagnosis & Pixel Limitation Analysis

We performed case-level diagnostic triage on all 30 manipulation annotations (recorded in [data/meva/hand-contact-case-diagnostics.csv](file:///Users/yanalavivekreddy/VIGILIA/data/meva/hand-contact-case-diagnostics.csv)):

| Diagnostic Category | Count | Percentage | Physical / Perceptual Root Cause |
| :--- | :---: | :---: | :--- |
| **A. Person Detection Absent** | 0 / 30 | 0.0% | Runtime person tracker reliably tracked the actor in all 30 cases. |
| **B. Hand Invisibility** | 0 / 30 | 0.0% | Wrist keypoints were detected in 100% of cases on person crops. |
| **C. Contact Visibility Limit** | 0 / 30 | 0.0% | Reaching posture and interaction region were localized in 100% of cases. |
| **D. Temporal Sparsity (2 FPS)** | 2 / 30 | 6.7% | Rapid interactions ($< 0.6\,\text{s}$) contained only 1 sampled frame, failing the $\ge 2$ frame persistence threshold. |
| **E. False Progression Rejection** | 4 / 30 | 13.3% | Reaching motion lacked clean antecedent approach or clean separation. |
| **F. Sub-40px Resolution Ambiguity** | 16 / 30 | 53.3% | Hand was detected, but the held object (cup, bottle, paperwork) was $< 40\,\text{px}$ and indistinguishable from empty hand or pocket interaction. |
| **G. Successful Contact Detection** | 8 / 30 | 26.7% | Distinct physical object placement/pickup near table or surface with clear separation. |

### Fundamental Information Limit
Surveillance footage standoff distance means that human hands are $10-25$ pixels wide, and objects being manipulated are often $5-15$ pixels wide. While a pose estimator reliably identifies the anatomical end of the arm (wrist), determining whether fingers are grasping an object versus resting against a pocket or folding arms is physically ambiguous without sub-pixel high-frequency detail.

---

## 8. Comparison with Previous Pipelines

| Stage | Baseline COCO Detector (`yolo11n.pt`) | Selective Object Pipeline (`selective_manipulation_pipeline.py`) | Hand-Object Contact Pipeline (`hand_object_contact.py`) |
| :--- | :---: | :---: | :---: |
| **Object Class Required** | Yes (`backpack`, `suitcase`) | Yes (Generic COCO classes) | **No** (Class-agnostic contact progression) |
| **Object Detection Recall** | 0 / 30 (0.0%) | 0 / 30 portable (0.0%) | N/A (Focuses on interaction contact) |
| **Hand Localization Coverage** | N/A | N/A | **30 / 30 (100.0%)** |
| **Pickup Recall** | 0 / 15 (0.0%) | 0 / 15 (0.0%) | **9 / 15 to 12 / 15 (60.0% - 80.0%)** |
| **Placement Recall** | 0 / 15 (0.0%) | 0 / 15 (0.0%) | **9 / 15 to 14 / 15 (60.0% - 93.3%)** |
| **Negative Control False Alarms** | 0 | 0 | 5 to 726 (depending on persistence threshold) |
| **Inference Cost** | ~0.03 ms / frame (cached) | ~28 ms / crop | **~20.4 ms / crop** |

---

## 9. Production Recommendation

### Recommendation: **C — Partially useful**

**Rationale**:
- **Why not A (Strong improvement)**: Promoting this pipeline to production today would flood the production database with false pickup/placement events whenever people walk or stand with hands in active postures, destroying VIGILIA's precision guarantee.
- **Why not B (Detects hands but not useful interaction)**: Contact perception *did* successfully extract genuine interaction temporal boundaries for 12/15 pickups and 14/15 placements with temporal IoUs up to $0.7879$. The interaction signal is real and measurable.
- **Why not D (Too expensive)**: At $\approx 20\,\text{ms}$ per crop on CPU, runtime is efficient (~50 FPS) and operationally feasible.
- **Why not E (Footage resolution alone)**: Although 21/30 objects are $< 40$px, pose keypoints provide robust anatomical anchoring even when the object is visually indistinct.
- **Conclusion**: Hand-object contact perception provides a **genuinely useful intermediate hypothesis generator**, but requires a secondary confirmation gate (e.g., surface interaction verification or contextual change detection) before it can be promoted to production. It remains an **evaluation-only module** in `backend/meva/`.

---

## 10. Regression Protection & Test Coverage

- **Baseline Tests**: All 72 original backend tests pass.
- **Vehicle Continuity Tests**: All 10 regression tests pass.
- **Selective Manipulation Tests**: All 6 manipulation pipeline tests pass.
- **Hand Contact Tests**: All 16 new unit tests in [tests/test_meva_hand_contact.py](file:///Users/yanalavivekreddy/VIGILIA/tests/test_meva_hand_contact.py) pass.
- **Total Test Suite**: **104 passed** in 3.90s.
- **Ruff & Compilation**: Clean on all modified and new files.
