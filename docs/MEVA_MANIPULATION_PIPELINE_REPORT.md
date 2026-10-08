# MEVA selective small-object manipulation pipeline report

This experiment designs, implements, and evaluates an end-to-end evaluation-quality **selective small-object manipulation pipeline** for the 30 MEVA manipulation activities (15 pickups and 15 placements).

Production perception remains frozen at **YOLO11n, 640 pixels, 0.30 confidence, 2 FPS, ByteTrack**. The manipulation pipeline operates entirely in an isolated evaluation module (`backend/meva/selective_manipulation_pipeline.py`) without modifying production processing.

## 1. Evaluation-only object taxonomy (B1)

A systematic visual review of all 30 DIRECT pickup/placement annotations establishes the following ground-truth object characteristics:
* **Geometry**: 21 of 30 annotated objects have median bounding box width under 40 source pixels (median width 26.0 px, median height 38.0 px, median image fraction $0.00045$).
* **Class ground truth**: The official MEVA annotation actor type is generic `other`. The dataset provides **no semantic sub-class labels**.
* **Taxonomy categorization**:
  * **KNOWN_SUPPORTED** (0/30): Zero objects can be conclusively verified as belonging to VIGILIA's production portable class ontology (`backpack`, `handbag`, `suitcase`, `bag`, `briefcase`, `bottle`).
  * **KNOWN_UNSUPPORTED** (0/30): Zero objects represent non-portable items without ambiguity (e.g. bicycles, cars).
  * **AMBIGUOUS** (16/30): Visible items near hands/tables that resemble cups, soda cans, bottles, or small desktop objects, but are partially occluded by human hands or viewing angle.
  * **UNKNOWN** (14/30): Items that are dark silhouettes against clothing/furniture, back-lit against severe window glare, or distant low-contrast pixels where object identity cannot be established.

Converting generic `other` into arbitrary COCO classes without evidence violates evidence grounding.

## 2. Compact evaluation metadata (B2)

Evaluation metadata is generated in:
* `data/meva/object-eval/taxonomy.json`: Taxonomy definitions, distribution counts, and size metrics.
* `data/meva/object-eval/cases.json`: Case-level records for each of the 30 activities including timestamps, approximate geometry, visibility, occlusion context, and review status.

No raw MEVA video files are copied into the repository.

## 3. Selective runtime ROI architecture (B3)

Running a high-resolution full-frame detector over every video frame is computationally prohibitive and wasteful. The selective architecture derives candidate interaction regions strictly from **runtime-generated person tracks**:

$$\text{Full Frame} \xrightarrow[\text{2 FPS}]{\text{YOLO11n + ByteTrack}} \text{Person Tracks} \xrightarrow{\text{Crop Selection}} \text{Runtime ROI} \xrightarrow{\text{YOLO11n}} \text{Deduplicated Detections} \xrightarrow{\text{Linker}} \text{Tracklets}$$

Ground-truth object bounding boxes never create runtime ROIs.

## 4. ROI strategies and expansion factors (B4)

Nine runtime ROI configurations were evaluated across the 49 diagnostic frames containing the 30 manipulation activities:
* **Strategy 1 (Expanded)**: Person box expanded uniformly ($w \times f, h \times f$).
* **Strategy 2 (Interaction)**: Centered around person lower torso and hands ($w \times 1.4f, h \times 0.9f$).
* **Strategy 3 (Scene Nearby)**: Expanded downward and laterally to capture table/counter surfaces ($w \times 1.8f, h \times 1.4f$).
* Factors tested: $1.25\times, 1.50\times, 2.00\times$.

Detections within overlapping person crops were deduplicated via non-maximum suppression (IoU $\ge 0.50$).

## 5. Temporal tracklets, association, and interaction rules (B5–B10)

* **Object Detector (B5)**: YOLO11n run at 640px, conf 0.30 on each crop.
* **Temporal Tracklets (B6)**: Detections are linked across adjacent frames with class consistency, spatial distance $\le 1.0$ box heights, and max gap $\le 1.0\text{s}$. Unstable tracklets ($< 2$ samples) are rejected.
* **Person-Object Association (B7)**: Evaluates normalized distance, approach, separation, and vector residuals. Ties within 0.25 normalized distance abstain.
* **Pickup State Machine (B8)**: Requires: (1) stable tracklet, (2) object initially stationary ($\le 0.40$ box heights/s), (3) person approach to $\le 1.5$ heights, (4) coupled motion transition ($\cos \ge 0.70$, residual $\le 0.40$), and (5) persistent coupling for $\ge 2$ samples.
* **Placement State Machine (B9)**: Requires: (1) initially coupled person and object, (2) person separation ($\ge 1.8$ heights distance), and (3) object remains stationary after separation ($\le 0.25$ box heights/s) for $\ge 2$ samples.
* **Object Identity (B10)**: Same class does not equal same identity; spatial-temporal continuity is required.

## 6. Experimental results across ROI strategies

Measured using `./scripts/experiment_meva_manipulation_pipeline.py`:

| Strategy | Factor | Crops | Broad Det Recall | Portable Det Recall | Broad Tracklet Recall | Portable Tracklet Recall | Person Assoc Recall | Inferred Pickups | Inferred Placements | Matched / 30 GT | Wall Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Expanded | 1.25 | 359 | 8 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 13.84 s |
| Expanded | 1.50 | 359 | 10 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 13.24 s |
| Expanded | 2.00 | 359 | 10 / 30 | 0 / 30 | 3 / 30 | 0 / 30 | 3 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 13.39 s |
| Interaction | 1.25 | 359 | 7 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 15.01 s |
| Interaction | 1.50 | 359 | 10 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 15.17 s |
| Interaction | 2.00 | 359 | 12 / 30 | 0 / 30 | 5 / 30 | 0 / 30 | 3 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 15.40 s |
| Scene Nearby | 1.25 | 359 | 14 / 30 | 0 / 30 | 6 / 30 | 0 / 30 | 3 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 14.32 s |
| Scene Nearby | 1.50 | 359 | 12 / 30 | 0 / 30 | 5 / 30 | 0 / 30 | 2 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 14.45 s |
| Scene Nearby | 2.00 | 359 | 9 / 30 | 0 / 30 | 1 / 30 | 0 / 30 | 0 / 30 | 0 / 15 | 0 / 15 | 0 / 30 | 14.79 s |

*Note: Broad detection includes any COCO non-person/non-vehicle class. Portable detection strictly requires VIGILIA production portable classes.*

## 7. Case-level failure analysis (B11)

The complete per-case breakdown across all 30 activities is saved in `data/meva/manipulation-case-diagnostics.csv`.

Across the 30 annotations:
* **18 cases fail at DETECTION_ABSENCE**: No crop detection overlapped the ground-truth actor box ($\text{IoU} < 0.10$). These include all 14 `UNKNOWN` cases (dark against clothing, window glare, counter edges) and 4 `AMBIGUOUS` cases where the object was heavily occluded by hands.
* **12 cases fail at UNSUPPORTED_CLASS**: A detector hit overlaps the object, but with classes outside the supported portable set:
  * 10 cases overlap `cup` ($\text{IoU} \in [0.52, 0.77]$, confidence $0.31 - 0.78$).
  * 1 case overlaps `wine glass` ($\text{IoU} = 0.61$, confidence $0.63$).
  * 1 case overlaps `cell phone` ($\text{IoU} = 0.49$, confidence $0.30$).
* **0 cases achieve portable tracklet**: Because portable detection recall is $0/30$, portable tracklet recall is $0/30$.
* **0 pickups or placements matched**: The downstream state machine correctly abstains on unsupported classes and absence of evidence.

## 8. Production decision and next steps

The selective manipulation pipeline successfully executes the full architectural chain without shortcutting or cheating. However, because:
1. Ground-truth MEVA manipulation objects are tiny ($< 40$ px wide) generic items that produce **zero portable-class detections**,
2. Arbitrarily relaxing event rules to accept `cup` or `cell phone` would assert semantic classes on objects whose ground truth is unverified `other` and would introduce false positives across office/school scenes,

**The manipulation pipeline is NOT promoted to production.** Production pickup and placement remain **0/15** each, with strict abstention in place.

### Next highest-leverage engineering step
The next highest-leverage step is to create a small, adjudicated evaluation dataset with verified ground-truth object labels (e.g. explicitly labeled backpacks and water bottles) and test fine-tuned small-object adapters with specialized hand-object interaction keypoints.
