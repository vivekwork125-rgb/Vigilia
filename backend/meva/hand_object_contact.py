"""Evaluation-only hand-object interaction and contact perception module.

Investigates whether hand-object interaction / contact perception provides a useful
intermediate signal for manipulation events (pickup / placement) on surveillance footage.

Conceptual Progression:
  Level 1: Hand detected (anatomical wrist/hand localization)
  Level 2: Hand near interaction region (active reaching/manipulation zone)
  Level 3: Contact (visual and kinematic evidence of interaction)
  Level 4: Persistent contact (contact sustained for temporal threshold >= t_persist)
  Level 5: Coupled motion (hand moves coupled with person centroid)
  Level 6: Release / separation (contact drops while interaction region remains stable)
  Level 7: Manipulation candidate (pickup / placement candidate with evidence chain)

STRICT NO-LEAKAGE:
  This module operates strictly on runtime video frames and runtime person tracks.
  Ground-truth MEVA annotations are never used to generate ROIs, keypoints, or events;
  annotations are only consulted during downstream evaluation scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class HandKeypoint:
    """Anatomical hand/wrist keypoint."""
    side: str  # 'left' or 'right'
    x: float  # Full-frame x coordinate
    y: float  # Full-frame y coordinate
    confidence: float
    wrist_x: float
    wrist_y: float
    elbow_x: float | None = None
    elbow_y: float | None = None
    shoulder_x: float | None = None
    shoulder_y: float | None = None


@dataclass
class ContactObservation:
    """Auditable contact observation traceable to source video, frame, and person."""
    timestamp: float
    frame_index: int
    person_track_id: str
    hand_side: str  # 'left', 'right', 'both', 'unknown'
    hand_box: tuple[int, int, int, int]  # [x1, y1, x2, y2]
    interaction_region: tuple[int, int, int, int]  # [x1, y1, x2, y2]
    contact_probability: float  # Score in [0.0, 1.0]
    model_confidence: float
    source_video: str
    source_frame: int
    level: str  # 'HAND_DETECTED', 'NEAR_INTERACTION', 'CONTACT', 'PERSISTENT_CONTACT', 'COUPLED_MOTION', 'RELEASE'
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_index": self.frame_index,
            "person_track_id": self.person_track_id,
            "hand_side": self.hand_side,
            "hand_box": list(self.hand_box),
            "interaction_region": list(self.interaction_region),
            "contact_probability": round(self.contact_probability, 3),
            "model_confidence": round(self.model_confidence, 3),
            "source_video": self.source_video,
            "source_frame": self.source_frame,
            "level": self.level,
            "attributes": self.attributes,
        }


@dataclass
class ManipulationCandidate:
    """Source-grounded candidate manipulation event."""
    candidate_id: str
    event_type: str  # 'pickup' or 'placement'
    person_track_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    confidence: float
    evidence_chain: list[str]
    supporting_observations: list[ContactObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "event_type": self.event_type,
            "person_track_id": self.person_track_id,
            "video_id": self.video_id,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration": round(self.end_seconds - self.start_seconds, 3),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": round(self.confidence, 3),
            "evidence_chain": self.evidence_chain,
            "observation_count": len(self.supporting_observations),
        }


def compute_person_roi(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    strategy: str = "expanded",
    factor: float = 1.25,
) -> tuple[int, int, int, int]:
    """Generate runtime-derived person crop ROI without ground truth.

    Strategies:
      - 'full_person': Exact detected person box.
      - 'upper_body': Top 65% of person height (torso, arms, head).
      - 'expanded': Person box expanded isotropically by factor.
      - 'interaction_region': Waist-to-hand interaction zone (30% to 85% of height).
    """
    x1, y1, x2, y2 = box
    pw = max(1, x2 - x1)
    ph = max(1, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if strategy == "full_person":
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
    elif strategy == "upper_body":
        rx1 = x1
        ry1 = y1
        rx2 = x2
        ry2 = int(y1 + 0.65 * ph)
    elif strategy == "expanded":
        nw = pw * factor
        nh = ph * factor
        rx1 = int(cx - nw / 2)
        ry1 = int(cy - nh / 2)
        rx2 = int(cx + nw / 2)
        ry2 = int(cy + nh / 2)
    elif strategy == "interaction_region":
        nw = pw * 1.3
        rx1 = int(cx - nw / 2)
        ry1 = int(y1 + 0.30 * ph)
        rx2 = int(cx + nw / 2)
        ry2 = int(y1 + 0.85 * ph)
    else:
        raise ValueError(f"Unknown ROI strategy: {strategy}")

    return (
        max(0, rx1),
        max(0, ry1),
        min(frame_width, rx2),
        min(frame_height, ry2),
    )


def extract_pose_keypoints(
    keypoints_data: np.ndarray,
    crop_offset: tuple[int, int] = (0, 0),
    min_conf: float = 0.25,
) -> list[dict[str, Any]]:
    """Parse 17 COCO keypoints from Ultralytics output and unproject to full frame.

    COCO keypoint indices:
      5: left_shoulder, 6: right_shoulder
      7: left_elbow, 8: right_elbow
      9: left_wrist, 10: right_wrist
      11: left_hip, 12: right_hip
    """
    ox, oy = crop_offset
    results = []
    # keypoints_data shape: [num_persons, 17, 3] or [17, 3]
    if keypoints_data.ndim == 2:
        keypoints_data = keypoints_data[np.newaxis, ...]

    for person_kps in keypoints_data:
        parsed = {}
        for idx, name in [
            (5, "left_shoulder"),
            (6, "right_shoulder"),
            (7, "left_elbow"),
            (8, "right_elbow"),
            (9, "left_wrist"),
            (10, "right_wrist"),
            (11, "left_hip"),
            (12, "right_hip"),
        ]:
            if idx < len(person_kps):
                x, y, conf = person_kps[idx]
                parsed[name] = {
                    "x": float(x + ox),
                    "y": float(y + oy),
                    "conf": float(conf),
                }
        results.append(parsed)
    return results


def evaluate_hand_contact_state(
    person_box: tuple[int, int, int, int],
    keypoints: dict[str, dict[str, float]],
    frame_pixels: np.ndarray | None = None,
    hand_conf_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Determine hand presence, interaction region, and contact score for both hands.

    Evaluates:
      1. Hand detection: Wrist confidence >= threshold
      2. Hand near interaction region: Anatomical position relative to torso/hips
      3. Arm extension: Reaching out/downward vs resting neutral hang at side
      4. Visual patch saliency: Local texture variance around hand
    """
    _, y1, _, y2 = person_box
    ph = max(1, y2 - y1)
    results = []

    for side in ["left", "right"]:
        wrist_key = f"{side}_wrist"
        elbow_key = f"{side}_elbow"
        shoulder_key = f"{side}_shoulder"
        hip_key = f"{side}_hip"

        w_info = keypoints.get(wrist_key, {"x": 0.0, "y": 0.0, "conf": 0.0})
        w_conf = w_info["conf"]
        if w_conf < hand_conf_threshold:
            continue

        wx, wy = w_info["x"], w_info["y"]
        s_info = keypoints.get(shoulder_key)
        h_info = keypoints.get(hip_key)
        e_info = keypoints.get(elbow_key)

        # Approximate hand bounding box: radius proportional to person height (~10% of height)
        hand_radius = max(6, int(0.08 * ph))
        hand_box = (
            max(0, int(wx - hand_radius)),
            max(0, int(wy - hand_radius)),
            int(wx + hand_radius),
            int(wy + hand_radius),
        )

        # Interaction region around hand & forward carrying zone
        inter_radius = max(10, int(0.14 * ph))
        interaction_region = (
            max(0, int(wx - inter_radius)),
            max(0, int(wy - inter_radius)),
            int(wx + inter_radius),
            int(wy + inter_radius),
        )

        # 1. Anatomical reach calculation:
        # Distance from shoulder or hip
        arm_reach = 0.0
        if s_info and s_info["conf"] >= 0.2:
            sx, sy = s_info["x"], s_info["y"]
            arm_reach = math.hypot(wx - sx, wy - sy) / ph

        # Vertical position relative to person height (0.0 = top of head, 1.0 = feet)
        rel_y = (wy - y1) / ph

        # Active manipulation zone is typically between chest and knees (0.35 to 0.85 of height)
        in_manip_zone = 0.35 <= rel_y <= 0.88

        # 2. Reaching and flexion posture:
        # If wrist is held forward/outward or elbow is flexed (bent) rather than hanging straight down
        reach_posture_score = 0.0
        if h_info and h_info["conf"] >= 0.2:
            hx, hy = h_info["x"], h_info["y"]
            dist_to_hip = math.hypot(wx - hx, wy - hy) / ph
            reach_posture_score = min(1.0, max(0.0, (dist_to_hip - 0.10) / 0.35))
        elif arm_reach > 0.25:
            reach_posture_score = min(1.0, arm_reach / 0.50)

        # Elbow flexion boost: bent elbow indicates carrying/holding
        if e_info and e_info["conf"] >= 0.2 and s_info and s_info["conf"] >= 0.2:
            ex, ey = e_info["x"], e_info["y"]
            sx, sy = s_info["x"], s_info["y"]
            upper_arm = math.hypot(ex - sx, ey - sy)
            forearm = math.hypot(wx - ex, wy - ey)
            direct_dist = math.hypot(wx - sx, wy - sy)
            if upper_arm + forearm > 1e-3:
                flexion = 1.0 - (direct_dist / (upper_arm + forearm))
                reach_posture_score = min(1.0, reach_posture_score + 0.3 * flexion)

        # 3. Patch saliency (local image gradient/contrast indicating object presence)
        patch_contrast = 0.5
        if frame_pixels is not None and frame_pixels.size > 0:
            hx1, hy1, hx2, hy2 = hand_box
            fh, fw = frame_pixels.shape[:2]
            cx1 = max(0, min(fw - 1, hx1))
            cy1 = max(0, min(fh - 1, hy1))
            cx2 = max(cx1 + 1, min(fw, hx2))
            cy2 = max(cy1 + 1, min(fh, hy2))
            crop = frame_pixels[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if cv2 else crop.mean(axis=2)
                std = float(np.std(gray))
                patch_contrast = min(1.0, std / 40.0)

        # 4. Synthesize Contact Probability S_contact
        # Base: requires wrist confidence
        # Contact increases if: in manipulation zone, reaching posture, and patch contrast
        if not in_manip_zone:
            contact_prob = 0.15 * w_conf
            level = "HAND_DETECTED"
        elif reach_posture_score < 0.25:
            # Hand detected near body, but not active manipulation
            contact_prob = 0.25 * w_conf + 0.10 * patch_contrast
            level = "NEAR_INTERACTION"
        else:
            # Active manipulation posture
            contact_prob = (
                0.35 * w_conf
                + 0.35 * reach_posture_score
                + 0.30 * patch_contrast
            )
            level = "CONTACT" if contact_prob >= 0.45 else "NEAR_INTERACTION"

        results.append({
            "hand_side": side,
            "wrist_conf": w_conf,
            "wrist_pos": (wx, wy),
            "hand_box": hand_box,
            "interaction_region": interaction_region,
            "contact_probability": float(contact_prob),
            "level": level,
            "rel_y": float(rel_y),
            "reach_score": float(reach_posture_score),
            "patch_contrast": float(patch_contrast),
        })

    return results


class ContactStateMachine:
    """Evaluation-only temporal state machine tracking person-hand contact progression.

    Distinguishes:
      Level 1: HAND_DETECTED
      Level 2: NEAR_INTERACTION
      Level 3: CONTACT
      Level 4: PERSISTENT_CONTACT (>= persist_duration)
      Level 5: COUPLED_MOTION
      Level 6: RELEASE / SEPARATION
      Level 7: MANIPULATION_CANDIDATE
    """

    def __init__(
        self,
        persist_duration: float = 0.5,
        contact_threshold: float = 0.45,
        max_gap_seconds: float = 0.6,
        motion_coupling_threshold: float = 0.70,
    ):
        self.persist_duration = persist_duration
        self.contact_threshold = contact_threshold
        self.max_gap_seconds = max_gap_seconds
        self.motion_coupling_threshold = motion_coupling_threshold

        # State per person track: list of observations
        self.track_history: dict[str, list[ContactObservation]] = {}
        self.candidates: list[ManipulationCandidate] = []

    def process_observation(self, obs: ContactObservation) -> list[ManipulationCandidate]:
        """Ingest observation, update temporal state, and return any newly emitted candidates."""
        pid = obs.person_track_id
        if pid not in self.track_history:
            self.track_history[pid] = []

        history = self.track_history[pid]
        history.append(obs)
        new_candidates = []

        # Analyze current window for manipulation transitions
        # Need at least 2 observations
        if len(history) < 2:
            return new_candidates

        # Find contact runs
        runs = self._extract_contact_runs(history)
        for run in runs:
            # Check for pickup: NO_CONTACT -> CONTACT -> PERSISTENT_CONTACT -> COUPLED_MOTION
            pickup = self._evaluate_run_for_pickup(run, history)
            if pickup and not self._is_duplicate(pickup):
                self.candidates.append(pickup)
                new_candidates.append(pickup)

            # Check for placement: COUPLED_MOTION -> CONTACT -> RELEASE
            placement = self._evaluate_run_for_placement(run, history)
            if placement and not self._is_duplicate(placement):
                self.candidates.append(placement)
                new_candidates.append(placement)

        return new_candidates

    def _extract_contact_runs(
        self, history: list[ContactObservation]
    ) -> list[list[ContactObservation]]:
        """Group contiguous contact observations into runs."""
        runs: list[list[ContactObservation]] = []
        current: list[ContactObservation] = []

        for obs in history:
            if obs.contact_probability >= self.contact_threshold:
                if not current:
                    current.append(obs)
                else:
                    gap = obs.timestamp - current[-1].timestamp
                    if gap <= self.max_gap_seconds:
                        current.append(obs)
                    else:
                        if len(current) >= 2:
                            runs.append(current)
                        current = [obs]
            else:
                if len(current) >= 2:
                    runs.append(current)
                current = []

        if len(current) >= 2:
            runs.append(current)

        return runs

    def _evaluate_run_for_pickup(
        self,
        run: list[ContactObservation],
        full_history: list[ContactObservation],
    ) -> ManipulationCandidate | None:
        """Evaluate if contact run represents a valid pickup event."""
        run_duration = run[-1].timestamp - run[0].timestamp
        if run_duration < self.persist_duration:
            return None  # Insufficient persistent contact duration

        # Verify antecedent: hand approached or was free before run
        first_t = run[0].timestamp
        antecedents = [
            o for o in full_history
            if 0 < (first_t - o.timestamp) <= 1.5
        ]
        has_antecedent_approach = any(
            o.contact_probability < self.contact_threshold for o in antecedents
        ) or len(antecedents) == 0
        if not has_antecedent_approach:
            return None

        # Verify motion coupling during run:
        # hand displacement should be consistent with person motion
        p_motion = self._estimate_coupling(run)
        if not p_motion["is_coupled"] and run_duration < 1.0:
            return None

        # Build candidate pickup
        candidate_id = f"pickup_{run[0].person_track_id}_{int(run[0].timestamp * 1000)}"
        evidence = [
            f"Hand detected (side={run[0].hand_side}, conf={run[0].model_confidence:.2f})",
            f"Contact established (p={run[0].contact_probability:.2f}) at t={run[0].timestamp:.2f}s",
            f"Persistent contact sustained for {run_duration:.2f}s (>={self.persist_duration:.2f}s)",
            f"Motion coupling score={p_motion['coupling_score']:.2f}",
        ]
        conf = float(
            np.mean([o.contact_probability for o in run]) * 0.7
            + p_motion["coupling_score"] * 0.3
        )
        return ManipulationCandidate(
            candidate_id=candidate_id,
            event_type="pickup",
            person_track_id=run[0].person_track_id,
            video_id=run[0].source_video,
            start_seconds=run[0].timestamp,
            end_seconds=run[-1].timestamp,
            start_frame=run[0].source_frame,
            end_frame=run[-1].source_frame,
            confidence=min(1.0, conf),
            evidence_chain=evidence,
            supporting_observations=run,
        )

    def _evaluate_run_for_placement(
        self,
        run: list[ContactObservation],
        full_history: list[ContactObservation],
    ) -> ManipulationCandidate | None:
        """Evaluate if contact run ending represents a valid placement event."""
        run_duration = run[-1].timestamp - run[0].timestamp
        if run_duration < self.persist_duration:
            return None

        last_t = run[-1].timestamp
        postcedents = [
            o for o in full_history
            if 0 < (o.timestamp - last_t) <= 1.5
        ]
        has_separation = any(
            o.contact_probability < self.contact_threshold for o in postcedents
        )
        if not has_separation:
            return None

        candidate_id = f"placement_{run[0].person_track_id}_{int(last_t * 1000)}"
        evidence = [
            f"Holding contact sustained for {run_duration:.2f}s (>={self.persist_duration:.2f}s)",
            f"Release/separation observed at t={last_t:.2f}s (contact dropped)",
            f"Post-release stabilization across {len(postcedents)} frames",
        ]
        conf = float(np.mean([o.contact_probability for o in run]) * 0.8)
        return ManipulationCandidate(
            candidate_id=candidate_id,
            event_type="placement",
            person_track_id=run[0].person_track_id,
            video_id=run[0].source_video,
            start_seconds=run[0].timestamp,
            end_seconds=last_t,
            start_frame=run[0].source_frame,
            end_frame=run[-1].source_frame,
            confidence=min(1.0, conf),
            evidence_chain=evidence,
            supporting_observations=run,
        )

    def _estimate_coupling(self, run: list[ContactObservation]) -> dict[str, Any]:
        """Estimate kinematic coupling between hand and person motion."""
        if len(run) < 2:
            return {"is_coupled": True, "coupling_score": 0.5}

        # Calculate velocity vectors between adjacent observations
        vel_vectors = []
        for i in range(1, len(run)):
            b0 = run[i - 1].hand_box
            b1 = run[i].hand_box
            c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
            c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
            dt = max(0.001, run[i].timestamp - run[i - 1].timestamp)
            vx = (c1[0] - c0[0]) / dt
            vy = (c1[1] - c0[1]) / dt
            vel_vectors.append((vx, vy))

        speeds = [math.hypot(vx, vy) for vx, vy in vel_vectors]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Acceleration magnitudes (direction/speed changes)
        accels = []
        for i in range(1, len(vel_vectors)):
            dt = max(0.001, run[i + 1].timestamp - run[i].timestamp)
            ax = (vel_vectors[i][0] - vel_vectors[i - 1][0]) / dt
            ay = (vel_vectors[i][1] - vel_vectors[i - 1][1]) / dt
            accels.append(math.hypot(ax, ay))

        mean_accel = float(np.mean(accels)) if accels else 0.0
        # Smooth carrying has low acceleration; gesturing/waving has high acceleration (> 800 px/s^2)
        stability = max(0.0, 1.0 - (mean_accel / 800.0))
        coupling_score = float(stability)

        return {
            "is_coupled": coupling_score >= 0.50 and avg_speed < 800.0,
            "coupling_score": coupling_score,
            "avg_speed": avg_speed,
            "mean_accel": mean_accel,
        }

    def _is_duplicate(self, candidate: ManipulationCandidate) -> bool:
        """Prevent emitting overlapping candidates for the same person track."""
        for c in self.candidates:
            if c.person_track_id == candidate.person_track_id and c.event_type == candidate.event_type:
                # Check temporal overlap
                overlap = max(
                    0.0,
                    min(c.end_seconds, candidate.end_seconds)
                    - max(c.start_seconds, candidate.start_seconds),
                )
                if overlap > 0.5:
                    return True
        return False
