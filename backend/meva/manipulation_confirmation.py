"""Evaluation-only secondary manipulation confirmation gate.

Investigates whether independent visual and temporal confirmation signals can reduce
the false manipulation hypotheses produced by upstream hand/interaction-region pipelines.

Confirmation Signals Evaluated:
  1. Localized motion (motion inside interaction region vs person body / background)
  2. Relative motion & coupling (kinematic hand-to-region displacement and stability)
  3. Temporal appearance change (pre-interaction vs post-interaction local patch difference)
  4. Region persistence (spatial coherence of the interaction region across frames)
  5. Approach / separation evidence (hand displacement away from release/pickup site)

STRICT NO-LEAKAGE:
  Operates exclusively on candidate intervals generated from runtime observations,
  runtime person tracks, and raw video frames. No MEVA ground truth is used.
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

from .hand_object_contact import ContactObservation, ManipulationCandidate


@dataclass
class ConfirmationFeatures:
    """Independent confirmation measurements computed on a candidate interval."""
    localized_motion: float  # [0.0, 1.0] - degree of motion concentrated in interaction region
    relative_motion_coupling: float  # [0.0, 1.0] - hand-to-region kinematic consistency
    appearance_change: float  # [0.0, 1.0] - local patch appearance delta before vs after
    region_persistence: float  # [0.0, 1.0] - spatial coherence of interaction zone
    separation_evidence: float  # [0.0, 1.0] - post-event separation of hand from anchor
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "localized_motion": round(self.localized_motion, 3),
            "relative_motion_coupling": round(self.relative_motion_coupling, 3),
            "appearance_change": round(self.appearance_change, 3),
            "region_persistence": round(self.region_persistence, 3),
            "separation_evidence": round(self.separation_evidence, 3),
            "details": self.details,
        }


@dataclass
class ConfirmationDecision:
    """Gate outcome on an upstream manipulation candidate."""
    candidate_id: str
    status: str  # 'CONFIRMED', 'WEAK', 'REJECTED'
    score: float  # Combined confirmation score in [0.0, 1.0]
    features: ConfirmationFeatures
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "score": round(self.score, 3),
            "features": self.features.to_dict(),
            "explanation": self.explanation,
        }


def compute_localized_motion(
    frames: list[np.ndarray],
    interaction_boxes: list[tuple[int, int, int, int]],
    person_boxes: list[tuple[int, int, int, int]],
) -> tuple[float, dict[str, float]]:
    """Measure whether motion energy is concentrated in the interaction region vs the body."""
    if len(frames) < 2 or not interaction_boxes:
        return 0.5, {"ratio": 1.0, "inter_motion": 0.0, "body_motion": 0.0}

    inter_diffs: list[float] = []
    body_diffs: list[float] = []

    for i in range(1, min(len(frames), len(interaction_boxes))):
        f0 = frames[i - 1]
        f1 = frames[i]
        diff = cv2.absdiff(f1, f0) if cv2 else np.abs(f1.astype(float) - f0.astype(float))
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if cv2 else diff.mean(axis=2)
        fh, fw = gray.shape[:2]

        # Interaction region motion
        ix1, iy1, ix2, iy2 = interaction_boxes[i]
        ix1, iy1 = max(0, ix1), max(0, iy1)
        ix2, iy2 = min(fw, ix2), min(fh, iy2)
        if ix2 > ix1 and iy2 > iy1:
            im = float(np.mean(gray[iy1:iy2, ix1:ix2]))
            inter_diffs.append(im)

        # Person body motion
        if i < len(person_boxes):
            px1, py1, px2, py2 = person_boxes[i]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(fw, px2), min(fh, py2)
            if px2 > px1 and py2 > py1:
                bm = float(np.mean(gray[py1:py2, px1:px2]))
                body_diffs.append(bm)

    avg_inter = float(np.mean(inter_diffs)) if inter_diffs else 0.0
    avg_body = float(np.mean(body_diffs)) if body_diffs else 0.0

    # If body is static but hand is moving (typical manipulation): ratio > 1.0
    # If entire body is walking fast: avg_body is very high (> 25), ratio <= 1.0
    ratio = avg_inter / max(1e-3, avg_body)

    # Normalize to [0.0, 1.0]
    # Optimal manipulation ratio is in [0.8, 2.5]
    if avg_body > 35.0:
        # Rapid whole-body motion / walking
        score = max(0.0, 0.6 - (avg_body - 35.0) / 50.0)
    else:
        score = min(1.0, max(0.0, (ratio - 0.5) / 1.5))

    return score, {
        "ratio": float(ratio),
        "inter_motion": float(avg_inter),
        "body_motion": float(avg_body),
    }


def compute_relative_motion_coupling(
    observations: list[ContactObservation],
    person_boxes: list[tuple[int, int, int, int]],
) -> tuple[float, dict[str, float]]:
    """Measure kinematic coupling and smooth coordination between hand and person."""
    if len(observations) < 2:
        return 0.5, {"coupling_stability": 0.5, "speed_variance": 0.0}

    vel_vectors: list[tuple[float, float]] = []
    for i in range(1, len(observations)):
        b0 = observations[i - 1].hand_box
        b1 = observations[i].hand_box
        dt = max(0.001, observations[i].timestamp - observations[i - 1].timestamp)
        c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        vx = (c1[0] - c0[0]) / dt
        vy = (c1[1] - c0[1]) / dt
        vel_vectors.append((vx, vy))

    speeds = [math.hypot(vx, vy) for vx, vy in vel_vectors]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0

    accels = []
    for i in range(1, len(vel_vectors)):
        dt = max(0.001, observations[i + 1].timestamp - observations[i].timestamp)
        ax = (vel_vectors[i][0] - vel_vectors[i - 1][0]) / dt
        ay = (vel_vectors[i][1] - vel_vectors[i - 1][1]) / dt
        accels.append(math.hypot(ax, ay))

    speed_var = float(np.var(speeds)) if len(speeds) > 1 else 0.0
    mean_accel = float(np.mean(accels)) if accels else 0.0
    stability = max(0.0, 1.0 - (mean_accel / 600.0))
    speed_penalty = 1.0 if mean_speed < 600.0 else max(0.0, 1.0 - (mean_speed - 600.0) / 400.0)
    coupling_score = min(1.0, max(0.0, stability * speed_penalty))

    return coupling_score, {
        "coupling_stability": float(stability),
        "mean_speed": float(mean_speed),
        "speed_variance": float(speed_var),
        "mean_accel": float(mean_accel),
    }


def compute_appearance_change(
    pre_frame: np.ndarray | None,
    post_frame: np.ndarray | None,
    anchor_box: tuple[int, int, int, int],
) -> tuple[float, dict[str, float]]:
    """Measure local appearance / texture change at the interaction site before vs after candidate."""
    if pre_frame is None or post_frame is None:
        return 0.4, {"delta_appearance": 0.0, "patch_contrast": 0.0}

    fh, fw = pre_frame.shape[:2]
    x1, y1, x2, y2 = anchor_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.4, {"delta_appearance": 0.0, "patch_contrast": 0.0}

    patch_pre = pre_frame[y1:y2, x1:x2]
    patch_post = post_frame[y1:y2, x1:x2]

    if patch_pre.size == 0 or patch_post.size == 0:
        return 0.4, {"delta_appearance": 0.0, "patch_contrast": 0.0}

    diff = cv2.absdiff(patch_post, patch_pre) if cv2 else np.abs(patch_post.astype(float) - patch_pre.astype(float))
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if cv2 else diff.mean(axis=2)
    mean_delta = float(np.mean(gray_diff))

    # Also compute texture standard deviation in patch
    gray_pre = cv2.cvtColor(patch_pre, cv2.COLOR_BGR2GRAY) if cv2 else patch_pre.mean(axis=2)
    std_pre = float(np.std(gray_pre))

    # Normalized score: meaningful local change is typically 6-30 intensity levels
    score = min(1.0, max(0.0, (mean_delta - 2.0) / 20.0))

    return score, {
        "delta_appearance": float(mean_delta),
        "patch_contrast": float(std_pre),
    }


def compute_region_persistence(
    interaction_boxes: list[tuple[int, int, int, int]],
) -> tuple[float, dict[str, float]]:
    """Measure spatial coherence and overlap stability of the interaction region across frames."""
    if len(interaction_boxes) < 2:
        return 0.5, {"mean_overlap_iou": 0.5}

    ious: list[float] = []
    for i in range(1, len(interaction_boxes)):
        b0 = interaction_boxes[i - 1]
        b1 = interaction_boxes[i]
        # Bounding box IoU
        ix1 = max(b0[0], b1[0])
        iy1 = max(b0[1], b1[1])
        ix2 = min(b0[2], b1[2])
        iy2 = min(b0[3], b1[3])
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        a0 = max(1, (b0[2] - b0[0]) * (b0[3] - b0[1]))
        a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        union = a0 + a1 - inter
        ious.append(inter / union if union > 0 else 0.0)

    mean_iou = float(np.mean(ious)) if ious else 0.5
    # Coherent interaction region has substantial overlap (0.3 to 0.8)
    persistence_score = min(1.0, max(0.0, mean_iou / 0.70))

    return persistence_score, {"mean_overlap_iou": float(mean_iou)}


def compute_separation_evidence(
    candidate: ManipulationCandidate,
    post_observations: list[ContactObservation],
    person_height: float = 200.0,
) -> tuple[float, dict[str, float]]:
    """Measure post-event separation between hand and interaction site (release/pull-away)."""
    if not candidate.supporting_observations or not post_observations:
        # If no postcedent observations available, return neutral
        return 0.45, {"separation_dist_px": 0.0, "norm_separation": 0.0}

    last_obs = candidate.supporting_observations[-1]
    anchor_box = last_obs.hand_box
    ax = (anchor_box[0] + anchor_box[2]) / 2
    ay = (anchor_box[1] + anchor_box[3]) / 2

    # Find maximum hand separation in post-event window (within 1.5s after end)
    max_dist = 0.0
    for p_obs in post_observations:
        hb = p_obs.hand_box
        hx = (hb[0] + hb[2]) / 2
        hy = (hb[1] + hb[3]) / 2
        d = math.hypot(hx - ax, hy - ay)
        max_dist = max(max_dist, d)

    norm_sep = max_dist / max(1.0, person_height)
    # Physical separation is typically > 0.15 of person height (hand pulls away from table/surface)
    # If hand stays glued at same position (clasped hands), norm_sep < 0.08
    score = min(1.0, max(0.0, (norm_sep - 0.08) / 0.25))

    return score, {
        "separation_dist_px": float(max_dist),
        "norm_separation": float(norm_sep),
    }


class ManipulationConfirmationGate:
    """Secondary confirmation gate evaluating candidate manipulation events.

    Modes:
      - 'baseline': Pass-through (no confirmation filtering)
      - 'localized_motion': Gated on localized motion only
      - 'relative_motion': Gated on relative motion coupling only
      - 'appearance_change': Gated on appearance change only
      - 'separation': Gated on separation evidence only
      - 'best_two': Weighted combination of appearance change + separation
      - 'best_three': Weighted combination of localized motion + appearance change + separation
      - 'all': Full multi-feature weighted confirmation
    """

    def __init__(
        self,
        mode: str = "all",
        confirm_threshold: float = 0.45,
        weak_threshold: float = 0.30,
    ):
        self.mode = mode
        self.confirm_threshold = confirm_threshold
        self.weak_threshold = weak_threshold

    def evaluate_candidate(
        self,
        candidate: ManipulationCandidate,
        candidate_frames: list[np.ndarray],
        pre_frame: np.ndarray | None = None,
        post_frame: np.ndarray | None = None,
        post_observations: list[ContactObservation] | None = None,
        person_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> ConfirmationDecision:
        """Evaluate confirmation features and produce an auditable decision."""
        if self.mode == "baseline":
            # Pass-through: retains upstream candidate unchanged
            features = ConfirmationFeatures(
                localized_motion=1.0,
                relative_motion_coupling=1.0,
                appearance_change=1.0,
                region_persistence=1.0,
                separation_evidence=1.0,
            )
            return ConfirmationDecision(
                candidate_id=candidate.candidate_id,
                status="CONFIRMED",
                score=candidate.confidence,
                features=features,
                explanation=f"{candidate.event_type.capitalize()} candidate (baseline pass-through)",
            )

        if not candidate.supporting_observations:
            features = ConfirmationFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
            return ConfirmationDecision(
                candidate_id=candidate.candidate_id,
                status="REJECTED",
                score=0.0,
                features=features,
                explanation=f"{candidate.event_type.capitalize()} candidate rejected (zero supporting observations)",
            )

        # Extract interaction boxes and observations
        obs_list = candidate.supporting_observations
        inter_boxes = [obs.interaction_region for obs in obs_list]
        p_boxes = person_boxes or [obs.hand_box for obs in obs_list]

        # 1. Localized motion
        s_motion, d_motion = compute_localized_motion(candidate_frames, inter_boxes, p_boxes)

        # 2. Relative motion coupling
        s_coupling, d_coupling = compute_relative_motion_coupling(obs_list, p_boxes)

        # 3. Appearance change
        anchor_box = inter_boxes[-1] if inter_boxes else (0, 0, 10, 10)
        s_appear, d_appear = compute_appearance_change(pre_frame, post_frame, anchor_box)

        # 4. Region persistence
        s_persist, d_persist = compute_region_persistence(inter_boxes)

        # 5. Separation evidence
        ph = 200.0
        if p_boxes:
            ph = float(max(20, p_boxes[-1][3] - p_boxes[-1][1]))
        s_sep, d_sep = compute_separation_evidence(candidate, post_observations or [], person_height=ph)

        all_details = {
            "motion": d_motion,
            "coupling": d_coupling,
            "appearance": d_appear,
            "persistence": d_persist,
            "separation": d_sep,
        }
        features = ConfirmationFeatures(
            localized_motion=s_motion,
            relative_motion_coupling=s_coupling,
            appearance_change=s_appear,
            region_persistence=s_persist,
            separation_evidence=s_sep,
            details=all_details,
        )

        # Compute mode score
        if self.mode == "localized_motion":
            score = s_motion
        elif self.mode == "relative_motion":
            score = s_coupling
        elif self.mode == "appearance_change":
            score = s_appear
        elif self.mode == "separation":
            score = s_sep
        elif self.mode == "best_two":
            # Combine appearance change (40%) and separation (60%)
            score = 0.40 * s_appear + 0.60 * s_sep
        elif self.mode == "best_three":
            # Localized motion (25%) + appearance change (35%) + separation (40%)
            score = 0.25 * s_motion + 0.35 * s_appear + 0.40 * s_sep
        elif self.mode == "all":
            # Transparent weighted combination
            score = (
                0.20 * s_motion
                + 0.15 * s_coupling
                + 0.25 * s_appear
                + 0.15 * s_persist
                + 0.25 * s_sep
            )
        else:
            raise ValueError(f"Unknown confirmation mode: {self.mode}")

        # Determine status
        if score >= self.confirm_threshold:
            status = "CONFIRMED"
        elif score >= self.weak_threshold:
            status = "WEAK"
        else:
            status = "REJECTED"

        # Evidence explanation string
        dur = round(candidate.end_seconds - candidate.start_seconds, 2)
        explanation = (
            f"{candidate.event_type.capitalize()} interaction candidate "
            f"({status}, score={score:.2f}) over {dur}s: "
            f"loc_motion={s_motion:.2f}, coupling={s_coupling:.2f}, "
            f"appear_change={s_appear:.2f}, persist={s_persist:.2f}, separation={s_sep:.2f}"
        )

        return ConfirmationDecision(
            candidate_id=candidate.candidate_id,
            status=status,
            score=score,
            features=features,
            explanation=explanation,
        )
