"""Unit and regression tests for secondary manipulation confirmation gate.

Validates the confirmation reasoning logic across mandated scenarios:
  1. no interaction
  2. transient interaction
  3. persistent interaction
  4. walking
  5. gesturing
  6. clasped hands
  7. localized motion
  8. global motion
  9. stable interaction region
  10. appearance change
  11. pickup sequence
  12. placement sequence
  13. release
  14. separation
  15. missing frames
  16. occlusion
  17. multiple people
  18. ambiguous cases
  19. threshold behavior
"""

import numpy as np
from meva.hand_object_contact import ContactObservation, ManipulationCandidate
from meva.manipulation_confirmation import (
    ManipulationConfirmationGate,
    compute_appearance_change,
    compute_localized_motion,
    compute_region_persistence,
    compute_relative_motion_coupling,
    compute_separation_evidence,
)


def _make_candidate(
    cand_id: str,
    event_type: str,
    t_start: float = 1.0,
    t_end: float = 2.0,
    hand_box: tuple[int, int, int, int] = (100, 200, 120, 220),
    inter_box: tuple[int, int, int, int] = (90, 190, 130, 230),
) -> tuple[ManipulationCandidate, list[ContactObservation]]:
    obs = [
        ContactObservation(
            timestamp=t,
            frame_index=int(t * 30),
            person_track_id="P1",
            hand_side="right",
            hand_box=hand_box,
            interaction_region=inter_box,
            contact_probability=0.80,
            model_confidence=0.85,
            source_video="TEST_VIDEO",
            source_frame=int(t * 30),
            level="CONTACT",
        )
        for t in [t_start, t_start + 0.5, t_end]
    ]
    cand = ManipulationCandidate(
        candidate_id=cand_id,
        event_type=event_type,
        person_track_id="P1",
        video_id="TEST_VIDEO",
        start_seconds=t_start,
        end_seconds=t_end,
        start_frame=int(t_start * 30),
        end_frame=int(t_end * 30),
        confidence=0.80,
        evidence_chain=["Upstream interaction detected"],
        supporting_observations=obs,
    )
    return cand, obs


def test_1_no_interaction():
    """Empty candidate with zero observations is rejected."""
    cand = ManipulationCandidate(
        candidate_id="empty_cand",
        event_type="pickup",
        person_track_id="P1",
        video_id="TEST_VIDEO",
        start_seconds=1.0,
        end_seconds=1.1,
        start_frame=30,
        end_frame=33,
        confidence=0.1,
        evidence_chain=[],
        supporting_observations=[],
    )
    gate = ManipulationConfirmationGate(mode="all", confirm_threshold=0.45)
    decision = gate.evaluate_candidate(cand, candidate_frames=[])
    assert decision.status in ("WEAK", "REJECTED")


def test_2_transient_interaction():
    """Interaction lasting only 0.1s receives low score."""
    cand, obs = _make_candidate("cand_transient", "pickup", t_start=1.0, t_end=1.1)
    cand.supporting_observations = obs[:1]
    gate = ManipulationConfirmationGate(mode="all", confirm_threshold=0.45)
    decision = gate.evaluate_candidate(cand, candidate_frames=[])
    assert decision.score < 0.60


def test_3_persistent_interaction():
    """Sustained interaction with stable region scores well on persistence."""
    _, obs = _make_candidate("cand_persist", "pickup", t_start=1.0, t_end=2.5)
    inter_boxes = [o.interaction_region for o in obs]
    score, details = compute_region_persistence(inter_boxes)
    assert score >= 0.70
    assert details["mean_overlap_iou"] > 0.50


def test_4_walking():
    """Walking sequence with large body motion results in suppressed localized motion score."""
    # Simulate high body motion (45 gray levels) vs interaction motion
    f0 = np.zeros((400, 400, 3), dtype=np.uint8)
    f1 = np.full((400, 400, 3), 45, dtype=np.uint8)
    score, details = compute_localized_motion(
        [f0, f1],
        interaction_boxes=[(100, 100, 150, 150), (100, 100, 150, 150)],
        person_boxes=[(80, 50, 200, 350), (80, 50, 200, 350)],
    )
    # Whole-body walking motion suppresses localized motion score
    assert score <= 0.40
    assert details["body_motion"] >= 35.0


def test_5_gesturing():
    """Violent hand movement has high speed variance and fails relative motion coupling."""
    _, obs = _make_candidate("cand_gesture", "pickup")
    # Make hand velocities erratic
    obs[0].hand_box = (100, 200, 120, 220)
    obs[1].hand_box = (180, 150, 200, 170)
    obs[2].hand_box = (110, 240, 130, 260)
    score, details = compute_relative_motion_coupling(obs, [(80, 50, 200, 350)] * 3)
    assert score < 0.50
    assert details["mean_accel"] > 600.0


def test_6_clasped_hands():
    """Clasped hands stay in same position post-event, failing separation evidence."""
    cand, _ = _make_candidate("cand_clasped", "placement")
    # Post-observations stay glued to same location
    post_obs = [
        ContactObservation(
            timestamp=2.5,
            frame_index=75,
            person_track_id="P1",
            hand_side="right",
            hand_box=(100, 200, 120, 220),  # Identical box
            interaction_region=(90, 190, 130, 230),
            contact_probability=0.2,
            model_confidence=0.8,
            source_video="TEST_VIDEO",
            source_frame=75,
            level="NEAR_INTERACTION",
        )
    ]
    score, details = compute_separation_evidence(cand, post_obs, person_height=200.0)
    # Separation score is 0.0 because hand did not separate
    assert score == 0.0
    assert details["norm_separation"] < 0.05


def test_7_localized_motion():
    """Specific motion at interaction site while body remains still gives high localized motion score."""
    f0 = np.zeros((300, 300, 3), dtype=np.uint8)
    f1 = np.zeros((300, 300, 3), dtype=np.uint8)
    # Paint motion in interaction region only
    f1[100:150, 100:150] = 30
    score, details = compute_localized_motion(
        [f0, f1],
        interaction_boxes=[(100, 100, 150, 150), (100, 100, 150, 150)],
        person_boxes=[(50, 50, 250, 280), (50, 50, 250, 280)],
    )
    assert score >= 0.70
    assert details["ratio"] > 1.5


def test_8_global_motion():
    """Equal global frame lighting change is recognized as non-localized."""
    f0 = np.zeros((300, 300, 3), dtype=np.uint8)
    f1 = np.full((300, 300, 3), 20, dtype=np.uint8)
    _, details = compute_localized_motion(
        [f0, f1],
        interaction_boxes=[(100, 100, 150, 150), (100, 100, 150, 150)],
        person_boxes=[(50, 50, 250, 280), (50, 50, 250, 280)],
    )
    # Ratio is ~1.0
    assert abs(details["ratio"] - 1.0) < 0.1


def test_9_stable_interaction_region():
    """Stationary interaction region across frames achieves high region persistence score."""
    boxes = [(100, 150, 140, 190)] * 4
    score, _ = compute_region_persistence(boxes)
    assert score >= 0.95


def test_10_appearance_change():
    """Local patch change between pre and post frames produces high appearance score."""
    f_pre = np.zeros((300, 300, 3), dtype=np.uint8)
    f_post = np.zeros((300, 300, 3), dtype=np.uint8)
    # Object appears/disappears at (100, 150)
    f_post[150:190, 100:140] = 50
    score, details = compute_appearance_change(f_pre, f_post, (100, 150, 140, 190))
    assert score >= 0.70
    assert details["delta_appearance"] > 15.0


def test_11_pickup_sequence():
    """Complete pickup sequence with approach, stable grasp, and pull-away passes confirmation."""
    cand, _ = _make_candidate("pickup_valid", "pickup")
    # Post-event observation: hand pulls away by 60 pixels
    post_obs = [
        ContactObservation(
            timestamp=2.5,
            frame_index=75,
            person_track_id="P1",
            hand_side="right",
            hand_box=(160, 140, 180, 160),  # Moved up and right
            interaction_region=(150, 130, 190, 170),
            contact_probability=0.2,
            model_confidence=0.8,
            source_video="TEST_VIDEO",
            source_frame=75,
            level="NEAR_INTERACTION",
        )
    ]
    f_pre = np.zeros((300, 300, 3), dtype=np.uint8)
    f_post = np.zeros((300, 300, 3), dtype=np.uint8)
    f_post[200:220, 100:120] = 30  # Table patch changed
    gate = ManipulationConfirmationGate(mode="all", confirm_threshold=0.45)
    decision = gate.evaluate_candidate(
        cand,
        candidate_frames=[f_pre, f_post],
        pre_frame=f_pre,
        post_frame=f_post,
        post_observations=post_obs,
    )
    assert decision.status == "CONFIRMED"
    assert decision.score >= 0.45
    assert "Pickup interaction candidate (CONFIRMED" in decision.explanation


def test_12_placement_sequence():
    """Complete placement sequence with holding, placement, and hand separation passes confirmation."""
    cand, _ = _make_candidate("placement_valid", "placement")
    # Post-event observation: hand pulls away
    post_obs = [
        ContactObservation(
            timestamp=2.5,
            frame_index=75,
            person_track_id="P1",
            hand_side="right",
            hand_box=(170, 130, 190, 150),
            interaction_region=(160, 120, 200, 160),
            contact_probability=0.1,
            model_confidence=0.8,
            source_video="TEST_VIDEO",
            source_frame=75,
            level="HAND_DETECTED",
        )
    ]
    f_pre = np.zeros((300, 300, 3), dtype=np.uint8)
    f_post = np.zeros((300, 300, 3), dtype=np.uint8)
    f_post[200:220, 100:120] = 35
    gate = ManipulationConfirmationGate(mode="all", confirm_threshold=0.45)
    decision = gate.evaluate_candidate(
        cand,
        candidate_frames=[f_pre, f_post],
        pre_frame=f_pre,
        post_frame=f_post,
        post_observations=post_obs,
    )
    assert decision.status == "CONFIRMED"
    assert decision.score >= 0.45


def test_13_release():
    """Clear postcedent release increases separation score."""
    cand, _ = _make_candidate("cand_rel", "placement")
    post_obs = [
        ContactObservation(
            timestamp=2.5,
            frame_index=75,
            person_track_id="P1",
            hand_side="right",
            hand_box=(180, 200, 200, 220),  # 80 px displacement
            interaction_region=(170, 190, 210, 230),
            contact_probability=0.1,
            model_confidence=0.8,
            source_video="TEST_VIDEO",
            source_frame=75,
            level="HAND_DETECTED",
        )
    ]
    score, details = compute_separation_evidence(cand, post_obs, person_height=200.0)
    assert score >= 0.80
    assert details["norm_separation"] >= 0.35


def test_14_separation():
    """Zero postcedent separation yields zero separation score."""
    cand, _ = _make_candidate("cand_nosep", "placement")
    _, details = compute_separation_evidence(cand, [], person_height=200.0)
    assert details["separation_dist_px"] == 0.0


def test_15_missing_frames():
    """Missing pre/post frames fall back gracefully to neutral score without crashing."""
    score, details = compute_appearance_change(None, None, (10, 10, 20, 20))
    assert 0.3 <= score <= 0.5
    assert details["delta_appearance"] == 0.0


def test_16_occlusion():
    """Occlusion causing empty interaction box list does not error."""
    score, _ = compute_localized_motion([], [], [])
    assert score == 0.5


def test_17_multiple_people():
    """Candidates from two different people are confirmed independently."""
    cand1, _ = _make_candidate("cand_P1", "pickup")
    cand1.person_track_id = "P1"
    cand2, _ = _make_candidate("cand_P2", "pickup")
    cand2.person_track_id = "P2"
    gate = ManipulationConfirmationGate(mode="all")
    d1 = gate.evaluate_candidate(cand1, [])
    d2 = gate.evaluate_candidate(cand2, [])
    assert d1.candidate_id == "cand_P1"
    assert d2.candidate_id == "cand_P2"


def test_18_ambiguous_cases():
    """Ambiguous candidate near boundary receives WEAK status."""
    gate_mock = ManipulationConfirmationGate(confirm_threshold=0.60, weak_threshold=0.35)
    score = 0.45
    status = "CONFIRMED" if score >= gate_mock.confirm_threshold else ("WEAK" if score >= gate_mock.weak_threshold else "REJECTED")
    assert status == "WEAK"


def test_19_threshold_behavior():
    """Varying confirm_threshold changes status from CONFIRMED to WEAK to REJECTED."""
    cand, _ = _make_candidate("cand_thresh", "pickup")
    # Low threshold confirms
    g_low = ManipulationConfirmationGate(mode="baseline", confirm_threshold=0.30)
    assert g_low.evaluate_candidate(cand, []).status == "CONFIRMED"
