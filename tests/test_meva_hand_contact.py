"""Unit and regression tests for hand-object contact perception and temporal state machine.

Validates the state machine across all mandated scenarios:
  1. no contact
  2. transient contact
  3. persistent contact
  4. walking
  5. gesturing
  6. hand near interaction region
  7. approach without contact
  8. contact followed by movement
  9. release
  10. pickup candidate
  11. placement candidate
  12. ambiguous contact
  13. missing hand observation
  14. multiple people
  15. occlusion
  16. temporal gaps
"""

from meva.hand_object_contact import (
    ContactObservation,
    ContactStateMachine,
    evaluate_hand_contact_state,
)


def _make_observation(
    track_id: str,
    t: float,
    contact_prob: float,
    hand_box: tuple[int, int, int, int] = (100, 200, 120, 220),
    hand_conf: float = 0.85,
    level: str = "CONTACT",
) -> ContactObservation:
    f_idx = int(t * 30.0)
    return ContactObservation(
        timestamp=t,
        frame_index=f_idx,
        person_track_id=track_id,
        hand_side="right",
        hand_box=hand_box,
        interaction_region=(hand_box[0] - 10, hand_box[1] - 10, hand_box[2] + 10, hand_box[3] + 10),
        contact_probability=contact_prob,
        model_confidence=hand_conf,
        source_video="TEST_VIDEO",
        source_frame=f_idx,
        level=level,
    )


def test_1_no_contact():
    """Hands detected with low contact probability produce zero manipulation candidates."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    for t in [1.0, 1.5, 2.0, 2.5, 3.0]:
        obs = _make_observation("P1", t, contact_prob=0.20, level="HAND_DETECTED")
        cands = sm.process_observation(obs)
        assert len(cands) == 0
    assert len(sm.candidates) == 0


def test_2_transient_contact():
    """Contact lasting only 0.2s (< 0.5s threshold) is rejected as transient."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    # Brief contact for 0.2s, then drops
    for t, p in [(1.0, 0.2), (1.1, 0.7), (1.2, 0.8), (1.3, 0.2), (1.4, 0.2)]:
        obs = _make_observation("P1", t, contact_prob=p)
        sm.process_observation(obs)
    assert len(sm.candidates) == 0


def test_3_persistent_contact():
    """Contact sustained for >= 0.5s successfully forms persistent contact runs."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    # Contact from 1.0s to 1.8s (0.8s duration)
    for t in [1.0, 1.2, 1.4, 1.6, 1.8]:
        obs = _make_observation("P1", t, contact_prob=0.80)
        sm.process_observation(obs)
    runs = sm._extract_contact_runs(sm.track_history["P1"])
    assert len(runs) == 1
    assert (runs[0][-1].timestamp - runs[0][0].timestamp) >= 0.5


def test_4_walking():
    """Walking posture with arms hanging at sides has low contact score and produces no candidate."""
    person_box = (100, 100, 200, 400)  # height = 300
    # Wrists at bottom of torso near hips, low reach
    kps = {
        "left_wrist": {"x": 130.0, "y": 280.0, "conf": 0.80},
        "right_wrist": {"x": 170.0, "y": 285.0, "conf": 0.85},
        "left_hip": {"x": 135.0, "y": 275.0, "conf": 0.75},
        "right_hip": {"x": 165.0, "y": 275.0, "conf": 0.75},
        "left_shoulder": {"x": 130.0, "y": 150.0, "conf": 0.80},
        "right_shoulder": {"x": 170.0, "y": 150.0, "conf": 0.80},
    }
    states = evaluate_hand_contact_state(person_box, kps, frame_pixels=None)
    for st in states:
        assert st["level"] in ("HAND_DETECTED", "NEAR_INTERACTION")
        assert st["contact_probability"] < 0.45


def test_5_gesturing():
    """Arm in air without sustained stable interaction region does not yield persistent pickup."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    # Gesturing: erratic hand displacement, rapid speed variation
    for i, t in enumerate([1.0, 1.2, 1.4, 1.6, 1.8]):
        # Hand jumps violently across frame
        hx = 100 + i * 80
        hy = 150 - (i % 2) * 60
        obs = _make_observation("P1", t, contact_prob=0.75, hand_box=(hx, hy, hx + 20, hy + 20))
        sm.process_observation(obs)
    # Because motion coupling between hand and body is erratic (high variance), pickup is rejected
    assert len(sm.candidates) == 0


def test_6_hand_near_interaction_region():
    """Hand at waist level without active reach or object contrast is classified as NEAR_INTERACTION."""
    person_box = (100, 100, 200, 400)
    kps = {
        "right_wrist": {"x": 160.0, "y": 260.0, "conf": 0.70},
        "right_hip": {"x": 160.0, "y": 265.0, "conf": 0.70},
        "right_shoulder": {"x": 160.0, "y": 160.0, "conf": 0.70},
    }
    states = evaluate_hand_contact_state(person_box, kps)
    assert len(states) == 1
    assert states[0]["level"] == "NEAR_INTERACTION"


def test_7_approach_without_contact():
    """Person approaches table but contact never occurs -> no pickup candidate."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    for t in [1.0, 1.5, 2.0, 2.5]:
        obs = _make_observation("P1", t, contact_prob=0.30, level="NEAR_INTERACTION")
        sm.process_observation(obs)
    assert len(sm.candidates) == 0


def test_8_contact_followed_by_movement():
    """Hand approaches -> persistent contact for 0.8s -> coupled motion produces candidate pickup."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    # Antecedent approach: free hand
    sm.process_observation(_make_observation("P1", 0.5, contact_prob=0.20, level="NEAR_INTERACTION"))
    # Contact run with smooth displacement
    for i, t in enumerate([1.0, 1.2, 1.4, 1.6, 1.8]):
        hx = 100 + i * 5
        hy = 200 + i * 5
        obs = _make_observation("P1", t, contact_prob=0.75, hand_box=(hx, hy, hx + 20, hy + 20))
        sm.process_observation(obs)
    pickups = [c for c in sm.candidates if c.event_type == "pickup"]
    assert len(pickups) >= 1
    assert pickups[0].event_type == "pickup"
    assert len(pickups[0].evidence_chain) >= 3


def test_9_release():
    """Carrying object -> contact ends (drops) with separation observed produces placement candidate."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    # Initial contact run (holding)
    for t in [1.0, 1.2, 1.4, 1.6]:
        sm.process_observation(_make_observation("P1", t, contact_prob=0.80))
    # Postcedent release: hand separates and contact drops
    for t in [1.8, 2.0]:
        sm.process_observation(_make_observation("P1", t, contact_prob=0.15, level="HAND_DETECTED"))
    placements = [c for c in sm.candidates if c.event_type == "placement"]
    assert len(placements) >= 1
    assert placements[0].event_type == "placement"


def test_10_pickup_candidate():
    """Pickup candidate contains complete auditable evidence chain."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    sm.process_observation(_make_observation("P1", 0.5, contact_prob=0.10))
    for t in [1.0, 1.2, 1.4, 1.6]:
        sm.process_observation(_make_observation("P1", t, contact_prob=0.85))
    pickups = [c for c in sm.candidates if c.event_type == "pickup"]
    assert len(pickups) == 1
    p = pickups[0]
    assert p.start_seconds == 1.0
    assert p.end_seconds == 1.6
    assert p.confidence > 0.60
    assert any("Persistent contact" in ev for ev in p.evidence_chain)


def test_11_placement_candidate():
    """Placement candidate contains complete auditable evidence chain."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    for t in [1.0, 1.2, 1.4, 1.6]:
        sm.process_observation(_make_observation("P1", t, contact_prob=0.85))
    for t in [1.8, 2.0]:
        sm.process_observation(_make_observation("P1", t, contact_prob=0.10))
    placements = [c for c in sm.candidates if c.event_type == "placement"]
    assert len(placements) == 1
    pl = placements[0]
    assert pl.start_seconds == 1.0
    assert pl.end_seconds == 1.6
    assert any("Release/separation" in ev for ev in pl.evidence_chain)


def test_12_ambiguous_contact():
    """Flickering contact right on threshold boundary (0.44-0.46) does not form persistent run."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    # Oscillates between 0.44 and 0.46
    for t, p in [(1.0, 0.44), (1.2, 0.46), (1.4, 0.43), (1.6, 0.46), (1.8, 0.42)]:
        sm.process_observation(_make_observation("P1", t, contact_prob=p))
    runs = sm._extract_contact_runs(sm.track_history["P1"])
    assert len(runs) == 0  # No run achieved >= 2 consecutive qualifying frames


def test_13_missing_hand_observation():
    """Low wrist confidence (< 0.30) produces no hand states or phantom hand boxes."""
    person_box = (100, 100, 200, 400)
    kps = {
        "left_wrist": {"x": 120.0, "y": 250.0, "conf": 0.15},
        "right_wrist": {"x": 180.0, "y": 250.0, "conf": 0.22},
    }
    states = evaluate_hand_contact_state(person_box, kps, hand_conf_threshold=0.30)
    assert len(states) == 0


def test_14_multiple_people():
    """Observations from two people are tracked independently without cross-talk."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45)
    # P1 is picking up an object
    sm.process_observation(_make_observation("P1", 0.5, contact_prob=0.10))
    for t in [1.0, 1.2, 1.4, 1.6]:
        sm.process_observation(_make_observation("P1", t, contact_prob=0.85))
        # P2 is just walking
        sm.process_observation(_make_observation("P2", t, contact_prob=0.15, level="HAND_DETECTED"))

    assert "P1" in sm.track_history
    assert "P2" in sm.track_history
    p1_cands = [c for c in sm.candidates if c.person_track_id == "P1"]
    p2_cands = [c for c in sm.candidates if c.person_track_id == "P2"]
    assert len(p1_cands) == 1
    assert len(p2_cands) == 0


def test_15_occlusion():
    """Temporal gap caused by person occlusion stops run and prevents false candidate."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45, max_gap_seconds=0.6)
    # Brief contact at 1.0s, then 2.0s gap (occlusion), then contact at 3.0s
    sm.process_observation(_make_observation("P1", 1.0, contact_prob=0.80))
    sm.process_observation(_make_observation("P1", 1.1, contact_prob=0.80))
    # Occlusion gap of 2 seconds
    sm.process_observation(_make_observation("P1", 3.1, contact_prob=0.80))
    sm.process_observation(_make_observation("P1", 3.2, contact_prob=0.80))
    runs = sm._extract_contact_runs(sm.track_history["P1"])
    # Two separate brief runs, neither exceeds 0.5s duration
    assert len(runs) == 2
    for r in runs:
        assert (r[-1].timestamp - r[0].timestamp) < 0.5


def test_16_temporal_gaps():
    """Gaps exceeding max_gap_seconds reset run rather than falsely bridging."""
    sm = ContactStateMachine(persist_duration=0.5, contact_threshold=0.45, max_gap_seconds=0.6)
    sm.process_observation(_make_observation("P1", 1.0, contact_prob=0.80))
    sm.process_observation(_make_observation("P1", 1.8, contact_prob=0.80))  # 0.8s gap > 0.6s
    runs = sm._extract_contact_runs(sm.track_history["P1"])
    assert len(runs) == 0  # Neither segment had >= 2 contiguous frames
