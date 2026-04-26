from services.cues import clear_cue_state, evaluate_cues


def _frame(ts, **overrides):
    frame = {
        "timestamp": ts,
        "phase": "drive",
        "knee_angle_l": 100,
        "knee_angle_r": 100,
        "trunk_lean": 12,
    }
    frame.update(overrides)
    return frame


def test_drive_knee_higher_triggers_after_three_bad_reps():
    session_id = "cue_session_drive"
    clear_cue_state(session_id)

    frames = [
        _frame(1.0, phase="drive", knee_angle_l=80, knee_angle_r=82),
        _frame(1.2, phase="flight", knee_angle_l=80, knee_angle_r=82),
        _frame(6.0, phase="drive", knee_angle_l=79, knee_angle_r=83),
        _frame(6.2, phase="flight", knee_angle_l=79, knee_angle_r=83),
        _frame(11.0, phase="drive", knee_angle_l=78, knee_angle_r=80),
        _frame(11.2, phase="flight", knee_angle_l=78, knee_angle_r=80),
    ]

    cues = []
    for frame in frames:
        cues.extend(evaluate_cues(session_id, "athlete_01", "sprint", frame))

    assert any(cue["text"] == "drive your knee higher" for cue in cues)


def test_good_sequence_stays_silent():
    session_id = "cue_session_good"
    clear_cue_state(session_id)

    frames = [
        _frame(1.0, phase="drive", knee_angle_l=95, knee_angle_r=96),
        _frame(1.2, phase="flight", knee_angle_l=95, knee_angle_r=96),
        _frame(4.0, phase="landing", knee_angle_l=130, knee_angle_r=132, trunk_lean=10),
        _frame(8.0, phase="drive", knee_angle_l=98, knee_angle_r=99),
        _frame(8.2, phase="flight", knee_angle_l=98, knee_angle_r=99),
        _frame(12.0, phase="landing", knee_angle_l=128, knee_angle_r=129, trunk_lean=9),
    ]

    cues = []
    for frame in frames:
        cues.extend(evaluate_cues(session_id, "athlete_01", "sprint", frame))

    assert cues == []


def test_warning_cue_uses_warning_urgency():
    session_id = "cue_session_warning"
    clear_cue_state(session_id)

    cues = evaluate_cues(
        session_id,
        "athlete_01",
        "sprint",
        _frame(1.0, phase="landing", knee_angle_l=60, knee_angle_r=62, trunk_lean=25),
    )

    assert cues
    assert cues[0]["text"] == "pause — check your landing"
    assert cues[0]["urgency"] == "warning"
