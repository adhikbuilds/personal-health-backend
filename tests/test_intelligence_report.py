from __future__ import annotations

"""Tests for the athlete intelligence report."""

import pytest

from routes.intelligence_report import (
    _action_items,
    _biomechanical_profile,
    _performance_trajectory,
    _periodization,
    _readiness_timeline,
    _tier_gap,
    _training_load_analysis,
)

# trajectory


def test_trajectory_empty():
    t = _performance_trajectory([], 30)
    assert t["trend_direction"] == "insufficient_data"
    assert t["projected_form_2w"] is None


def test_trajectory_improving():
    sessions = []
    for i in range(8):
        sessions.append({"summary": {"avg_form_score": 60 + i * 3}})
    t = _performance_trajectory(sessions, 30)
    assert t["trend_direction"] == "improving"
    assert t["projected_form_2w"] is not None
    assert t["projected_form_2w"] > 75
    assert len(t["segments"]) == 4


def test_trajectory_declining():
    sessions = []
    for i in range(8):
        sessions.append({"summary": {"avg_form_score": 85 - i * 4}})
    t = _performance_trajectory(sessions, 30)
    assert t["trend_direction"] == "declining"


def test_trajectory_stable():
    sessions = [{"summary": {"avg_form_score": 70}} for _ in range(8)]
    t = _performance_trajectory(sessions, 30)
    assert t["trend_direction"] == "stable"


# biomechanical profile


def test_profile_empty():
    p = _biomechanical_profile([], "vertical_jump")
    assert p["joints"] == []
    assert p["balance"] == []


def test_profile_with_frames():
    sessions = [
        {
            "frames": [
                {"knee_angle_l": 100, "knee_angle_r": 105, "hip_angle_l": 90, "hip_angle_r": 92, "trunk_lean": 10},
                {"knee_angle_l": 102, "knee_angle_r": 107, "hip_angle_l": 88, "hip_angle_r": 90, "trunk_lean": 12},
            ]
        }
    ]
    p = _biomechanical_profile(sessions, "vertical_jump")
    assert len(p["joints"]) > 0
    assert len(p["balance"]) > 0
    # check knee balance
    knee_bal = next((b for b in p["balance"] if b["joint"] == "knee_angle"), None)
    assert knee_bal is not None
    assert knee_bal["asymmetry_deg"] > 0


# training load


def test_load_empty():
    load = _training_load_analysis([], 30)
    assert load["acwr"] is None
    assert load["training_days"] == 0


def test_load_with_sessions():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    sessions = []
    for i in range(10):
        sessions.append(
            {
                "started_at": (now - timedelta(days=i)).isoformat(),
                "summary": {"avg_form_score": 75, "total_frames": 50},
            }
        )
    load = _training_load_analysis(sessions, 14)
    assert load["acwr"] is not None
    assert load["training_days"] > 0
    assert load["density"] in ("high", "moderate", "low", "very_low")


# periodization


def test_periodization_recovery():
    p = _periodization(
        {"trend_direction": "declining"},
        {"acwr": 1.4, "avg_sessions_per_week": 5},
        {"risk": "high"},
    )
    assert p["current_phase"] == "recovery"
    assert len(p["four_week_plan"]) == 4


def test_periodization_build():
    p = _periodization(
        {"trend_direction": "improving"},
        {"acwr": 1.0, "avg_sessions_per_week": 4},
        {"risk": "low"},
    )
    assert p["current_phase"] == "transmutation"


def test_periodization_accumulation():
    p = _periodization(
        {"trend_direction": "stable"},
        {"acwr": 0.9, "avg_sessions_per_week": 1},
        {"risk": "low"},
    )
    assert p["current_phase"] == "accumulation"


# peer benchmarks


def test_tier_gap():
    gap = _tier_gap("District", 8000)
    assert gap is not None
    assert gap["next_tier"] == "State"
    assert gap["bpi_needed"] == 9501 - 8000


def test_tier_gap_elite():
    gap = _tier_gap("Elite", 20000)
    assert gap is None


# readiness timeline


def test_timeline_improving():
    t = _readiness_timeline(
        {"trend_direction": "improving"},
        {"risk": "low"},
        {"avg_sessions_per_week": 4},
    )
    assert t["estimated_weeks"] == 2
    assert t["confidence"] == "high"


def test_timeline_injured():
    t = _readiness_timeline(
        {"trend_direction": "stable"},
        {"risk": "high"},
        {"avg_sessions_per_week": 3},
    )
    assert t["estimated_weeks"] == 6


# action items


def test_actions_high_risk():
    actions = _action_items(
        {"risk": "high", "reason": "symmetry bad"},
        [{"joint": "knee_angle", "deviation_deg": 15, "mean_deg": 100, "ideal_min": 90, "ideal_max": 130}],
        {"acwr": 1.0, "acwr_status": "optimal", "avg_sessions_per_week": 3},
        {"trend_direction": "stable"},
        [{}] * 5,
    )
    assert actions[0]["category"] == "injury"
    assert actions[0]["priority"] == 1


def test_actions_no_sessions():
    actions = _action_items(
        {"risk": "unknown"},
        [],
        {"acwr": None, "acwr_status": "no_data", "avg_sessions_per_week": 0},
        {"trend_direction": "insufficient_data"},
        [],
    )
    assert any(a["category"] == "consistency" for a in actions)


# integration


def test_report_404(client):
    r = client.get("/athlete/nonexistent/intelligence-report")
    assert r.status_code == 404


def test_report_full_structure(client):
    r = client.get("/athlete/athlete_01/intelligence-report?days=30")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()

    assert body["athlete_id"] == "athlete_01"
    assert "overall_grade" in body
    assert body["overall_grade"]["letter"] in ("A+", "A", "B+", "B", "C+", "C", "D")
    assert "trajectory" in body
    assert "biomechanical_profile" in body
    assert "training_load" in body
    assert "periodization" in body
    assert body["periodization"]["current_phase"] in ("accumulation", "transmutation", "realization", "recovery")
    assert len(body["periodization"]["four_week_plan"]) == 4
    assert "peer_benchmarks" in body
    assert "competition_timeline" in body
    assert body["competition_timeline"]["estimated_weeks"] >= 1
    assert "action_items" in body
    assert len(body["action_items"]) >= 1
    assert "injury_risk" in body
    assert "generated_at" in body
    assert "next_report_at" in body
