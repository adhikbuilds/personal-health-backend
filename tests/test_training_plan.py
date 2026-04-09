from __future__ import annotations

"""Tests for the dynamic training plan generator."""

import pytest

from routes.training_plan import (
    _classify_load_phase,
    _generate_daily_schedule,
    _pick_drills,
    _volume_recommendation,
)


def test_classify_recover_on_high_injury():
    assert _classify_load_phase(5.0, 6, 14, "high", 0.8) == "recover"


def test_classify_recover_on_low_hrv():
    assert _classify_load_phase(5.0, 6, 14, "low", 0.2) == "recover"


def test_classify_overreach():
    assert _classify_load_phase(-8.0, 10, 14, "low", 0.7) == "overreach"


def test_classify_build():
    assert _classify_load_phase(5.0, 5, 14, "low", 0.7) == "build"


def test_classify_maintain():
    assert _classify_load_phase(0.5, 3, 14, "low", 0.6) == "maintain"


def test_daily_schedule_has_7_days():
    for phase in ["recover", "overreach", "build", "maintain"]:
        schedule = _generate_daily_schedule(phase, "vertical_jump")
        assert len(schedule) == 7
        assert schedule[0]["day"] == "Monday"
        assert schedule[6]["day"] == "Sunday"
        assert all("type" in d and "prescription" in d for d in schedule)


def test_recover_schedule_has_mostly_rest():
    schedule = _generate_daily_schedule("recover", "sprint")
    types = [d["type"] for d in schedule]
    assert types.count("rest") >= 3


def test_build_schedule_has_train_days():
    schedule = _generate_daily_schedule("build", "snatch")
    types = [d["type"] for d in schedule]
    assert types.count("train") >= 3


def test_pick_drills_targets_weak_joints():
    weak = [
        {"joint": "knee_angle", "deviation_deg": 15.0},
        {"joint": "hip_angle", "deviation_deg": 8.0},
    ]
    drills = _pick_drills("vertical_jump", weak, count=3)
    assert len(drills) == 3
    assert any("knee" in d.get("reason", "").lower() for d in drills)


def test_pick_drills_fallback_for_unknown_sport():
    drills = _pick_drills("underwater_hockey", [], count=2)
    assert len(drills) >= 1


def test_volume_recommendation_phases():
    for phase in ["recover", "overreach", "build", "maintain"]:
        vol = _volume_recommendation(phase, 4.0)
        assert "sessions_target" in vol
        assert "rep_intensity" in vol
        assert "note" in vol
        assert vol["sessions_target"] >= 1


def test_recover_reduces_volume():
    vol = _volume_recommendation("recover", 5.0)
    assert vol["sessions_target"] <= 3


def test_build_increases_volume():
    vol = _volume_recommendation("build", 3.0)
    assert vol["sessions_target"] >= 4


# ─── Integration tests ─────────────────────────────────────────────────────


def test_training_plan_404_for_unknown(client):
    r = client.get("/training-plan/does_not_exist")
    assert r.status_code == 404


def test_training_plan_returns_full_structure(client):
    r = client.get("/training-plan/athlete_01?days=14")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == "athlete_01"
    assert body["phase"] in {"recover", "overreach", "build", "maintain"}
    assert len(body["schedule"]) == 7
    assert len(body["drills"]) <= 3
    assert "volume" in body
    assert "insights" in body
    assert "data_summary" in body
    assert "generated_at" in body
