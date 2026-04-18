from __future__ import annotations

"""Tests for coach roster, streaks, and session replay."""

import uuid

import pytest

from routes.streaks import _compute_streaks

# streaks unit tests


def test_streaks_empty():
    s = _compute_streaks([])
    assert s["current_streak"] == 0
    assert s["longest_streak"] == 0


def test_streaks_consecutive():
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(5)]
    s = _compute_streaks(sorted(dates))
    assert s["current_streak"] == 5
    assert s["longest_streak"] == 5
    assert s["total_training_days"] == 5


def test_streaks_gap_breaks_current():
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc).date()
    # trained 3 days ago, 4 days ago, 5 days ago (gap of 2 days)
    dates = [(today - timedelta(days=i)).isoformat() for i in range(3, 6)]
    s = _compute_streaks(sorted(dates))
    assert s["current_streak"] == 0  # gap too big
    assert s["longest_streak"] == 3


# integration tests


def test_coach_roster_empty(client):
    r = client.get("/coach/nonexistent_coach/athletes")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_coach_add_unknown_athlete(client):
    r = client.post("/coach/test_coach/athletes?athlete_id=fake_athlete")
    assert r.status_code == 404


def test_coach_add_and_list(client):
    r = client.post("/coach/test_coach_x/athletes?athlete_id=athlete_01")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("added", "already_added")

    r2 = client.get("/coach/test_coach_x/athletes")
    assert r2.status_code == 200
    assert r2.json()["count"] >= 1


def test_coach_dashboard(client):
    r = client.get("/coach/test_coach_x/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body


def test_streaks_endpoint(client):
    r = client.get("/athlete/athlete_01/streaks")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert "current_streak" in body
    assert "weekly_volume" in body
    assert len(body["weekly_volume"]) == 4


def test_achievements_endpoint(client):
    r = client.get("/athlete/athlete_01/achievements")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert "unlocked" in body
    assert "locked" in body
    assert body["total_achievements"] > 0


def test_replay_404(client):
    r = client.get(f"/sessions/{uuid.uuid4()}/replay")
    assert r.status_code == 404


def test_replay_with_session(client):
    from database import SESSION_DB

    sid = f"test_replay_{uuid.uuid4().hex[:8]}"
    SESSION_DB[sid] = {
        "session_id": sid,
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "status": "completed",
        "started_at": "2026-04-09T10:00:00Z",
        "frames": [
            {"frame_num": 0, "form_score": 70, "phase": "setup", "knee_angle_l": 170, "knee_angle_r": 168},
            {"frame_num": 1, "form_score": 82, "phase": "descent", "knee_angle_l": 130, "knee_angle_r": 128},
            {"frame_num": 2, "form_score": 65, "phase": "bottom", "knee_angle_l": 95, "knee_angle_r": 92},
        ],
        "summary": {"avg_form_score": 72.3},
    }
    try:
        r = client.get(f"/sessions/{sid}/replay")
        assert r.status_code == 200
        body = r.json()
        assert body["total_frames"] == 3
        assert len(body["frames"]) == 3
        assert len(body["phases"]) == 3  # 3 different phases
    finally:
        SESSION_DB.pop(sid, None)


def test_highlights_with_session(client):
    from database import SESSION_DB

    sid = f"test_hl_{uuid.uuid4().hex[:8]}"
    SESSION_DB[sid] = {
        "session_id": sid,
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "status": "completed",
        "frames": [
            {
                "frame_num": 0,
                "form_score": 85,
                "form_quality": "good",
                "primary_feedback": "nice",
                "estimated_jump_height": 40,
                "limb_symmetry_idx": 0.98,
            },
            {
                "frame_num": 1,
                "form_score": 45,
                "form_quality": "poor",
                "primary_feedback": "knee cave",
                "estimated_jump_height": 35,
                "limb_symmetry_idx": 0.72,
            },
        ],
    }
    try:
        r = client.get(f"/sessions/{sid}/highlights")
        assert r.status_code == 200
        body = r.json()
        types = [h["type"] for h in body["highlights"]]
        assert "best_form" in types
        assert "worst_form" in types
    finally:
        SESSION_DB.pop(sid, None)


def test_streaks_404_unknown(client):
    r = client.get("/athlete/nonexistent/streaks")
    assert r.status_code == 404


def test_achievements_404_unknown(client):
    r = client.get("/athlete/nonexistent/achievements")
    assert r.status_code == 404
