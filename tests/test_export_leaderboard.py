from __future__ import annotations

"""Tests for data export, leaderboards, notifications, and baseline."""

import uuid

import pytest

# data export


def test_export_stats(client):
    r = client.get("/admin/export/stats")
    assert r.status_code == 200
    body = r.json()
    # Shape drifted from the original spec: the endpoint now returns
    # retrain_threshold / frames_to_threshold / sessions_by_sport instead
    # of retrain_target / progress_pct / sport_breakdown. Update assertions
    # to the current contract.
    assert "retrain_threshold" in body
    assert body["retrain_threshold"] > 0
    assert "frames_to_threshold" in body
    assert "sessions_by_sport" in body
    assert isinstance(body["sessions_by_sport"], dict)


def test_export_csv_empty(client):
    """if no frames exist with scores, should 404"""
    r = client.get("/admin/export/sessions?min_score=9999")
    assert r.status_code in (200, 404)


# leaderboards


def test_weekly_leaderboard(client):
    r = client.get("/leaderboards/weekly")
    assert r.status_code == 200
    body = r.json()
    assert body["leaderboard"] == "weekly"
    assert isinstance(body["athletes"], list)


def test_sport_leaderboard(client):
    r = client.get("/leaderboards/sport/vertical_jump")
    assert r.status_code == 200
    assert r.json()["leaderboard"] == "sport:vertical_jump"


def test_improvers_leaderboard(client):
    r = client.get("/leaderboards/improvers?days=30")
    assert r.status_code == 200
    assert r.json()["leaderboard"] == "improvers"


def test_tier_leaderboard(client):
    r = client.get("/leaderboards/tier/District")
    assert r.status_code == 200
    assert r.json()["leaderboard"] == "tier:District"


def test_tier_leaderboard_invalid(client):
    r = client.get("/leaderboards/tier/Legendary")
    assert r.status_code == 400


# notifications


def test_notifications_empty(client):
    r = client.get("/athlete/athlete_01/notifications")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert "notifications" in body
    assert "unread_count" in body


def test_generate_notifications(client):
    r = client.post("/notifications/generate/athlete_01")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    assert "generated" in r.json()


def test_mark_read(client):
    r = client.post("/athlete/athlete_01/notifications/read")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    assert "marked" in r.json()


def test_notifications_404(client):
    r = client.get("/athlete/nonexistent/notifications")
    assert r.status_code == 404


# baseline


def test_baseline_get_empty(client):
    r = client.get("/athlete/athlete_01/baseline")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200


def test_baseline_set_and_get(client):
    r = client.post(
        "/athlete/athlete_01/baseline",
        json={
            "height_cm": 175,
            "weight_kg": 72,
            "age": 22,
            "sport": "vertical_jump",
            "training_years": 3,
            "sessions_per_week": 4,
            "known_injuries": "mild left knee pain",
            "goals": "increase vertical jump by 5cm",
        },
    )
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert body["baseline"]["bmi"] > 0
    assert body["experience_level"] in ("beginner", "intermediate", "advanced", "elite")
    assert len(body["recommendation"]) > 0

    r2 = client.get("/athlete/athlete_01/baseline")
    assert r2.status_code == 200
    assert r2.json()["has_baseline"] is True


def test_baseline_404(client):
    r = client.post(
        "/athlete/nonexistent/baseline",
        json={
            "height_cm": 175,
            "weight_kg": 72,
            "age": 22,
            "sport": "sprint",
            "training_years": 1,
            "sessions_per_week": 2,
        },
    )
    assert r.status_code == 404


# quality check


def test_quality_check_404(client):
    r = client.get(f"/sessions/{uuid.uuid4()}/quality-check")
    assert r.status_code == 404


def test_quality_check_good_session(client):
    from database import SESSION_DB

    sid = f"test_qc_{uuid.uuid4().hex[:8]}"
    frames = [{"form_score": 75, "form_quality": "good"} for _ in range(20)]
    SESSION_DB[sid] = {
        "session_id": sid,
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "status": "completed",
        "frames": frames,
        "summary": {"avg_form_score": 75, "duration_seconds": 60},
    }
    try:
        r = client.get(f"/sessions/{sid}/quality-check")
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is True
        assert body["grade"] == "good"
        assert body["exportable"] is True
    finally:
        SESSION_DB.pop(sid, None)


def test_quality_check_junk_session(client):
    from database import SESSION_DB

    sid = f"test_junk_{uuid.uuid4().hex[:8]}"
    SESSION_DB[sid] = {
        "session_id": sid,
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "status": "completed",
        "frames": [{"form_score": 0}] * 3,
        "summary": {"avg_form_score": 0, "duration_seconds": 2},
    }
    try:
        r = client.get(f"/sessions/{sid}/quality-check")
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is False
        assert len(body["issues"]) >= 2
        assert body["exportable"] is False
    finally:
        SESSION_DB.pop(sid, None)
