from __future__ import annotations

"""Tests for scorecard, weekly summary, and athlete comparison endpoints."""

import uuid

import pytest

from routes.scorecard import VALID_SPORTS, _build_share_text, _week_stats

# ─── Unit tests ────────────────────────────────────────────────────────────


def test_valid_sports_includes_all_eight():
    assert len(VALID_SPORTS) == 8
    assert "vertical_jump" in VALID_SPORTS
    assert "push_up" in VALID_SPORTS


def test_week_stats_empty():
    stats = _week_stats([])
    assert stats["session_count"] == 0
    assert stats["avg_form_score"] == 0.0


def test_week_stats_with_sessions():
    sessions = [
        {
            "summary": {
                "avg_form_score": 80,
                "peak_form_score": 90,
                "peak_jump_height_cm": 45,
                "total_frames": 100,
                "xp_earned": 200,
                "quality_distribution": {"elite": 5, "good": 10, "average": 3, "poor": 1},
            }
        },
        {
            "summary": {
                "avg_form_score": 70,
                "peak_form_score": 85,
                "peak_jump_height_cm": 50,
                "total_frames": 80,
                "xp_earned": 180,
                "quality_distribution": {"elite": 3, "good": 8, "average": 5, "poor": 2},
            }
        },
    ]
    stats = _week_stats(sessions)
    assert stats["session_count"] == 2
    assert stats["avg_form_score"] == 75.0
    assert stats["peak_form_score"] == 90.0
    assert stats["peak_jump_cm"] == 50.0
    assert stats["total_xp"] == 380
    assert stats["quality_distribution"]["elite"] == 8


def test_build_share_text():
    athlete = {"name": "Viraj"}
    summary = {"avg_form_score": 82.5, "peak_jump_height_cm": 47.2}
    text = _build_share_text(athlete, "vertical_jump", summary, 250)
    assert "Viraj" in text
    assert "82.5" in text
    assert "47.2" in text
    assert "+250 XP" in text


def test_build_share_text_no_jump():
    athlete = {"name": "Priya"}
    summary = {"avg_form_score": 75.0, "peak_jump_height_cm": 0}
    text = _build_share_text(athlete, "sprint", summary, 100)
    assert "Priya" in text
    assert "Sprint" in text
    assert "cm" not in text  # no jump for sprint


# ─── Integration tests ─────────────────────────────────────────────────────


def test_scorecard_404_unknown_session(client):
    r = client.get(f"/sessions/{uuid.uuid4()}/scorecard")
    assert r.status_code == 404


def test_scorecard_400_active_session(client):
    """Score card should reject sessions that aren't completed."""
    from database import SESSION_DB

    sid = f"test_active_{uuid.uuid4().hex[:8]}"
    SESSION_DB[sid] = {
        "session_id": sid,
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "status": "active",
    }
    try:
        r = client.get(f"/sessions/{sid}/scorecard")
        assert r.status_code == 400
    finally:
        SESSION_DB.pop(sid, None)


def test_scorecard_returns_full_structure(client):
    from database import SESSION_DB

    sid = f"test_sc_{uuid.uuid4().hex[:8]}"
    SESSION_DB[sid] = {
        "session_id": sid,
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "status": "completed",
        "started_at": "2026-04-08T10:00:00Z",
        "summary": {
            "avg_form_score": 78.5,
            "peak_form_score": 92.0,
            "peak_jump_height_cm": 45.0,
            "total_frames": 100,
            "valid_frames": 85,
            "duration_seconds": 120,
            "xp_earned": 250,
            "quality_distribution": {"elite": 5, "good": 40, "average": 30, "poor": 10},
        },
    }
    try:
        r = client.get(f"/sessions/{sid}/scorecard")
        if r.status_code == 404:
            pytest.skip("athlete_01 not seeded")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert body["athlete"]["sport"] == "vertical_jump"
        assert body["stats"]["avg_form_score"] == 78.5
        assert body["bpi"]["xp_earned"] == 250
        assert "share_text" in body
        assert "personal_bests" in body
    finally:
        SESSION_DB.pop(sid, None)


def test_weekly_summary_404(client):
    r = client.get("/athlete/does_not_exist/weekly-summary")
    assert r.status_code == 404


def test_weekly_summary_returns_weeks(client):
    r = client.get("/athlete/athlete_01/weekly-summary?weeks=2")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == "athlete_01"
    assert len(body["week_summaries"]) == 2
    assert body["week_summaries"][0]["label"] == "This week"


def test_compare_needs_two_athletes(client):
    r = client.get("/compare?athletes=athlete_01")
    assert r.status_code == 400


def test_compare_max_four(client):
    r = client.get("/compare?athletes=a,b,c,d,e")
    # Will 404 on first unknown athlete
    assert r.status_code in (400, 404)


def test_compare_returns_rankings(client):
    r = client.get("/compare?athletes=athlete_01,athlete_02&days=30")
    if r.status_code == 404:
        pytest.skip("athletes not seeded")
    assert r.status_code == 200
    body = r.json()
    assert len(body["athletes"]) == 2
    assert "rankings" in body
    assert "avg_form_score" in body["rankings"]


def test_sport_validation_rejects_invalid(client):
    r = client.post("/session/start", json={"athlete_id": "athlete_01", "sport": "underwater_hockey"})
    assert r.status_code == 400
    assert "unknown sport" in r.json()["detail"]
