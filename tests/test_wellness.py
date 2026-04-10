from __future__ import annotations

"""Tests for the wellness check-in and scoring endpoints."""

import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

BEST_INPUTS = {
    "sleep_hours": 8.0,
    "water_glasses": 8,
    "mood": 10,
    "energy": 10,
    "stress": 1,
    "soreness": 1,
}

WORST_INPUTS = {
    "sleep_hours": 0.0,
    "water_glasses": 0,
    "mood": 1,
    "energy": 1,
    "stress": 10,
    "soreness": 10,
}


def _seed_athlete(client) -> str:
    """Register a throw-away athlete and return its ID."""
    r = client.post(
        "/athlete",
        json={"name": "Wellness Tester", "sport": "vertical_jump"},
    )
    assert r.status_code in (200, 201)
    return r.json()["id"]


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_best_inputs_sum_to_100(client):
    aid = _seed_athlete(client)
    r = client.post(f"/athlete/{aid}/wellness/checkin", json=BEST_INPUTS)
    assert r.status_code == 200
    body = r.json()
    assert body["wellness_score"] == 100
    assert sum(body["breakdown"].values()) == 100


def test_worst_inputs_score_is_low(client):
    aid = _seed_athlete(client)
    r = client.post(f"/athlete/{aid}/wellness/checkin", json=WORST_INPUTS)
    assert r.status_code == 200
    body = r.json()
    # Worst inputs should produce a very low score (0 is acceptable)
    assert body["wellness_score"] <= 10


def test_no_checkin_returns_null_score(client):
    aid = _seed_athlete(client)
    r = client.get(f"/athlete/{aid}/wellness/score")
    assert r.status_code == 200
    body = r.json()
    assert body["wellness_score"] is None
    assert "message" in body


def test_invalid_athlete_404(client):
    r = client.get("/athlete/does_not_exist_xyz/wellness/score")
    assert r.status_code == 404

    r2 = client.post("/athlete/does_not_exist_xyz/wellness/checkin", json=BEST_INPUTS)
    assert r2.status_code == 404


def test_invalid_date_rejected(client):
    aid = _seed_athlete(client)
    r = client.post(f"/athlete/{aid}/wellness/checkin", json={**BEST_INPUTS, "date": "banana"})
    assert r.status_code == 422


def test_get_score_after_checkin(client):
    aid = _seed_athlete(client)
    post_r = client.post(f"/athlete/{aid}/wellness/checkin", json=BEST_INPUTS)
    assert post_r.status_code == 200

    get_r = client.get(f"/athlete/{aid}/wellness/score")
    assert get_r.status_code == 200
    body = get_r.json()
    assert body["athlete_id"] == aid
    assert body["wellness_score"] == 100
    assert body["recovery_ready"] is True
    assert "breakdown" in body
    assert "recommendation" in body
