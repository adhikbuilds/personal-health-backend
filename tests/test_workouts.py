from __future__ import annotations

"""Tests for structured workouts and session templates."""

import pytest


def test_list_all_workouts(client):
    r = client.get("/workouts")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    assert all("name" in w for w in body["workouts"])


def test_list_by_sport(client):
    r = client.get("/workouts?sport=vertical_jump")
    assert r.status_code == 200
    body = r.json()
    assert body["sport"] == "vertical_jump"
    assert body["count"] >= 3  # beginner, intermediate, advanced


def test_sport_workouts(client):
    r = client.get("/workouts/sprint")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_sport_workouts_404(client):
    r = client.get("/workouts/underwater_hockey")
    assert r.status_code == 404


def test_recommended_workout(client):
    r = client.get("/workouts/recommended/athlete_01")
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == "athlete_01"
    assert body["workout"] is not None
    assert "exercises" in body["workout"]
    assert body["recommended_difficulty"] in ("beginner", "intermediate", "advanced")


def test_recommended_workout_404(client):
    r = client.get("/workouts/recommended/nonexistent")
    assert r.status_code == 404


def test_create_custom_workout(client):
    r = client.post(
        "/workouts/custom",
        json={
            "name": "My Custom Leg Day",
            "sport": "squat",
            "duration_min": 30,
            "description": "custom squat workout",
            "exercises": [
                {"name": "Back squat", "sets": 5, "reps": 5, "rest_s": 180, "cue": "full depth"},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "created"
    assert body["workout"]["name"] == "My Custom Leg Day"
    assert body["workout"]["id"].startswith("custom_")


def test_create_custom_workout_empty_exercises(client):
    r = client.post(
        "/workouts/custom",
        json={
            "name": "Empty",
            "sport": "squat",
            "duration_min": 10,
            "exercises": [],
        },
    )
    assert r.status_code == 422  # validation error
