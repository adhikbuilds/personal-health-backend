from __future__ import annotations

"""
Tests for WN-03 Nutrition endpoints.
Run with: pytest tests/test_nutrition.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

MOCK_ATHLETES = {
    "athlete_01": {
        "id": "athlete_01",
        "name": "Test Athlete",
        "sport": "sprint",
    }
}

MOCK_FOODS = {
    "f_001": {
        "name": "Oats",
        "per_100g": {
            "calories": 389, "protein_g": 16.9,
            "carbs_g": 66.3, "fat_g": 6.9, "fiber_g": 10.6,
        },
    },
    "f_003": {
        "name": "Chicken Breast",
        "per_100g": {
            "calories": 165, "protein_g": 31.0,
            "carbs_g": 0.0, "fat_g": 3.6, "fiber_g": 0.0,
        },
    },
    "f_004": {
        "name": "Brown Rice",
        "per_100g": {
            "calories": 216, "protein_g": 4.5,
            "carbs_g": 45.0, "fat_g": 1.8, "fiber_g": 3.5,
        },
    },
    "f_005": {
        "name": "Whole Egg",
        "per_100g": {
            "calories": 155, "protein_g": 13.0,
            "carbs_g": 1.1, "fat_g": 11.0, "fiber_g": 0.0,
        },
    },
    "f_006": {
        "name": "Paneer",
        "per_100g": {
            "calories": 265, "protein_g": 18.3,
            "carbs_g": 1.2, "fat_g": 20.8, "fiber_g": 0.0,
        },
    },
}

patch("database.ATHLETE_DB", MOCK_ATHLETES).start()
patch("database.FOOD_DB", MOCK_FOODS).start()
patch("routes.nutrition.ATHLETE_DB", MOCK_ATHLETES).start()
patch("routes.nutrition.FOOD_DB", MOCK_FOODS).start()

from api_server import app  # noqa: E402

client = TestClient(app)

ATHLETE_ID = "athlete_01"
VALID_DATE  = "2026-04-07"


def test_log_meal_success():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": VALID_DATE, "slot": "breakfast", "time": "08:30",
        "items": [{"food_id": "f_001", "qty_g": 200}],
    })
    assert resp.status_code == 201, resp.json()
    data = resp.json()
    assert "meal_id" in data
    assert data["slot"] == "breakfast"
    assert data["date"] == VALID_DATE
    assert data["total"]["calories"] == pytest.approx(778.0, rel=1e-2)


def test_log_meal_two_items_total_is_sum():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": VALID_DATE, "slot": "lunch",
        "items": [
            {"food_id": "f_003", "qty_g": 150},
            {"food_id": "f_004", "qty_g": 200},
        ],
    })
    assert resp.status_code == 201, resp.json()
    assert resp.json()["total"]["calories"] == pytest.approx(679.5, rel=1e-2)


def test_log_meal_overwrites_same_slot():
    client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": "2026-04-10", "slot": "dinner",
        "items": [{"food_id": "f_005", "qty_g": 100}],
    })
    client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": "2026-04-10", "slot": "dinner",
        "items": [{"food_id": "f_006", "qty_g": 100}],
    })
    resp = client.get(f"/athlete/{ATHLETE_ID}/meals", params={"date": "2026-04-10"})
    assert resp.status_code == 200, resp.json()
    dinner_items = resp.json()["meals"]["dinner"]["items"]
    assert len(dinner_items) == 1
    assert dinner_items[0]["food_id"] == "f_006"


def test_invalid_slot_returns_422():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": VALID_DATE, "slot": "brunch",
        "items": [{"food_id": "f_001", "qty_g": 100}],
    })
    assert resp.status_code == 422


def test_unknown_food_id_returns_404():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": VALID_DATE, "slot": "breakfast",
        "items": [{"food_id": "f_999", "qty_g": 100}],
    })
    assert resp.status_code == 404
    assert "Food not found" in resp.json()["detail"]


def test_unknown_athlete_returns_404():
    resp = client.post("/athlete/fake_999/meals", json={
        "date": VALID_DATE, "slot": "breakfast",
        "items": [{"food_id": "f_001", "qty_g": 100}],
    })
    assert resp.status_code == 404
    assert "Athlete not found" in resp.json()["detail"]


def test_zero_qty_returns_422():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": VALID_DATE, "slot": "breakfast",
        "items": [{"food_id": "f_001", "qty_g": 0}],
    })
    assert resp.status_code == 422


def test_empty_items_returns_422():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": VALID_DATE, "slot": "breakfast",
        "items": [],
    })
    assert resp.status_code == 422


def test_invalid_date_returns_422():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": "banana", "slot": "breakfast",
        "items": [{"food_id": "f_001", "qty_g": 100}],
    })
    assert resp.status_code == 422


def test_invalid_time_format_returns_422():
    resp = client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": VALID_DATE, "slot": "breakfast", "time": "99:99",
        "items": [{"food_id": "f_001", "qty_g": 100}],
    })
    assert resp.status_code == 422


def test_get_meals_returns_grouped_by_slot():
    client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": "2026-04-09", "slot": "breakfast",
        "items": [{"food_id": "f_001", "qty_g": 100}],
    })
    client.post(f"/athlete/{ATHLETE_ID}/meals", json={
        "date": "2026-04-09", "slot": "lunch",
        "items": [{"food_id": "f_003", "qty_g": 100}],
    })
    resp = client.get(f"/athlete/{ATHLETE_ID}/meals", params={"date": "2026-04-09"})
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert "breakfast" in data["meals"]
    assert "lunch" in data["meals"]
    assert "day_total" in data


def test_get_meals_unknown_athlete_returns_404():
    resp = client.get("/athlete/fake_999/meals", params={"date": VALID_DATE})
    assert resp.status_code == 404
