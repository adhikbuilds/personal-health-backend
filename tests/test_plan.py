from __future__ import annotations

"""Tests for the Dynamic Training Plan feature (routes/plan.py + services/training_plan.py)."""

from datetime import date, timedelta

import pytest

# ─── Service-level tests (pure, no HTTP) ─────────────────────────────────────


def test_generate_plan_shape_balanced_low_risk():
    from services.training_plan import generate_plan

    ctx = {
        "athlete_id": "athlete_test",
        "athlete_name": "Test",
        "sport": "vertical_jump",
        "tier": "State",
        "session_count": 4,
        "avg_form_score": 78.0,
        "form_trend_pct": 2.5,
        "form_bucket": "steady",
        "bpi_delta": 120,
        "injury_risk": "low",
        "injury_reason": "",
        "weak_joints": [{"joint": "hip_angle", "deviation_deg": 6.0}],
        "volume_state": "balanced",
    }
    plan = generate_plan(ctx, week_start=date(2026, 4, 6))  # Monday
    assert plan.week_start == "2026-04-06"
    assert plan.week_end == "2026-04-12"
    assert len(plan.days) == 7
    # Every day has a label + rationale + valid intensity.
    for day in plan.days:
        assert day.label
        assert day.rationale
        assert day.intensity in {"low", "high", "rest"}
    # At least one rest day in a balanced week.
    assert any(d.type == "rest" for d in plan.days)
    # Adherence starts at 0.
    assert plan.adherence_pct == 0.0
    # Source defaults to deterministic when no LLM is wired.
    assert plan.source == "deterministic"


def test_generate_plan_high_risk_skews_recovery():
    from services.training_plan import generate_plan

    ctx = {
        "athlete_id": "athlete_test",
        "athlete_name": "Test",
        "sport": "sprint",
        "tier": "Block",
        "session_count": 3,
        "avg_form_score": 62.0,
        "form_trend_pct": -5.0,
        "form_bucket": "rough",
        "bpi_delta": -10,
        "injury_risk": "high",
        "injury_reason": "12% symmetry deviation",
        "weak_joints": [{"joint": "knee_angle", "deviation_deg": 14.0}],
        "volume_state": "balanced",
    }
    plan = generate_plan(ctx, week_start=date(2026, 4, 6))
    day_types = [d.type for d in plan.days]
    # No high-intensity power/speed days when risk is elevated.
    assert "power" not in day_types
    assert "speed" not in day_types
    # At least two recovery-or-rest days.
    assert sum(1 for t in day_types if t in {"recovery", "rest"}) >= 2
    assert "Asymmetry" in plan.summary or "recovery" in plan.summary.lower()


def test_pick_drills_biases_weak_joint():
    from services.training_plan import _pick_drills

    drills = _pick_drills("vertical_jump", "strength", weak_joint="hip_angle", count=3)
    assert len(drills) == 3
    # First drill should target the weak joint.
    assert drills[0]["joint"] == "hip_angle"


def test_pick_drills_falls_back_for_missing_day_type():
    from services.training_plan import _pick_drills

    # snatch has no speed drills — should fall back to technique.
    drills = _pick_drills("snatch", "speed", weak_joint=None, count=2)
    assert len(drills) == 2


def test_compute_adherence_excludes_rest_days():
    from services.training_plan import compute_adherence, generate_plan

    ctx = {
        "athlete_id": "a",
        "athlete_name": "A",
        "sport": "vertical_jump",
        "session_count": 4,
        "avg_form_score": 75,
        "form_trend_pct": 0,
        "form_bucket": "steady",
        "bpi_delta": 0,
        "injury_risk": "low",
        "injury_reason": "",
        "weak_joints": [],
        "volume_state": "balanced",
    }
    plan = generate_plan(ctx, week_start=date(2026, 4, 6))
    workable = [d for d in plan.days if d.type != "rest"]
    # Mark half of workable days complete.
    for d in workable[: len(workable) // 2]:
        d.completed = True
    adherence = compute_adherence(plan)
    assert 0 < adherence <= 100
    # Math sanity.
    assert adherence == round(len(workable) // 2 / len(workable) * 100, 1)


def test_mark_day_complete_returns_none_for_missing_date():
    from services.training_plan import generate_plan, mark_day_complete

    ctx = {
        "athlete_id": "a",
        "athlete_name": "A",
        "sport": "vertical_jump",
        "session_count": 3,
        "avg_form_score": 70,
        "form_trend_pct": 0,
        "form_bucket": "steady",
        "bpi_delta": 0,
        "injury_risk": "low",
        "injury_reason": "",
        "weak_joints": [],
        "volume_state": "balanced",
    }
    plan = generate_plan(ctx, week_start=date(2026, 4, 6))
    assert mark_day_complete(plan, "1999-01-01") is None
    assert mark_day_complete(plan, "2026-04-06") is not None


# ─── HTTP endpoint tests ─────────────────────────────────────────────────────


def _seed_athlete(client, headers=None) -> str:
    """Return an existing seeded athlete id (seed_athletes creates athlete_01..30)."""
    r = client.get("/athletes", headers=headers)
    if r.status_code == 200:
        data = r.json()
        athletes = data if isinstance(data, list) else data.get("athletes", [])
        if athletes:
            first = athletes[0]
            return first.get("id") or first.get("athlete_id")
    # Fallback — register a fresh one.
    r = client.post("/athlete", json={"name": "Plan Tester", "sport": "vertical_jump"}, headers=headers)
    assert r.status_code in (200, 201)
    body = r.json()
    return body.get("id") or body.get("athlete_id")


def test_weekly_plan_endpoint_happy_path(client, admin_client):
    aid = _seed_athlete(client, admin_client["headers"])
    r = client.get(f"/plan/{aid}/weekly", headers=admin_client["headers"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["athlete_id"] == aid
    assert len(body["days"]) == 7
    assert body["source"] in {"deterministic", "anthropic", "anthropic-fallback"}
    assert "week_start" in body and "week_end" in body
    # Cached second read should be identical.
    r2 = client.get(f"/plan/{aid}/weekly", headers=admin_client["headers"])
    assert r2.status_code == 200
    assert r2.json()["generated_at"] == body["generated_at"]


def test_weekly_plan_404_for_unknown_athlete(client, admin_client):
    r = client.get("/plan/definitely_not_a_real_athlete/weekly", headers=admin_client["headers"])
    assert r.status_code == 404


def test_weekly_plan_rejects_bad_week_start(client, admin_client):
    aid = _seed_athlete(client, admin_client["headers"])
    r = client.get(f"/plan/{aid}/weekly", params={"week_start": "not-a-date"}, headers=admin_client["headers"])
    assert r.status_code == 400


def test_regenerate_produces_new_plan(client, admin_client):
    aid = _seed_athlete(client, admin_client["headers"])
    r1 = client.get(f"/plan/{aid}/weekly", headers=admin_client["headers"])
    ts1 = r1.json()["generated_at"]

    r2 = client.post(f"/plan/{aid}/regenerate", headers=admin_client["headers"])
    assert r2.status_code == 200
    ts2 = r2.json()["generated_at"]
    assert len(r2.json()["days"]) == 7
    assert ts2 >= ts1


def test_day_complete_updates_adherence(client, admin_client):
    aid = _seed_athlete(client, admin_client["headers"])
    r = client.get(f"/plan/{aid}/weekly", headers=admin_client["headers"])
    assert r.status_code == 200
    plan = r.json()

    # Pick the first non-rest day.
    workable = [d for d in plan["days"] if d["type"] != "rest"]
    assert workable, "plan should have at least one workable day"
    target = workable[0]

    r2 = client.post(f"/plan/{aid}/day/{target['date']}/complete", headers=admin_client["headers"])
    assert r2.status_code == 200
    body = r2.json()
    assert body["completed"] is True
    assert body["adherence_pct"] > 0

    # Read back — the day should be marked complete.
    r3 = client.get(f"/plan/{aid}/weekly", headers=admin_client["headers"])
    assert r3.status_code == 200
    for day in r3.json()["days"]:
        if day["date"] == target["date"]:
            assert day["completed"] is True
            break
    else:
        pytest.fail(f"target date {target['date']} missing from plan after completion")


def test_day_complete_404_without_plan(client, admin_client):
    aid = _seed_athlete(client, admin_client["headers"])
    # Use a date in a week we've never generated a plan for.
    future = (date.today() + timedelta(days=60)).isoformat()
    r = client.post(f"/plan/{aid}/day/{future}/complete", headers=admin_client["headers"])
    assert r.status_code == 404


def test_plan_history_returns_generated_week(client, admin_client):
    aid = _seed_athlete(client, admin_client["headers"])
    client.get(f"/plan/{aid}/weekly", headers=admin_client["headers"])
    r = client.get(f"/plan/{aid}/history", headers=admin_client["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert isinstance(body["weeks"], list)
    assert body["weeks"][0]["athlete_id"] == aid
