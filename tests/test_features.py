from __future__ import annotations

"""Tests for the mega-feature batch: scorecard, weekly summary, progressive load, huddle, data export."""


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _get_athlete(client) -> str:
    r = client.get("/athletes")
    if r.status_code == 200:
        data = r.json()
        athletes = data if isinstance(data, list) else data.get("athletes", [])
        if athletes:
            return athletes[0].get("id") or athletes[0].get("athlete_id")
    r = client.post("/athlete", json={"name": "Tester", "sport": "vertical_jump"})
    assert r.status_code in (200, 201)
    return r.json().get("id") or r.json().get("athlete_id")


# ─── Weekly Summary ──────────────────────────────────────────────────────────


def test_weekly_summary_happy(client):
    aid = _get_athlete(client)
    r = client.get(f"/athlete/{aid}/weekly-summary?days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == aid
    assert "volume" in body
    assert "coaching_note" in body
    assert "daily_scores" in body
    assert isinstance(body["weak_joints"], list)


def test_weekly_summary_404(client):
    r = client.get("/athlete/nonexistent_xyz/weekly-summary")
    assert r.status_code == 404


# ─── Progressive Load ────────────────────────────────────────────────────────


def test_load_recommendation_happy(client):
    aid = _get_athlete(client)
    r = client.get(f"/athlete/{aid}/load-recommendation")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == aid
    assert "acwr" in body
    assert "zone" in body
    assert "next_week_targets" in body
    assert body["next_week_targets"]["sessions"] > 0


def test_load_recommendation_404(client):
    r = client.get("/athlete/nonexistent_xyz/load-recommendation")
    assert r.status_code == 404


def test_progressive_load_unit():
    from services.progressive_load import compute_progressive_load

    result = compute_progressive_load("test", [], sport="sprint")
    assert result["zone"] == "inactive"
    assert result["acwr"] == 0.0
    assert result["next_week_targets"]["sessions"] == 3


# ─── Scorecard ───────────────────────────────────────────────────────────────


def test_scorecard_json_requires_completed(client):
    aid = _get_athlete(client)
    # Start a session but don't end it
    r = client.post("/session/start", json={"athlete_id": aid, "sport": "vertical_jump"})
    if r.status_code in (200, 201):
        sid = r.json().get("session_id")
        r2 = client.get(f"/session/{sid}/scorecard")
        assert r2.status_code == 400  # not completed yet
        # Clean up — end the session
        client.post(f"/session/{sid}/end")


def test_scorecard_404(client):
    r = client.get("/session/fake_session_xyz/scorecard")
    assert r.status_code == 404


def test_scorecard_png_404(client):
    r = client.get("/session/fake_session_xyz/scorecard.png")
    assert r.status_code == 404


# ─── Scorecard image generation unit test ────────────────────────────────────


def test_scorecard_image_generation():
    from services.scorecard import generate_scorecard

    summary = {
        "session_id": "test123",
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "avg_form_score": 82.5,
        "peak_form_score": 95.0,
        "peak_jump_height_cm": 48.2,
        "avg_symmetry": 0.95,
        "xp_earned": 280,
        "duration_seconds": 300,
        "rep_count": 12,
        "quality_distribution": {"elite": 5, "good": 20, "average": 15, "poor": 2},
        "total_frames": 150,
    }
    athlete = {"id": "athlete_01", "name": "Test", "sport": "vertical_jump", "tier": "State", "bpi": 15000}
    png_bytes = generate_scorecard(summary, athlete)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000  # should be a real PNG
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


# ─── Huddle ──────────────────────────────────────────────────────────────────


def test_huddle_create_join_start_end(client):
    aid = _get_athlete(client)

    # Create
    r = client.post("/huddle/create", json={"name": "Test Huddle", "sport": "vertical_jump"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    hid = body.get("huddle_id") or (body.get("huddle", {}) or {}).get("huddle_id")
    assert hid

    # Join
    r2 = client.post(f"/huddle/{hid}/join", json={"athlete_id": aid})
    assert r2.status_code == 200, r2.text

    # Start
    r3 = client.post(f"/huddle/{hid}/start")
    assert r3.status_code == 200, r3.text
    start_body = r3.json()
    huddle_data = start_body.get("huddle", start_body)
    assert huddle_data.get("status") == "active"

    # Live view
    r4 = client.get(f"/huddle/{hid}/live")
    assert r4.status_code == 200

    # End
    r5 = client.post(f"/huddle/{hid}/end")
    assert r5.status_code == 200
    end_body = r5.json()
    end_data = end_body.get("huddle", end_body)
    assert end_data.get("status") == "ended"

    # Get
    r6 = client.get(f"/huddle/{hid}")
    assert r6.status_code == 200
    get_data = r6.json().get("huddle", r6.json())
    assert get_data.get("status") == "ended" or r6.json().get("status") == "ended"


def test_huddle_404(client):
    r = client.get("/huddle/nonexistent")
    assert r.status_code == 404


def test_huddle_join_bad_athlete(client):
    r = client.post("/huddle/create", json={"name": "T", "sport": "sprint"})
    body = r.json()
    hid = body.get("huddle_id") or (body.get("huddle", {}) or {}).get("huddle_id")
    r2 = client.post(f"/huddle/{hid}/join", json={"athlete_id": "fake_athlete_xyz"})
    assert r2.status_code == 400 or r2.status_code == 404


def test_huddles_list(client):
    r = client.get("/huddles")
    assert r.status_code == 200
    assert isinstance(r.json(), list) or "huddles" in r.json()


# ─── Data Export ─────────────────────────────────────────────────────────────


def test_export_stats(client):
    r = client.get("/admin/export/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_sessions" in body
    assert "total_frames" in body
    assert "ready_for_retrain" in body


def test_export_sessions_json(client, admin_client):
    r = client.get("/admin/export/sessions?format=json", headers=admin_client["headers"])
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    assert "frames" in body


def test_export_sessions_csv(client, admin_client):
    r = client.get("/admin/export/sessions?format=csv", headers=admin_client["headers"])
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")


def test_export_sessions_requires_admin(client, authed):
    """Non-admin athlete cannot dump full session data."""
    r = client.get("/admin/export/sessions?format=json", headers=authed["headers"])
    assert r.status_code == 403


def test_export_athlete(client, authed):
    """Owner can export their own athlete data."""
    aid = authed["athlete_id"]
    r = client.get(f"/athlete/{aid}/export?format=json", headers=authed["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == aid


def test_export_athlete_rejects_non_owner(client, authed):
    """An athlete cannot export another athlete's data."""
    other_aid = _get_athlete(client)  # uses old un-owned creation path → likely athlete_01
    if other_aid == authed["athlete_id"]:
        return  # nothing to test if seeded data overlaps
    r = client.get(f"/athlete/{other_aid}/export?format=json", headers=authed["headers"])
    assert r.status_code == 403


def test_export_athlete_404(client, authed):
    """Even with a valid token, exporting a non-existent athlete is 403
    (we treat missing == not-yours rather than leak existence)."""
    r = client.get("/athlete/nonexistent_xyz/export", headers=authed["headers"])
    assert r.status_code in (403, 404)
