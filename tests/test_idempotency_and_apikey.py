"""Tests for idempotency replay, API-key auth, and daily-tracker history."""

import uuid


def _email() -> str:
    return f"idem_{uuid.uuid4().hex[:10]}@example.com"


def test_idempotent_register_replay(client):
    """Same Idempotency-Key → second POST returns the cached response."""
    email = _email()
    key = "test-idem-" + uuid.uuid4().hex
    body = {"email": email, "name": "Idem", "password": "Sup3rsecret!"}
    r1 = client.post("/auth/register", json=body, headers={"Idempotency-Key": key})
    assert r1.status_code == 201
    access1 = r1.json()["access_token"]

    # Replay same key — should NOT 409, should return the same access token
    r2 = client.post("/auth/register", json=body, headers={"Idempotency-Key": key})
    assert r2.status_code == 201
    assert r2.headers.get("X-Idempotent-Replay") == "true"
    assert r2.json()["access_token"] == access1


def test_api_key_round_trip():
    """Unit test the api key helpers end-to-end against sqlite."""
    from auth import create_api_key, verify_api_key

    raw = create_api_key("test-label")
    assert raw.startswith("phk_")
    row = verify_api_key(raw)
    assert row is not None
    assert row["label"] == "test-label"
    assert verify_api_key("phk_garbage") is None
    assert verify_api_key("") is None


def test_daily_tracker_history_endpoint(client, authed):
    """Daily-tracker is now ownership-gated; we use a freshly-registered
    athlete via the authed fixture so the bearer-token user matches the
    {athlete_id} path param."""
    aid = authed["athlete_id"]
    payload = {
        "steps": 8500,
        "active_minutes": 45,
        "distance_km": 6.2,
        "calories_burned": 420,
        "calorie_intake": 1900,
        "water_glasses": 6,
        "sleep_hours": 7.5,
        "date": "2026-04-01",
    }
    r = client.post(f"/athlete/{aid}/daily-tracker", json=payload, headers=authed["headers"])
    assert r.status_code == 200, r.text
    hist = client.get(
        f"/athlete/{aid}/daily-tracker/history?days=30",
        headers=authed["headers"],
    )
    assert hist.status_code == 200
    body = hist.json()
    assert "history" in body
    assert isinstance(body["history"], list)
    assert body["total"] >= 1


def test_daily_tracker_rejects_non_owner(client, authed):
    """Another athlete cannot read or write your daily tracker."""
    aid = authed["athlete_id"]
    # Create a second user
    import uuid as _uuid
    other_email = f"o_{_uuid.uuid4().hex[:8]}@example.com"
    r2 = client.post("/auth/register", json={
        "email": other_email, "name": "Other", "password": "Sup3rsecret!"})
    other_token = r2.json()["access_token"]
    other_headers = {"Authorization": f"Bearer {other_token}"}

    # Other athlete can't write to ours
    w = client.post(
        f"/athlete/{aid}/daily-tracker",
        json={"steps": 1000, "active_minutes": 5, "distance_km": 0,
              "calories_burned": 0, "calorie_intake": 0, "water_glasses": 0,
              "sleep_hours": 0, "date": "2026-04-02"},
        headers=other_headers,
    )
    assert w.status_code == 403

    # Other athlete can't read ours
    r = client.get(f"/athlete/{aid}/daily-tracker", headers=other_headers)
    assert r.status_code == 403
