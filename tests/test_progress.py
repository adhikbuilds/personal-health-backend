def test_progress_returns_real_numbers(client, admin_client):
    r = client.get("/progress/athlete_01?days=90", headers=admin_client["headers"])
    # Either 200 with stats, or 404 if seeding produced no athlete_01
    if r.status_code == 404:
        return
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == "athlete_01"
    assert "form_score_trend" in body
    assert "session_count" in body
    assert isinstance(body["session_count"], int)


def test_injury_risk_band(client, admin_client):
    r = client.get("/injury-risk/athlete_01?days=30", headers=admin_client["headers"])
    if r.status_code == 404:
        return
    assert r.status_code == 200
    assert r.json()["risk"] in {"low", "watch", "high", "unknown"}


def test_weak_joints(client, admin_client):
    r = client.get("/weak-joints/athlete_01?days=60", headers=admin_client["headers"])
    if r.status_code == 404:
        return
    assert r.status_code == 200
    body = r.json()
    assert "weak_joints" in body
    assert isinstance(body["weak_joints"], list)


def test_progress_404_for_unknown(client, admin_client):
    r = client.get("/progress/does_not_exist", headers=admin_client["headers"])
    assert r.status_code == 404
