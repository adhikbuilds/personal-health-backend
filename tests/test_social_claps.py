from __future__ import annotations

"""Tests for the SQLite-backed clap reaction endpoints (POST /athlete/{id}/clap/{tid},
GET /claps/{tid}). Covers idempotency, count, batch consistency, and the
ownership check on the caller's athlete_id."""

import uuid


def _new_user(client) -> dict:
    email = f"u_{uuid.uuid4().hex[:10]}@example.com"
    r = client.post(
        "/auth/register",
        json={"email": email, "name": "Clap User", "password": "Sup3rsecret!"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return {
        "athlete_id": body["user"]["athlete_id"],
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


def test_clap_increments_count(client):
    me = _new_user(client)
    r = client.post(f"/athlete/{me['athlete_id']}/clap/post_x", headers=me["headers"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_id"] == "post_x"
    assert body["count"] == 1
    assert body["you_clapped"] is True


def test_clap_is_idempotent(client):
    me = _new_user(client)
    r1 = client.post(f"/athlete/{me['athlete_id']}/clap/post_y", headers=me["headers"])
    assert r1.json()["count"] == 1
    # Same caller taps again — count should NOT advance
    r2 = client.post(f"/athlete/{me['athlete_id']}/clap/post_y", headers=me["headers"])
    assert r2.status_code == 200
    body = r2.json()
    assert body["count"] == 1
    assert body["you_clapped"] is True


def test_clap_distinct_callers_each_count_once(client):
    callers = [_new_user(client) for _ in range(5)]
    for c in callers:
        r = client.post(f"/athlete/{c['athlete_id']}/clap/post_z", headers=c["headers"])
        assert r.status_code == 200, r.text
    g = client.get("/claps/post_z", headers=callers[0]["headers"])
    assert g.status_code == 200
    body = g.json()
    assert body["count"] == 5
    # Caller-specific you_clapped check
    g2 = client.get(f"/claps/post_z?athlete_id={callers[2]['athlete_id']}", headers=callers[2]["headers"])
    assert g2.json()["you_clapped"] is True
    g3 = client.get("/claps/post_z?athlete_id=athlete_never_clapped_xyz", headers=callers[0]["headers"])
    assert g3.json()["you_clapped"] is False


def test_clap_rejects_caller_who_doesnt_own_athlete_id(client):
    """Cannot clap on behalf of another athlete — server enforces ownership."""
    me = _new_user(client)
    other = _new_user(client)
    r = client.post(
        f"/athlete/{other['athlete_id']}/clap/some_target",
        headers=me["headers"],  # me's token, other's athlete_id
    )
    assert r.status_code == 403


def test_clap_requires_auth(client):
    r = client.post("/athlete/athlete_a/clap/post_x")
    assert r.status_code == 401


def test_get_claps_for_unknown_target(client):
    me = _new_user(client)
    r = client.get("/claps/never_exists_post", headers=me["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["you_clapped"] is False


def test_clap_with_empty_inputs_is_safe(client):
    me = _new_user(client)
    r = client.get("/claps/some_target?athlete_id=", headers=me["headers"])
    assert r.status_code == 200
    assert r.json()["you_clapped"] is False


def test_claps_persist_in_feed(client):
    a = _new_user(client)
    b = _new_user(client)
    client.post(f"/athlete/{a['athlete_id']}/clap/feed_target_42", headers=a["headers"])
    client.post(f"/athlete/{b['athlete_id']}/clap/feed_target_42", headers=b["headers"])
    g = client.get("/claps/feed_target_42", headers=a["headers"])
    assert g.json()["count"] == 2
