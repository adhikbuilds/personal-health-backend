from __future__ import annotations

"""Tests for the SQLite-backed clap reaction endpoints (POST /athlete/{id}/clap/{tid},
GET /claps/{tid}). Covers idempotency, count, batch consistency."""


def test_clap_increments_count(client):
    r = client.post("/athlete/athlete_a/clap/post_x")
    assert r.status_code == 200
    body = r.json()
    assert body["target_id"] == "post_x"
    assert body["count"] == 1
    assert body["you_clapped"] is True


def test_clap_is_idempotent(client):
    # First tap
    r1 = client.post("/athlete/athlete_b/clap/post_y")
    assert r1.json()["count"] == 1
    # Same caller taps again — count should NOT advance
    r2 = client.post("/athlete/athlete_b/clap/post_y")
    assert r2.status_code == 200
    body = r2.json()
    assert body["count"] == 1
    assert body["you_clapped"] is True


def test_clap_distinct_callers_each_count_once(client):
    for i in range(5):
        r = client.post(f"/athlete/athlete_caller_{i}/clap/post_z")
        assert r.status_code == 200
    g = client.get("/claps/post_z")
    assert g.status_code == 200
    body = g.json()
    assert body["count"] == 5
    # Caller-specific you_clapped check
    g2 = client.get("/claps/post_z?athlete_id=athlete_caller_2")
    assert g2.json()["you_clapped"] is True
    g3 = client.get("/claps/post_z?athlete_id=athlete_never_clapped")
    assert g3.json()["you_clapped"] is False


def test_get_claps_for_unknown_target(client):
    r = client.get("/claps/never_exists_post")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["you_clapped"] is False


def test_clap_with_empty_inputs_is_safe(client):
    # The route enforces non-empty path params via FastAPI, so this just
    # confirms the GET path handles unknowns + empty viewer correctly.
    r = client.get("/claps/some_target?athlete_id=")
    assert r.status_code == 200
    assert r.json()["you_clapped"] is False


def test_claps_persist_in_feed(client):
    # Tap a synthetic target, then verify counts roll into a fresh GET
    client.post("/athlete/athlete_persist/clap/feed_target_42")
    client.post("/athlete/athlete_persist_2/clap/feed_target_42")
    g = client.get("/claps/feed_target_42")
    assert g.json()["count"] == 2
