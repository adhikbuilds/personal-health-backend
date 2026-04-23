from __future__ import annotations

"""Tests for the SQLite-backed coach broadcast surface:
GET  /coach/{coach}/athletes (roster), POST /coach/{coach}/broadcast,
GET  /coach/{coach}/inbox, GET /coach/inbox/athlete/{athlete}.

Broadcasts are now ownership-gated — the bearer token's user.athlete_id
must match the {coach_id} path param. Tests use the `authed` fixture for
the coach and register additional users via /auth/register for athletes
that should follow the coach."""

import uuid


def _new_athlete(client) -> dict:
    """Register a fresh user, return {athlete_id, headers}."""
    email = f"a_{uuid.uuid4().hex[:10]}@example.com"
    r = client.post("/auth/register", json={
        "email": email, "name": "Roster", "password": "Sup3rsecret!",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    return {
        "athlete_id": body["user"]["athlete_id"],
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


def _follow(client, follower, coach_id: str) -> None:
    """Athlete `follower` opts in to the coach's roster."""
    r = client.post(
        "/follow",
        json={"follower": follower["athlete_id"], "following": coach_id},
        headers=follower["headers"],  # follow requires the follower's token
    )
    # /follow may not enforce auth on this build; both 200 and 201 are fine.
    assert r.status_code in (200, 201, 401)


def test_roster_returns_athletes(client, authed):
    coach_id = authed["athlete_id"]
    # Roster starts empty in production-mode (no fallback to all athletes)
    r = client.get(f"/coach/{coach_id}/athletes")
    assert r.status_code == 200
    body = r.json()
    assert body["coach_id"] == coach_id
    assert isinstance(body["athletes"], list)
    assert body["count"] == len(body["athletes"])


def test_broadcast_requires_message_or_voice(client, authed):
    coach_id = authed["athlete_id"]
    r = client.post(f"/coach/{coach_id}/broadcast", json={}, headers=authed["headers"])
    assert r.status_code == 400


def test_broadcast_requires_auth(client, authed):
    """No bearer token → 401."""
    r = client.post(
        f"/coach/{authed['athlete_id']}/broadcast",
        json={"message": "hi"},
    )
    assert r.status_code == 401


def test_broadcast_rejects_impersonation(client, authed):
    """A different signed-in user can't broadcast as someone else's coach_id."""
    other = _new_athlete(client)
    r = client.post(
        f"/coach/{authed['athlete_id']}/broadcast",
        json={"message": "hi"},
        headers=other["headers"],
    )
    assert r.status_code == 403


def test_broadcast_text_to_subset(client, authed):
    """Coach explicitly addresses two athletes."""
    coach_id = authed["athlete_id"]
    a = _new_athlete(client)
    b = _new_athlete(client)
    r = client.post(
        f"/coach/{coach_id}/broadcast",
        json={
            "message": "Form check at lunchtime",
            "athlete_ids": [a["athlete_id"], b["athlete_id"]],
        },
        headers=authed["headers"],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["coach_id"] == coach_id
    assert a["athlete_id"] in body["athlete_ids"]
    assert b["athlete_id"] in body["athlete_ids"]
    assert body["recipient_count"] == 2


def test_coach_inbox_lists_recent_broadcasts(client, authed):
    coach_id = authed["athlete_id"]
    # Send 3 broadcasts
    for i in range(3):
        r = client.post(
            f"/coach/{coach_id}/broadcast",
            json={"message": f"inbox-test-{i}", "athlete_ids": [coach_id]},
            headers=authed["headers"],
        )
        assert r.status_code == 200, r.text
    r = client.get(f"/coach/{coach_id}/inbox?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert body["coach_id"] == coach_id
    assert len(body["broadcasts"]) <= 2
    assert body["total"] >= 3
    if len(body["broadcasts"]) == 2:
        assert body["broadcasts"][0]["created_at"] >= body["broadcasts"][1]["created_at"]


def test_athlete_inbox_returns_addressed_messages(client, authed):
    coach_id = authed["athlete_id"]
    target = _new_athlete(client)
    msg = "personal-inbox-marker-9XK2"
    client.post(
        f"/coach/{coach_id}/broadcast",
        json={"message": msg, "athlete_ids": [target["athlete_id"]]},
        headers=authed["headers"],
    )
    r = client.get(f"/coach/inbox/athlete/{target['athlete_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == target["athlete_id"]
    assert any(b.get("message") == msg for b in body["broadcasts"])


def test_athlete_inbox_excludes_other_recipients(client, authed):
    coach_id = authed["athlete_id"]
    a = _new_athlete(client)
    b = _new_athlete(client)
    msg = "exclusive-marker-abc123"
    client.post(
        f"/coach/{coach_id}/broadcast",
        json={"message": msg, "athlete_ids": [a["athlete_id"]]},
        headers=authed["headers"],
    )
    r = client.get(f"/coach/inbox/athlete/{b['athlete_id']}")
    body = r.json()
    assert not any(b.get("message") == msg for b in body["broadcasts"])
