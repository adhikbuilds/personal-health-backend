from __future__ import annotations

"""Tests for the SQLite-backed coach broadcast surface:
GET /coach/{coach}/athletes (roster), POST /coach/{coach}/broadcast,
GET /coach/{coach}/inbox, GET /coach/inbox/athlete/{athlete}."""


def _get_athlete(client) -> str:
    r = client.get("/athletes")
    if r.status_code == 200:
        data = r.json()
        athletes = data if isinstance(data, list) else data.get("athletes", [])
        if athletes:
            return athletes[0].get("id") or athletes[0].get("athlete_id")
    r = client.post("/athlete", json={"name": "TestCoach", "sport": "vertical_jump"})
    return r.json().get("id") or r.json().get("athlete_id")


def test_roster_returns_athletes(client):
    coach_id = _get_athlete(client)
    r = client.get(f"/coach/{coach_id}/athletes")
    assert r.status_code == 200
    body = r.json()
    assert body["coach_id"] == coach_id
    assert isinstance(body["athletes"], list)
    assert body["count"] == len(body["athletes"])
    # Every entry has the expected shape
    for a in body["athletes"]:
        assert "id" in a and "name" in a


def test_broadcast_requires_message_or_voice(client):
    coach_id = _get_athlete(client)
    r = client.post(f"/coach/{coach_id}/broadcast", json={})
    assert r.status_code == 400


def test_broadcast_text_to_whole_roster(client):
    coach_id = _get_athlete(client)
    roster = client.get(f"/coach/{coach_id}/athletes").json()["athletes"]
    expected_count = len(roster)
    r = client.post(
        f"/coach/{coach_id}/broadcast",
        json={"message": "Sprint drills tomorrow at 6am, lights on."},
    )
    assert r.status_code == 200
    bcast = r.json()
    assert bcast["coach_id"] == coach_id
    assert bcast["message"] == "Sprint drills tomorrow at 6am, lights on."
    assert bcast["recipient_count"] == expected_count
    assert isinstance(bcast["athlete_ids"], list)
    assert "id" in bcast and len(bcast["id"]) >= 8


def test_broadcast_to_subset(client):
    coach_id = _get_athlete(client)
    roster = client.get(f"/coach/{coach_id}/athletes").json()["athletes"]
    if len(roster) < 2:
        # Ensure at least 2 athletes for the subset test
        client.post("/athlete", json={"name": "Filler1", "sport": "sprint"})
        client.post("/athlete", json={"name": "Filler2", "sport": "sprint"})
        roster = client.get(f"/coach/{coach_id}/athletes").json()["athletes"]
    target = roster[0]["id"]
    r = client.post(
        f"/coach/{coach_id}/broadcast",
        json={"message": "Form check at lunchtime", "athlete_ids": [target]},
    )
    assert r.status_code == 200
    body = r.json()
    assert target in body["athlete_ids"]
    assert body["recipient_count"] == 1


def test_coach_inbox_lists_recent_broadcasts(client):
    coach_id = _get_athlete(client)
    # Send 3 broadcasts
    for i in range(3):
        client.post(
            f"/coach/{coach_id}/broadcast",
            json={"message": f"inbox-test-{i}"},
        )
    r = client.get(f"/coach/{coach_id}/inbox?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert body["coach_id"] == coach_id
    assert len(body["broadcasts"]) <= 2
    assert body["total"] >= 3
    # Most-recent first ordering
    if len(body["broadcasts"]) == 2:
        assert body["broadcasts"][0]["created_at"] >= body["broadcasts"][1]["created_at"]


def test_athlete_inbox_returns_messages_addressed_to_them(client):
    coach_id = _get_athlete(client)
    roster = client.get(f"/coach/{coach_id}/athletes").json()["athletes"]
    if not roster:
        return
    target = roster[0]["id"]
    # Targeted broadcast to this one athlete
    msg = "personal-inbox-marker-9XK2"
    client.post(
        f"/coach/{coach_id}/broadcast",
        json={"message": msg, "athlete_ids": [target]},
    )
    r = client.get(f"/coach/inbox/athlete/{target}")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == target
    assert any(b.get("message") == msg for b in body["broadcasts"])


def test_athlete_inbox_excludes_other_recipients(client):
    coach_id = _get_athlete(client)
    roster = client.get(f"/coach/{coach_id}/athletes").json()["athletes"]
    if len(roster) < 2:
        return
    target = roster[0]["id"]
    other  = roster[1]["id"]
    msg = "exclusive-marker-abc123"
    client.post(
        f"/coach/{coach_id}/broadcast",
        json={"message": msg, "athlete_ids": [target]},
    )
    r = client.get(f"/coach/inbox/athlete/{other}")
    body = r.json()
    assert not any(b.get("message") == msg for b in body["broadcasts"])
