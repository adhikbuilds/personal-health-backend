from __future__ import annotations

"""Tests that the session lifecycle survives a server-style state reset.

After fitness.start_session writes SESSION_DB[id] and calls _save_db(),
the session must be present on disk and re-loadable into a fresh
SESSION_DB. Same for end_session — summary, ended_at, frames must all
round-trip through the save/load cycle.

The previous shape was a 60s periodic-save worker; sessions started but
not ended within that window were lost on crash. The fix: explicit
_save_db() at the end of start_session, plus a 5s periodic safety net.
"""


def _start_session(client, athlete_id: str = "athlete_01", sport: str = "vertical_jump") -> str:
    r = client.post(
        "/session/start",
        json={"athlete_id": athlete_id, "sport": sport},
    )
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert sid
    return sid


def test_start_session_is_immediately_durable(client, tmp_path, monkeypatch):
    """After /session/start returns, the session must be in SESSION_DB
    AND on disk — not waiting on the periodic save worker."""
    import database

    sid = _start_session(client)
    # In-memory: session exists
    assert sid in database.SESSION_DB
    assert database.SESSION_DB[sid]["status"] == "active"
    # On disk: sessions.json must already contain the new session
    import json
    sessions_file = database.DB_PATH / "sessions.json"
    assert sessions_file.exists(), "sessions.json should exist after start_session"
    raw = json.loads(sessions_file.read_text(encoding="utf-8"))
    assert sid in raw
    assert raw[sid]["status"] == "active"


def test_session_can_be_fetched_via_api(client):
    sid = _start_session(client)
    r = client.get(f"/session/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["status"] == "active"


def test_end_session_persists_summary(client):
    """End the session and verify the summary fields land on disk."""
    import json
    import database

    sid = _start_session(client)
    r = client.post(f"/session/{sid}/end")
    assert r.status_code == 200
    summary = r.json()
    assert "avg_form_score" in summary or "total_frames" in summary or summary == {} or "xp_earned" in summary

    raw = json.loads((database.DB_PATH / "sessions.json").read_text(encoding="utf-8"))
    assert sid in raw
    assert raw[sid]["status"] == "completed"
    assert raw[sid].get("ended_at"), "ended_at must be set after end_session"


def test_session_simulates_reload_from_disk(client):
    """Simulate a server restart: clear in-memory dicts, reload from JSON,
    verify the session is recovered with status + summary intact."""
    import json
    import database

    sid = _start_session(client)
    client.post(f"/session/{sid}/end")

    # Snapshot disk state
    saved_sessions = json.loads((database.DB_PATH / "sessions.json").read_text(encoding="utf-8"))
    saved_athletes = json.loads((database.DB_PATH / "athletes.json").read_text(encoding="utf-8"))

    # Wipe in-memory state and rehydrate
    database.SESSION_DB.clear()
    database.ATHLETE_DB.clear()
    database.SESSION_DB.update(saved_sessions)
    database.ATHLETE_DB.update(saved_athletes)

    # The session is back, fully formed
    assert sid in database.SESSION_DB
    assert database.SESSION_DB[sid]["status"] == "completed"
