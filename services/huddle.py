from __future__ import annotations

"""
Huddle Mode — Group training sessions where 10-20 athletes train
simultaneously with a shared live leaderboard.
"""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from database import ATHLETE_DB, SESSION_DB, _load_json, _save_json
from logging_setup import get_logger

logger = get_logger("services.huddle")

HUDDLES_FILE = "huddles.json"


@dataclass
class Huddle:
    huddle_id: str
    name: str
    sport: str
    coach_id: str | None
    created_at: str
    status: str  # 'waiting' | 'active' | 'ended'
    max_athletes: int
    athletes: list[str] = field(default_factory=list)
    sessions: dict[str, str] = field(default_factory=dict)
    leaderboard: list[dict] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None


# ─── Persistence helpers ───────────────────────────────────────────────────


def _load_huddles() -> dict[str, dict]:
    return _load_json(HUDDLES_FILE)


def _save_huddles(huddles: dict[str, dict]) -> None:
    _save_json(HUDDLES_FILE, huddles)


def _get_huddle(huddle_id: str) -> Huddle:
    huddles = _load_huddles()
    raw = huddles.get(huddle_id)
    if raw is None:
        raise KeyError(f"Huddle {huddle_id} not found")
    return Huddle(**raw)


def _persist(huddle: Huddle) -> None:
    huddles = _load_huddles()
    huddles[huddle.huddle_id] = asdict(huddle)
    _save_huddles(huddles)


# ─── Public API ────────────────────────────────────────────────────────────


def create_huddle(
    name: str,
    sport: str,
    coach_id: str | None = None,
    max_athletes: int = 20,
) -> Huddle:
    """Create a new huddle in 'waiting' state."""
    huddle_id = f"huddle_{uuid.uuid4().hex[:10]}"
    huddle = Huddle(
        huddle_id=huddle_id,
        name=name,
        sport=sport,
        coach_id=coach_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="waiting",
        max_athletes=max_athletes,
    )
    _persist(huddle)
    logger.info("Huddle created", extra={"huddle_id": huddle_id, "sport": sport})
    return huddle


def join_huddle(huddle_id: str, athlete_id: str) -> dict:
    """Add an athlete to a huddle. Returns confirmation dict."""
    huddle = _get_huddle(huddle_id)

    if huddle.status == "ended":
        raise ValueError("Cannot join an ended huddle")

    if len(huddle.athletes) >= huddle.max_athletes:
        raise ValueError(f"Huddle is full ({huddle.max_athletes} max)")

    if athlete_id in huddle.athletes:
        raise ValueError("Athlete already in this huddle")

    athlete = ATHLETE_DB.get(athlete_id)
    if athlete is None:
        raise KeyError(f"Athlete {athlete_id} not found")

    huddle.athletes.append(athlete_id)
    _persist(huddle)
    logger.info(
        "Athlete joined huddle",
        extra={"huddle_id": huddle_id, "athlete_id": athlete_id},
    )
    return {
        "huddle_id": huddle_id,
        "athlete_id": athlete_id,
        "position": len(huddle.athletes),
        "status": huddle.status,
    }


def leave_huddle(huddle_id: str, athlete_id: str) -> dict:
    """Remove an athlete from a huddle. Safe to call on ended huddles (no-op)."""
    huddle = _get_huddle(huddle_id)
    if athlete_id not in huddle.athletes:
        raise ValueError("Athlete not in huddle")
    huddle.athletes.remove(athlete_id)
    huddle.sessions.pop(athlete_id, None)
    _persist(huddle)
    logger.info("Athlete left huddle", extra={"huddle_id": huddle_id, "athlete_id": athlete_id})
    return {
        "huddle_id": huddle_id,
        "athlete_id": athlete_id,
        "remaining": len(huddle.athletes),
        "status": huddle.status,
    }


def bind_session_to_huddle(huddle_id: str, athlete_id: str, session_id: str) -> None:
    """Link an athlete's session to a huddle so the live leaderboard tracks it."""
    try:
        huddle = _get_huddle(huddle_id)
    except KeyError:
        return
    if athlete_id not in huddle.athletes:
        return
    huddle.sessions[athlete_id] = session_id
    _persist(huddle)


def start_huddle(huddle_id: str) -> Huddle:
    """Transition huddle from 'waiting' to 'active'."""
    huddle = _get_huddle(huddle_id)

    if huddle.status != "waiting":
        raise ValueError(f"Cannot start huddle in '{huddle.status}' state")

    if not huddle.athletes:
        raise ValueError("Cannot start huddle with no athletes")

    huddle.status = "active"
    huddle.started_at = datetime.now(timezone.utc).isoformat()
    _persist(huddle)
    logger.info("Huddle started", extra={"huddle_id": huddle_id, "count": len(huddle.athletes)})
    return huddle


def end_huddle(huddle_id: str) -> Huddle:
    """End a huddle and compute the final leaderboard."""
    huddle = _get_huddle(huddle_id)

    if huddle.status == "ended":
        raise ValueError("Huddle already ended")

    huddle.status = "ended"
    huddle.ended_at = datetime.now(timezone.utc).isoformat()
    huddle.leaderboard = compute_huddle_leaderboard(huddle_id, huddle=huddle)
    _persist(huddle)
    logger.info("Huddle ended", extra={"huddle_id": huddle_id})
    return huddle


def get_huddle_live(huddle_id: str) -> dict:
    """Return live per-athlete stats and ranking."""
    huddle = _get_huddle(huddle_id)
    entries: list[dict] = []

    for aid in huddle.athletes:
        athlete = ATHLETE_DB.get(aid, {})
        # Find sessions belonging to this athlete during the huddle
        athlete_sessions = [
            s
            for s in SESSION_DB.values()
            if s.get("athlete_id") == aid and s.get("session_id") in huddle.sessions.values()
        ]
        # Also include any session mapped in huddle.sessions for this athlete
        sid = huddle.sessions.get(aid)
        if sid and sid in SESSION_DB:
            mapped = SESSION_DB[sid]
            if mapped not in athlete_sessions:
                athlete_sessions.append(mapped)

        latest_score = 0.0
        total_frames = 0
        scores: list[float] = []
        for sess in athlete_sessions:
            summary = sess.get("summary", {}) or {}
            avg = summary.get("avg_form_score", 0)
            if avg:
                scores.append(avg)
            total_frames += summary.get("total_frames", 0)
            if not total_frames:
                total_frames += len(sess.get("frames", []))

        latest_score = scores[-1] if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        entries.append(
            {
                "athlete_id": aid,
                "name": athlete.get("name", aid),
                "latest_form_score": round(latest_score, 1),
                "avg_form_score": round(avg_score, 1),
                "total_frames": total_frames,
            }
        )

    # Sort by avg_form_score descending
    entries.sort(key=lambda e: e["avg_form_score"], reverse=True)
    for i, entry in enumerate(entries, 1):
        entry["rank"] = i

    return {
        "huddle_id": huddle_id,
        "status": huddle.status,
        "athlete_count": len(huddle.athletes),
        "leaderboard": entries,
    }


def compute_huddle_leaderboard(
    huddle_id: str,
    *,
    huddle: Huddle | None = None,
) -> list[dict]:
    """Compute leaderboard sorted by avg_form_score desc."""
    if huddle is None:
        huddle = _get_huddle(huddle_id)

    entries: list[dict] = []

    for aid in huddle.athletes:
        athlete = ATHLETE_DB.get(aid, {})
        # Gather all completed sessions for this athlete
        athlete_sessions = [
            s for s in SESSION_DB.values() if s.get("athlete_id") == aid and s.get("status") == "completed"
        ]
        # If huddle tracks specific sessions, narrow to those
        sid = huddle.sessions.get(aid)
        if sid and sid in SESSION_DB:
            athlete_sessions = [SESSION_DB[sid]]

        scores: list[float] = []
        xp_total = 0
        for sess in athlete_sessions:
            summary = sess.get("summary", {}) or {}
            avg = summary.get("avg_form_score", 0)
            if avg:
                scores.append(avg)
            xp_total += summary.get("xp_earned", 0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        entries.append(
            {
                "athlete_id": aid,
                "name": athlete.get("name", aid),
                "avg_form_score": round(avg_score, 1),
                "xp": xp_total,
            }
        )

    entries.sort(key=lambda e: e["avg_form_score"], reverse=True)
    for i, entry in enumerate(entries, 1):
        entry["rank"] = i

    return entries
