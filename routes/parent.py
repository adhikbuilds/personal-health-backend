from __future__ import annotations

"""
Parent safety-summary and weekly-digest endpoints.

These serve the token-gated parent views (parent.ejs, parent-digest.ejs)
where a parent sees only safety-relevant signals — no raw scores or
session IDs.

  GET /parent/{consent_id}/safety-summary?token=
  GET /parent/{token}/weekly-digest
  POST /parent/consent              (create a consent link)
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from routes.progress import _athlete_sessions, _compute_injury_risk, _compute_progress
from routes.weekly_summary import _session_volume
from sqlite_store import cursor

router = APIRouter(prefix="/parent", tags=["Parent"])
log = get_logger("routes.parent")


def _ensure_table() -> None:
    with cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_consents (
              consent_id TEXT PRIMARY KEY,
              token TEXT NOT NULL UNIQUE,
              athlete_id TEXT NOT NULL,
              parent_label TEXT,
              created_at REAL NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pc_token ON parent_consents(token)")


_ensure_table()


def _lookup_consent(consent_id: str, token: str) -> dict | None:
    with cursor() as cur:
        row = cur.execute(
            "SELECT * FROM parent_consents WHERE consent_id = ? AND token = ?",
            (consent_id, token),
        ).fetchone()
        return dict(row) if row else None


def _lookup_by_token(token: str) -> dict | None:
    with cursor() as cur:
        row = cur.execute(
            "SELECT * FROM parent_consents WHERE token = ?",
            (token,),
        ).fetchone()
        return dict(row) if row else None


def _volume_label(session_count: int) -> str:
    if session_count <= 2:
        return "light"
    if session_count <= 5:
        return "moderate"
    return "heavy"


def _safety_sentence(athlete_name: str, risk_level: str, session_count: int, sport: str) -> str:
    sport_pretty = (sport or "training").replace("_", " ")
    if risk_level == "low" and session_count > 0:
        return (
            f"{athlete_name} trained {session_count} time{'s' if session_count != 1 else ''} "
            f"this week in {sport_pretty} with no injury flags. Movement quality looks solid."
        )
    if risk_level == "low":
        return f"{athlete_name} has no recent sessions and no injury concerns."
    if risk_level == "medium":
        return (
            f"{athlete_name} is training actively in {sport_pretty}. Some biomechanical patterns "
            f"are worth monitoring — nothing urgent, but a check-in would be welcome."
        )
    return (
        f"{athlete_name} shows elevated injury risk signals in {sport_pretty}. "
        f"Consider reaching out or speaking with their coach about load management."
    )


@router.get("/{consent_id}/safety-summary")
async def safety_summary(consent_id: str, token: str = Query(...)):
    consent = _lookup_consent(consent_id, token)
    if not consent:
        raise HTTPException(403, "consent not found or token mismatch")

    athlete_id = consent["athlete_id"]
    athlete = ATHLETE_DB.get(athlete_id)
    if not athlete:
        raise HTTPException(404, "athlete not found")

    sessions = _athlete_sessions(athlete_id, 7)
    risk = _compute_injury_risk(athlete_id, 7)
    risk_level = risk.get("risk", "low")
    name = athlete.get("name", "Athlete")
    sport = athlete.get("sport", "general")

    last_date = None
    if sessions:
        last_session = max(sessions, key=lambda s: s.get("started_at", ""))
        last_date = last_session.get("started_at", "")[:10]

    return {
        "injury_risk_level": risk_level,
        "athlete_name": name,
        "sport": sport,
        "injury_risk_reason": risk.get("reason", "No data available"),
        "sessions_last_7_days": len(sessions),
        "last_session_date": last_date or "No sessions yet",
        "training_volume": _volume_label(len(sessions)),
        "summary_sentence": _safety_sentence(name, risk_level, len(sessions), sport),
    }


@router.get("/{token}/weekly-digest")
async def weekly_digest(token: str):
    consent = _lookup_by_token(token)
    if not consent:
        raise HTTPException(403, "digest link not found or expired")

    athlete_id = consent["athlete_id"]
    athlete = ATHLETE_DB.get(athlete_id)
    if not athlete:
        raise HTTPException(404, "athlete not found")

    sessions = _athlete_sessions(athlete_id, 7)
    risk = _compute_injury_risk(athlete_id, 7)
    progress = _compute_progress(athlete_id, 7)
    volume = _session_volume(sessions)
    name = athlete.get("name", "Athlete")
    sport = athlete.get("sport", "general")
    risk_level = risk.get("risk", "low")
    today = datetime.now(timezone.utc).date()

    return {
        "injury_risk": risk_level,
        "sessions_this_week": len(sessions),
        "avg_form_score": progress.get("avg_form_score", 0),
        "total_training_minutes": round(volume.get("total_duration_min", 0)),
        "total_reps": volume.get("total_reps", 0),
        "athlete_name": name,
        "sport": sport,
        "week_ending": today.isoformat(),
        "safety_note": _safety_sentence(name, risk_level, len(sessions), sport),
    }


from pydantic import BaseModel
from uuid import uuid4
import time


class ConsentCreate(BaseModel):
    athlete_id: str
    parent_label: str | None = None


@router.post("/consent")
async def create_consent(body: ConsentCreate):
    if body.athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    consent_id = uuid4().hex[:12]
    token = uuid4().hex[:16]
    with cursor() as cur:
        cur.execute(
            "INSERT INTO parent_consents(consent_id, token, athlete_id, parent_label, created_at) VALUES(?,?,?,?,?)",
            (consent_id, token, body.athlete_id, body.parent_label, time.time()),
        )
    return {
        "consent_id": consent_id,
        "token": token,
        "athlete_id": body.athlete_id,
        "safety_url": f"/parent/{consent_id}?token={token}",
        "digest_url": f"/digest/{token}",
    }
