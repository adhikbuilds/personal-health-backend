from __future__ import annotations

"""
Personal Health — Score Card endpoint.

VISION.md GTM Step 2: shareable session score card.

  GET /session/{session_id}/scorecard      → JSON summary for app rendering
  GET /session/{session_id}/scorecard.png  → rendered 1080x1080 PNG image
"""

from fastapi import APIRouter, Depends, HTTPException

from auth import current_user
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from services.scorecard import generate_scorecard_response

router = APIRouter(tags=["Scorecard"])
log = get_logger("routes.scorecard")


def _session_and_athlete(session_id: str) -> tuple[dict, dict]:
    session = SESSION_DB.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    if session.get("status") != "completed":
        raise HTTPException(400, "session not completed — end it first")
    athlete_id = session.get("athlete_id", "")
    athlete = ATHLETE_DB.get(athlete_id, {"id": athlete_id, "name": athlete_id})
    return session, athlete


@router.get("/session/{session_id}/scorecard", dependencies=[Depends(current_user)])
async def get_scorecard_data(session_id: str):
    """JSON score card data — for the Android app to render natively."""
    session, athlete = _session_and_athlete(session_id)
    summary = session.get("summary", session)
    return {
        "session_id": session_id,
        "athlete_id": athlete.get("id"),
        "athlete_name": athlete.get("name"),
        "sport": session.get("sport"),
        "tier": athlete.get("tier"),
        "bpi": athlete.get("bpi", 0),
        "avg_form_score": float(summary.get("avg_form_score", 0) or 0),
        "peak_form_score": float(summary.get("peak_form_score", 0) or 0),
        "peak_jump_height_cm": float(summary.get("peak_jump_height_cm", 0) or 0),
        "avg_symmetry": float(summary.get("avg_symmetry", 0) or 0),
        "xp_earned": int(summary.get("xp_earned", 0) or 0),
        "rep_count": int(summary.get("rep_count", 0) or 0),
        "duration_seconds": float(summary.get("duration_seconds", 0) or 0),
        "quality_distribution": summary.get("quality_distribution", {}),
        "total_frames": int(summary.get("total_frames", 0) or 0),
        "shareable": True,
    }


@router.get("/session/{session_id}/scorecard.png", dependencies=[Depends(current_user)])
async def get_scorecard_image(session_id: str):
    """Rendered 1080x1080 PNG score card for sharing."""
    session, athlete = _session_and_athlete(session_id)
    summary = session.get("summary", session)
    return generate_scorecard_response(summary, athlete)
