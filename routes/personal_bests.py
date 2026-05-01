from __future__ import annotations

"""
Personal Health — Personal Bests endpoint.

GET /athlete/{id}/personal-bests

Scans all completed sessions for an athlete and returns their all-time
personal best across key metrics: form score, jump height, rep count in
a single session, longest streak, total XP, and best session by sport.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from auth import require_athlete_owner
from cache import progress_cache
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger

router = APIRouter(tags=["Progress"])
log = get_logger("routes.personal_bests")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


@router.get("/athlete/{athlete_id}/personal-bests", dependencies=[Depends(require_athlete_owner())])
async def personal_bests(athlete_id: str):
    """All-time personal bests across key performance metrics."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    cache_key = f"pbs:{athlete_id}"
    cached = progress_cache.get(cache_key)
    if cached:
        return cached

    sessions = [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"]

    if not sessions:
        result = {
            "athlete_id": athlete_id,
            "has_data": False,
            "message": "No completed sessions yet.",
        }
        progress_cache.set(cache_key, result, ttl=120.0)
        return result

    best_form = max(sessions, key=lambda s: float(s.get("avg_form_score") or 0), default=None)
    best_jump = max(sessions, key=lambda s: float(s.get("peak_jump_height_cm") or 0), default=None)
    best_reps = max(sessions, key=lambda s: int(s.get("rep_count") or 0), default=None)
    best_xp = max(sessions, key=lambda s: int(s.get("xp_earned") or 0), default=None)

    # Per-sport best form score
    sport_bests: dict[str, dict] = {}
    for s in sessions:
        sport = s.get("sport", "unknown")
        score = float(s.get("avg_form_score") or 0)
        if sport not in sport_bests or score > sport_bests[sport]["score"]:
            sport_bests[sport] = {
                "sport": sport,
                "score": round(score, 1),
                "session_id": s.get("session_id"),
                "date": (s.get("started_at") or "")[:10],
            }

    # Total career stats
    total_reps = sum(int(s.get("rep_count") or 0) for s in sessions)
    total_xp = sum(int(s.get("xp_earned") or 0) for s in sessions)

    def _snap(s: dict | None, metric_key: str) -> dict | None:
        if not s:
            return None
        return {
            "value": s.get(metric_key) or s.get("avg_form_score"),
            "session_id": s.get("session_id"),
            "date": (s.get("started_at") or "")[:10],
            "sport": s.get("sport"),
        }

    result = {
        "athlete_id": athlete_id,
        "has_data": True,
        "total_sessions": len(sessions),
        "total_reps": total_reps,
        "total_xp": total_xp,
        "best_form_score": _snap(best_form, "avg_form_score"),
        "best_jump_height_cm": _snap(best_jump, "peak_jump_height_cm"),
        "best_reps_single_session": _snap(best_reps, "rep_count"),
        "best_xp_single_session": _snap(best_xp, "xp_earned"),
        "sport_bests": list(sport_bests.values()),
    }
    progress_cache.set(cache_key, result, ttl=120.0)
    return result
