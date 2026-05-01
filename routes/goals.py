from __future__ import annotations

"""
Personal Health — Goals Routes

GET  /athlete/{id}/goals          — current week progress vs goals
GET  /athlete/{id}/goals/defaults — suggested default goals for athlete's sport
PUT  /athlete/{id}/goals          — save custom goal targets
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import require_athlete_owner
from cache import progress_cache
from database import ATHLETE_DB, SESSION_DB, _save_db
from logging_setup import get_logger
from services.goal_engine import default_goals, evaluate_goals
from services.intelligence import StreakTracker

router = APIRouter(tags=["Goals"])
log = get_logger("routes.goals")


class GoalsUpdate(BaseModel):
    sessions_per_week: int | None = None
    form_score_target: float | None = None
    streak_days: int | None = None
    weekly_reps: int | None = None


def _get_sessions(athlete_id: str) -> list[dict]:
    return [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id]


@router.get("/athlete/{athlete_id}/goals", dependencies=[Depends(require_athlete_owner())])
async def get_goals(athlete_id: str):
    """Current week goal progress for the athlete."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    cache_key = f"goals:{athlete_id}"
    cached = progress_cache.get(cache_key)
    if cached:
        return cached

    athlete = ATHLETE_DB[athlete_id]
    sessions = _get_sessions(athlete_id)
    streak_data = StreakTracker().compute(sessions)
    result = evaluate_goals(athlete, sessions, streak_data["current_streak"])
    result["athlete_id"] = athlete_id
    progress_cache.set(cache_key, result, ttl=120.0)
    return result


@router.get("/athlete/{athlete_id}/goals/defaults", dependencies=[Depends(require_athlete_owner())])
async def get_default_goals(athlete_id: str):
    """Sport-specific suggested goal defaults."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")
    sport = ATHLETE_DB[athlete_id].get("sport", "vertical_jump")
    return {"athlete_id": athlete_id, "sport": sport, "defaults": default_goals(sport)}


@router.put("/athlete/{athlete_id}/goals", dependencies=[Depends(require_athlete_owner())])
async def update_goals(athlete_id: str, body: GoalsUpdate):
    """Save custom goal targets for the athlete."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    current = dict(athlete.get("goals") or {})

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "no goal fields provided")

    current.update(updates)
    ATHLETE_DB[athlete_id]["goals"] = current
    _save_db()

    progress_cache._data.pop(f"goals:{athlete_id}", None)
    log.info("goals updated", extra={"athlete_id": athlete_id, "updates": updates})
    return {"athlete_id": athlete_id, "goals": current, "message": "Goals saved"}
