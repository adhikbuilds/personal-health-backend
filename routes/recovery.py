from __future__ import annotations

"""
Personal Health — Recovery Routes

GET /athlete/{athlete_id}/recovery/score
  Returns recovery readiness score + ACWR + load status.

GET /athlete/{athlete_id}/recovery/recommendation
  Returns the training recommendation with human-readable detail.
"""

from fastapi import APIRouter, Depends, HTTPException

from auth import require_athlete_owner
from cache import progress_cache
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from services.recovery_engine import compute_recovery_score, enrich_recommendation

router = APIRouter(tags=["Recovery"])
log = get_logger("routes.recovery")


def _get_athlete_sessions(athlete_id: str) -> list[dict]:
    return [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id]


def _get_wellness_today(athlete_id: str) -> dict | None:
    athlete = ATHLETE_DB.get(athlete_id, {})
    entries = athlete.get("wellness_entries") or []
    if not entries:
        return None
    latest = sorted(entries, key=lambda e: e.get("logged_at", ""), reverse=True)
    if not latest:
        return None
    entry = latest[0]
    mental = entry.get("mental") or {}
    physical = entry.get("physical") or {}
    sleep = entry.get("sleep") or {}
    return {
        "soreness": physical.get("soreness"),
        "energy": mental.get("energy"),
        "sleep_quality": sleep.get("quality"),
    }


@router.get("/athlete/{athlete_id}/recovery/score", dependencies=[Depends(require_athlete_owner())])
async def get_recovery_score(athlete_id: str):
    """Recovery readiness score (0-100) with ACWR and load breakdown."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    cache_key = f"recovery:score:{athlete_id}"
    cached = progress_cache.get(cache_key)
    if cached:
        return cached

    sessions = _get_athlete_sessions(athlete_id)
    wellness = _get_wellness_today(athlete_id)
    result = compute_recovery_score(sessions, wellness)
    result["athlete_id"] = athlete_id
    progress_cache.set(cache_key, result, ttl=180.0)
    return result


@router.get("/athlete/{athlete_id}/recovery/recommendation", dependencies=[Depends(require_athlete_owner())])
async def get_recovery_recommendation(athlete_id: str):
    """Training recommendation with label, description, and coaching tip."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    cache_key = f"recovery:rec:{athlete_id}"
    cached = progress_cache.get(cache_key)
    if cached:
        return cached

    sessions = _get_athlete_sessions(athlete_id)
    wellness = _get_wellness_today(athlete_id)
    score_data = compute_recovery_score(sessions, wellness)
    detail = enrich_recommendation(score_data["recommendation"])

    result = {
        "athlete_id": athlete_id,
        "score": score_data["score"],
        "recommendation": score_data["recommendation"],
        **detail,
        "reasons": score_data["reasons"],
        "acwr": score_data["acwr"],
        "load_status": score_data["load_status"],
    }
    progress_cache.set(cache_key, result, ttl=180.0)
    return result
