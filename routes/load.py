from __future__ import annotations

"""
Personal Health — Progressive Load endpoint.

VISION.md Layer 2 Item 5: volume/intensity recommendation based on ACWR.

  GET /athlete/{id}/load-recommendation
"""

from fastapi import APIRouter, Depends, HTTPException

from auth import require_athlete_or_admin
from cache import progress_cache
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from services.progressive_load import compute_progressive_load

router = APIRouter(tags=["Load"])
log = get_logger("routes.load")


@router.get("/athlete/{athlete_id}/load-recommendation")
async def get_load_recommendation(athlete_id: str, _: dict = Depends(require_athlete_or_admin("athlete_id"))):
    """ACWR-based training load recommendation."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    cache_key = f"load:{athlete_id}"
    cached = progress_cache.get(cache_key)
    if cached:
        return cached

    athlete = ATHLETE_DB[athlete_id]
    # Gather all completed sessions for this athlete
    sessions = [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"]
    sessions.sort(key=lambda x: x.get("started_at", ""))

    payload = compute_progressive_load(
        athlete_id=athlete_id,
        all_sessions=sessions,
        sport=athlete.get("sport", "vertical_jump"),
    )
    progress_cache.set(cache_key, payload, ttl=300)
    return payload
