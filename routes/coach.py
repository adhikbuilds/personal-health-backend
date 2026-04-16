from __future__ import annotations

"""
Personal Health — AI Coach route (/coach/*).
Wraps ai_coach module which uses Anthropic with deterministic fallback.
"""

from fastapi import APIRouter, HTTPException, Query

from ai_coach import generate_coach_note
from cache import coach_cache
from database import ATHLETE_DB
from logging_setup import get_logger
from routes.progress import _compute_injury_risk, _compute_progress, _compute_weak_joints

router = APIRouter(prefix="/coach", tags=["Coach"])
log = get_logger("routes.coach")


@router.get("/{athlete_id}/weekly-note")
async def weekly_note(athlete_id: str, days: int = Query(default=7, ge=1, le=30), refresh: bool = False):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    cache_key = f"coach:{athlete_id}:{days}"
    if not refresh:
        cached = coach_cache.get(cache_key)
        if cached:
            return cached

    progress = _compute_progress(athlete_id, days)
    risk = _compute_injury_risk(athlete_id, days)
    weak = _compute_weak_joints(athlete_id, days)

    stats = {
        "session_count": progress["session_count"],
        "avg_form_score": progress["avg_form_score"],
        "form_trend_pct": progress["form_trend_pct"],
        "bpi_delta": progress["bpi_delta"],
        "best_jump_cm": progress["best_jump_cm"],
        "injury_risk": risk["risk"],
        "injury_reason": risk["reason"],
        "weak_joints": weak[:3],
    }

    athlete = ATHLETE_DB[athlete_id]
    note = generate_coach_note(athlete.get("name", "Athlete"), athlete.get("sport", "vertical_jump"), stats)
    payload = {
        "athlete_id": athlete_id,
        "window_days": days,
        **note,
    }
    coach_cache.set(cache_key, payload)
    return payload
