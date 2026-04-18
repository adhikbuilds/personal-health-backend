from __future__ import annotations

"""
Personal Health — Weekly Session Summary endpoint.

VISION.md Milestone 4: structured weekly recap with AI coaching note.
Aggregates the last 7 days of sessions into a single summary object
that the generative layer (ai_coach) can consume.

  GET /athlete/{id}/weekly-summary?days=7
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_coach import generate_coach_note
from auth import require_athlete_or_admin
from cache import progress_cache
from database import ATHLETE_DB
from logging_setup import get_logger
from routes.progress import (
    _athlete_sessions,
    _compute_injury_risk,
    _compute_progress,
    _compute_weak_joints,
)

router = APIRouter(tags=["Summary"])
log = get_logger("routes.weekly_summary")


def _session_volume(sessions: list[dict]) -> dict:
    """Compute volume metrics from a list of sessions."""
    total_frames = sum(s.get("total_frames", 0) for s in sessions)
    total_reps = sum(s.get("rep_count", 0) or s.get("summary", {}).get("rep_count", 0) or 0 for s in sessions)
    total_xp = sum(s.get("xp_earned", 0) or s.get("summary", {}).get("xp_earned", 0) or 0 for s in sessions)
    durations = []
    for s in sessions:
        d = s.get("duration_seconds", 0) or s.get("summary", {}).get("duration_seconds", 0) or 0
        if d > 0:
            durations.append(float(d))
    total_duration_min = round(sum(durations) / 60, 1) if durations else 0
    return {
        "total_frames": total_frames,
        "total_reps": total_reps,
        "total_xp": total_xp,
        "total_duration_min": total_duration_min,
    }


def _quality_breakdown(sessions: list[dict]) -> dict[str, int]:
    """Aggregate quality distribution across sessions."""
    totals: dict[str, int] = {"elite": 0, "good": 0, "average": 0, "poor": 0}
    for s in sessions:
        dist = s.get("quality_distribution") or s.get("summary", {}).get("quality_distribution", {})
        for k in totals:
            totals[k] += int(dist.get(k, 0))
    return totals


def _daily_scores(sessions: list[dict]) -> list[dict]:
    """Group sessions by day and compute daily average form score."""
    from collections import defaultdict

    from routes.progress import _parse_dt

    by_day: dict[str, list[float]] = defaultdict(list)
    for s in sessions:
        ts = _parse_dt(s.get("started_at"))
        day = ts.date().isoformat() if ts else "unknown"
        score = float(s.get("avg_form_score", 0) or s.get("summary", {}).get("avg_form_score", 0) or 0)
        if score > 0:
            by_day[day].append(score)
    return [
        {"date": day, "avg_score": round(sum(scores) / len(scores), 1), "sessions": len(scores)}
        for day, scores in sorted(by_day.items())
    ]


def build_weekly_summary(athlete_id: str, days: int = 7) -> dict:
    """Build the full weekly summary payload."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    sessions = _athlete_sessions(athlete_id, days)
    progress = _compute_progress(athlete_id, days)
    risk = _compute_injury_risk(athlete_id, days)
    weak = _compute_weak_joints(athlete_id, days)
    volume = _session_volume(sessions)
    quality = _quality_breakdown(sessions)
    daily = _daily_scores(sessions)

    # Coach note (uses Anthropic with deterministic fallback)
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
    note = generate_coach_note(
        athlete.get("name", "Athlete"),
        athlete.get("sport", "vertical_jump"),
        stats,
    )

    # Streaks: consecutive days with at least one session
    streak = 0
    if daily:
        today = datetime.now(timezone.utc).date()
        check = today
        day_set = {d["date"] for d in daily}
        while check.isoformat() in day_set or (check == today and check.isoformat() not in day_set):
            if check.isoformat() in day_set:
                streak += 1
            check -= timedelta(days=1)
            if check < today - timedelta(days=days):
                break

    return {
        "athlete_id": athlete_id,
        "athlete_name": athlete.get("name"),
        "sport": athlete.get("sport"),
        "tier": athlete.get("tier"),
        "window_days": days,
        "session_count": len(sessions),
        "volume": volume,
        "quality_breakdown": quality,
        "daily_scores": daily,
        "avg_form_score": progress["avg_form_score"],
        "form_trend_pct": progress["form_trend_pct"],
        "best_jump_cm": progress["best_jump_cm"],
        "bpi_current": athlete.get("bpi", 0),
        "bpi_delta": progress["bpi_delta"],
        "injury_risk": risk,
        "weak_joints": weak[:5],
        "streak_days": streak,
        "coaching_note": note,
    }


@router.get("/athlete/{athlete_id}/weekly-summary")
async def get_weekly_summary(
    athlete_id: str,
    days: int = Query(default=7, ge=1, le=30),
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """Structured weekly session summary with coaching note."""
    cache_key = f"weekly_summary:{athlete_id}:{days}"
    cached = progress_cache.get(cache_key)
    if cached:
        return cached

    payload = build_weekly_summary(athlete_id, days)
    progress_cache.set(cache_key, payload, ttl=300)
    return payload
