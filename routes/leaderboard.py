from __future__ import annotations

"""
Advanced leaderboards — per-sport, per-tier, weekly, and improvement-based.

The existing /leaderboard endpoint just returns all athletes sorted by BPI.
This module adds the leaderboards that actually drive engagement:

  GET /leaderboards/weekly      — who trained the most this week
  GET /leaderboards/sport/{s}   — sport-specific rankings
  GET /leaderboards/improvers   — biggest form score improvement in last 14 days
  GET /leaderboards/tier/{t}    — ranking within a tier (Block, District, State, etc.)
"""

import statistics
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query

from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from routes.progress import _athlete_sessions

router = APIRouter(prefix="/leaderboards", tags=["Leaderboards"])
log = get_logger("routes.leaderboard")


def _all_athletes_with_stats(days: int) -> list[dict]:
    """Build athlete list with recent session stats."""
    results = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=days)

    for aid, a in ATHLETE_DB.items():
        sessions = []
        for s in SESSION_DB.values():
            if s.get("athlete_id") != aid or s.get("status") != "completed":
                continue
            started = s.get("started_at", "")
            if started:
                try:
                    dt = datetime.fromisoformat(started.replace("Z", "+00:00")).replace(tzinfo=None)
                    if dt >= cutoff:
                        sessions.append(s)
                except (ValueError, TypeError):
                    continue

        form_scores = []
        total_xp = 0
        for s in sessions:
            sm = s.get("summary") or {}
            af = float(sm.get("avg_form_score") or 0)
            if af > 0:
                form_scores.append(af)
            total_xp += int(sm.get("xp_earned") or 0)

        results.append(
            {
                "athlete_id": aid,
                "name": a.get("name", "Unknown"),
                "sport": a.get("sport"),
                "tier": a.get("tier"),
                "bpi": a.get("bpi", 0),
                "avatar": a.get("avatar", "?"),
                "sessions_count": len(sessions),
                "avg_form_score": round(statistics.mean(form_scores), 1) if form_scores else 0.0,
                "xp_earned": total_xp,
            }
        )
    return results


@router.get("/weekly")
async def weekly_leaderboard(limit: int = Query(default=20, ge=1, le=100)):
    """Who trained the most this week? Ranked by session count, then XP."""
    athletes = _all_athletes_with_stats(days=7)
    athletes = [a for a in athletes if a["sessions_count"] > 0]
    athletes.sort(key=lambda a: (a["sessions_count"], a["xp_earned"]), reverse=True)

    for i, a in enumerate(athletes):
        a["rank"] = i + 1

    return {
        "leaderboard": "weekly",
        "period": "last 7 days",
        "athletes": athletes[:limit],
        "total": len(athletes),
    }


@router.get("/sport/{sport}")
async def sport_leaderboard(sport: str, limit: int = Query(default=20, ge=1, le=100)):
    """Rankings within a specific sport by BPI."""
    athletes = _all_athletes_with_stats(days=30)
    athletes = [a for a in athletes if a["sport"] == sport]
    athletes.sort(key=lambda a: a["bpi"], reverse=True)

    for i, a in enumerate(athletes):
        a["rank"] = i + 1

    return {
        "leaderboard": f"sport:{sport}",
        "athletes": athletes[:limit],
        "total": len(athletes),
    }


@router.get("/improvers")
async def improvers_leaderboard(
    days: int = Query(default=14, ge=7, le=60),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Biggest form score improvers. Compares recent half of sessions to earlier half.
    This is the leaderboard that rewards getting better, not just being good.
    """
    results = []
    for aid, a in ATHLETE_DB.items():
        sessions = _athlete_sessions(aid, days)
        if len(sessions) < 4:
            continue

        form_scores = []
        for s in sessions:
            sm = s.get("summary") or {}
            af = float(sm.get("avg_form_score") or 0)
            if af > 0:
                form_scores.append(af)

        if len(form_scores) < 4:
            continue

        half = len(form_scores) // 2
        earlier = statistics.mean(form_scores[:half])
        later = statistics.mean(form_scores[half:])
        if earlier <= 0:
            continue

        improvement = round((later - earlier) / earlier * 100, 1)
        results.append(
            {
                "athlete_id": aid,
                "name": a.get("name", "Unknown"),
                "sport": a.get("sport"),
                "tier": a.get("tier"),
                "bpi": a.get("bpi", 0),
                "avatar": a.get("avatar", "?"),
                "improvement_pct": improvement,
                "earlier_avg": round(earlier, 1),
                "later_avg": round(later, 1),
                "sessions_count": len(sessions),
            }
        )

    results.sort(key=lambda r: r["improvement_pct"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {
        "leaderboard": "improvers",
        "period": f"last {days} days",
        "athletes": results[:limit],
        "total": len(results),
    }


@router.get("/tier/{tier}")
async def tier_leaderboard(tier: str, limit: int = Query(default=20, ge=1, le=100)):
    """Rankings within a tier (Block, District, State, National, Elite)."""
    valid_tiers = {"Block", "District", "State", "National", "Elite"}
    # case insensitive match
    matched = next((t for t in valid_tiers if t.lower() == tier.lower()), None)
    if not matched:
        raise HTTPException(400, f"unknown tier '{tier}'. valid: {', '.join(sorted(valid_tiers))}")

    athletes = _all_athletes_with_stats(days=30)
    athletes = [a for a in athletes if a["tier"] == matched]
    athletes.sort(key=lambda a: a["bpi"], reverse=True)

    for i, a in enumerate(athletes):
        a["rank"] = i + 1

    return {
        "leaderboard": f"tier:{matched}",
        "athletes": athletes[:limit],
        "total": len(athletes),
    }
