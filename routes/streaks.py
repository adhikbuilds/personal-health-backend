from __future__ import annotations

"""
Training streaks and achievements.

VISION.md data flywheel: "More athletes use the app → more real sessions →
retrain → better scoring → athletes get better results → athletes tell others."

Gamification drives the flywheel. Streaks and achievements keep athletes
coming back, which means more sessions, which means more data.

Endpoints:
  GET /athlete/{id}/streaks     — current streak, longest streak, weekly counts
  GET /athlete/{id}/achievements — unlocked achievements based on session history
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from auth import require_athlete_or_admin
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger

router = APIRouter(tags=["Streaks"])
log = get_logger("routes.streaks")


def _session_dates(athlete_id: str) -> list[str]:
    """Get sorted list of unique dates (YYYY-MM-DD) athlete trained on."""
    dates = set()
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id:
            continue
        if s.get("status") != "completed":
            continue
        started = s.get("started_at", "")
        if started:
            dates.add(started[:10])
    return sorted(dates)


def _compute_streaks(dates: list[str]) -> dict:
    """Compute current streak and longest streak from sorted date strings."""
    if not dates:
        return {"current_streak": 0, "longest_streak": 0, "total_training_days": 0}

    today = datetime.now(timezone.utc).date()

    # parse all dates
    parsed = []
    for d in dates:
        try:
            parsed.append(datetime.fromisoformat(d).date())
        except (ValueError, TypeError):
            continue

    if not parsed:
        return {"current_streak": 0, "longest_streak": 0, "total_training_days": 0}

    parsed = sorted(set(parsed))

    # longest streak
    longest = 1
    current = 1
    for i in range(1, len(parsed)):
        if (parsed[i] - parsed[i - 1]).days == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    # current streak (counting back from today or yesterday)
    current_streak = 0
    check_date = today
    # allow today or yesterday as most recent training day
    if parsed[-1] < today - timedelta(days=1):
        current_streak = 0
    else:
        check_date = today if parsed[-1] == today else parsed[-1]
        date_set = set(parsed)
        while check_date in date_set:
            current_streak += 1
            check_date -= timedelta(days=1)

    return {
        "current_streak": current_streak,
        "longest_streak": longest,
        "total_training_days": len(parsed),
    }


def _weekly_volume(athlete_id: str, weeks: int = 4) -> list[dict]:
    """Sessions per week for the last N weeks."""
    now = datetime.now(timezone.utc).date()
    week_counts: dict[int, int] = defaultdict(int)
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id or s.get("status") != "completed":
            continue
        started = s.get("started_at", "")
        if not started:
            continue
        try:
            d = datetime.fromisoformat(started[:10]).date()
        except (ValueError, TypeError):
            continue
        days_ago = (now - d).days
        week_idx = days_ago // 7
        if week_idx < weeks:
            week_counts[week_idx] += 1

    return [
        {
            "week": i,
            "label": "this week" if i == 0 else f"{i}w ago",
            "sessions": week_counts.get(i, 0),
        }
        for i in range(weeks)
    ]


@router.get("/athlete/{athlete_id}/streaks")
async def get_streaks(athlete_id: str, _: dict = Depends(require_athlete_or_admin("athlete_id"))):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    dates = _session_dates(athlete_id)
    streaks = _compute_streaks(dates)
    weekly = _weekly_volume(athlete_id, weeks=4)

    return {
        "athlete_id": athlete_id,
        **streaks,
        "weekly_volume": weekly,
    }


# achievement definitions
ACHIEVEMENTS = [
    {"id": "first_session", "name": "First Step", "desc": "complete your first session", "check": lambda s, a: s >= 1},
    {"id": "ten_sessions", "name": "Getting Serious", "desc": "complete 10 sessions", "check": lambda s, a: s >= 10},
    {"id": "fifty_sessions", "name": "Dedicated", "desc": "complete 50 sessions", "check": lambda s, a: s >= 50},
    {"id": "hundred_sessions", "name": "Centurion", "desc": "complete 100 sessions", "check": lambda s, a: s >= 100},
    {
        "id": "streak_3",
        "name": "Hat Trick",
        "desc": "3 day training streak",
        "check": lambda s, a: a.get("longest_streak", 0) >= 3,
    },
    {
        "id": "streak_7",
        "name": "Week Warrior",
        "desc": "7 day training streak",
        "check": lambda s, a: a.get("longest_streak", 0) >= 7,
    },
    {
        "id": "streak_14",
        "name": "Fortnight Force",
        "desc": "14 day training streak",
        "check": lambda s, a: a.get("longest_streak", 0) >= 14,
    },
    {
        "id": "streak_30",
        "name": "Monthly Machine",
        "desc": "30 day training streak",
        "check": lambda s, a: a.get("longest_streak", 0) >= 30,
    },
    {"id": "bpi_5k", "name": "Rising Star", "desc": "reach 5000 BPI", "check": lambda s, a: a.get("bpi", 0) >= 5000},
    {
        "id": "bpi_10k",
        "name": "Elite Prospect",
        "desc": "reach 10000 BPI",
        "check": lambda s, a: a.get("bpi", 0) >= 10000,
    },
    {
        "id": "bpi_20k",
        "name": "National Caliber",
        "desc": "reach 20000 BPI",
        "check": lambda s, a: a.get("bpi", 0) >= 20000,
    },
    {
        "id": "form_80",
        "name": "Clean Form",
        "desc": "avg form score above 80 in a session",
        "check": lambda s, a: a.get("best_form", 0) >= 80,
    },
    {
        "id": "form_90",
        "name": "Technical Master",
        "desc": "avg form score above 90 in a session",
        "check": lambda s, a: a.get("best_form", 0) >= 90,
    },
]


@router.get("/athlete/{athlete_id}/achievements")
async def get_achievements(athlete_id: str, _: dict = Depends(require_athlete_or_admin("athlete_id"))):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    dates = _session_dates(athlete_id)
    streaks = _compute_streaks(dates)
    session_count = int(athlete.get("sessions", 0)) or len(dates)

    # find best form score across all sessions
    best_form = 0.0
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id or s.get("status") != "completed":
            continue
        sm = s.get("summary") or {}
        af = float(sm.get("avg_form_score") or 0)
        if af > best_form:
            best_form = af

    check_data = {**streaks, "bpi": athlete.get("bpi", 0), "best_form": best_form}
    unlocked = []
    locked = []
    for ach in ACHIEVEMENTS:
        entry = {"id": ach["id"], "name": ach["name"], "description": ach["desc"]}
        if ach["check"](session_count, check_data):
            unlocked.append(entry)
        else:
            locked.append(entry)

    return {
        "athlete_id": athlete_id,
        "unlocked": unlocked,
        "locked": locked,
        "total_unlocked": len(unlocked),
        "total_achievements": len(ACHIEVEMENTS),
    }
