from __future__ import annotations

"""
Athletes domain — CRUD, Progress, Insights, Daily Tracker
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import ATHLETE_DB, FRAME_BUFFER, SESSION_DB, _save_db

router = APIRouter()


class NewAthleteRequest(BaseModel):
    name: str
    sport: str = "vertical_jump"
    tier: str = "Block"
    height_cm: Optional[float] = None  # PF-03: athlete body height for jump estimation


class DailyTrackerUpdate(BaseModel):
    steps: int = 0
    active_minutes: int = 0
    distance_km: float = 0.0
    calories_burned: int = 0
    calorie_intake: int = 0
    water_glasses: int = 0
    sleep_hours: float = 0.0
    date: str = ""


# ─── CRUD ───────────────────────────────────────────────────────────────────


@router.get("/athletes", tags=["Athletes"])
async def list_athletes():
    athletes = list(ATHLETE_DB.values())
    athletes.sort(key=lambda x: x.get("bpi", 0), reverse=True)
    return {"athletes": athletes, "total": len(athletes)}


@router.post("/athlete", tags=["Athletes"])
async def create_athlete(req: NewAthleteRequest):
    athlete_id = f"athlete_{uuid.uuid4().hex[:8]}"
    initials = "".join(w[0].upper() for w in req.name.strip().split()[:2])
    athlete = {
        "id": athlete_id,
        "name": req.name,
        "sport": req.sport,
        "tier": req.tier,
        "bpi": 1000,
        "sessions": 0,
        "avatar": initials,
        "height_cm": req.height_cm or 170,
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
    }
    ATHLETE_DB[athlete_id] = athlete
    _save_db()
    return athlete


@router.get("/athlete/{athlete_id}", tags=["Athletes"])
async def get_athlete(athlete_id: str):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")
    athlete = dict(ATHLETE_DB[athlete_id])
    athlete["recent_sessions"] = sorted(
        [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
        key=lambda x: x.get("started_at", ""),
        reverse=True,
    )[:10]
    return athlete


# ─── Intelligence ───────────────────────────────────────────────────────────


@router.get("/athlete/{athlete_id}/insights", tags=["Intelligence"])
async def athlete_insights(athlete_id: str):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")
    try:
        from services.intelligence import generate_insights
    except ImportError:
        return {"error": "intelligence.py not found", "athlete_id": athlete_id}
    athlete = ATHLETE_DB[athlete_id]
    sessions = sorted(
        [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
        key=lambda x: x.get("started_at", ""),
    )
    return generate_insights(athlete, sessions)


@router.get("/session/{session_id}/coaching", tags=["Intelligence"])
async def session_coaching(session_id: str):
    if session_id not in SESSION_DB:
        raise HTTPException(404, "Session not found")
    session = SESSION_DB[session_id]
    # Use FRAME_BUFFER for active sessions, fall back to stored frames for completed ones
    frames = FRAME_BUFFER.get(session_id) or session.get("frames") or []
    try:
        from services.intelligence import CoachRecommender
    except ImportError:
        return {"error": "intelligence.py not found"}
    sport = session.get("sport", "vertical_jump")
    summary = session.get("summary", {})
    avg_score = summary.get("avg_form_score", 0) if summary else 0
    trend_g = "improving" if avg_score >= 75 else "stable" if avg_score >= 55 else "declining"
    joint_keys = ["knee_angle_l", "knee_angle_r", "hip_angle_l", "hip_angle_r"]
    avg_angles = {}
    if frames:
        for key in joint_keys:
            vals = [f.get(key, 0) for f in frames if f.get(key, 0) > 0]
            if vals:
                avg_angles[key.replace("angle_", "").upper()] = sum(vals) / len(vals)
    coach = CoachRecommender().recommend(sport, avg_angles, trend_g)
    return {"session_id": session_id, "sport": sport, "avg_score": round(avg_score, 1), "coaching": coach}


# ─── Progress ───────────────────────────────────────────────────────────────


@router.get("/athlete/{athlete_id}/progress", tags=["Athletes"])
async def get_athlete_progress(athlete_id: str):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")
    athlete_sessions = sorted(
        [
            s
            for s in SESSION_DB.values()
            if s.get("athlete_id") == athlete_id and s.get("status") == "completed" and s.get("summary")
        ],
        key=lambda s: s.get("started_at", ""),
    )
    form_trend = []
    for s in athlete_sessions:
        avg = s.get("summary", {}).get("avg_form_score")
        if avg is not None:
            form_trend.append({"date": str(s.get("started_at", ""))[:10], "avg_form_score": round(avg, 1)})
    improvement_pct = 0.0
    if len(form_trend) >= 4:
        half = len(form_trend) // 2
        first = sum(t["avg_form_score"] for t in form_trend[:half]) / half
        second = sum(t["avg_form_score"] for t in form_trend[half:]) / (len(form_trend) - half)
        if first > 0:
            improvement_pct = round((second - first) / first * 100, 1)
    now = datetime.now(timezone.utc)
    sessions_this_week = sessions_last_week = 0
    for s in athlete_sessions:
        started_str = s.get("started_at", "")
        if not started_str:
            continue
        try:
            started = datetime.fromisoformat(str(started_str).replace("Z", "+00:00")).replace(tzinfo=None)
            days_ago = (now - started).days
            if days_ago < 7:
                sessions_this_week += 1
            elif days_ago < 14:
                sessions_last_week += 1
        except Exception:
            pass
    all_frames = []
    for s in athlete_sessions[-5:]:
        all_frames.extend(s.get("frames", []))
    joint_keys = ["knee_angle_l", "knee_angle_r", "hip_angle_l", "hip_angle_r", "trunk_lean", "limb_symmetry_idx"]
    joint_avgs = {}
    for key in joint_keys:
        vals = [f.get(key, 0) for f in all_frames if f.get(key, 0) > 0]
        if vals:
            joint_avgs[key] = round(sum(vals) / len(vals), 1)
    return {
        "athlete_id": athlete_id,
        "form_trend": form_trend,
        "improvement_pct": improvement_pct,
        "sessions_this_week": sessions_this_week,
        "sessions_last_week": sessions_last_week,
        "joint_averages": joint_avgs,
        "total_sessions": len(athlete_sessions),
    }


# ─── Daily Tracker ──────────────────────────────────────────────────────────


@router.get("/athlete/{athlete_id}/daily-tracker", tags=["Daily Tracker"])
async def get_daily_tracker(athlete_id: str):
    athlete = ATHLETE_DB.get(athlete_id, {})
    today = datetime.now(timezone.utc).date().isoformat()
    tracker = athlete.get("daily_tracker", {}).get(today, {})
    return {"athlete_id": athlete_id, "date": today, "tracker": tracker}


@router.post("/athlete/{athlete_id}/daily-tracker", tags=["Daily Tracker"])
async def update_daily_tracker(athlete_id: str, data: DailyTrackerUpdate):
    if athlete_id not in ATHLETE_DB:
        ATHLETE_DB[athlete_id] = {"id": athlete_id, "daily_tracker": {}}
    athlete = ATHLETE_DB[athlete_id]
    if "daily_tracker" not in athlete:
        athlete["daily_tracker"] = {}
    date_key = data.date or datetime.now(timezone.utc).date().isoformat()
    athlete["daily_tracker"][date_key] = {
        "steps": data.steps,
        "active_minutes": data.active_minutes,
        "distance_km": data.distance_km,
        "calories_burned": data.calories_burned,
        "calorie_intake": data.calorie_intake,
        "water_glasses": data.water_glasses,
        "sleep_hours": data.sleep_hours,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_db()
    # Mirror to SQLite v2 store so the /history endpoint has data even after restart
    try:
        from sqlite_store import upsert_daily_tracker

        upsert_daily_tracker(athlete_id, date_key, athlete["daily_tracker"][date_key])
    except Exception:
        pass
    return {"athlete_id": athlete_id, "date": date_key, "ok": True}


@router.get("/athlete/{athlete_id}/daily-tracker/history", tags=["Daily Tracker"])
async def daily_tracker_history(athlete_id: str, days: int = 30):
    """Returns up to `days` most recent daily-tracker entries, newest first.

    Reads from SQLite v2 store first; falls back to the legacy JSON dict
    on the athlete record so existing seeded data is still visible.
    """
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")
    rows: list[dict] = []
    try:
        from sqlite_store import daily_tracker_history as _hist

        rows = _hist(athlete_id, days)
    except Exception:
        rows = []
    if not rows:
        legacy = ATHLETE_DB[athlete_id].get("daily_tracker", {}) or {}
        rows = [{"date": k, **v} for k, v in sorted(legacy.items(), reverse=True)[:days]]
    return {"athlete_id": athlete_id, "days": days, "history": rows, "total": len(rows)}
