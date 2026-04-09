from __future__ import annotations

"""
Personal Health — Dynamic Training Plan Generator (VISION Layer 3).

  GET /training-plan/{athlete_id}?days=14

Generates a personalized weekly training plan based on the athlete's recent
session data. Uses form trends, injury risk, weak joints, volume history,
and recovery state (HRV) to prescribe:
  - Daily schedule (train / rest / active-recovery / skill-work)
  - Volume targets (session count, rep ranges)
  - Intensity guidelines (tempo, load %)
  - Focus drills for weakest joints
  - Progressive overload or deload recommendations

Falls back to a deterministic template when no LLM is available, so
the endpoint *never* breaks in dev.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from cache import progress_cache
from database import ATHLETE_DB
from logging_setup import get_logger
from routes.analytics import _hrv_component
from routes.progress import (
    _compute_injury_risk,
    _compute_progress,
    _compute_weak_joints,
)

router = APIRouter(tags=["Training Plan"])
log = get_logger("routes.training_plan")


# ─── Sport-specific drill library ──────────────────────────────────────────

DRILL_LIBRARY: dict[str, dict[str, list[dict]]] = {
    "vertical_jump": {
        "knee_angle": [
            {"name": "Banded goblet squat pause", "sets": "3x8", "note": "2s pause at bottom, knees tracking toes"},
            {"name": "Box jump step-down", "sets": "3x5", "note": "focus on soft landing with aligned knees"},
        ],
        "hip_angle": [
            {"name": "Hip hinge with band", "sets": "3x10", "note": "push hips back, flat spine"},
            {"name": "Single-leg RDL", "sets": "3x8/side", "note": "control descent, hip crease to 90deg"},
        ],
        "ankle_dorsiflexion": [
            {"name": "Wall ankle mobilisation", "sets": "3x30s/side", "note": "knee past toe, heel flat"},
            {"name": "Banded ankle distraction", "sets": "2x45s/side", "note": "band pulls talus back"},
        ],
        "trunk_lean": [
            {"name": "Dead bug hold", "sets": "3x30s", "note": "flatten lower back to floor throughout"},
            {"name": "Front plank with reach", "sets": "3x8/side", "note": "no rotation when reaching"},
        ],
        "general": [
            {"name": "Depth jump to vertical", "sets": "4x3", "note": "minimal ground contact time"},
            {"name": "Countermovement jump filmed", "sets": "3x5", "note": "review knee tracking after"},
        ],
    },
    "sprint": {
        "knee_angle": [
            {"name": "A-skip drill", "sets": "3x30m", "note": "drive knee to 90deg hip flexion"},
        ],
        "hip_angle": [
            {"name": "Wall drive", "sets": "4x8/side", "note": "45deg lean, full hip extension"},
        ],
        "trunk_lean": [
            {"name": "Sled push", "sets": "4x20m", "note": "maintain forward lean angle"},
        ],
        "general": [
            {"name": "Flying 30m sprint", "sets": "3x30m", "note": "build up zone first, then hold form"},
            {"name": "Wicket runs", "sets": "4x40m", "note": "consistent stride length over hurdles"},
        ],
    },
    "snatch": {
        "knee_angle": [
            {"name": "Overhead squat", "sets": "3x5", "note": "bar behind ears, heels flat"},
        ],
        "hip_angle": [
            {"name": "Snatch-grip RDL", "sets": "3x6", "note": "hamstrings loaded, bar close"},
        ],
        "shoulder_angle": [
            {"name": "Snatch press in squat", "sets": "3x5", "note": "elbows locked, active shoulders"},
        ],
        "trunk_lean": [
            {"name": "Overhead carry", "sets": "3x30m", "note": "ribs down, core braced"},
        ],
        "general": [
            {"name": "High-hang muscle snatch", "sets": "5x3 @ 40%", "note": "fast elbows, pause at catch"},
        ],
    },
    "squat": {
        "knee_angle": [
            {"name": "Tempo back squat", "sets": "3x5 @ 65%", "note": "3s down, 1s pause, 2s up"},
        ],
        "hip_angle": [
            {"name": "Hip thrust", "sets": "3x10", "note": "full lockout, squeeze glutes"},
        ],
        "general": [
            {"name": "Goblet squat to box", "sets": "3x10", "note": "sit back, control descent"},
        ],
    },
    "push_up": {
        "elbow_angle": [
            {"name": "Eccentric push-up", "sets": "3x6", "note": "5s lowering, reset at top"},
        ],
        "general": [
            {"name": "Banded push-up", "sets": "3x8", "note": "band around back for overload at top"},
        ],
    },
    "pull_up": {
        "elbow_angle": [
            {"name": "Eccentric pull-up", "sets": "3x5", "note": "5s lowering from chin over bar"},
        ],
        "general": [
            {"name": "Scapular pull-up", "sets": "3x8", "note": "hang and depress scapulae only"},
        ],
    },
    "javelin": {
        "shoulder_angle": [
            {"name": "Single-arm med-ball throw", "sets": "3x6", "note": "elbow above shoulder"},
        ],
        "trunk_lean": [
            {"name": "Rotational slam", "sets": "3x8/side", "note": "hip drives rotation"},
        ],
        "general": [
            {"name": "Standing throw with step", "sets": "4x5", "note": "block with front leg"},
        ],
    },
    "cricket_bat": {
        "elbow_angle": [
            {"name": "Shadow drive", "sets": "3x8", "note": "head over front knee, check in mirror"},
        ],
        "shoulder_angle": [
            {"name": "Bat speed drill", "sets": "3x10", "note": "light bat, max hand speed"},
        ],
        "general": [
            {"name": "Front-foot drive off tee", "sets": "3x12", "note": "film from side, check head position"},
        ],
    },
}

# Fallback drills for unknown sports
DEFAULT_DRILLS = {
    "general": [
        {"name": "Tempo reps of main movement", "sets": "3x8 @ 60%", "note": "film from side, review form"},
    ],
}


# ─── Plan logic ────────────────────────────────────────────────────────────


def _classify_load_phase(
    form_trend_pct: float,
    session_count: int,
    days: int,
    injury_risk: str,
    hrv_score: float,
) -> str:
    """
    Classify the athlete's current training phase:
      overreach  — too much, needs deload
      build      — progressing well, increase load
      maintain   — stable, hold current volume
      recover    — injured / depleted, reduce everything
    """
    sessions_per_week = session_count / max(1, days / 7)

    if injury_risk == "high" or hrv_score < 0.3:
        return "recover"
    if form_trend_pct < -5 and sessions_per_week > 4:
        return "overreach"
    if form_trend_pct > 2 and sessions_per_week >= 2:
        return "build"
    return "maintain"


def _pick_drills(sport: str, weak_joints: list[dict], count: int = 3) -> list[dict]:
    """Pick targeted drills for the athlete's weakest joints."""
    sport_drills = DRILL_LIBRARY.get(sport, DEFAULT_DRILLS)
    picked: list[dict] = []

    # First: drills for weak joints
    for joint_info in weak_joints[:count]:
        joint = joint_info.get("joint", "")
        joint_drills = sport_drills.get(joint, [])
        if joint_drills and len(picked) < count:
            drill = joint_drills[0].copy()
            drill["reason"] = (
                f"{joint.replace('_', ' ')} deviates {joint_info.get('deviation_deg', 0):.0f}deg from ideal"
            )
            picked.append(drill)

    # Fill remaining with general drills
    general = sport_drills.get("general", DEFAULT_DRILLS["general"])
    for d in general:
        if len(picked) >= count:
            break
        picked.append(d.copy())

    return picked[:count]


def _generate_daily_schedule(phase: str, sport: str) -> list[dict]:
    """
    Generate a 7-day schedule based on the load phase.
    Each day has a type and brief prescription.
    """
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if phase == "recover":
        pattern = ["rest", "active-recovery", "rest", "skill-work", "rest", "active-recovery", "rest"]
        descriptions = {
            "rest": "Full rest. Light walking OK. Prioritise sleep.",
            "active-recovery": "20min foam rolling + mobility circuit. No load.",
            "skill-work": "Film 10 reps at 50% effort. Review form only, no fatigue.",
        }
    elif phase == "overreach":
        pattern = ["rest", "skill-work", "active-recovery", "train-light", "rest", "skill-work", "rest"]
        descriptions = {
            "rest": "Full rest. Hydrate and sleep 8h+.",
            "active-recovery": "20min mobility + foam roll. Focus on tight areas.",
            "skill-work": "Drill work only — no max effort. Film and review.",
            "train-light": f"Short {sport.replace('_', ' ')} session at 60% intensity. Cap at 20min.",
        }
    elif phase == "build":
        pattern = ["train", "skill-work", "train", "active-recovery", "train", "skill-work", "rest"]
        descriptions = {
            "train": f"Full {sport.replace('_', ' ')} session. Push intensity 5% above last week.",
            "skill-work": "Drill work targeting weak joints. Film for form check.",
            "active-recovery": "Mobility circuit + light cardio. Keep HR below 120.",
            "rest": "Full rest. Recovery is where adaptation happens.",
        }
    else:  # maintain
        pattern = ["train", "active-recovery", "train", "rest", "skill-work", "train", "rest"]
        descriptions = {
            "train": f"{sport.replace('_', ' ')} session at current intensity. Maintain volume.",
            "active-recovery": "20min mobility + foam roll.",
            "skill-work": "Targeted drill work. Film one set for review.",
            "rest": "Full rest day.",
        }

    schedule = []
    for i, day_name in enumerate(days_of_week):
        day_type = pattern[i]
        schedule.append(
            {
                "day": day_name,
                "type": day_type,
                "prescription": descriptions.get(day_type, ""),
            }
        )

    return schedule


def _volume_recommendation(phase: str, current_sessions_per_week: float) -> dict:
    """Recommend volume targets for next week."""
    if phase == "recover":
        target = max(1, int(current_sessions_per_week * 0.4))
        return {
            "sessions_target": target,
            "rep_intensity": "50-60%",
            "note": "Minimal volume. Focus on movement quality over quantity.",
        }
    if phase == "overreach":
        target = max(2, int(current_sessions_per_week * 0.6))
        return {
            "sessions_target": target,
            "rep_intensity": "60-70%",
            "note": "Reduce volume by 40%. Let the body absorb previous training.",
        }
    if phase == "build":
        target = min(6, int(current_sessions_per_week + 1))
        return {
            "sessions_target": target,
            "rep_intensity": "75-85%",
            "note": "Add one session or increase intensity 5%. Progressive overload.",
        }
    # maintain
    target = max(2, int(current_sessions_per_week))
    return {
        "sessions_target": target,
        "rep_intensity": "70-80%",
        "note": "Hold current volume. Consistency is the goal this week.",
    }


def _generate_plan(athlete_id: str, days: int) -> dict:
    """Core plan generator. Assembles all components into a structured plan."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    sport = athlete.get("sport", "vertical_jump")

    # Gather all analytics
    progress = _compute_progress(athlete_id, days)
    risk = _compute_injury_risk(athlete_id, days)
    weak = _compute_weak_joints(athlete_id, days)
    hrv_score, hrv_samples = _hrv_component(athlete_id, days)

    sessions_per_week = progress["session_count"] / max(1, days / 7)

    # Determine training phase
    phase = _classify_load_phase(
        form_trend_pct=progress["form_trend_pct"],
        session_count=progress["session_count"],
        days=days,
        injury_risk=risk["risk"],
        hrv_score=hrv_score,
    )

    # Build plan components
    schedule = _generate_daily_schedule(phase, sport)
    volume = _volume_recommendation(phase, sessions_per_week)
    drills = _pick_drills(sport, weak, count=3)

    # Key insights
    insights: list[str] = []
    if progress["session_count"] < 3:
        insights.append(
            f"Only {progress['session_count']} session(s) in the last {days} days. "
            "Log more sessions for a better-calibrated plan."
        )
    if progress["form_trend_pct"] > 3:
        insights.append(
            f"Form is trending up {progress['form_trend_pct']:+.1f}% — good momentum. "
            "This plan pushes intensity slightly higher."
        )
    elif progress["form_trend_pct"] < -3:
        insights.append(
            f"Form dropped {progress['form_trend_pct']:.1f}% recently. This plan reduces load and adds recovery days."
        )
    if risk["risk"] == "high":
        insights.append(
            f"Injury risk is elevated: {risk['reason']}. Priority is asymmetry correction, not performance."
        )
    elif risk["risk"] == "watch":
        insights.append(f"Mild asymmetry detected: {risk['reason']}. Monitor closely.")
    if weak and weak[0].get("deviation_deg", 0) > 10:
        j = weak[0]
        insights.append(
            f"Biggest weakness: {j['joint'].replace('_', ' ')} "
            f"({j['deviation_deg']:.0f}deg off ideal). Drills below target this."
        )

    return {
        "athlete_id": athlete_id,
        "athlete_name": athlete.get("name"),
        "sport": sport,
        "window_days": days,
        "phase": phase,
        "phase_rationale": {
            "recover": "High injury risk or poor recovery signals. Prioritise rest and mobility.",
            "overreach": "Form declining despite high volume. Time to absorb training with a deload.",
            "build": "Form trending up, risk low. Safe to push intensity and volume.",
            "maintain": "Stable performance. Hold current load and focus on consistency.",
        }[phase],
        "schedule": schedule,
        "volume": volume,
        "drills": drills,
        "insights": insights,
        "data_summary": {
            "session_count": progress["session_count"],
            "sessions_per_week": round(sessions_per_week, 1),
            "avg_form_score": progress["avg_form_score"],
            "form_trend_pct": progress["form_trend_pct"],
            "best_jump_cm": progress["best_jump_cm"],
            "injury_risk": risk["risk"],
            "hrv_score": round(hrv_score * 100, 1),
            "hrv_samples": hrv_samples,
            "weak_joints_top3": [{"joint": j["joint"], "deviation_deg": j["deviation_deg"]} for j in weak[:3]],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Route ─────────────────────────────────────────────────────────────────


@router.get("/training-plan/{athlete_id}")
async def training_plan(athlete_id: str, days: int = Query(default=14, ge=3, le=90)):
    key = f"plan:{athlete_id}:{days}"
    cached = progress_cache.get(key)
    if cached:
        return cached
    payload = _generate_plan(athlete_id, days)
    progress_cache.set(key, payload)
    return payload
