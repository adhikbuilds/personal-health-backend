from __future__ import annotations

"""
Athlete baseline assessment and session quality guard.

Baseline: when an athlete first joins, run a short assessment to establish
their starting point. This gives the coaching layer context and makes the
first training plan meaningful.

Quality guard: automatically flag sessions that are too short, have too few
valid frames, or have consistently poor scores. These "junk sessions" pollute
the training data and shouldn't count toward streaks or be exported.

Endpoints:
  POST /athlete/{id}/baseline          — record baseline assessment
  GET  /athlete/{id}/baseline          — get baseline
  GET  /sessions/{id}/quality-check    — run quality guard on a session
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import require_athlete_or_admin
from database import ATHLETE_DB, SESSION_DB, _save_db
from logging_setup import get_logger

router = APIRouter(tags=["Baseline"])
log = get_logger("routes.baseline")


class BaselineRequest(BaseModel):
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=20, lt=300)
    age: int = Field(gt=5, lt=100)
    sport: str
    training_years: float = Field(ge=0, lt=50)
    sessions_per_week: int = Field(ge=0, le=14)
    known_injuries: str = ""
    goals: str = ""


@router.post("/athlete/{athlete_id}/baseline")
async def set_baseline(
    athlete_id: str,
    req: BaselineRequest,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    baseline = {
        "height_cm": req.height_cm,
        "weight_kg": req.weight_kg,
        "bmi": round(req.weight_kg / (req.height_cm / 100) ** 2, 1),
        "age": req.age,
        "sport": req.sport,
        "training_years": req.training_years,
        "sessions_per_week": req.sessions_per_week,
        "known_injuries": req.known_injuries,
        "goals": req.goals,
        "assessed_at": datetime.now(timezone.utc).isoformat(),
    }

    # also update athlete height for jump calculations
    athlete["height_cm"] = req.height_cm
    athlete["baseline"] = baseline
    _save_db()

    log.info("baseline recorded", extra={"athlete_id": athlete_id})

    # classify experience level
    if req.training_years < 1:
        level = "beginner"
    elif req.training_years < 3:
        level = "intermediate"
    elif req.training_years < 7:
        level = "advanced"
    else:
        level = "elite"

    return {
        "athlete_id": athlete_id,
        "baseline": baseline,
        "experience_level": level,
        "recommendation": _baseline_recommendation(level, req),
    }


def _baseline_recommendation(level: str, req: BaselineRequest) -> str:
    if level == "beginner":
        return (
            "start with 2-3 sessions per week focusing on form over intensity. "
            "the app will build your baseline over the first 10 sessions"
        )
    if level == "intermediate":
        return (
            f"youre at {req.sessions_per_week} sessions/week which is solid. "
            f"focus on the weak joint feedback after each session to keep improving"
        )
    if level == "advanced":
        return (
            f"with {req.training_years:.0f} years experience, the injury risk tracking "
            f"and training plan features will be most valuable for you"
        )
    return (
        f"elite level athlete with {req.training_years:.0f} years. "
        f"use the weekly summary and readiness score to manage load"
    )


@router.get("/athlete/{athlete_id}/baseline")
async def get_baseline(athlete_id: str, _: dict = Depends(require_athlete_or_admin("athlete_id"))):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    baseline = athlete.get("baseline")
    if not baseline:
        return {
            "athlete_id": athlete_id,
            "has_baseline": False,
            "message": "no baseline recorded yet. POST /athlete/{id}/baseline to set one",
        }

    return {
        "athlete_id": athlete_id,
        "has_baseline": True,
        "baseline": baseline,
    }


# session quality thresholds
MIN_FRAMES = 10
MIN_VALID_RATIO = 0.3
MIN_AVG_SCORE = 15.0
MIN_DURATION_SECONDS = 10


@router.get("/sessions/{session_id}/quality-check")
async def session_quality_check(session_id: str):
    """
    Run quality guard on a session. Flags junk sessions that shouldnt be
    exported for retraining or counted toward streaks.

    A session fails quality check if:
    - less than 10 frames
    - less than 30% valid frames (with pose detected)
    - average form score below 15 (camera probably not pointed at athlete)
    - duration under 10 seconds
    """
    if session_id not in SESSION_DB:
        raise HTTPException(404, "session not found")

    session = SESSION_DB[session_id]
    summary = session.get("summary") or {}
    frames = session.get("frames") or []

    issues = []
    total_frames = len(frames)
    valid_frames = sum(1 for f in frames if f.get("form_score", 0) > 0)
    avg_score = float(summary.get("avg_form_score") or 0)
    duration = float(summary.get("duration_seconds") or 0)

    if total_frames < MIN_FRAMES:
        issues.append(f"only {total_frames} frames (minimum {MIN_FRAMES})")

    if total_frames > 0 and valid_frames / total_frames < MIN_VALID_RATIO:
        ratio = round(valid_frames / total_frames * 100, 0)
        issues.append(f"only {ratio}% valid frames (minimum {MIN_VALID_RATIO * 100:.0f}%)")

    if 0 < avg_score < MIN_AVG_SCORE:
        issues.append(f"avg form score {avg_score:.1f} is suspiciously low (minimum {MIN_AVG_SCORE})")
    elif avg_score == 0 and total_frames > 0:
        issues.append("no valid form scores detected — camera may not have captured the athlete")

    if duration > 0 and duration < MIN_DURATION_SECONDS:
        issues.append(f"session only lasted {duration:.0f}s (minimum {MIN_DURATION_SECONDS}s)")

    passed = len(issues) == 0
    grade = "good" if passed else "poor" if len(issues) >= 3 else "marginal"

    return {
        "session_id": session_id,
        "passed": passed,
        "grade": grade,
        "issues": issues,
        "stats": {
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "valid_ratio": round(valid_frames / max(total_frames, 1), 2),
            "avg_form_score": avg_score,
            "duration_seconds": duration,
        },
        "exportable": passed,
        "counts_for_streak": passed or grade == "marginal",
    }
