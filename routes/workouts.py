from __future__ import annotations

"""
Structured workouts and session templates.

VISION.md priorities: "The session flow — from open app to form score on
screen in under 60 seconds, reliably." and "Progressive load modelling."

Athletes need structure, not just freeform sessions. This module provides:

  GET  /workouts                          — list available workout templates
  GET  /workouts/{sport}                  — templates for a specific sport
  GET  /workouts/recommended/{athlete_id} — personalized workout based on recent data
  POST /workouts/custom                   — create a custom workout template

A workout template defines:
  - Sport, difficulty, estimated duration
  - Sequence of exercises with sets, reps, rest, and form cues
  - Target metrics (form score, rep count)
  - Progression rules (when to level up)
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from routes.progress import _athlete_sessions

router = APIRouter(prefix="/workouts", tags=["Workouts"])
log = get_logger("routes.workouts")


# ─── Built-in workout templates ───────────────────────────────────────────


TEMPLATES: dict[str, list[dict]] = {
    "vertical_jump": [
        {
            "id": "vj_beginner",
            "name": "Jump Foundations",
            "difficulty": "beginner",
            "duration_min": 15,
            "description": "build the base movement pattern for vertical jump. focus on knee tracking and landing mechanics",
            "exercises": [
                {
                    "name": "Bodyweight squat",
                    "sets": 3,
                    "reps": 10,
                    "rest_s": 60,
                    "cue": "knees track over toes, chest up, full depth",
                },
                {
                    "name": "Box step-up",
                    "sets": 3,
                    "reps": 8,
                    "rest_s": 60,
                    "cue": "drive through heel, control descent",
                },
                {
                    "name": "Countermovement jump",
                    "sets": 3,
                    "reps": 5,
                    "rest_s": 90,
                    "cue": "arm swing, soft landing, knee alignment",
                },
                {
                    "name": "Ankle mobilisation",
                    "sets": 2,
                    "reps": 30,
                    "rest_s": 30,
                    "cue": "wall drill, knee past toe, heel flat",
                },
            ],
            "target_form_score": 60,
            "progression": "move to intermediate when avg form score > 65 for 3 sessions",
        },
        {
            "id": "vj_intermediate",
            "name": "Power Development",
            "difficulty": "intermediate",
            "duration_min": 25,
            "description": "develop explosive power and landing stability",
            "exercises": [
                {
                    "name": "Goblet squat",
                    "sets": 3,
                    "reps": 8,
                    "rest_s": 90,
                    "cue": "pause at bottom, drive fast on the way up",
                },
                {
                    "name": "Box jump",
                    "sets": 4,
                    "reps": 5,
                    "rest_s": 120,
                    "cue": "step down dont jump down, soft landing",
                },
                {
                    "name": "Depth jump",
                    "sets": 3,
                    "reps": 3,
                    "rest_s": 120,
                    "cue": "minimal ground contact, immediate takeoff",
                },
                {
                    "name": "Single-leg RDL",
                    "sets": 3,
                    "reps": 8,
                    "rest_s": 60,
                    "cue": "flat back, hip hinge, control the descent",
                },
                {
                    "name": "Banded clamshell",
                    "sets": 2,
                    "reps": 15,
                    "rest_s": 45,
                    "cue": "glute activation, no rotation",
                },
            ],
            "target_form_score": 75,
            "progression": "move to advanced when avg form score > 78 and peak jump > 40cm",
        },
        {
            "id": "vj_advanced",
            "name": "Peak Performance",
            "difficulty": "advanced",
            "duration_min": 35,
            "description": "max effort vertical jump training with plyometrics and reactive strength",
            "exercises": [
                {
                    "name": "Back squat",
                    "sets": 4,
                    "reps": 5,
                    "rest_s": 180,
                    "cue": "75-85% 1RM, full depth, fast concentric",
                },
                {
                    "name": "Depth jump to max vertical",
                    "sets": 4,
                    "reps": 3,
                    "rest_s": 180,
                    "cue": "30cm box, minimize ground time, max height",
                },
                {
                    "name": "Weighted CMJ",
                    "sets": 3,
                    "reps": 5,
                    "rest_s": 120,
                    "cue": "light DB/vest, full arm swing allowed",
                },
                {
                    "name": "Altitude landing",
                    "sets": 3,
                    "reps": 4,
                    "rest_s": 90,
                    "cue": "drop from box, absorb with bent knees, freeze",
                },
                {
                    "name": "Hip flexor stretch",
                    "sets": 2,
                    "reps": 30,
                    "rest_s": 30,
                    "cue": "90-90 position, posterior tilt",
                },
            ],
            "target_form_score": 85,
            "progression": "maintain and compete. focus on peaking for events",
        },
    ],
    "sprint": [
        {
            "id": "sp_beginner",
            "name": "Sprint Mechanics",
            "difficulty": "beginner",
            "duration_min": 20,
            "description": "learn proper sprint mechanics. posture, arm action, foot strike",
            "exercises": [
                {
                    "name": "A-skip drill",
                    "sets": 3,
                    "reps": 30,
                    "rest_s": 60,
                    "cue": "high knee, toe up, quick ground contact",
                },
                {"name": "Wall drive", "sets": 4, "reps": 8, "rest_s": 60, "cue": "45 degree lean, full hip extension"},
                {
                    "name": "Falling start to 20m",
                    "sets": 4,
                    "reps": 1,
                    "rest_s": 120,
                    "cue": "lean until you have to step, then explode",
                },
                {"name": "Calf raise", "sets": 3, "reps": 15, "rest_s": 45, "cue": "full range, slow eccentric"},
            ],
            "target_form_score": 60,
            "progression": "move to intermediate when stride pattern is consistent for 3 sessions",
        },
        {
            "id": "sp_intermediate",
            "name": "Speed Endurance",
            "difficulty": "intermediate",
            "duration_min": 30,
            "description": "build speed endurance and maintain form under fatigue",
            "exercises": [
                {
                    "name": "Wicket runs",
                    "sets": 4,
                    "reps": 40,
                    "rest_s": 180,
                    "cue": "maintain stride length over mini hurdles",
                },
                {
                    "name": "Flying 30m sprint",
                    "sets": 3,
                    "reps": 1,
                    "rest_s": 180,
                    "cue": "build up zone then hold max form",
                },
                {
                    "name": "Resisted sprint (sled/band)",
                    "sets": 3,
                    "reps": 20,
                    "rest_s": 180,
                    "cue": "45 degree trunk, drive through ground",
                },
                {
                    "name": "Nordic hamstring curl",
                    "sets": 3,
                    "reps": 5,
                    "rest_s": 90,
                    "cue": "slow eccentric, catch yourself at the bottom",
                },
            ],
            "target_form_score": 72,
            "progression": "move to advanced when 30m time drops below personal target",
        },
    ],
    "squat": [
        {
            "id": "sq_beginner",
            "name": "Squat Fundamentals",
            "difficulty": "beginner",
            "duration_min": 15,
            "description": "master the squat pattern before adding load",
            "exercises": [
                {
                    "name": "Goblet squat",
                    "sets": 3,
                    "reps": 10,
                    "rest_s": 60,
                    "cue": "elbows inside knees, upright torso",
                },
                {
                    "name": "Wall squat hold",
                    "sets": 3,
                    "reps": 30,
                    "rest_s": 45,
                    "cue": "back flat on wall, thighs parallel",
                },
                {
                    "name": "Ankle mobility drill",
                    "sets": 2,
                    "reps": 10,
                    "rest_s": 30,
                    "cue": "banded distraction, knee over toe",
                },
            ],
            "target_form_score": 65,
            "progression": "move to intermediate when depth and knee tracking are consistent",
        },
    ],
    "push_up": [
        {
            "id": "pu_beginner",
            "name": "Push-Up Progression",
            "difficulty": "beginner",
            "duration_min": 12,
            "description": "build push-up strength with proper form",
            "exercises": [
                {
                    "name": "Incline push-up",
                    "sets": 3,
                    "reps": 10,
                    "rest_s": 60,
                    "cue": "hands on bench, full range, body straight",
                },
                {
                    "name": "Negative push-up",
                    "sets": 3,
                    "reps": 5,
                    "rest_s": 60,
                    "cue": "5 second lowering, reset at top",
                },
                {"name": "Plank hold", "sets": 3, "reps": 30, "rest_s": 45, "cue": "neutral spine, squeeze everything"},
            ],
            "target_form_score": 60,
            "progression": "move to intermediate when you can do 3x10 full push-ups with good form",
        },
    ],
    "snatch": [
        {
            "id": "sn_beginner",
            "name": "Snatch Skill Work",
            "difficulty": "beginner",
            "duration_min": 20,
            "description": "learn the movement pattern with light load",
            "exercises": [
                {
                    "name": "Overhead squat (PVC)",
                    "sets": 3,
                    "reps": 8,
                    "rest_s": 60,
                    "cue": "bar behind ears, heels flat, full depth",
                },
                {
                    "name": "Snatch-grip RDL",
                    "sets": 3,
                    "reps": 8,
                    "rest_s": 60,
                    "cue": "wide grip, bar close to shins, hamstrings loaded",
                },
                {
                    "name": "Muscle snatch from hang",
                    "sets": 5,
                    "reps": 3,
                    "rest_s": 90,
                    "cue": "fast elbows, punch up, pause at catch",
                },
                {"name": "Overhead hold", "sets": 3, "reps": 20, "rest_s": 45, "cue": "active shoulders, ribs down"},
            ],
            "target_form_score": 55,
            "progression": "add load when overhead position is stable for 3 sessions",
        },
    ],
}


@router.get("")
async def list_workouts(sport: str | None = Query(default=None)):
    """List all available workout templates, optionally filtered by sport."""
    if sport:
        templates = TEMPLATES.get(sport, [])
        return {"sport": sport, "workouts": templates, "count": len(templates)}

    all_workouts = []
    for s, templates in TEMPLATES.items():
        for t in templates:
            all_workouts.append({**t, "sport": s})
    return {"workouts": all_workouts, "count": len(all_workouts)}


@router.get("/{sport}")
async def sport_workouts(sport: str):
    """Get workout templates for a specific sport."""
    templates = TEMPLATES.get(sport)
    if templates is None:
        available = sorted(TEMPLATES.keys())
        raise HTTPException(404, f"no templates for '{sport}'. available: {', '.join(available)}")
    return {"sport": sport, "workouts": templates, "count": len(templates)}


@router.get("/recommended/{athlete_id}")
async def recommended_workout(athlete_id: str):
    """
    Pick the right workout for this athlete based on their current level.

    Looks at recent form scores to determine difficulty, then picks the
    matching template for their sport.
    """
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    sport = athlete.get("sport", "vertical_jump")
    templates = TEMPLATES.get(sport, [])

    if not templates:
        return {
            "athlete_id": athlete_id,
            "sport": sport,
            "workout": None,
            "reason": f"no templates available for {sport} yet",
        }

    # determine difficulty from recent form scores
    sessions = _athlete_sessions(athlete_id, 14)
    form_scores = []
    for s in sessions:
        sm = s.get("summary") or {}
        af = float(sm.get("avg_form_score") or 0)
        if af > 0:
            form_scores.append(af)

    avg_form = sum(form_scores) / len(form_scores) if form_scores else 0

    # pick difficulty
    if avg_form >= 78 or len(form_scores) > 20:
        target_diff = "advanced"
    elif avg_form >= 60 or len(form_scores) > 5:
        target_diff = "intermediate"
    else:
        target_diff = "beginner"

    # find matching template
    workout = next((t for t in templates if t["difficulty"] == target_diff), None)
    if not workout:
        workout = templates[-1]  # fallback to hardest available

    return {
        "athlete_id": athlete_id,
        "sport": sport,
        "recommended_difficulty": target_diff,
        "avg_form_score": round(avg_form, 1),
        "recent_sessions": len(form_scores),
        "workout": workout,
    }


class CustomWorkout(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    sport: str
    difficulty: str = "custom"
    duration_min: int = Field(ge=5, le=120)
    description: str = ""
    exercises: list[dict] = Field(min_length=1)


@router.post("/custom")
async def create_custom_workout(workout: CustomWorkout):
    """Create a custom workout template. Stored in db/custom_workouts.json."""
    custom = _load_json("custom_workouts.json")
    wid = f"custom_{len(custom) + 1}"

    entry = {
        "id": wid,
        **workout.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    custom[wid] = entry
    _save_json("custom_workouts.json", custom)
    log.info("custom workout created", extra={"id": wid, "sport": workout.sport})

    return {"status": "created", "workout": entry}
