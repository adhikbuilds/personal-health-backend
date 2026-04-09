from __future__ import annotations

"""
Wellness domain — Morning check-in and composite wellness score
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database import ATHLETE_DB, _save_db

router = APIRouter()

# Max points per component — designed so total max = 100
_MAX: dict[str, int] = {
    "sleep": 30,
    "hydration": 15,
    "mood": 15,
    "energy": 20,
    "stress": 10,
    "soreness": 10,
}

_RECOVERY_READY_THRESHOLD = 60
_HYDRATION_TARGET_GLASSES = 8


class WellnessCheckin(BaseModel):
    sleep_hours: float = Field(default=7.0, ge=0, le=24)
    water_glasses: int = Field(default=6, ge=0)
    mood: int = Field(default=7, ge=1, le=10)      # 1–10, higher = better
    energy: int = Field(default=7, ge=1, le=10)    # 1–10, higher = better
    stress: int = Field(default=3, ge=1, le=10)    # 1–10, higher = more stressed
    soreness: int = Field(default=3, ge=1, le=10)  # 1–10, higher = more sore
    date: str = ""

    model_config = {
        "json_schema_extra": {
            "example": {
                "sleep_hours": 6.5,
                "water_glasses": 5,
                "mood": 7,
                "energy": 6,
                "stress": 4,
                "soreness": 3,
            }
        }
    }


def _compute_breakdown(entry: dict) -> dict[str, int]:
    """Compute per-component wellness scores from a raw log entry."""
    sleep_hours = float(entry.get("sleep_hours", 0))
    water_glasses = int(entry.get("water_glasses", 0))
    mood = max(1, min(10, int(entry.get("mood", 5))))
    energy = max(1, min(10, int(entry.get("energy", 5))))
    stress = max(1, min(10, int(entry.get("stress", 5))))
    soreness = max(1, min(10, int(entry.get("soreness", 5))))

    sleep_score = round(min(30.0, (min(sleep_hours, 8.0) / 8.0) * 30))
    hydration_score = round(
        min(15.0, (min(water_glasses, _HYDRATION_TARGET_GLASSES) / _HYDRATION_TARGET_GLASSES) * 15)
    )
    mood_score = round((mood / 10) * 15)
    energy_score = round((energy / 10) * 20)
    # Stress and soreness are inverted: high input → low score
    stress_score = round(((10 - stress) / 9) * 10)
    soreness_score = round(((10 - soreness) / 9) * 10)

    return {
        "sleep": sleep_score,
        "hydration": hydration_score,
        "mood": mood_score,
        "energy": energy_score,
        "stress": stress_score,
        "soreness": soreness_score,
    }


def _generate_recommendation(breakdown: dict[str, int], entry: dict) -> str:
    """
    Find the component with the lowest relative contribution and return
    the matching coaching message if it crosses its actionable threshold.
    Falls through to check all thresholds before returning the default.
    """
    relative = {k: breakdown[k] / _MAX[k] for k in _MAX}
    worst = min(relative, key=relative.get)

    stress_input = int(entry.get("stress", 5))
    soreness_input = int(entry.get("soreness", 5))
    target_ml = _HYDRATION_TARGET_GLASSES * 250

    rules: dict[str, tuple[bool, str]] = {
        "sleep": (
            breakdown["sleep"] < 18,
            "Your sleep score is low. Try to get 7-8 hours tonight.",
        ),
        "hydration": (
            breakdown["hydration"] < 10,
            f"You're under-hydrated. Aim for {target_ml}ml today.",
        ),
        "stress": (
            stress_input > 7,
            "High stress detected. Consider a lighter session or active recovery.",
        ),
        "soreness": (
            soreness_input > 7,
            "High soreness. Rest day recommended to prevent injury.",
        ),
    }

    # Check worst component first; fall through to remaining components
    priority = [worst] + [k for k in rules if k != worst]
    for component in priority:
        if component in rules:
            condition, message = rules[component]
            if condition:
                return message

    return "Looking good! You're ready for a solid session."


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/athlete/{athlete_id}/wellness/checkin", tags=["Wellness"])
async def log_wellness_checkin(athlete_id: str, data: WellnessCheckin):
    """Log a morning wellness check-in for an athlete."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    if "wellness_log" not in athlete:
        athlete["wellness_log"] = {}

    date_key = data.date or datetime.utcnow().date().isoformat()
    entry = {
        "sleep_hours": data.sleep_hours,
        "water_glasses": data.water_glasses,
        "mood": data.mood,
        "energy": data.energy,
        "stress": data.stress,
        "soreness": data.soreness,
        "logged_at": datetime.utcnow().isoformat() + "Z",
    }

    athlete["wellness_log"][date_key] = entry
    _save_db()

    breakdown = _compute_breakdown(entry)
    wellness_score = sum(breakdown.values())

    return {
        "athlete_id": athlete_id,
        "date": date_key,
        "logged": True,
        "wellness_score": wellness_score,
        "breakdown": breakdown,
    }


@router.get("/athlete/{athlete_id}/wellness/score", tags=["Wellness"])
async def get_wellness_score(athlete_id: str):
    """Return today's composite wellness score (0–100)."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    wellness_log: dict = athlete.get("wellness_log", {})

    today = datetime.utcnow().date().isoformat()
    yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()

    if today in wellness_log:
        date_used, entry = today, wellness_log[today]
    elif yesterday in wellness_log:
        date_used, entry = yesterday, wellness_log[yesterday]
    else:
        return {
            "wellness_score": None,
            "message": "No recent wellness data. Log your morning check-in!",
        }

    breakdown = _compute_breakdown(entry)
    wellness_score = sum(breakdown.values())
    recovery_ready = wellness_score >= _RECOVERY_READY_THRESHOLD
    recommendation = _generate_recommendation(breakdown, entry)

    return {
        "date": date_used,
        "wellness_score": wellness_score,
        "recovery_ready": recovery_ready,
        "breakdown": breakdown,
        "recommendation": recommendation,
    }
