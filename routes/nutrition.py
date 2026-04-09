from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from database import _load_json, _save_json

router = APIRouter(
    prefix="/athlete/{athlete_id}/nutrition",
    tags=["nutrition"]
)

def get_default_goals(sport: str):
    if sport in ["vertical_jump", "sprint", "javelin"]:
        return {"daily_calories": 2600, "protein_g": 130, "carbs_g": 320, "fat_g": 70}
    elif sport in ["squat", "push_up", "pull_up"]:
        return {"daily_calories": 2800, "protein_g": 150, "carbs_g": 280, "fat_g": 80}
    elif sport == "cricket_bat":
        return {"daily_calories": 2400, "protein_g": 110, "carbs_g": 310, "fat_g": 65}
    else:
        return {"daily_calories": 2400, "protein_g": 120, "carbs_g": 300, "fat_g": 60}


@router.post("/goals")
def set_goals(athlete_id: str, payload: dict):
    data = _load_json("nutrition.json") or {}

    # validation
    for key in ["daily_calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]:
        if key not in payload or not isinstance(payload[key], int) or payload[key] <= 0:
            raise HTTPException(status_code=400, detail="All values must be positive integers")

    if athlete_id not in data:
        data[athlete_id] = {}

    data[athlete_id]["goals"] = payload
    data[athlete_id]["goals"]["updated_at"] = datetime.now(timezone.utc).isoformat()

    _save_json("nutrition.json", data)

    return {"status": "success", "data": data[athlete_id]["goals"]}


@router.get("/goals")
def get_goals(athlete_id: str, sport: str = "sprint"):
    data = _load_json("nutrition.json") or {}

    if athlete_id in data and "goals" in data[athlete_id]:
        return {"status": "success", "data": data[athlete_id]["goals"]}

    default = get_default_goals(sport)
    default["fiber_g"] = 30
    default["updated_at"] = datetime.now(timezone.utc).isoformat()

    return {"status": "default", "data": default}