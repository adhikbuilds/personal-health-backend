from __future__ import annotations

"""Nutrition domain — goal setting and food APIs."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from database import _load_json, _save_json, ATHLETE_DB, FOOD_DB
from logging_setup import get_logger

log = get_logger("routes.nutrition")

router = APIRouter(tags=["nutrition"])


# ================== GOALS API ==================

class NutritionGoals(BaseModel):
    daily_calories: float = Field(gt=0)
    protein_g: float = Field(gt=0)
    carbs_g: float = Field(gt=0)
    fat_g: float = Field(gt=0)
    fiber_g: float = Field(gt=0)


def get_default_goals(sport: str):
    if sport in ["vertical_jump", "sprint", "javelin"]:
        return {
            "daily_calories": 2600,
            "protein_g": 130,
            "carbs_g": 320,
            "fat_g": 70,
            "fiber_g": 30,
        }
    elif sport in ["squat", "push_up", "pull_up"]:
        return {
            "daily_calories": 2800,
            "protein_g": 150,
            "carbs_g": 280,
            "fat_g": 80,
            "fiber_g": 30,
        }
    elif sport == "cricket_bat":
        return {
            "daily_calories": 2400,
            "protein_g": 110,
            "carbs_g": 310,
            "fat_g": 65,
            "fiber_g": 30,
        }
    else:
        return {
            "daily_calories": 2400,
            "protein_g": 120,
            "carbs_g": 300,
            "fat_g": 60,
            "fiber_g": 30,
        }


@router.post("/athlete/{athlete_id}/nutrition/goals")
async def set_goals(athlete_id: str, payload: NutritionGoals):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")

    data = _load_json("nutrition.json") or {}

    if athlete_id not in data:
        data[athlete_id] = {}

    data[athlete_id]["goals"] = payload.model_dump()
    data[athlete_id]["goals"]["updated_at"] = datetime.now(timezone.utc).isoformat()

    _save_json("nutrition.json", data)

    log.info("nutrition goals updated", extra={"athlete_id": athlete_id})

    return {"status": "success", "data": data[athlete_id]["goals"]}


@router.get("/athlete/{athlete_id}/nutrition/goals")
async def get_goals(athlete_id: str, sport: str = "sprint"):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")

    data = _load_json("nutrition.json") or {}

    if athlete_id in data and "goals" in data[athlete_id]:
        return {"status": "success", "data": data[athlete_id]["goals"]}

    default = get_default_goals(sport)
    default["updated_at"] = datetime.now(timezone.utc).isoformat()

    return {"status": "default", "data": default}


# ================== FOOD API ==================

@router.get("/foods")
async def list_foods(
    category: str | None = Query(None),
    cuisine: str | None = Query(None),
    tag: str | None = Query(None),
    q: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    results = list(FOOD_DB.values())
    if category:
        results = [f for f in results if f["category"] == category]
    if cuisine:
        results = [f for f in results if f["cuisine"] == cuisine]
    if tag:
        results = [f for f in results if tag in f.get("tags", [])]
    if q:
        q_lower = q.lower()
        results = [f for f in results if q_lower in f["name"].lower()]

    total = len(results)
    results = results[offset : offset + limit]

    return {"count": total, "limit": limit, "offset": offset, "foods": results}


@router.get("/foods/{food_id}")
async def get_food(food_id: str):
    food = FOOD_DB.get(food_id)
    if not food:
        raise HTTPException(404, detail=f"Food '{food_id}' not found")
    return food