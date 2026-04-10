

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from database import FOOD_DB, ATHLETE_DB, _load_json

from datetime import datetime, date


router = APIRouter(prefix="/foods", tags=["Nutrition"])


@router.get("/")
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


@router.get("/{food_id}")
async def get_food(food_id: str):
    food = FOOD_DB.get(food_id)
    if not food:
        raise HTTPException(404, detail=f"Food '{food_id}' not found")
    return food

nutrition_router = APIRouter(tags=["Nutrition"])

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


@nutrition_router.get("/athlete/{athlete_id}/nutrition/summary")
async def get_nutrition_summary(
    athlete_id: str,
    on_date: date | None = Query(default=None),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")

    data = _load_json("nutrition.json") or {}

    if on_date is None:
        on_date = datetime.utcnow().date()

    if on_date > datetime.utcnow().date():
        raise HTTPException(status_code=400, detail="Date cannot be in the future")

    date_str = on_date.strftime("%Y-%m-%d")

    athlete_data = data.get(athlete_id, {})
    logs = athlete_data.get("logs", {})
    day_log = logs.get(date_str, {})

    total_calories = 0
    protein = 0
    carbs = 0
    fat = 0
    fiber = 0

    slots = {}

    for slot, slot_data in day_log.items():
        if not isinstance(slot_data, dict):
            continue

        total = slot_data.get("total", {}) or {}

        cals = total.get("calories", 0)
        prot = total.get("protein_g", 0)
        carb = total.get("carbs_g", 0)
        fat_g = total.get("fat_g", 0)
        fib = total.get("fiber_g", 0)

        total_calories += cals
        protein += prot
        carbs += carb
        fat += fat_g
        fiber += fib

        slots[slot] = {
            "calories": cals,
            "protein_g": prot,
            "carbs_g": carb,
            "fat_g": fat_g,
            "fiber_g": fib,

        }

    meals_logged = len(day_log)

    raw_goals = athlete_data.get("goals", {})

    if not raw_goals:
        sport = ATHLETE_DB[athlete_id].get("sport", "vertical_jump")
        raw_goals = get_default_goals(sport)

    athlete_goals = {
        "calories": raw_goals.get("daily_calories", 0),
        "protein_g": raw_goals.get("protein_g", 0),
        "carbs_g": raw_goals.get("carbs_g", 0),
        "fat_g": raw_goals.get("fat_g", 0),
        "fiber_g": raw_goals.get("fiber_g", 0),
    }

    pct_of_goal = {
        "calories": round(
            (total_calories / max(athlete_goals["calories"], 1)) * 100
        ),
        "protein_g": round(
            (protein / max(athlete_goals["protein_g"], 1)) * 100
        ),
        "carbs_g": round(
            (carbs / max(athlete_goals["carbs_g"], 1)) * 100
        ),
        "fat_g": round(
            (fat / max(athlete_goals["fat_g"], 1)) * 100
        ),
        "fiber_g": round(
            (fiber / max(athlete_goals["fiber_g"], 1)) * 100
        ),
    }

    return {
        "date": date_str,
        "total_calories": total_calories,
        "protein_g": protein,
        "carbs_g": carbs,
        "fat_g": fat,
        "fiber_g": fiber,
        "meals_logged": meals_logged,
        "slots": slots,

        "goals": athlete_goals,
        "pct_of_goal": pct_of_goal,
    }


