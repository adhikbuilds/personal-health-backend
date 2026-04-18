from __future__ import annotations

"""
Nutrition domain — consolidated module.

Endpoints:
  GET  /foods                              list/filter foods           (PR #21)
  GET  /foods/{food_id}                    single food                 (PR #21)
  POST /athlete/{id}/nutrition/goals       set macro goals             (PR #6,  WN-06)
  GET  /athlete/{id}/nutrition/goals       read goals (with default)   (PR #6,  WN-06)
  GET  /athlete/{id}/nutrition/summary     daily totals + pct-of-goal  (PR #9,  WN-05)
  POST /athlete/{id}/meals                 log a meal for a slot       (PR #17, WN-03)
  GET  /athlete/{id}/meals                 read meals for a date       (PR #17, WN-03)

Two routers are exported:
  router          — /foods + /athlete/{id}/nutrition/goals + /athlete/{id}/meals
  nutrition_router — /athlete/{id}/nutrition/summary (separate namespace)

Storage shape (db/nutrition.json):
{
  "athlete_id": {
    "goals": { "daily_calories": ..., "protein_g": ..., "updated_at": "..." },
    "logs": {
      "YYYY-MM-DD": {
        "breakfast"|"lunch"|"dinner"|"snack": {
          "meal_id": "uuid",
          "time": "HH:MM" | None,
          "items": [{"food_id", "food_name", "qty_g", "macros": {...}}],
          "total": {"calories", "protein_g", "carbs_g", "fat_g", "fiber_g"}
        }
      }
    }
  }
}
"""

import asyncio
import re
import uuid
from datetime import date as Date
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from auth import require_athlete_or_admin
from database import ATHLETE_DB, FOOD_DB, _load_json, _save_json
from logging_setup import get_logger

log = get_logger("routes.nutrition")

router = APIRouter(tags=["nutrition"])
nutrition_router = APIRouter(tags=["Nutrition"])


# ─── Shared helpers ──────────────────────────────────────────────────────────


def get_default_goals(sport: str) -> dict:
    if sport in ["vertical_jump", "sprint", "javelin"]:
        return {"daily_calories": 2600, "protein_g": 130, "carbs_g": 320, "fat_g": 70, "fiber_g": 30}
    if sport in ["squat", "push_up", "pull_up"]:
        return {"daily_calories": 2800, "protein_g": 150, "carbs_g": 280, "fat_g": 80, "fiber_g": 30}
    if sport == "cricket_bat":
        return {"daily_calories": 2400, "protein_g": 110, "carbs_g": 310, "fat_g": 65, "fiber_g": 30}
    return {"daily_calories": 2400, "protein_g": 120, "carbs_g": 300, "fat_g": 60, "fiber_g": 30}


VALID_SLOTS = {"breakfast", "lunch", "dinner", "snack"}

_NUTRITION_LOCKS: dict[str, asyncio.Lock] = {}


def _get_lock(athlete_id: str) -> asyncio.Lock:
    if athlete_id not in _NUTRITION_LOCKS:
        _NUTRITION_LOCKS[athlete_id] = asyncio.Lock()
    return _NUTRITION_LOCKS[athlete_id]


def _compute_macros(food: dict, qty_g: float) -> dict:
    per100 = food["per_100g"]
    factor = qty_g / 100.0
    return {
        "calories": round(per100["calories"] * factor, 2),
        "protein_g": round(per100["protein_g"] * factor, 2),
        "carbs_g": round(per100["carbs_g"] * factor, 2),
        "fat_g": round(per100["fat_g"] * factor, 2),
        "fiber_g": round(per100["fiber_g"] * factor, 2),
    }


def _sum_macros(items: list[dict]) -> dict:
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    for item in items:
        for key in totals:
            totals[key] += item.get("macros", {}).get(key, 0.0)
    return {k: round(v, 2) for k, v in totals.items()}


# ─── Pydantic models ─────────────────────────────────────────────────────────


class NutritionGoals(BaseModel):
    daily_calories: float = Field(gt=0)
    protein_g: float = Field(gt=0)
    carbs_g: float = Field(gt=0)
    fat_g: float = Field(gt=0)
    fiber_g: float = Field(gt=0)


class MealItem(BaseModel):
    food_id: str
    qty_g: float = Field(..., gt=0, description="Quantity in grams")


class LogMealRequest(BaseModel):
    date: Date = Field(..., description="ISO 8601 date, e.g. 2026-04-18")
    slot: Literal["breakfast", "lunch", "dinner", "snack"] = Field(...)
    time: Optional[str] = Field(None, description="Optional HH:MM")
    items: List[MealItem] = Field(..., min_length=1)

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v):
        if v is None:
            return v
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("time must be HH:MM")
        hh, mm = int(v[:2]), int(v[3:])
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError("time must be a valid 24h time")
        return v


# ─── Goals endpoints (PR #6, WN-06) ──────────────────────────────────────────


@router.post("/athlete/{athlete_id}/nutrition/goals")
async def set_goals(
    athlete_id: str,
    payload: NutritionGoals,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")
    async with _get_lock(athlete_id):
        data = _load_json("nutrition.json") or {}
        data.setdefault(athlete_id, {})
        data[athlete_id]["goals"] = payload.model_dump()
        data[athlete_id]["goals"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_json("nutrition.json", data)
    log.info("nutrition goals updated", extra={"athlete_id": athlete_id})
    return {"status": "success", "data": data[athlete_id]["goals"]}


@router.get("/athlete/{athlete_id}/nutrition/goals")
async def get_goals(
    athlete_id: str,
    sport: str | None = None,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")
    data = _load_json("nutrition.json") or {}
    if athlete_id in data and "goals" in data[athlete_id]:
        return {"status": "success", "data": data[athlete_id]["goals"]}
    chosen_sport = sport or ATHLETE_DB[athlete_id].get("sport", "sprint")
    default = get_default_goals(chosen_sport)
    default["updated_at"] = datetime.now(timezone.utc).isoformat()
    return {"status": "default", "data": default}


# ─── Foods endpoints (PR #21) ────────────────────────────────────────────────


@router.get("/foods")
async def list_foods(
    category: str | None = Query(None),
    cuisine: str | None = Query(None),
    tag: str | None = Query(None),
    q: str | None = Query(None),
    search: str | None = Query(None, alias="search"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    results = list(FOOD_DB.values())
    if category:
        results = [f for f in results if f.get("category") == category]
    if cuisine:
        results = [f for f in results if f.get("cuisine") == cuisine]
    if tag:
        results = [f for f in results if tag in f.get("tags", [])]
    effective_q = q or search
    if effective_q:
        ql = effective_q.lower().strip()
        tokens = ql.split()

        def _fuzzy_match(name: str) -> bool:
            nl = name.lower()
            if all(t in nl for t in tokens):
                return True
            it = iter(nl)
            matched = sum(1 for ch in ql if ch in it)
            return matched >= max(1, int(len(ql) * 0.7))

        results = [f for f in results if _fuzzy_match(f.get("name", ""))]
    total = len(results)
    results = results[offset : offset + limit]
    return {"count": total, "limit": limit, "offset": offset, "foods": results}


@router.get("/foods/{food_id}")
async def get_food(food_id: str):
    food = FOOD_DB.get(food_id)
    if not food:
        raise HTTPException(404, detail=f"Food '{food_id}' not found")
    return food


# ─── Meals endpoints (PR #17, WN-03) ─────────────────────────────────────────


@router.post("/athlete/{athlete_id}/meals", status_code=201)
async def log_meal(
    athlete_id: str,
    body: LogMealRequest,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """
    Log a meal for an athlete. Posting to the same slot on the same date
    overwrites the previous entry (idempotent upsert).
    """
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")

    logged_items = []
    for item in body.items:
        food = FOOD_DB.get(item.food_id)
        if not food:
            raise HTTPException(status_code=404, detail=f"Food not found: {item.food_id}")
        logged_items.append(
            {
                "food_id": item.food_id,
                "food_name": food["name"],
                "qty_g": item.qty_g,
                "macros": _compute_macros(food, item.qty_g),
            }
        )

    slot_total = _sum_macros(logged_items)
    meal_id = str(uuid.uuid4())
    date_str = body.date.isoformat()
    meal_record = {"meal_id": meal_id, "time": body.time, "items": logged_items, "total": slot_total}

    async with _get_lock(athlete_id):
        nutrition_db = _load_json("nutrition.json") or {}
        nutrition_db.setdefault(athlete_id, {"goals": {}, "logs": {}})
        nutrition_db[athlete_id].setdefault("goals", {})
        nutrition_db[athlete_id].setdefault("logs", {})
        nutrition_db[athlete_id]["logs"].setdefault(date_str, {})
        nutrition_db[athlete_id]["logs"][date_str][body.slot] = meal_record
        _save_json("nutrition.json", nutrition_db)

    return {
        "meal_id": meal_id,
        "athlete_id": athlete_id,
        "date": date_str,
        "slot": body.slot,
        "items": logged_items,
        "total": slot_total,
    }


@router.get("/athlete/{athlete_id}/meals")
async def get_meals(
    athlete_id: str,
    date: Date,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")

    date_str = date.isoformat()
    nutrition_db = _load_json("nutrition.json") or {}
    athlete_logs = nutrition_db.get(athlete_id, {}).get("logs", {})
    date_logs = athlete_logs.get(date_str, {})

    grouped = {slot: date_logs[slot] for slot in VALID_SLOTS if slot in date_logs}

    all_items: list[dict] = []
    for slot_data in grouped.values():
        all_items.extend(slot_data.get("items", []))
    day_total = _sum_macros(all_items)

    return {"athlete_id": athlete_id, "date": date_str, "meals": grouped, "day_total": day_total}


@router.delete("/athlete/{athlete_id}/meals/{date_str}/{slot}", status_code=200)
async def delete_meal(
    athlete_id: str,
    date_str: str,
    slot: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """Remove a meal slot entry for an athlete on a given date. (WN-04)"""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")
    if slot not in VALID_SLOTS:
        raise HTTPException(status_code=400, detail=f"Invalid slot '{slot}'. Must be one of: {sorted(VALID_SLOTS)}")
    async with _get_lock(athlete_id):
        nutrition_db = _load_json("nutrition.json") or {}
        day_log = nutrition_db.get(athlete_id, {}).get("logs", {}).get(date_str, {})
        if slot not in day_log:
            raise HTTPException(status_code=404, detail=f"No meal found for slot '{slot}' on {date_str}")
        del nutrition_db[athlete_id]["logs"][date_str][slot]
        _save_json("nutrition.json", nutrition_db)
    log.info("meal deleted athlete=%s date=%s slot=%s", athlete_id, date_str, slot)
    return {"deleted": True, "athlete_id": athlete_id, "date": date_str, "slot": slot}


@router.get("/nutrition/team-summary")
async def get_team_nutrition_summary():
    """Return today's nutrition compliance for all athletes. (WN-27)"""
    today = datetime.now(timezone.utc).date().isoformat()
    data = _load_json("nutrition.json") or {}
    results = []
    for athlete_id, athlete in ATHLETE_DB.items():
        athlete_data = data.get(athlete_id, {})
        day_log = athlete_data.get("logs", {}).get(today, {})
        raw_goals = athlete_data.get("goals") or {}
        if not raw_goals:
            sport = athlete.get("sport", "sprint")
            raw_goals = get_default_goals(sport)
        goal_cals = raw_goals.get("daily_calories", 2400)
        goal_prot = raw_goals.get("protein_g", 120)
        total_cals = 0.0
        total_prot = 0.0
        for slot_data in day_log.values():
            if isinstance(slot_data, dict):
                t = slot_data.get("total", {})
                total_cals += t.get("calories", 0) or 0
                total_prot += t.get("protein_g", 0) or 0
        cal_pct = round((total_cals / max(goal_cals, 1)) * 100)
        prot_pct = round((total_prot / max(goal_prot, 1)) * 100)
        results.append({
            "id": athlete_id,
            "name": athlete.get("name", "Unknown"),
            "calories_pct": cal_pct,
            "protein_pct": prot_pct,
            "meals_logged": len(day_log),
            "flag": "red" if cal_pct < 40 else ("yellow" if cal_pct < 60 else "ok"),
        })
    team_avg = round(sum(r["calories_pct"] for r in results) / len(results)) if results else 0
    return {"date": today, "athletes": results, "team_avg_calories_pct": team_avg}


# ─── Summary endpoint (PR #9, WN-05) ─────────────────────────────────────────


@nutrition_router.get("/athlete/{athlete_id}/nutrition/summary")
async def get_nutrition_summary(
    athlete_id: str,
    on_date: Date | None = Query(default=None),
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail="Athlete not found")

    today = datetime.now(timezone.utc).date()
    if on_date is None:
        on_date = today
    if on_date > today:
        raise HTTPException(status_code=400, detail="Date cannot be in the future")

    data = _load_json("nutrition.json") or {}
    date_str = on_date.strftime("%Y-%m-%d")
    athlete_data = data.get(athlete_id, {})
    day_log = (athlete_data.get("logs") or {}).get(date_str, {})

    total_calories = 0.0
    protein = 0.0
    carbs = 0.0
    fat = 0.0
    fiber = 0.0
    slots: dict[str, dict] = {}

    for slot, slot_data in day_log.items():
        if not isinstance(slot_data, dict):
            continue
        total = slot_data.get("total") or {}
        cals = total.get("calories", 0) or 0
        prot = total.get("protein_g", 0) or 0
        carb = total.get("carbs_g", 0) or 0
        fat_g = total.get("fat_g", 0) or 0
        fib = total.get("fiber_g", 0) or 0
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

    raw_goals = athlete_data.get("goals") or {}
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
        "calories": round((total_calories / max(athlete_goals["calories"], 1)) * 100),
        "protein_g": round((protein / max(athlete_goals["protein_g"], 1)) * 100),
        "carbs_g": round((carbs / max(athlete_goals["carbs_g"], 1)) * 100),
        "fat_g": round((fat / max(athlete_goals["fat_g"], 1)) * 100),
        "fiber_g": round((fiber / max(athlete_goals["fiber_g"], 1)) * 100),
    }

    return {
        "date": date_str,
        "athlete_id": athlete_id,
        "calories": total_calories,
        "protein_g": protein,
        "carbs_g": carbs,
        "fat_g": fat,
        "fiber_g": fiber,
        "meals_logged": len(day_log),
        "slots": slots,
        "goals": athlete_goals,
        "pct_of_goal": pct_of_goal,
    }
