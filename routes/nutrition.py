from __future__ import annotations

"""
Nutrition Routes — PR #21 + WN-03
GET  /foods               → list all foods (PR #21)
GET  /foods/{food_id}     → get single food (PR #21)
POST /athlete/{id}/meals  → log a meal for a slot (WN-03)
GET  /athlete/{id}/meals  → read meals for a date grouped by slot (WN-03)

Storage shape (db/nutrition.json):
{
  "athlete_id": {
    "goals": { ... },        <- WN-06 (Navin)
    "logs": {
      "date": {
        "slot": {
          "meal_id": "...",
          "time": "HH:MM",
          "items": [{"food_id": "...", "food_name": "...", "qty_g": 100,
                     "macros": {"calories": 0, "protein_g": 0,
                                "carbs_g": 0, "fat_g": 0, "fiber_g": 0}}],
          "total": {"calories": 0, "protein_g": 0,
                    "carbs_g": 0, "fat_g": 0, "fiber_g": 0}
        }
      }
    }                        <- WN-03 (Khadija) + WN-05 (Soumya)
  }
}
"""

import asyncio
import re
import uuid
from datetime import date as Date
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from database import ATHLETE_DB, FOOD_DB, _load_json, _save_json

router = APIRouter(tags=["Nutrition"])

# ─── Valid meal slots ─────────────────────────────────────────────────────────

VALID_SLOTS = {"breakfast", "lunch", "dinner", "snack"}

# ─── Per-athlete write locks ──────────────────────────────────────────────────

_NUTRITION_LOCKS: dict[str, asyncio.Lock] = {}


def _get_lock(athlete_id: str) -> asyncio.Lock:
    if athlete_id not in _NUTRITION_LOCKS:
        _NUTRITION_LOCKS[athlete_id] = asyncio.Lock()
    return _NUTRITION_LOCKS[athlete_id]


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class MealItem(BaseModel):
    food_id: str
    qty_g: float = Field(..., gt=0, description="Quantity in grams, must be > 0")


class LogMealRequest(BaseModel):
    date: Date = Field(
        ...,
        json_schema_extra={"example": "2026-04-07"},
        description="ISO 8601 date string e.g. 2026-04-07",
    )
    slot: Literal["breakfast", "lunch", "dinner", "snack"] = Field(
        ...,
        description="One of: breakfast, lunch, dinner, snack",
    )
    time: Optional[str] = Field(
        None,
        json_schema_extra={"example": "08:30"},
        description="Optional time in HH:MM format e.g. 08:30",
    )
    items: List[MealItem] = Field(..., min_length=1, description="At least one food item required")

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v):
        if v is not None:
            if not re.match(r"^\d{2}:\d{2}$", v):
                raise ValueError("time must be in HH:MM format e.g. '08:30'")
            hh, mm = int(v[:2]), int(v[3:])
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError("time must be a valid time e.g. '08:30'")
        return v


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_macros(food: dict, qty_g: float) -> dict:
    """Pro-rate macros from per_100g based on qty_g."""
    per100 = food["per_100g"]
    factor = qty_g / 100.0
    return {
        "calories":  round(per100["calories"]  * factor, 2),
        "protein_g": round(per100["protein_g"] * factor, 2),
        "carbs_g":   round(per100["carbs_g"]   * factor, 2),
        "fat_g":     round(per100["fat_g"]      * factor, 2),
        "fiber_g":   round(per100["fiber_g"]    * factor, 2),
    }


def _sum_macros(items: list[dict]) -> dict:
    """Sum macros across a list of logged items."""
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    for item in items:
        for key in totals:
            totals[key] += item["macros"].get(key, 0.0)
    return {k: round(v, 2) for k, v in totals.items()}


# ─── Foods endpoints (PR #21) ─────────────────────────────────────────────────

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
        results = [f for f in results if f.get("category") == category]
    if cuisine:
        results = [f for f in results if f.get("cuisine") == cuisine]
    if tag:
        results = [f for f in results if tag in f.get("tags", [])]
    if q:
        q_lower = q.lower()
        results = [f for f in results if q_lower in f["name"].lower()]
    total = len(results)
    results = results[offset: offset + limit]
    return {"count": total, "limit": limit, "offset": offset, "foods": results}


@router.get("/foods/{food_id}")
async def get_food(food_id: str):
    food = FOOD_DB.get(food_id)
    if not food:
        raise HTTPException(404, detail=f"Food '{food_id}' not found")
    return food


# ─── POST /athlete/{athlete_id}/meals ────────────────────────────────────────

@router.post("/athlete/{athlete_id}/meals", status_code=201)
async def log_meal(athlete_id: str, body: LogMealRequest):
    """
    Log a meal for an athlete.

    - **slot**: one of breakfast, lunch, dinner, snack
    - **date**: ISO 8601 format e.g. 2026-04-07
    - **time**: optional HH:MM format e.g. 08:30
    - **items**: list of food items with food_id and qty_g (grams)
    - Posting to the same slot on the same date **overwrites** the previous entry.
    """

    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")

    logged_items = []
    for item in body.items:
        food = FOOD_DB.get(item.food_id)
        if not food:
            raise HTTPException(status_code=404, detail=f"Food not found: {item.food_id}")
        macros = _compute_macros(food, item.qty_g)
        logged_items.append({
            "food_id":   item.food_id,
            "food_name": food["name"],
            "qty_g":     item.qty_g,
            "macros":    macros,
        })

    slot_total = _sum_macros(logged_items)
    meal_id = str(uuid.uuid4())
    date_str = body.date.isoformat()

    meal_record = {
        "meal_id": meal_id,
        "time":    body.time,
        "items":   logged_items,
        "total":   slot_total,
    }

    async with _get_lock(athlete_id):
        nutrition_db = _load_json("nutrition.json")
        nutrition_db.setdefault(athlete_id, {"goals": {}, "logs": {}})
        nutrition_db[athlete_id].setdefault("goals", {})
        nutrition_db[athlete_id].setdefault("logs", {})
        nutrition_db[athlete_id]["logs"].setdefault(date_str, {})
        nutrition_db[athlete_id]["logs"][date_str][body.slot] = meal_record
        _save_json("nutrition.json", nutrition_db)

    return {
        "meal_id":    meal_id,
        "athlete_id": athlete_id,
        "date":       date_str,
        "slot":       body.slot,
        "items":      logged_items,
        "total":      slot_total,
    }


# ─── GET /athlete/{athlete_id}/meals ─────────────────────────────────────────

@router.get("/athlete/{athlete_id}/meals")
async def get_meals(athlete_id: str, date: Date):
    """
    Get all meals for an athlete on a specific date, grouped by slot.

    - **date** (query param): ISO 8601 format e.g. ?date=2026-04-07

    Returns all logged slots for that date plus a day_total summing all macros.
    """

    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")

    date_str = date.isoformat()
    nutrition_db = _load_json("nutrition.json")
    athlete_logs = nutrition_db.get(athlete_id, {}).get("logs", {})
    date_logs = athlete_logs.get(date_str, {})

    grouped = {slot: date_logs[slot] for slot in VALID_SLOTS if slot in date_logs}

    all_items = []
    for slot_data in grouped.values():
        all_items.extend(slot_data.get("items", []))
    day_total = _sum_macros(all_items)

    return {
        "athlete_id": athlete_id,
        "date":       date_str,
        "meals":      grouped,
        "day_total":  day_total,
    }
