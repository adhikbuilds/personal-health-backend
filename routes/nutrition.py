from __future__ import annotations

"""
Nutrition Routes — WN-03
POST /athlete/{id}/meals  → Log a meal for a slot
GET  /athlete/{id}/meals  → Read all meals for a date, grouped by slot
"""

import json
import uuid
import re
from datetime import date as Date
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from database import ATHLETE_DB, _load_json, _save_json

router = APIRouter(tags=["Nutrition"])

# ─── Valid meal slots ─────────────────────────────────────────────────────────

VALID_SLOTS = {"breakfast", "lunch", "dinner", "snacks"}

# ─── Load foods.json once at module level (static data) ──────────────────────

_FOODS_PATH = Path(__file__).parent.parent / "db" / "foods.json"

def _load_foods_once() -> dict:
    if _FOODS_PATH.exists():
        with open(_FOODS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

FOODS_CACHE: dict = _load_foods_once()


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class MealItem(BaseModel):
    food_id: str
    qty_g: float = Field(..., gt=0, description="Quantity in grams, must be > 0")


class LogMealRequest(BaseModel):
    date: Date = Field(..., example="2026-04-07")
    slot: str = Field(..., example="breakfast")
    time: Optional[str] = Field(None, example="08:30")
    items: List[MealItem] = Field(..., min_length=1)

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v):
        if v is not None:
            if not re.match(r"^\d{2}:\d{2}$", v):
                raise ValueError("time must be in HH:MM format, e.g. '08:30'")
            hh, mm = int(v[:2]), int(v[3:])
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError("time must be a valid time, e.g. '08:30'")
        return v


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_macros(food: dict, qty_g: float) -> dict:
    """Pro-rate all macros from per_100g values based on actual qty_g."""
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
    """Sum macro dicts from a list of logged items."""
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    for item in items:
        for key in totals:
            totals[key] += item["macros"].get(key, 0.0)
    return {k: round(v, 2) for k, v in totals.items()}


# ─── POST /athlete/{athlete_id}/meals ────────────────────────────────────────

@router.post("/athlete/{athlete_id}/meals", status_code=201)
async def log_meal(athlete_id: str, body: LogMealRequest):
    """Log a meal for an athlete."""

    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")

    if body.slot not in VALID_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid slot '{body.slot}'. Must be one of: {', '.join(sorted(VALID_SLOTS))}"
        )

    logged_items = []
    for item in body.items:
        if item.food_id not in FOODS_CACHE:
            raise HTTPException(status_code=404, detail=f"Food not found: {item.food_id}")
        food = FOODS_CACHE[item.food_id]
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

    nutrition_db = _load_json("nutrition.json")
    nutrition_db.setdefault(athlete_id, {"logs": {}})
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
    """Get all meals for an athlete on a specific date, grouped by slot."""

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
