from __future__ import annotations

"""
Nutrition Routes — WN-03
POST /athlete/{id}/meals  → Log a meal for a slot
GET  /athlete/{id}/meals  → Read all meals for a date, grouped by slot
"""

import json
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import database
from database import ATHLETE_DB, _load_json, _save_json

router = APIRouter(tags=["nutrition"])

# ─── Valid meal slots ─────────────────────────────────────────────────────────

VALID_SLOTS = {"breakfast", "lunch", "dinner", "snacks"}

# ─── Path to foods.json (sits next to db/) ───────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
_FOODS_PATH = _PROJECT_ROOT / "db" / "foods.json"


def _load_foods() -> dict:
    """Load foods reference data from db/foods.json."""
    if _FOODS_PATH.exists():
        with open(_FOODS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class MealItem(BaseModel):
    food_id: str
    qty_g: float = Field(..., gt=0, description="Quantity in grams, must be > 0")


class LogMealRequest(BaseModel):
    date: str = Field(..., example="2026-04-07")
    slot: str = Field(..., example="breakfast")
    time: Optional[str] = Field(None, example="08:30")
    items: List[MealItem]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_macros(food: dict, qty_g: float) -> dict:
    """
    Pro-rate all macros from per_100g values based on actual qty_g.
    Example: 200g of food with 135 cal/100g → 270 cal
    """
    per100 = food["per_100g"]
    factor = qty_g / 100.0
    return {
        "calories":   round(per100["calories"]   * factor, 2),
        "protein_g":  round(per100["protein_g"]  * factor, 2),
        "carbs_g":    round(per100["carbs_g"]     * factor, 2),
        "fat_g":      round(per100["fat_g"]       * factor, 2),
        "fiber_g":    round(per100["fiber_g"]     * factor, 2),
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
def log_meal(athlete_id: str, body: LogMealRequest):
    """
    Log a meal for an athlete.
    - Validates athlete exists
    - Validates slot is one of: breakfast, lunch, dinner, snacks
    - Computes macros for each item from foods.json
    - Saves to db/nutrition.json under athlete_id → logs → date → slot
    - Returns meal_id (UUID) and computed totals
    """

    # 1. Validate athlete exists
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")

    # 2. Validate slot
    if body.slot not in VALID_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid slot '{body.slot}'. Must be one of: {', '.join(sorted(VALID_SLOTS))}"
        )

    # 3. Load foods reference data
    foods = _load_foods()

    # 4. Compute macros for each item
    logged_items = []
    for item in body.items:
        if item.food_id not in foods:
            raise HTTPException(status_code=404, detail=f"Food not found: {item.food_id}")

        food = foods[item.food_id]
        macros = _compute_macros(food, item.qty_g)
        logged_items.append({
            "food_id":   item.food_id,
            "food_name": food["name"],
            "qty_g":     item.qty_g,
            "macros":    macros,
        })

    # 5. Compute slot totals
    slot_total = _sum_macros(logged_items)

    # 6. Generate meal_id
    meal_id = str(uuid.uuid4())

    # 7. Build the meal record
    meal_record = {
        "meal_id": meal_id,
        "time":    body.time,
        "items":   logged_items,
        "total":   slot_total,
    }

    # 8. Load current nutrition DB, update, and save
    #    Structure: { athlete_id: { "logs": { date: { slot: meal_record } } } }
    nutrition_db = _load_json("nutrition.json")

    nutrition_db.setdefault(athlete_id, {"logs": {}})
    nutrition_db[athlete_id]["logs"].setdefault(body.date, {})

    # Overwrite the slot (a new log for the same slot replaces the old one)
    nutrition_db[athlete_id]["logs"][body.date][body.slot] = meal_record

    _save_json("nutrition.json", nutrition_db)

    # 9. Return response
    return {
        "meal_id":  meal_id,
        "athlete_id": athlete_id,
        "date":     body.date,
        "slot":     body.slot,
        "items":    logged_items,
        "total":    slot_total,
    }


# ─── GET /athlete/{athlete_id}/meals ─────────────────────────────────────────

@router.get("/athlete/{athlete_id}/meals")
def get_meals(athlete_id: str, date: str):
    """
    Get all meals for an athlete on a specific date, grouped by slot.
    Usage: GET /athlete/{id}/meals?date=2026-04-07
    """

    # 1. Validate athlete exists
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(status_code=404, detail=f"Athlete not found: {athlete_id}")

    # 2. Load nutrition DB
    nutrition_db = _load_json("nutrition.json")

    # 3. Fetch logs for this athlete + date
    athlete_logs = nutrition_db.get(athlete_id, {}).get("logs", {})
    date_logs = athlete_logs.get(date, {})

    # 4. Build response grouped by slot (return all valid slots, empty if no data)
    grouped = {}
    for slot in VALID_SLOTS:
        if slot in date_logs:
            grouped[slot] = date_logs[slot]

    # 5. Compute grand total for the day across all slots
    all_items = []
    for slot_data in grouped.values():
        all_items.extend(slot_data.get("items", []))
    day_total = _sum_macros(all_items)

    return {
        "athlete_id": athlete_id,
        "date":       date,
        "meals":      grouped,
        "day_total":  day_total,
    }