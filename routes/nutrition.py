from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from database import FOOD_DB

router = APIRouter(prefix="/foods", tags=["Nutrition"])


@router.get("/")
def list_foods(
    category: str | None = Query(None),
    cuisine: str | None = Query(None),
    tag: str | None = Query(None),
    q: str | None = Query(None),
):
    results = list(FOOD_DB.values())   # ← indent fix: yeh line function ke andar hai
    if category:
        results = [f for f in results if f["category"] == category]
    if cuisine:
        results = [f for f in results if f["cuisine"] == cuisine]
    if tag:
        results = [f for f in results if tag in f.get("tags", [])]
    if q:
        q_lower = q.lower()
        results = [f for f in results if q_lower in f["name"].lower()]
    return {"count": len(results), "foods": results}


@router.get("/{food_id}")
def get_food(food_id: str):
    food = FOOD_DB.get(food_id)
    if not food:
        raise HTTPException(404, detail=f"Food '{food_id}' not found")
    return food