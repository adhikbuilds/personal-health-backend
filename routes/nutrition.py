from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from database import FOOD_DB

router = APIRouter(prefix="/foods", tags=["Nutrition"])

# Sport-specific top food recommendations (high-impact foods per sport type)
_SPORT_FOOD_TIPS: dict[str, dict] = {
    "vertical_jump": {
        "focus": "explosive power & fast-twitch muscle recovery",
        "top_nutrients": ["creatine", "protein", "fast carbs"],
        "pre_workout": ["banana", "white rice", "sports drink"],
        "post_workout": ["chicken breast", "brown rice", "low-fat milk"],
        "avoid": ["heavy fats before training", "excess fibre pre-session"],
    },
    "sprint": {
        "focus": "glycolytic energy + fast recovery between reps",
        "top_nutrients": ["carbohydrates", "beta-alanine", "sodium"],
        "pre_workout": ["oats", "toast with jam", "dates"],
        "post_workout": ["eggs", "sweet potato", "paneer"],
        "avoid": ["high-fat meals 2 h before", "excess protein pre-session"],
    },
    "snatch": {
        "focus": "joint mobility support + strength endurance",
        "top_nutrients": ["protein", "omega-3", "magnesium"],
        "pre_workout": ["rice with curd", "fruit smoothie"],
        "post_workout": ["fish (rohu/hilsa)", "dhal", "Greek yoghurt"],
        "avoid": ["refined sugar spikes", "alcohol (hinders recovery)"],
    },
    "cricket_bat": {
        "focus": "sustained energy + concentration",
        "top_nutrients": ["complex carbs", "B-vitamins", "iron"],
        "pre_workout": ["idli + sambar", "multigrain toast", "coconut water"],
        "post_workout": ["rajma", "quinoa", "sprouts"],
        "avoid": ["heavy ghee-laden meals on match days"],
    },
    "javelin": {
        "focus": "rotational power + shoulder health",
        "top_nutrients": ["protein", "vitamin C", "collagen"],
        "pre_workout": ["ragi mudde", "poha", "fruit"],
        "post_workout": ["chicken soup", "dal + rice", "amla juice"],
        "avoid": ["inflammatory foods (excess refined oil)"],
    },
    "squat": {
        "focus": "leg strength + connective tissue repair",
        "top_nutrients": ["protein", "calcium", "vitamin D"],
        "pre_workout": ["banana", "toast", "oat bar"],
        "post_workout": ["milk + turmeric", "paneer bhurji", "boiled eggs"],
        "avoid": ["training fasted for heavy leg days"],
    },
    "push_up": {
        "focus": "chest / tricep hypertrophy + endurance",
        "top_nutrients": ["protein", "carbohydrates", "vitamin E"],
        "pre_workout": ["light snack: fruit + nuts"],
        "post_workout": ["chicken + chapati", "tofu bhurji", "curd + seeds"],
        "avoid": ["skipping post-workout protein window (>90 min delay)"],
    },
    "pull_up": {
        "focus": "back strength + grip endurance",
        "top_nutrients": ["protein", "potassium", "zinc"],
        "pre_workout": ["peanut butter toast", "banana"],
        "post_workout": ["moong dhal", "fish curry + rice", "soy milk"],
        "avoid": ["insufficient calories if training for hypertrophy"],
    },
}


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


@router.get("/tips/{sport}")
async def get_sport_nutrition_tips(sport: str):
    """Sport-specific nutrition guidance — pre/post-workout foods and key nutrients."""
    tips = _SPORT_FOOD_TIPS.get(sport)
    if not tips:
        supported = list(_SPORT_FOOD_TIPS.keys())
        raise HTTPException(404, detail=f"No tips for sport '{sport}'. Supported: {supported}")

    # Enrich with matching foods from FOOD_DB
    suggestions: list[dict] = []
    keywords = [w.lower() for w in tips.get("post_workout", [])]
    for food in FOOD_DB.values():
        name_lower = food["name"].lower()
        if any(k in name_lower for k in keywords):
            suggestions.append({"id": food["id"], "name": food["name"], "calories": food.get("calories_per_100g")})
        if len(suggestions) >= 6:
            break

    return {"sport": sport, **tips, "food_db_suggestions": suggestions}
