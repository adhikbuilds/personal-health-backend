"""WN-32: Seed 90 days of realistic nutrition data for all athletes."""

import json
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

DB = Path(__file__).parent.parent / "db"


def load(name):
    p = DB / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save(name, data):
    (DB / name).write_text(json.dumps(data, indent=2))


SLOT_TIMES = {
    "breakfast": ["07:00", "07:30", "08:00", "08:30"],
    "lunch": ["12:30", "13:00", "13:30", "14:00"],
    "dinner": ["19:00", "19:30", "20:00", "20:30"],
    "snack": ["10:30", "16:00", "17:00"],
}

SLOT_CATEGORIES = {
    "breakfast": ["breakfast"],
    "lunch": ["lunch", "main"],
    "dinner": ["dinner", "main"],
    "snack": ["snack", "breakfast"],
}


def macros_for(food, qty_g):
    p = food["per_100g"]
    r = qty_g / 100
    return {
        "calories": round(p["calories"] * r, 1),
        "protein_g": round(p["protein_g"] * r, 2),
        "carbs_g": round(p["carbs_g"] * r, 2),
        "fat_g": round(p["fat_g"] * r, 2),
        "fiber_g": round(p.get("fiber_g", 0) * r, 2),
    }


def sum_macros(items):
    totals = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fiber_g": 0}
    for item in items:
        for k in totals:
            totals[k] = round(totals[k] + item["macros"][k], 2)
    return totals


def build_meal(slot, foods_by_cat, all_foods):
    cats = SLOT_CATEGORIES[slot]
    pool = []
    for c in cats:
        pool.extend(foods_by_cat.get(c, []))
    if not pool:
        pool = all_foods
    n_items = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    chosen = random.sample(pool, min(n_items, len(pool)))
    items = []
    for food in chosen:
        base = food["serving_g"]
        qty = round(random.uniform(base * 0.7, base * 1.5) / 25) * 25
        qty = max(50, qty)
        m = macros_for(food, qty)
        items.append(
            {
                "food_id": food["food_id"],
                "food_name": food["name"],
                "qty_g": float(qty),
                "macros": m,
            }
        )
    return {
        "meal_id": str(uuid.uuid4()),
        "time": random.choice(SLOT_TIMES[slot]),
        "items": items,
        "total": sum_macros(items),
    }


def build_goals(sport_tier):
    tier = sport_tier or "amateur"
    if tier == "elite":
        cal = random.randint(2800, 3400)
    elif tier == "state":
        cal = random.randint(2400, 2900)
    else:
        cal = random.randint(2000, 2500)
    prot = round(cal * random.uniform(0.20, 0.30) / 4)
    carb = round(cal * random.uniform(0.45, 0.55) / 4)
    fat = round(cal * random.uniform(0.20, 0.30) / 9)
    return {
        "daily_calories": cal,
        "protein_g": prot,
        "carbs_g": carb,
        "fat_g": fat,
        "fiber_g": random.randint(25, 38),
    }


def main():
    athletes = load("athletes.json")
    foods_data = load("foods.json")
    nutrition = load("nutrition.json")

    all_foods = foods_data.get("foods", [])
    foods_by_cat = {}
    for f in all_foods:
        c = f.get("category", "snack")
        foods_by_cat.setdefault(c, []).append(f)

    today = date.today()
    start = today - timedelta(days=89)

    for athlete_id, athlete in athletes.items():
        if athlete_id not in nutrition:
            nutrition[athlete_id] = {"goals": {}, "logs": {}}
        entry = nutrition[athlete_id]

        if not entry.get("goals"):
            entry["goals"] = build_goals(athlete.get("tier"))

        logs = entry.setdefault("logs", {})
        adherence = random.uniform(0.7, 0.95)

        for i in range(90):
            d = (start + timedelta(days=i)).isoformat()
            if d in logs:
                continue
            if random.random() > adherence:
                continue
            day_log = {}
            for slot in ["breakfast", "lunch", "dinner", "snack"]:
                slot_prob = {"breakfast": 0.90, "lunch": 0.85, "dinner": 0.80, "snack": 0.60}
                if random.random() < slot_prob[slot]:
                    day_log[slot] = build_meal(slot, foods_by_cat, all_foods)
            if day_log:
                logs[d] = day_log

    save("nutrition.json", nutrition)
    total_days = sum(len(v.get("logs", {})) for v in nutrition.values())
    print(f"Seeded {len(athletes)} athletes, {total_days} log-days across 90 days.")


if __name__ == "__main__":
    main()
