"""WN-33: Seed 90 days of enhanced wellness data for all athletes."""

import json
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

DB = Path(__file__).parent.parent / "db"


def load(name):
    p = DB / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save(name, data):
    (DB / name).write_text(json.dumps(data, indent=2))


COMPONENT_MAX = {
    "sleep": 30,
    "hydration": 15,
    "mood": 15,
    "energy": 15,
    "stress": 10,
    "soreness": 15,
}


def sleep_score(bedtime_hhmm, waketime_hhmm, quality, interruptions):
    def to_min(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    bed = to_min(bedtime_hhmm)
    wake = to_min(waketime_hhmm)
    if wake <= bed:
        wake += 24 * 60
    duration_h = (wake - bed) / 60

    base = (min(duration_h, 8) / 8) * 40
    q_score = quality * 8
    penalty = interruptions * 5
    raw = base + q_score - penalty
    return max(0, min(100, round(raw)))


def wellness_score(sleep_sc, hydration_pct, mood, stress, energy, soreness):
    s = (
        (sleep_sc / 100) * COMPONENT_MAX["sleep"]
        + (min(hydration_pct, 100) / 100) * COMPONENT_MAX["hydration"]
        + (mood / 10) * COMPONENT_MAX["mood"]
        + (energy / 10) * COMPONENT_MAX["energy"]
        + ((10 - stress) / 10) * COMPONENT_MAX["stress"]
        + ((10 - soreness) / 10) * COMPONENT_MAX["soreness"]
    )
    return max(0, min(100, round(s)))


def recommendation(score):
    if score >= 80:
        return "Excellent recovery. Train hard and track your new peak."
    if score >= 70:
        return "Good to go. Prioritise skill work while energy is high."
    if score >= 60:
        return "Moderate readiness. Warm up longer and reduce max-intensity sets."
    if score >= 50:
        return "Listen to your body. Technical drills only today."
    return "Rest day recommended. Recovery is part of the program."


def generate_profile(tier):
    if tier == "elite":
        base_mood = random.uniform(6.5, 8.5)
        base_sleep_q = random.uniform(3.0, 4.5)
        base_hydration = random.randint(2200, 3000)
        base_soreness = random.uniform(3, 6)
    elif tier == "state":
        base_mood = random.uniform(5.5, 7.5)
        base_sleep_q = random.uniform(2.5, 4.0)
        base_hydration = random.randint(1800, 2500)
        base_soreness = random.uniform(3, 7)
    else:
        base_mood = random.uniform(4.5, 7.0)
        base_sleep_q = random.uniform(2.0, 3.5)
        base_hydration = random.randint(1500, 2200)
        base_soreness = random.uniform(4, 8)

    return {
        "base_mood": base_mood,
        "base_sleep_q": base_sleep_q,
        "base_hydration": base_hydration,
        "base_soreness": base_soreness,
        "energy_var": random.uniform(0.5, 2.0),
    }


BEDTIMES = ["22:00", "22:30", "23:00", "23:30", "00:00"]
WAKETIMES = ["05:30", "06:00", "06:30", "07:00", "07:30"]


def build_day(profile, day_idx):
    def noise(amp):
        return random.gauss(0, amp)

    bed = random.choice(BEDTIMES)
    wake = random.choice(WAKETIMES)
    quality = max(1, min(5, round(profile["base_sleep_q"] + noise(0.7))))
    interruptions = random.choices([0, 1, 2, 3], weights=[0.5, 0.3, 0.15, 0.05])[0]
    sl_sc = sleep_score(bed, wake, quality, interruptions)

    water = max(500, round(profile["base_hydration"] + noise(300), -2))
    daily_goal_ml = 2500
    hydration_pct = round((water / daily_goal_ml) * 100)

    mood = max(1, min(10, round(profile["base_mood"] + noise(1.5))))
    stress = max(1, min(10, round(5 + noise(2.0))))
    energy = max(
        1, min(10, round(profile["base_mood"] - 1 + noise(1.5) + profile["energy_var"] * math.sin(day_idx / 7)))
    )
    soreness = max(1, min(10, round(profile["base_soreness"] + noise(1.5))))

    body_weight = None
    if random.random() < 0.4:
        body_weight = round(random.uniform(55, 90) + noise(1), 1)

    ws = wellness_score(sl_sc, hydration_pct, mood, stress, energy, soreness)
    rec_ready = ws >= 70

    breakdown = {
        "sleep": round((sl_sc / 100) * COMPONENT_MAX["sleep"], 1),
        "hydration": round((min(hydration_pct, 100) / 100) * COMPONENT_MAX["hydration"], 1),
        "mood": round((mood / 10) * COMPONENT_MAX["mood"], 1),
        "energy": round((energy / 10) * COMPONENT_MAX["energy"], 1),
        "stress": round(((10 - stress) / 10) * COMPONENT_MAX["stress"], 1),
        "soreness": round(((10 - soreness) / 10) * COMPONENT_MAX["soreness"], 1),
    }

    return {
        "sleep": {
            "bedtime": bed,
            "wake_time": wake,
            "quality": quality,
            "interruptions": interruptions,
            "sleep_score": sl_sc,
        },
        "hydration": {
            "water_ml": water,
            "daily_goal_ml": daily_goal_ml,
            "pct": hydration_pct,
        },
        "mental": {
            "mood": mood,
            "stress": stress,
            "energy": energy,
            "journal_note": None,
        },
        "physical": {
            "soreness": soreness,
            "body_weight_kg": body_weight,
        },
        "wellness_score": ws,
        "breakdown": breakdown,
        "recovery_ready": rec_ready,
        "recommendation": recommendation(ws),
        "logged_at": datetime.now().isoformat(),
    }


def main():
    athletes = load("athletes.json")
    wellness = load("wellness.json")

    today = date.today()
    start = today - timedelta(days=89)

    for athlete_id, athlete in athletes.items():
        if athlete_id not in wellness:
            wellness[athlete_id] = {}

        profile = generate_profile(athlete.get("tier", "amateur"))
        adherence = random.uniform(0.72, 0.95)

        for i in range(90):
            d = (start + timedelta(days=i)).isoformat()
            if d in wellness[athlete_id]:
                continue
            if random.random() > adherence:
                continue
            wellness[athlete_id][d] = build_day(profile, i)

    save("wellness.json", wellness)
    total_days = sum(len(v) for v in wellness.values())
    print(f"Seeded {len(athletes)} athletes, {total_days} wellness log-days across 90 days.")


if __name__ == "__main__":
    main()
