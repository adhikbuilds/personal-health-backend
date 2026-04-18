"""
Personal Health — Athlete Seed Script
======================================
Populates db/athletes.json with realistic athlete profiles for development
and demo purposes. Overwrites the 5-athlete seed in api_server.py with 30
diverse athletes showing realistic BPI distributions and progression tiers.

Run:
  python seed_athletes.py

Output: db/athletes.json (loaded automatically by api_server.py on next start)
"""

import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

DB_PATH = Path(__file__).resolve().parent.parent / "db"
DB_PATH.mkdir(parents=True, exist_ok=True)

# ── Athlete profiles ──────────────────────────────────────────────────────────
# Tier → BPI range mapping (BPI = Biomechanics Performance Index)
TIER_BPI = {
    "Block": (3000, 6000),
    "District": (6001, 9500),
    "State": (9501, 13000),
    "National": (13001, 17000),
    "Elite": (17001, 22000),
}

SPORTS = [
    "vertical_jump",
    "vertical_jump",
    "vertical_jump",  # weight toward jump (demo sport)
    "sprint",
    "sprint",
    "squat",
    "squat",
    "push_up",
    "pull_up",
    "snatch",
]

NAMES = [
    ("Aryan Kapoor", "AK"),
    ("Priya Nair", "PN"),
    ("Rahul Verma", "RV"),
    ("Sneha Joshi", "SJ"),
    ("Karan Mehta", "KM"),
    ("Divya Sharma", "DS"),
    ("Rohan Patel", "RP"),
    ("Anjali Gupta", "AG"),
    ("Vikas Singh", "VS"),
    ("Meera Iyer", "MI"),
    ("Arjun Reddy", "AR"),
    ("Pooja Desai", "PD"),
    ("Siddharth Das", "SD"),
    ("Kavya Pillai", "KP"),
    ("Amit Tiwari", "AT"),
    ("Nisha Rao", "NR"),
    ("Deepak Kumar", "DK"),
    ("Riya Malhotra", "RM"),
    ("Suresh Nambiar", "SN"),
    ("Tanya Bose", "TB"),
    ("Ishaan Shah", "IS"),
    ("Lakshmi Menon", "LM"),
    ("Varun Chandra", "VC"),
    ("Shruti Ghosh", "SG"),
    ("Nikhil Pandey", "NP"),
    ("Aisha Khan", "AK"),
    ("Manish Dubey", "MD"),
    ("Rekha Yadav", "RY"),
    ("Sachin Jain", "SJ"),
    ("Urvi Trivedi", "UT"),
]

TIERS_WEIGHTED = ["Block"] * 8 + ["District"] * 10 + ["State"] * 7 + ["National"] * 4 + ["Elite"] * 1


def generate_athletes():
    athletes = {}
    for i, (name, avatar) in enumerate(NAMES):
        athlete_id = f"athlete_{i + 1:02d}"
        tier = TIERS_WEIGHTED[i]
        sport = random.choice(SPORTS)
        bpi_lo, bpi_hi = TIER_BPI[tier]
        bpi = random.randint(bpi_lo, bpi_hi)
        sessions = random.randint(3 if tier == "Block" else 8, 60 if tier == "Elite" else 30)

        athletes[athlete_id] = {
            "id": athlete_id,
            "name": name,
            "avatar": avatar,
            "sport": sport,
            "tier": tier,
            "bpi": bpi,
            "sessions": sessions,
            "rank": 0,  # set below after sort
        }

    # Assign global ranks by BPI descending
    sorted_ids = sorted(athletes, key=lambda aid: athletes[aid]["bpi"], reverse=True)
    for rank, aid in enumerate(sorted_ids, 1):
        athletes[aid]["rank"] = rank

    return athletes


def main():
    athletes = generate_athletes()
    out = DB_PATH / "athletes.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(athletes, f, indent=2)
    print(f"[SEED] {len(athletes)} athletes written to {out}")
    top5 = sorted(athletes.values(), key=lambda a: a["bpi"], reverse=True)[:5]
    print("[SEED] Top 5 by BPI:")
    for a in top5:
        print(f"  #{a['rank']:2d}  {a['name']:<18} {a['sport']:<16} {a['tier']:<10} {a['bpi']:,} BPI")


if __name__ == "__main__":
    main()
