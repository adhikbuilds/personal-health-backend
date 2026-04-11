from __future__ import annotations

"""
Personal Health — Shared Database Layer

All JSON persistence and in-memory state lives here.
Every route module imports from this single source of truth.
"""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "db"
DB_PATH.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "dataset"

# --- In-Memory State --------------------------------------------------------
SESSION_DB: dict[str, dict] = {}
ATHLETE_DB: dict[str, dict] = {}
FOOD_DB: dict[str, dict] = {}
FRAME_BUFFER: dict[str, list[dict]] = defaultdict(list)
WS_CONNECTIONS: dict[str, list] = defaultdict(list)
ANALYSIS_QUEUE: asyncio.Queue | None = None
RESULT_STORE: dict[str, dict] = {}
_POSE_ANALYZERS: dict[str, object] = {}
RPPG_STORE: dict[str, object] = {}
_FOLLOWS: dict[str, set] = defaultdict(set)
_RATE_LIMITS: dict[str, float] = {}

# --- Load / Save ------------------------------------------------------------

def _load_db():
    sessions_file = DB_PATH / "sessions.json"
    athletes_file = DB_PATH / "athletes.json"

    if sessions_file.exists():
        try:
            with open(sessions_file, encoding="utf-8") as f:
                SESSION_DB.update(json.load(f))
        except Exception as e:
            print(f"[DB WARN] Could not load sessions: {e}")

    if athletes_file.exists():
        try:
            with open(athletes_file, encoding="utf-8") as f:
                ATHLETE_DB.update(json.load(f))
        except Exception as e:
            print(f"[DB WARN] Could not load athletes: {e}")

    # Load foods database
    foods_file = DB_PATH / "foods.json"
    if foods_file.exists():
        try:
            with open(foods_file, encoding="utf-8") as f:
                raw = json.load(f)
                for food in raw.get("foods", []):
                    FOOD_DB[food["food_id"]] = food
            print(f"[DB] {len(FOOD_DB)} foods loaded")
        except Exception as e:
            print(f"[DB WARN] Could not load foods: {e}")

    # Seed athletes and sessions if DB is sparse
    if len(ATHLETE_DB) < 10:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from seeds.seed_athletes import generate_athletes
            from seeds.seed_sessions import generate_session
            athletes = generate_athletes()
            ATHLETE_DB.update(athletes)
            for athlete in athletes.values():
                n = athlete.get("sessions", 5)
                for i in range(n):
                    s = generate_session(athlete, i, n)
                    SESSION_DB[s["session_id"]] = s
            _save_db()
            print(f"[DB] Auto-seeded {len(athletes)} athletes, {len(SESSION_DB)} sessions")
        except Exception as e:
            print(f"[DB] Seed failed ({e}), using minimal defaults")
            ATHLETE_DB.update(
                {
                    "athlete_01": {
                        "id": "athlete_01",
                        "name": "Viraj Sharma",
                        "sport": "vertical_jump",
                        "tier": "District",
                        "bpi": 12450,
                        "sessions": 0,
                        "avatar": "VS",
                        "rank": 1,
                    },
                    "athlete_02": {
                        "id": "athlete_02",
                        "name": "Priya Desai",
                        "sport": "sprint",
                        "tier": "State",
                        "bpi": 11800,
                        "sessions": 0,
                        "avatar": "PD",
                        "rank": 2,
                    },
                    "athlete_03": {
                        "id": "athlete_03",
                        "name": "Rajan Mehta",
                        "sport": "snatch",
                        "tier": "National",
                        "bpi": 14200,
                        "sessions": 0,
                        "avatar": "RM",
                        "rank": 3,
                    },
                    "athlete_04": {
                        "id": "athlete_04",
                        "name": "Amita Joshi",
                        "sport": "javelin",
                        "tier": "District",
                        "bpi": 9300,
                        "sessions": 0,
                        "avatar": "AJ",
                        "rank": 4,
                    },
                    "athlete_05": {
                        "id": "athlete_05",
                        "name": "Karan Singh",
                        "sport": "cricket_bat",
                        "tier": "Block",
                        "bpi": 8600,
                        "sessions": 0,
                        "avatar": "KS",
                        "rank": 5,
                    },
                }
            )

    # Load follow relationships
    follows_file = DB_PATH / "follows.json"
    if follows_file.exists():
        try:
            with open(follows_file, encoding="utf-8") as f:
                raw_follows = json.load(f)
                for k, v in raw_follows.items():
                    _FOLLOWS[k] = set(v)
        except Exception as e:
            print(f"[DB WARN] Could not load follows: {e}")

    print(f"[DB] {len(SESSION_DB)} sessions, {len(ATHLETE_DB)} athletes loaded")


def _save_db():
    try:
        with open(DB_PATH / "sessions.json", "w", encoding="utf-8") as f:
            json.dump(SESSION_DB, f, indent=2, default=str)
        with open(DB_PATH / "athletes.json", "w", encoding="utf-8") as f:
            json.dump(ATHLETE_DB, f, indent=2, default=str)
        follows_data = {k: list(v) for k, v in _FOLLOWS.items()}
        with open(DB_PATH / "follows.json", "w", encoding="utf-8") as f:
            json.dump(follows_data, f, indent=2)
    except Exception as e:
        print(f"[DB WARN] Could not save db: {e}")


def _load_json(filename: str) -> dict:
    filepath = DB_PATH / filename
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(filename: str, data):
    with open(DB_PATH / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# --- Helpers ----------------------------------------------------------------

def _compute_xp(scores: list[float], jump_heights: list[float]) -> int:
    base = 50
    if scores:
        avg = sum(scores) / len(scores)
        base += 200 if avg >= 90 else 120 if avg >= 75 else 60 if avg >= 55 else 20
    if jump_heights:
        base += min(int(max(jump_heights) * 2), 100)
    return base
