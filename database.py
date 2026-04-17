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

from logging_setup import get_logger

_log = get_logger("database")

DB_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "db"
DB_PATH.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "dataset"

# ─── In-Memory State ────────────────────────────────────────────────────────

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
_RATE_LIMITS: dict[str, float] = {}  # PF-04: session_id → last frame timestamp


# ─── Load / Save ────────────────────────────────────────────────────────────


def _load_db():
    sessions_file = DB_PATH / "sessions.json"
    athletes_file = DB_PATH / "athletes.json"
    if sessions_file.exists():
        try:
            with open(sessions_file, encoding="utf-8") as f:
                SESSION_DB.update(json.load(f))
        except Exception as e:
            _log.warning("could not load sessions", extra={"error": str(e)})
    if athletes_file.exists():
        try:
            with open(athletes_file, encoding="utf-8") as f:
                ATHLETE_DB.update(json.load(f))
        except Exception as e:
            _log.warning("could not load athletes", extra={"error": str(e)})
    # Load foods database (independent of athletes file)
    foods_file = DB_PATH / "foods.json"
    if foods_file.exists():
        try:
            with open(foods_file, encoding="utf-8") as f:
                raw = json.load(f)
                for food in raw.get("foods", []):
                    FOOD_DB[food["food_id"]] = food
            _log.info("foods loaded", extra={"count": len(FOOD_DB)})
        except Exception as e:
            _log.warning("could not load foods", extra={"error": str(e)})

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
            _log.info("auto-seeded athletes/sessions", extra={"athletes": len(athletes), "sessions": len(SESSION_DB)})
        except Exception as e:
            _log.warning("seed failed, using minimal defaults", extra={"error": str(e)})
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
    # PF-12: Reload persisted frames for any still-active sessions so a restart
    # doesn't lose in-progress data. Frames for completed sessions stay on disk
    # until the 24h cleanup cron deletes them.
    frames_dir = DB_PATH / "frames"
    if frames_dir.exists():
        reloaded_sessions = 0
        reloaded_frames = 0
        for fp in frames_dir.glob("*.jsonl"):
            sid = fp.stem
            session = SESSION_DB.get(sid)
            if not session or session.get("status") != "active":
                continue
            try:
                recovered = []
                with open(fp, encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            recovered.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                if recovered:
                    FRAME_BUFFER[sid] = recovered
                    reloaded_sessions += 1
                    reloaded_frames += len(recovered)
            except Exception as e:
                _log.warning("frame reload failed", extra={"session_id": sid[:8], "error": str(e)})
        if reloaded_sessions:
            _log.info(
                "reloaded active-session frames",
                extra={"frames": reloaded_frames, "sessions": reloaded_sessions},
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
            _log.warning("could not load follows", extra={"error": str(e)})
    _log.info("db loaded", extra={"sessions": len(SESSION_DB), "athletes": len(ATHLETE_DB)})


def _save_db():
    """Atomically persist SESSION_DB/ATHLETE_DB/follows.

    Writes to a temp file first and renames on success so a crash during write
    can't leave a half-written JSON that kills the next startup.
    """
    targets = [
        (DB_PATH / "sessions.json", SESSION_DB),
        (DB_PATH / "athletes.json", ATHLETE_DB),
        (DB_PATH / "follows.json", {k: list(v) for k, v in _FOLLOWS.items()}),
    ]
    for path, data in targets:
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
        except Exception as e:
            _log.error("save db failed", extra={"path": str(path), "error": str(e)})


# ─── Per-key locks for safe async mutations ─────────────────────────────────


SESSION_LOCKS: dict[str, asyncio.Lock] = {}
ATHLETE_LOCKS: dict[str, asyncio.Lock] = {}


def session_lock(session_id: str) -> asyncio.Lock:
    lock = SESSION_LOCKS.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        SESSION_LOCKS[session_id] = lock
    return lock


def athlete_lock(athlete_id: str) -> asyncio.Lock:
    lock = ATHLETE_LOCKS.get(athlete_id)
    if lock is None:
        lock = asyncio.Lock()
        ATHLETE_LOCKS[athlete_id] = lock
    return lock


# ─── Periodic save worker ───────────────────────────────────────────────────


async def periodic_save_worker(interval_seconds: int = 60) -> None:
    """Flush in-memory DB to disk every `interval_seconds`.

    Without this, a crashed server loses every session started since the last
    shutdown. Saves are atomic via _save_db's temp-rename pattern.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            _save_db()
            _log.debug("periodic db save", extra={"sessions": len(SESSION_DB)})
        except asyncio.CancelledError:
            break
        except Exception as e:
            _log.warning("periodic save error", extra={"error": str(e)})
            await asyncio.sleep(5)


def _load_json(filename: str) -> dict:
    """Load any JSON file from db/ directory."""
    filepath = DB_PATH / filename
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(filename: str, data):
    """Save any data to a JSON file in db/ directory."""
    with open(DB_PATH / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _compute_xp(scores: list[float], jump_heights: list[float]) -> int:
    base = 50
    if scores:
        avg = sum(scores) / len(scores)
        base += 200 if avg >= 90 else 120 if avg >= 75 else 60 if avg >= 55 else 20
    if jump_heights:
        base += min(int(max(jump_heights) * 2), 100)
    return base
