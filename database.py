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

# ─── In-Memory State ────────────────────────────────────────────────────────

SESSION_DB: dict[str, dict] = {}
ATHLETE_DB: dict[str, dict] = {}
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
            print(f"[DB WARN] Could not load sessions: {e}")
    if athletes_file.exists():
        try:
            with open(athletes_file, encoding="utf-8") as f:
                ATHLETE_DB.update(json.load(f))
        except Exception as e:
            print(f"[DB WARN] Could not load athletes: {e}")
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
                print(f"[PF-12 WARN] Could not reload frames for {sid[:8]}: {e}")
        if reloaded_sessions:
            print(f"[PF-12] Reloaded {reloaded_frames} frames across {reloaded_sessions} active sessions")

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


def _atomic_write(filepath: Path, data, **kwargs):
    """Write JSON atomically: write to temp file, then rename.

    Prevents data corruption when concurrent requests call _save_db()
    simultaneously (critical for huddle scenario with 15+ concurrent sessions).
    os.replace() is atomic on POSIX and near-atomic on Windows.
    """
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)
    os.replace(tmp, filepath)


def _save_db():
    try:
        _atomic_write(DB_PATH / "sessions.json", SESSION_DB, indent=2, default=str)
        _atomic_write(DB_PATH / "athletes.json", ATHLETE_DB, indent=2, default=str)
        follows_data = {k: list(v) for k, v in _FOLLOWS.items()}
        _atomic_write(DB_PATH / "follows.json", follows_data, indent=2)
    except Exception as e:
        print(f"[DB WARN] Could not save db: {e}")


def _load_json(filename: str) -> dict:
    """Load any JSON file from db/ directory."""
    filepath = DB_PATH / filename
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(filename: str, data):
    """Save any data to a JSON file in db/ directory (atomic)."""
    _atomic_write(DB_PATH / filename, data, indent=2, default=str)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _compute_xp(scores: list[float], jump_heights: list[float]) -> int:
    base = 50
    if scores:
        avg = sum(scores) / len(scores)
        base += 200 if avg >= 90 else 120 if avg >= 75 else 60 if avg >= 55 else 20
    if jump_heights:
        base += min(int(max(jump_heights) * 2), 100)
    return base
