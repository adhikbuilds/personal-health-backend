from __future__ import annotations

"""
Personal Health — tiny SQLite store for production-only concerns.

Hand-rolled CRUD; no SQLAlchemy. Tables added here are *additive* —
existing JSON files (sessions/athletes/follows) are untouched. This file
only owns: users, refresh_tokens, api_keys, audit_log, idempotency_cache,
progress_cache, daily_tracker_v2.

Schema migrations are idempotent CREATE TABLE IF NOT EXISTS statements
guarded by a schema_version row.
"""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from config import settings
from logging_setup import get_logger

log = get_logger("sqlite_store")

_LOCK = threading.RLock()
_INITIALIZED = False
SCHEMA_VERSION = 1


def _connect() -> sqlite3.Connection:
    settings.db_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.users_sqlite), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.cursor()
            yield cur
        finally:
            conn.close()


def init_db() -> None:
    """Run idempotent migrations. Safe to call on every startup."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id TEXT PRIMARY KEY,
              email TEXT UNIQUE NOT NULL,
              name TEXT NOT NULL,
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL DEFAULT 'athlete',
              athlete_id TEXT,
              created_at REAL NOT NULL,
              last_login_at REAL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        # Idempotent ALTER — add `consents` column if missing (DPDP / injury-disclaimer flags)
        _cols = {r["name"] for r in cur.execute("PRAGMA table_info(users)").fetchall()}
        if "consents" not in _cols:
            cur.execute("ALTER TABLE users ADD COLUMN consents TEXT")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS refresh_tokens (
              jti TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              issued_at REAL NOT NULL,
              expires_at REAL NOT NULL,
              revoked INTEGER NOT NULL DEFAULT 0,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rt_user ON refresh_tokens(user_id)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
              key_hash TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              created_at REAL NOT NULL,
              last_used_at REAL,
              revoked INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL NOT NULL,
              user_id TEXT,
              action TEXT NOT NULL,
              resource TEXT,
              ip TEXT,
              request_id TEXT,
              detail TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_cache (
              key TEXT PRIMARY KEY,
              response_json TEXT NOT NULL,
              created_at REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS progress_cache (
              cache_key TEXT PRIMARY KEY,
              payload TEXT NOT NULL,
              created_at REAL NOT NULL,
              ttl_seconds INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_tracker_v2 (
              athlete_id TEXT NOT NULL,
              date TEXT NOT NULL,
              steps INTEGER NOT NULL DEFAULT 0,
              active_minutes INTEGER NOT NULL DEFAULT 0,
              distance_km REAL NOT NULL DEFAULT 0,
              calories_burned INTEGER NOT NULL DEFAULT 0,
              calorie_intake INTEGER NOT NULL DEFAULT 0,
              water_glasses INTEGER NOT NULL DEFAULT 0,
              sleep_hours REAL NOT NULL DEFAULT 0,
              updated_at REAL NOT NULL,
              PRIMARY KEY(athlete_id, date)
            )
            """
        )
        cur.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
    _INITIALIZED = True
    log.info("sqlite initialized", extra={"path": str(settings.users_sqlite), "version": SCHEMA_VERSION})


# ─── Users ──────────────────────────────────────────────────────────────────


def insert_user(user: dict) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO users(id, email, name, password_hash, role, athlete_id, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["email"],
                user["name"],
                user["password_hash"],
                user.get("role", "athlete"),
                user.get("athlete_id"),
                user.get("created_at", time.time()),
            ),
        )


def get_user_by_email(email: str) -> Optional[dict]:
    with cursor() as cur:
        row = cur.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict]:
    with cursor() as cur:
        row = cur.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def touch_login(user_id: str) -> None:
    with cursor() as cur:
        cur.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (time.time(), user_id))


def delete_user(user_id: str) -> None:
    """Hard-delete a user + their refresh tokens. DPDP right-to-delete."""
    with cursor() as cur:
        cur.execute("DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))


def set_user_consent(user_id: str, key: str, accepted: bool) -> None:
    """Record an arbitrary consent flag (e.g. injury_disclaimer, dpdp_notice)."""
    import json as _json

    with cursor() as cur:
        row = cur.execute("SELECT consents FROM users WHERE id = ?", (user_id,)).fetchone()
        consents = {}
        if row and row["consents"]:
            try:
                consents = _json.loads(row["consents"])
            except (ValueError, TypeError):
                consents = {}
        consents[key] = {"accepted": bool(accepted), "at": time.time()}
        cur.execute("UPDATE users SET consents = ? WHERE id = ?", (_json.dumps(consents), user_id))


# ─── Refresh tokens ─────────────────────────────────────────────────────────


def store_refresh(jti: str, user_id: str, ttl_seconds: int) -> None:
    now = time.time()
    with cursor() as cur:
        cur.execute(
            "INSERT INTO refresh_tokens(jti, user_id, issued_at, expires_at) VALUES(?, ?, ?, ?)",
            (jti, user_id, now, now + ttl_seconds),
        )


def revoke_refresh(jti: str) -> None:
    with cursor() as cur:
        cur.execute("UPDATE refresh_tokens SET revoked = 1 WHERE jti = ?", (jti,))


def is_refresh_valid(jti: str) -> bool:
    with cursor() as cur:
        row = cur.execute("SELECT revoked, expires_at FROM refresh_tokens WHERE jti = ?", (jti,)).fetchone()
        if not row:
            return False
        return row["revoked"] == 0 and row["expires_at"] > time.time()


# ─── Audit log ──────────────────────────────────────────────────────────────


def audit(action: str, **fields: Any) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit_log(ts, user_id, action, resource, ip, request_id, detail)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                fields.get("user_id"),
                action,
                fields.get("resource"),
                fields.get("ip"),
                fields.get("request_id"),
                json.dumps(fields.get("detail")) if fields.get("detail") is not None else None,
            ),
        )


def recent_audit(limit: int = 100) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute("SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ─── Progress cache ────────────────────────────────────────────────────────


def cache_get(key: str) -> Optional[dict]:
    with cursor() as cur:
        row = cur.execute(
            "SELECT payload, created_at, ttl_seconds FROM progress_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        if row["created_at"] + row["ttl_seconds"] < time.time():
            return None
        try:
            return json.loads(row["payload"])
        except Exception:
            return None


def cache_set(key: str, payload: dict, ttl_seconds: int) -> None:
    with cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO progress_cache(cache_key, payload, created_at, ttl_seconds) VALUES(?, ?, ?, ?)",
            (key, json.dumps(payload, default=str), time.time(), ttl_seconds),
        )


# ─── Daily tracker v2 ───────────────────────────────────────────────────────


def upsert_daily_tracker(athlete_id: str, date: str, fields: dict) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO daily_tracker_v2(athlete_id, date, steps, active_minutes, distance_km,
                                          calories_burned, calorie_intake, water_glasses, sleep_hours, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(athlete_id, date) DO UPDATE SET
              steps=excluded.steps,
              active_minutes=excluded.active_minutes,
              distance_km=excluded.distance_km,
              calories_burned=excluded.calories_burned,
              calorie_intake=excluded.calorie_intake,
              water_glasses=excluded.water_glasses,
              sleep_hours=excluded.sleep_hours,
              updated_at=excluded.updated_at
            """,
            (
                athlete_id,
                date,
                int(fields.get("steps", 0)),
                int(fields.get("active_minutes", 0)),
                float(fields.get("distance_km", 0)),
                int(fields.get("calories_burned", 0)),
                int(fields.get("calorie_intake", 0)),
                int(fields.get("water_glasses", 0)),
                float(fields.get("sleep_hours", 0)),
                time.time(),
            ),
        )


# ─── Idempotency ────────────────────────────────────────────────────────────


def idempotency_get(key: str, max_age_seconds: int = 86400) -> Optional[dict]:
    with cursor() as cur:
        row = cur.execute("SELECT response_json, created_at FROM idempotency_cache WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        if row["created_at"] + max_age_seconds < time.time():
            return None
        try:
            return json.loads(row["response_json"])
        except Exception:
            return None


def idempotency_set(key: str, response: dict) -> None:
    with cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO idempotency_cache(key, response_json, created_at) VALUES(?, ?, ?)",
            (key, json.dumps(response, default=str), time.time()),
        )


# ─── API keys ───────────────────────────────────────────────────────────────


def store_api_key(key_hash: str, label: str) -> None:
    with cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys(key_hash, label, created_at) VALUES(?, ?, ?)",
            (key_hash, label, time.time()),
        )


def verify_api_key_hash(key_hash: str) -> Optional[dict]:
    with cursor() as cur:
        row = cur.execute("SELECT * FROM api_keys WHERE key_hash = ? AND revoked = 0", (key_hash,)).fetchone()
        if not row:
            return None
        cur.execute("UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?", (time.time(), key_hash))
        return dict(row)


def daily_tracker_history(athlete_id: str, days: int = 30) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM daily_tracker_v2 WHERE athlete_id = ? ORDER BY date DESC LIMIT ?",
            (athlete_id, days),
        ).fetchall()
        return [dict(r) for r in rows]
