from __future__ import annotations

"""
Repository layer over database.py globals.

Why this exists: routes currently import ATHLETE_DB / SESSION_DB / FRAME_BUFFER
straight from database and mutate them directly. That couples every route to
an in-memory-dict storage model and makes it hard to swap in SQLite/Postgres
later, to add invariants (e.g. "always touch last_updated"), or to stub for
tests.

This module wraps each collection in a repository object that exposes a
small typed API. Routes can migrate one call site at a time — existing
code that still imports the globals keeps working because the repos read
and write the same underlying dicts.
"""

import asyncio
from datetime import datetime, timezone
from typing import Iterable, Optional

from database import (
    _FOLLOWS,
    ATHLETE_DB,
    FRAME_BUFFER,
    RESULT_STORE,
    SESSION_DB,
    _save_db,
    athlete_lock,
    session_lock,
)


# ─── Athletes ──────────────────────────────────────────────────────────────


class AthleteRepository:
    """Read/write athletes. All mutations go through athlete_lock() so two
    concurrent handlers can't step on each other's BPI updates.
    """

    def get(self, athlete_id: str) -> Optional[dict]:
        return ATHLETE_DB.get(athlete_id)

    def require(self, athlete_id: str) -> dict:
        athlete = self.get(athlete_id)
        if athlete is None:
            raise KeyError(f"Athlete {athlete_id!r} not found")
        return athlete

    def list(self, sport: Optional[str] = None) -> list[dict]:
        athletes = list(ATHLETE_DB.values())
        if sport:
            athletes = [a for a in athletes if a.get("sport") == sport]
        return athletes

    def exists(self, athlete_id: str) -> bool:
        return athlete_id in ATHLETE_DB

    def upsert(self, athlete: dict) -> dict:
        aid = athlete.get("id")
        if not aid:
            raise ValueError("athlete requires an 'id' field")
        ATHLETE_DB[aid] = {**ATHLETE_DB.get(aid, {}), **athlete}
        return ATHLETE_DB[aid]

    async def add_bpi(self, athlete_id: str, delta: int) -> dict:
        """Atomically increment a BPI score. Safe across concurrent session-end handlers."""
        async with athlete_lock(athlete_id):
            athlete = self.require(athlete_id)
            athlete["bpi"] = int(athlete.get("bpi", 0)) + int(delta)
            athlete["sessions"] = int(athlete.get("sessions", 0)) + 1
            return athlete

    # Follows live on the athlete graph but the data is stored in _FOLLOWS
    def followed_ids(self, athlete_id: str) -> set[str]:
        return set(_FOLLOWS.get(athlete_id, set()))

    def toggle_follow(self, follower: str, target: str) -> str:
        """Idempotent follow/unfollow. Returns the resulting action."""
        if target in _FOLLOWS[follower]:
            _FOLLOWS[follower].discard(target)
            return "unfollowed"
        _FOLLOWS[follower].add(target)
        return "followed"


# ─── Sessions ──────────────────────────────────────────────────────────────


class SessionRepository:
    """Reads and mutates the SESSION_DB in-memory store. Writes are guarded
    by session_lock() so concurrent frame ingestion + cleanup can't race.
    """

    def get(self, session_id: str) -> Optional[dict]:
        return SESSION_DB.get(session_id)

    def require(self, session_id: str) -> dict:
        sess = self.get(session_id)
        if sess is None:
            raise KeyError(f"Session {session_id!r} not found")
        return sess

    def exists(self, session_id: str) -> bool:
        return session_id in SESSION_DB

    def list(
        self,
        *,
        athlete_id: Optional[str] = None,
        sport: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        sessions: Iterable[dict] = SESSION_DB.values()
        if athlete_id:
            sessions = (s for s in sessions if s.get("athlete_id") == athlete_id)
        if sport:
            sessions = (s for s in sessions if s.get("sport") == sport)
        if status:
            sessions = (s for s in sessions if s.get("status") == status)
        result = list(sessions)
        result.sort(key=lambda s: s.get("started_at", ""), reverse=True)
        return result

    def active(self) -> list[dict]:
        return self.list(status="active")

    def completed(self) -> list[dict]:
        return self.list(status="completed")

    def save_sync(self) -> None:
        """Flush state to disk. Use sparingly — the periodic save worker
        handles the common case."""
        _save_db()

    async def mutate(self, session_id: str, mutator):
        """Apply an async mutator under the session lock and return the mutated dict.

        Usage:
            await repo.mutate(sid, lambda s: s.update(status='completed'))
        """
        async with session_lock(session_id):
            sess = self.require(session_id)
            result = mutator(sess)
            if asyncio.iscoroutine(result):
                await result
            sess["last_updated_at"] = datetime.now(timezone.utc).isoformat()
            return sess


# ─── Frames ────────────────────────────────────────────────────────────────


class FrameStore:
    """Per-session frame buffer. Keeps the FRAME_BUFFER shape so existing
    producers keep working.
    """

    def append(self, session_id: str, frame: dict) -> None:
        FRAME_BUFFER.setdefault(session_id, []).append(frame)

    def for_session(self, session_id: str) -> list[dict]:
        return FRAME_BUFFER.get(session_id, [])

    def latest_result(self, session_id: str) -> Optional[dict]:
        return RESULT_STORE.get(session_id)

    def clear(self, session_id: str) -> None:
        FRAME_BUFFER.pop(session_id, None)


# Singleton-ish handles so routes don't need to instantiate every time
athletes = AthleteRepository()
sessions = SessionRepository()
frames = FrameStore()
