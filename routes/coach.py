from __future__ import annotations

"""
Personal Health — AI Coach route (/coach/*).
Wraps ai_coach module which uses Anthropic with deterministic fallback.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ai_coach import generate_coach_note
from cache import coach_cache
from database import ATHLETE_DB, _FOLLOWS, _load_json, _save_json
from logging_setup import get_logger
from routes.progress import _compute_injury_risk, _compute_progress, _compute_weak_joints

router = APIRouter(prefix="/coach", tags=["Coach"])
log = get_logger("routes.coach")


# ─── Broadcast store ──────────────────────────────────────────────────────
# Coach → athletes one-shot messages (text or short voice note).
# Persisted to db/broadcasts.json so it survives restarts.

_BROADCASTS: dict[str, list[dict]] = {}


def _load_broadcasts() -> None:
    raw = _load_json("broadcasts.json")
    if isinstance(raw, dict):
        for coach_id, items in raw.items():
            if isinstance(items, list):
                _BROADCASTS[coach_id] = list(items)


_load_broadcasts()


def _save_broadcasts() -> None:
    try:
        _save_json("broadcasts.json", _BROADCASTS)
    except Exception as e:
        log.warning("could not persist broadcasts", extra={"error": str(e)})


def _coach_roster(coach_id: str) -> list[dict]:
    """Return the coach's roster — athletes who follow them. If empty
    (typical in MVP), fall back to all athletes so the demo isn't blank."""
    follower_ids = [aid for aid, follows in _FOLLOWS.items() if coach_id in follows]
    if not follower_ids:
        # Fallback: every athlete except the coach themselves
        follower_ids = [aid for aid in ATHLETE_DB.keys() if aid != coach_id]
    return [ATHLETE_DB[aid] for aid in follower_ids if aid in ATHLETE_DB]


@router.get("/{athlete_id}/weekly-note")
async def weekly_note(athlete_id: str, days: int = Query(default=7, ge=1, le=30), refresh: bool = False):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    cache_key = f"coach:{athlete_id}:{days}"
    if not refresh:
        cached = coach_cache.get(cache_key)
        if cached:
            return cached

    progress = _compute_progress(athlete_id, days)
    risk = _compute_injury_risk(athlete_id, days)
    weak = _compute_weak_joints(athlete_id, days)

    stats = {
        "session_count": progress["session_count"],
        "avg_form_score": progress["avg_form_score"],
        "form_trend_pct": progress["form_trend_pct"],
        "bpi_delta": progress["bpi_delta"],
        "best_jump_cm": progress["best_jump_cm"],
        "injury_risk": risk["risk"],
        "injury_reason": risk["reason"],
        "weak_joints": weak[:3],
    }

    athlete = ATHLETE_DB[athlete_id]
    note = generate_coach_note(athlete.get("name", "Athlete"), athlete.get("sport", "vertical_jump"), stats)
    payload = {
        "athlete_id": athlete_id,
        "window_days": days,
        **note,
    }
    coach_cache.set(cache_key, payload)
    return payload


# ─── Roster + broadcast endpoints ─────────────────────────────────────────


@router.get("/{coach_id}/athletes")
async def list_roster(coach_id: str):
    roster = _coach_roster(coach_id)
    items = [
        {
            "id": a.get("id"),
            "name": a.get("name") or a.get("id"),
            "sport": a.get("sport"),
            "tier": a.get("tier"),
            "bpi": a.get("bpi", 0),
            "sessions": a.get("sessions", 0),
        }
        for a in roster
    ]
    items.sort(key=lambda x: (x.get("name") or "").lower())
    return {"coach_id": coach_id, "athletes": items, "count": len(items)}


class BroadcastIn(BaseModel):
    message: Optional[str] = None
    voice_note_url: Optional[str] = None
    athlete_ids: Optional[list[str]] = None  # None / empty = entire roster


@router.post("/{coach_id}/broadcast")
async def send_broadcast(coach_id: str, body: BroadcastIn):
    text = (body.message or "").strip()
    voice = (body.voice_note_url or "").strip()
    if not text and not voice:
        raise HTTPException(400, "broadcast must have a message or voice_note_url")

    roster_ids = [a["id"] for a in _coach_roster(coach_id) if a.get("id")]
    if body.athlete_ids:
        roster_set = set(roster_ids)
        recipients = [aid for aid in body.athlete_ids if aid in roster_set]
        if not recipients:
            recipients = list(body.athlete_ids)  # MVP-permissive
    else:
        recipients = roster_ids

    bcast = {
        "id": uuid4().hex[:12],
        "coach_id": coach_id,
        "message": text or None,
        "voice_note_url": voice or None,
        "athlete_ids": recipients,
        "recipient_count": len(recipients),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _BROADCASTS.setdefault(coach_id, []).append(bcast)
    if len(_BROADCASTS[coach_id]) > 200:
        _BROADCASTS[coach_id] = _BROADCASTS[coach_id][-200:]
    _save_broadcasts()
    return bcast


@router.get("/{coach_id}/inbox")
async def coach_inbox(coach_id: str, limit: int = Query(default=10, ge=1, le=100)):
    items = list(_BROADCASTS.get(coach_id, []))
    items.sort(key=lambda b: b.get("created_at") or "", reverse=True)
    items = items[:limit]
    return {
        "coach_id": coach_id,
        "broadcasts": items,
        "total": len(_BROADCASTS.get(coach_id, [])),
    }


@router.get("/inbox/athlete/{athlete_id}")
async def athlete_inbox(athlete_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """Broadcasts addressed to this athlete across all coaches. Used by the
    Android app to surface coach messages on Home/ScoreCard."""
    items: list[dict] = []
    for _coach_id, bcasts in _BROADCASTS.items():
        for b in bcasts:
            if athlete_id in (b.get("athlete_ids") or []):
                items.append(b)
    items.sort(key=lambda b: b.get("created_at") or "", reverse=True)
    return {"athlete_id": athlete_id, "broadcasts": items[:limit], "total": len(items)}
