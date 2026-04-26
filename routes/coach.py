from __future__ import annotations

"""
Personal Health — AI Coach route (/coach/*).
Wraps ai_coach module which uses Anthropic with deterministic fallback.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from ai_coach import generate_coach_note
from auth import current_user, verify_athlete_owner
from cache import coach_cache
from database import _FOLLOWS, ATHLETE_DB, _load_json
from logging_setup import get_logger
from routes.progress import _compute_injury_risk, _compute_progress, _compute_weak_joints
from sqlite_store import (
    count_broadcasts_by_coach,
    insert_broadcast,
    list_broadcasts_by_coach,
    list_broadcasts_for_athlete,
)

router = APIRouter(prefix="/coach", tags=["Coach"])
log = get_logger("routes.coach")


# ─── Broadcast store ──────────────────────────────────────────────────────
# Coach → athletes one-shot messages (text or short voice note).
# Persisted via sqlite_store. A one-time JSON migration absorbs any prior
# db/broadcasts.json from before the SQLite move.


_LEGACY_BROADCASTS_MIGRATED = False


def _migrate_legacy_broadcasts_once() -> None:
    global _LEGACY_BROADCASTS_MIGRATED
    if _LEGACY_BROADCASTS_MIGRATED:
        return
    _LEGACY_BROADCASTS_MIGRATED = True
    try:
        raw = _load_json("broadcasts.json")
    except Exception:
        return
    if not isinstance(raw, dict) or not raw:
        return
    migrated = 0
    for _coach_id, items in raw.items():
        if not isinstance(items, list):
            continue
        for b in items:
            try:
                insert_broadcast(b)
                migrated += 1
            except Exception:
                pass
    if migrated:
        log.info("migrated legacy broadcasts to sqlite", extra={"count": migrated})


_migrate_legacy_broadcasts_once()


def _coach_roster(coach_id: str) -> list[dict]:
    """Return the coach's roster — athletes who follow them. Returns an empty
    list if no one has explicitly opted in via /follow. Previously fell back
    to ALL athletes which meant a fresh coach could broadcast to every user
    in the system unintentionally — that's a security/UX hole."""
    follower_ids = [aid for aid, follows in _FOLLOWS.items() if coach_id in follows]
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
    return {
        "coach_id": coach_id,
        "athletes": items,
        "items": items,
        "count": len(items),
        "total": len(items),
    }


class BroadcastIn(BaseModel):
    message: Optional[str] = None
    voice_note_url: Optional[str] = None
    athlete_ids: Optional[list[str]] = None  # None / empty = entire roster


@router.post("/{coach_id}/broadcast")
async def send_broadcast(coach_id: str, body: BroadcastIn, user: dict = Depends(current_user)):
    """Send a text or voice broadcast to the coach's roster. Caller must own
    the coach_id — you can't impersonate another coach."""
    verify_athlete_owner(user, coach_id)
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
    insert_broadcast(bcast)
    return bcast


@router.get("/{coach_id}/inbox")
async def coach_inbox(coach_id: str, limit: int = Query(default=10, ge=1, le=100)):
    items = list_broadcasts_by_coach(coach_id, limit)
    total = count_broadcasts_by_coach(coach_id)
    return {"coach_id": coach_id, "broadcasts": items, "items": items, "total": total}


@router.get("/inbox/athlete/{athlete_id}")
async def athlete_inbox(athlete_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """Broadcasts addressed to this athlete across all coaches. Used by the
    Android app to surface coach messages on Home/ScoreCard."""
    items = list_broadcasts_for_athlete(athlete_id, limit)
    return {"athlete_id": athlete_id, "broadcasts": items, "total": len(items)}


# ─── Billing (read-only, activity-derived) ───────────────────────────────


billing_router = APIRouter(tags=["Billing"])


@billing_router.get("/billing/coach/{coach_id}")
async def coach_billing(coach_id: str):
    from routes.progress import _athlete_sessions

    roster = _coach_roster(coach_id)
    overdue, paused, paid = [], [], []
    now = datetime.now(timezone.utc)

    for a in roster:
        aid = a.get("id", "")
        sessions_14d = _athlete_sessions(aid, 14)
        sessions_30d = _athlete_sessions(aid, 30)
        last_date = None
        if sessions_30d:
            last_s = max(sessions_30d, key=lambda s: s.get("started_at", ""))
            last_date = last_s.get("started_at", "")[:10]

        entry = {
            "athlete_id": aid,
            "athlete_name": a.get("name", aid),
            "amount_inr": 500,
            "due_date": now.strftime("%Y-%m-01"),
            "sessions_this_month": len(sessions_30d),
            "last_session_date": last_date,
        }
        if len(sessions_14d) > 0:
            entry["status"] = "active"
            paid.append(entry)
        elif len(sessions_30d) > 0:
            entry["status"] = "paused"
            paused.append(entry)
        else:
            entry["status"] = "overdue"
            overdue.append(entry)

    total = len(roster)
    return {
        "coach_id": coach_id,
        "summary": {
            "total": total,
            "paid": len(paid),
            "overdue": len(overdue),
            "paused": len(paused),
            "total_revenue": len(paid) * 500,
        },
        "overdue": overdue,
        "paused": paused,
        "paid": paid,
    }


@billing_router.post("/billing/override/{athlete_id}")
async def billing_override(athlete_id: str, coach_id: str = Query("")):
    # TODO: actual payment system needed — this is a placeholder
    return {"ok": True, "athlete_id": athlete_id, "action": "override", "note": "No billing system yet"}


@router.post("/{coach_id}/invite-link")
async def generate_invite_link(coach_id: str):
    token = uuid4().hex[:12]
    url = f"https://activebharat.in/join/{coach_id}/{token}"
    return {
        "invite_url": url,
        "coach_id": coach_id,
        "token": token,
        "expires_in": "7d",
        "whatsapp_share_text": f"Join my training roster on ActiveBharat: {url}",
    }


VOICE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "db" / "voice_notes"
VOICE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO = {"audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/wav", "audio/x-wav"}
MAX_VOICE_BYTES = 10 * 1024 * 1024

voice_router = APIRouter(tags=["Voice"])


@voice_router.post("/voice-note/upload")
async def upload_voice_note(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_AUDIO:
        raise HTTPException(400, f"unsupported audio type: {content_type}")

    file_id = uuid4().hex[:12]
    ext = ".webm"
    if content_type == "audio/ogg":
        ext = ".ogg"
    elif content_type in ("audio/mp4", "audio/mpeg"):
        ext = ".mp4"
    elif content_type in ("audio/wav", "audio/x-wav"):
        ext = ".wav"

    filename = f"{file_id}{ext}"
    dest = VOICE_DIR / filename

    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(64 * 1024):
            size += len(chunk)
            if size > MAX_VOICE_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "voice note too large (10MB max)")
            out.write(chunk)

    url = f"/voice-notes/{filename}"
    return {"url": url, "voice_url": url, "size_bytes": size}
