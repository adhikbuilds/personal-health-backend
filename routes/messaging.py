from __future__ import annotations

"""
1:1 private coach ↔ athlete messaging room.

Separate from broadcast/reply — this is for sensitive topics:
injury, missed sessions, payment issues. Searchable by coach.
Never visible to huddle or other athletes.

Endpoints:
  POST /messages/{coach_id}/{athlete_id}  — send a message (either party)
  GET  /messages/{coach_id}/{athlete_id}  — fetch thread, newest-last
  GET  /coach/{coach_id}/messages         — coach inbox: all threads, sorted by last message
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import require_coach_or_admin
from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from routes.coach_roster import _load_rosters

router = APIRouter(tags=["Messaging"])
log = get_logger("routes.messaging")

_MESSAGES_FILE = "messages.json"


def _load_messages() -> dict:
    return _load_json(_MESSAGES_FILE)


def _save_messages(data: dict):
    _save_json(_MESSAGES_FILE, data)


def _thread_key(coach_id: str, athlete_id: str) -> str:
    return f"{coach_id}::{athlete_id}"


class MessageRequest(BaseModel):
    sender: str  # "coach" or "athlete"
    text: Optional[str] = None
    voice_note_url: Optional[str] = None


# ─── Send message ────────────────────────────────────────────────────────────


@router.post("/messages/{coach_id}/{athlete_id}")
async def send_message(coach_id: str, athlete_id: str, req: MessageRequest):
    """
    Either party sends a message. `sender` must be "coach" or "athlete".
    No auth dependency here — caller passes their JWT separately;
    the sender field determines UI rendering. Validation via roster membership.
    """
    if not req.text and not req.voice_note_url:
        raise HTTPException(400, "provide text or voice_note_url")

    if req.sender not in ("coach", "athlete"):
        raise HTTPException(400, "sender must be 'coach' or 'athlete'")

    rosters = _load_rosters()
    if athlete_id not in rosters.get(coach_id, {}).get("athletes", []):
        raise HTTPException(403, "athlete not in this coach's roster")

    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    key = _thread_key(coach_id, athlete_id)

    messages = _load_messages()
    thread = messages.setdefault(key, {"coach_id": coach_id, "athlete_id": athlete_id, "msgs": []})
    thread["msgs"].append(
        {
            "msg_id": msg_id,
            "sender": req.sender,
            "text": req.text,
            "voice_note_url": req.voice_note_url,
            "sent_at": now,
        }
    )
    thread["last_msg_at"] = now
    _save_messages(messages)

    log.info("message sent", extra={"key": key, "sender": req.sender})
    return {"msg_id": msg_id, "sent_at": now}


# ─── Fetch thread ────────────────────────────────────────────────────────────


@router.get("/messages/{coach_id}/{athlete_id}")
async def get_thread(coach_id: str, athlete_id: str, limit: int = 50):
    """
    Full thread between coach and athlete, newest-last.
    Coach can search across all threads via GET /coach/{id}/messages?q=.
    """
    messages = _load_messages()
    key = _thread_key(coach_id, athlete_id)
    thread = messages.get(key)

    if not thread:
        return {"coach_id": coach_id, "athlete_id": athlete_id, "msgs": []}

    msgs = thread["msgs"][-limit:]
    athlete = ATHLETE_DB.get(athlete_id) or {}
    coach = ATHLETE_DB.get(coach_id) or {}

    return {
        "coach_id": coach_id,
        "coach_name": coach.get("name", coach_id),
        "athlete_id": athlete_id,
        "athlete_name": athlete.get("name", athlete_id),
        "msgs": msgs,
        "total": len(thread["msgs"]),
    }


# ─── Coach inbox (all threads) ───────────────────────────────────────────────


@router.get("/coach/{coach_id}/messages")
async def coach_message_inbox(
    coach_id: str,
    q: Optional[str] = None,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """
    Coach sees all 1:1 threads, sorted by most recent message.
    Optional ?q= searches text across all messages in all threads.
    """
    messages = _load_messages()
    threads = [v for v in messages.values() if v["coach_id"] == coach_id]
    threads.sort(key=lambda t: t.get("last_msg_at", ""), reverse=True)

    result = []
    for t in threads:
        athlete = ATHLETE_DB.get(t["athlete_id"]) or {}
        msgs = t["msgs"]

        if q:
            q_lower = q.lower()
            msgs = [m for m in msgs if q_lower in (m.get("text") or "").lower()]
            if not msgs:
                continue

        last = t["msgs"][-1] if t["msgs"] else {}
        result.append(
            {
                "athlete_id": t["athlete_id"],
                "athlete_name": athlete.get("name", t["athlete_id"]),
                "last_msg_at": t.get("last_msg_at"),
                "last_msg_preview": (last.get("text") or "[voice note]")[:80],
                "last_sender": last.get("sender"),
                "message_count": len(t["msgs"]),
                "matched_msgs": msgs if q else None,
            }
        )

    return {"coach_id": coach_id, "threads": result, "total": len(result)}
