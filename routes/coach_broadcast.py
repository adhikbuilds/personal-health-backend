from __future__ import annotations

"""
Coach broadcast, private replies, clap reactions, and PB reinforcement inbox.

Trainer-first v1 flows:
  POST /coach/{id}/broadcast            — send voice/text note to N athletes
  POST /athlete/{id}/reply              — athlete private reply to a broadcast
  POST /athlete/{id}/clap/{target_id}  — huddle-mate clap reaction (1 per pair per PB)
  GET  /coach/{id}/inbox               — coach reads all replies + clap counts
  POST /coach/{id}/drill-assignment    — bulk assign tomorrow's drill to athlete subset
"""

import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import require_athlete_or_admin, require_coach_or_admin
from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from routes.coach_roster import _load_rosters

router = APIRouter(tags=["Coach Broadcast"])
log = get_logger("routes.coach_broadcast")

_BROADCASTS_FILE = "coach_broadcasts.json"
_CLAPS_FILE = "huddle_claps.json"
_DRILLS_FILE = "coach_drill_assignments.json"


def _load_broadcasts() -> dict:
    return _load_json(_BROADCASTS_FILE)


def _save_broadcasts(data: dict):
    _save_json(_BROADCASTS_FILE, data)


def _load_claps() -> dict:
    return _load_json(_CLAPS_FILE)


def _save_claps(data: dict):
    _save_json(_CLAPS_FILE, data)


def _load_drills() -> dict:
    return _load_json(_DRILLS_FILE)


def _save_drills(data: dict):
    _save_json(_DRILLS_FILE, data)


# ─── Models ─────────────────────────────────────────────────────────────────


class BroadcastRequest(BaseModel):
    message: Optional[str] = None
    voice_note_url: Optional[str] = None
    athlete_ids: Optional[list[str]] = None  # None = entire roster


class ReplyRequest(BaseModel):
    broadcast_id: str
    message: Optional[str] = None
    voice_note_url: Optional[str] = None


class DrillAssignmentRequest(BaseModel):
    drill_name: str
    drill_description: Optional[str] = None
    sport: Optional[str] = None
    scheduled_for: str  # ISO date "2026-04-20"
    athlete_ids: list[str]  # subset of roster


# ─── Broadcast ──────────────────────────────────────────────────────────────


@router.post("/coach/{coach_id}/broadcast")
async def coach_broadcast(
    coach_id: str,
    req: BroadcastRequest,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """Send a voice note or text to N athletes in a coach's roster."""
    if not req.message and not req.voice_note_url:
        raise HTTPException(400, "provide message or voice_note_url")

    rosters = _load_rosters()
    roster_ids: list[str] = rosters.get(coach_id, {}).get("athletes", [])
    if not roster_ids:
        raise HTTPException(404, "coach roster is empty")

    target_ids = req.athlete_ids if req.athlete_ids else roster_ids
    unknown = [aid for aid in target_ids if aid not in roster_ids]
    if unknown:
        raise HTTPException(400, f"athletes not in roster: {unknown}")

    broadcast_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    broadcasts = _load_broadcasts()
    broadcasts[broadcast_id] = {
        "broadcast_id": broadcast_id,
        "coach_id": coach_id,
        "message": req.message,
        "voice_note_url": req.voice_note_url,
        "target_athlete_ids": target_ids,
        "reply_count": 0,
        "replies": {},
        "sent_at": now,
    }
    _save_broadcasts(broadcasts)

    log.info(
        "broadcast sent",
        extra={"coach_id": coach_id, "broadcast_id": broadcast_id, "recipients": len(target_ids)},
    )
    return {
        "broadcast_id": broadcast_id,
        "coach_id": coach_id,
        "recipients": len(target_ids),
        "sent_at": now,
    }


# ─── Athlete reply ───────────────────────────────────────────────────────────


@router.post("/athlete/{athlete_id}/reply")
async def athlete_reply(
    athlete_id: str,
    req: ReplyRequest,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """Athlete sends a private reply to a coach broadcast."""
    if not req.message and not req.voice_note_url:
        raise HTTPException(400, "provide message or voice_note_url")

    broadcasts = _load_broadcasts()
    bc = broadcasts.get(req.broadcast_id)
    if not bc:
        raise HTTPException(404, "broadcast not found")

    if athlete_id not in bc["target_athlete_ids"]:
        raise HTTPException(403, "athlete not in broadcast target list")

    reply_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # replies is a list — multiple replies from same athlete are preserved
    if not isinstance(bc["replies"], list):
        # migrate old dict format
        bc["replies"] = list(bc["replies"].values())
    bc["replies"].append(
        {
            "reply_id": reply_id,
            "athlete_id": athlete_id,
            "message": req.message,
            "voice_note_url": req.voice_note_url,
            "replied_at": now,
        }
    )
    bc["reply_count"] = len(bc["replies"])
    _save_broadcasts(broadcasts)

    log.info(
        "athlete replied to broadcast",
        extra={"athlete_id": athlete_id, "broadcast_id": req.broadcast_id},
    )
    return {"reply_id": reply_id, "broadcast_id": req.broadcast_id, "athlete_id": athlete_id}


# ─── Coach inbox ─────────────────────────────────────────────────────────────


@router.get("/coach/{coach_id}/inbox")
async def coach_inbox(
    coach_id: str,
    limit: int = 20,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """
    Return recent broadcasts with their per-athlete replies.
    Ordered newest-first. Each row is one broadcast with reply threads grouped
    by athlete so coach sees 8 replies not one chaotic thread.
    """
    broadcasts = _load_broadcasts()
    mine = sorted(
        [bc for bc in broadcasts.values() if bc["coach_id"] == coach_id],
        key=lambda bc: bc["sent_at"],
        reverse=True,
    )[:limit]

    result = []
    for bc in mine:
        replies_raw = bc["replies"] if isinstance(bc["replies"], list) else list(bc["replies"].values())
        replies_by_athlete = []
        for r in replies_raw:
            aid = r["athlete_id"]
            a = ATHLETE_DB.get(aid) or {}
            replies_by_athlete.append(
                {
                    "athlete_id": aid,
                    "athlete_name": a.get("name", aid),
                    "replied_at": r["replied_at"],
                    "message": r.get("message"),
                    "voice_note_url": r.get("voice_note_url"),
                }
            )
        replies_by_athlete.sort(key=lambda r: r["replied_at"])
        result.append(
            {
                "broadcast_id": bc["broadcast_id"],
                "sent_at": bc["sent_at"],
                "message": bc.get("message"),
                "voice_note_url": bc.get("voice_note_url"),
                "recipient_count": len(bc["target_athlete_ids"]),
                "reply_count": bc["reply_count"],
                "replies": replies_by_athlete,
            }
        )

    return {"coach_id": coach_id, "broadcasts": result, "total": len(result)}


# ─── Athlete inbox (see coach broadcasts + clap activity) ───────────────────


@router.get("/athlete/{athlete_id}/inbox")
async def athlete_inbox(
    athlete_id: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """Athlete's inbox: all broadcasts addressed to them + clap notifications."""
    broadcasts = _load_broadcasts()
    mine = sorted(
        [bc for bc in broadcasts.values() if athlete_id in bc["target_athlete_ids"]],
        key=lambda bc: bc["sent_at"],
        reverse=True,
    )[:20]

    result = []
    for bc in mine:
        replies_raw = bc["replies"] if isinstance(bc["replies"], list) else list(bc["replies"].values())
        my_replies = [r for r in replies_raw if r["athlete_id"] == athlete_id]
        my_reply = my_replies[-1] if my_replies else None
        result.append(
            {
                "broadcast_id": bc["broadcast_id"],
                "coach_id": bc["coach_id"],
                "sent_at": bc["sent_at"],
                "message": bc.get("message"),
                "voice_note_url": bc.get("voice_note_url"),
                "replied": my_reply is not None,
                "my_reply": my_reply,
            }
        )

    return {"athlete_id": athlete_id, "broadcasts": result}


# ─── Clap reactions ──────────────────────────────────────────────────────────


@router.post("/athlete/{athlete_id}/clap/{target_id}")
async def clap_pb(
    athlete_id: str,
    target_id: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """
    Huddle-mate clap reaction on a PB. One clap per (athlete_id, target_id) pair.
    No comment thread. Target sees 'Athlete B clapped.' — nothing else.
    """
    if athlete_id == target_id:
        raise HTTPException(400, "cannot clap your own PB")

    if target_id not in ATHLETE_DB:
        raise HTTPException(404, "target athlete not found")

    claps = _load_claps()
    key = f"{athlete_id}::{target_id}"
    if key in claps:
        return {"status": "already_clapped", "athlete_id": athlete_id, "target_id": target_id}

    now = datetime.now(timezone.utc).isoformat()
    claps[key] = {
        "from_athlete_id": athlete_id,
        "to_athlete_id": target_id,
        "clapped_at": now,
    }
    _save_claps(claps)

    clap_count = sum(1 for k in claps if k.endswith(f"::{target_id}"))
    target_name = ATHLETE_DB.get(target_id, {}).get("name", target_id)
    from_name = ATHLETE_DB.get(athlete_id, {}).get("name", athlete_id)

    log.info("clap registered", extra={"from": athlete_id, "to": target_id})
    return {
        "status": "clapped",
        "athlete_id": athlete_id,
        "target_id": target_id,
        "notification": f"{from_name} clapped for {target_name}.",
        "total_claps_for_target": clap_count,
    }


@router.get("/athlete/{athlete_id}/claps")
async def get_claps_for_athlete(
    athlete_id: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """Return everyone who has clapped for this athlete."""
    claps = _load_claps()
    received = [
        {
            "from_athlete_id": v["from_athlete_id"],
            "from_name": ATHLETE_DB.get(v["from_athlete_id"], {}).get("name", v["from_athlete_id"]),
            "clapped_at": v["clapped_at"],
        }
        for k, v in claps.items()
        if k.endswith(f"::{athlete_id}")
    ]
    received.sort(key=lambda c: c["clapped_at"], reverse=True)
    return {"athlete_id": athlete_id, "claps": received, "total": len(received)}


# ─── Bulk drill assignment ───────────────────────────────────────────────────


@router.post("/coach/{coach_id}/drill-assignment")
async def assign_drill(
    coach_id: str,
    req: DrillAssignmentRequest,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """
    Coach bulk-assigns tomorrow's drill to a subset of the roster.
    Each athlete sees the drill pre-selected in their next session.
    BE-ONLY: no push notification yet — athlete sees it when they open the app.
    """
    if not req.athlete_ids:
        raise HTTPException(400, "athlete_ids must be non-empty")

    rosters = _load_rosters()
    roster_ids = rosters.get(coach_id, {}).get("athletes", [])
    out_of_roster = [aid for aid in req.athlete_ids if aid not in roster_ids]
    if out_of_roster:
        raise HTTPException(400, f"athletes not in roster: {out_of_roster}")

    assignment_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    drills = _load_drills()
    drills[assignment_id] = {
        "assignment_id": assignment_id,
        "coach_id": coach_id,
        "drill_name": req.drill_name,
        "drill_description": req.drill_description,
        "sport": req.sport,
        "scheduled_for": req.scheduled_for,
        "athlete_ids": req.athlete_ids,
        "created_at": now,
    }
    _save_drills(drills)

    log.info(
        "drill assigned",
        extra={
            "coach_id": coach_id,
            "assignment_id": assignment_id,
            "athletes": len(req.athlete_ids),
            "scheduled_for": req.scheduled_for,
        },
    )
    return {
        "assignment_id": assignment_id,
        "drill_name": req.drill_name,
        "scheduled_for": req.scheduled_for,
        "assigned_to": len(req.athlete_ids),
    }


@router.get("/athlete/{athlete_id}/drill-assignments")
async def athlete_drill_assignments(
    athlete_id: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """Athlete fetches their upcoming drill assignments from coach."""
    drills = _load_drills()
    today = datetime.now(timezone.utc).date().isoformat()
    upcoming = sorted(
        [d for d in drills.values() if athlete_id in d.get("athlete_ids", []) and d.get("scheduled_for", "") >= today],
        key=lambda d: d["scheduled_for"],
    )
    return {"athlete_id": athlete_id, "assignments": upcoming}
