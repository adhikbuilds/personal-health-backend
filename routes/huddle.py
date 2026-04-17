from __future__ import annotations

"""
Huddle Mode — REST endpoints for group training sessions.
"""

import asyncio
import json
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from database import ATHLETE_DB
from logging_setup import get_logger
from services.huddle import (
    _get_huddle,
    _load_huddles,
    create_huddle,
    end_huddle,
    get_huddle_live,
    join_huddle,
    leave_huddle,
    start_huddle,
)

# Per-huddle WS listeners (dashboards, observer phones)
_HUDDLE_WATCHERS: dict[str, list[WebSocket]] = defaultdict(list)

logger = get_logger("routes.huddle")

router = APIRouter(prefix="/huddle", tags=["Huddle"])


# ─── Request Models ────────────────────────────────────────────────────────


class CreateHuddleRequest(BaseModel):
    name: str
    sport: str
    coach_id: Optional[str] = None
    max_athletes: int = 20


class JoinHuddleRequest(BaseModel):
    athlete_id: str


# ─── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/create")
async def api_create_huddle(req: CreateHuddleRequest):
    """Create a new huddle group training session."""
    if req.coach_id and req.coach_id not in ATHLETE_DB:
        raise HTTPException(400, f"Coach athlete_id '{req.coach_id}' not found")
    if req.max_athletes < 2:
        raise HTTPException(400, "max_athletes must be at least 2")
    huddle = create_huddle(
        name=req.name,
        sport=req.sport,
        coach_id=req.coach_id,
        max_athletes=req.max_athletes,
    )
    return {"ok": True, "huddle": huddle.__dict__}


@router.post("/{huddle_id}/join")
async def api_join_huddle(huddle_id: str, req: JoinHuddleRequest):
    """Join an existing huddle."""
    if req.athlete_id not in ATHLETE_DB:
        raise HTTPException(400, f"Athlete '{req.athlete_id}' not found")
    try:
        result = join_huddle(huddle_id, req.athlete_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    await _notify_watchers(huddle_id, "athlete_joined", {"athlete_id": req.athlete_id})
    return {"ok": True, **result}


@router.post("/{huddle_id}/leave")
async def api_leave_huddle(huddle_id: str, req: JoinHuddleRequest):
    """Leave a huddle."""
    try:
        result = leave_huddle(huddle_id, req.athlete_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    await _notify_watchers(huddle_id, "athlete_left", {"athlete_id": req.athlete_id})
    return {"ok": True, **result}


@router.post("/{huddle_id}/start")
async def api_start_huddle(huddle_id: str):
    """Start a huddle — transitions from waiting to active."""
    try:
        huddle = start_huddle(huddle_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    await _notify_watchers(huddle_id, "huddle_started", {})
    return {"ok": True, "huddle": huddle.__dict__}


@router.post("/{huddle_id}/end")
async def api_end_huddle(huddle_id: str):
    """End a huddle and compute final leaderboard."""
    try:
        huddle = end_huddle(huddle_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    await _notify_watchers(huddle_id, "huddle_ended", {"leaderboard": huddle.leaderboard})
    return {"ok": True, "huddle": huddle.__dict__}


@router.get("/{huddle_id}")
async def api_get_huddle(huddle_id: str):
    """Get full huddle state."""
    try:
        huddle = _get_huddle(huddle_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    return huddle.__dict__


@router.get("/{huddle_id}/live")
async def api_get_huddle_live(huddle_id: str):
    """Get live leaderboard and per-athlete stats."""
    try:
        return get_huddle_live(huddle_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


# NOTE: This route uses a separate router-level path so it doesn't clash
# with the /{huddle_id} path. We mount it at /huddles below.

_list_router = APIRouter(tags=["Huddle"])


@_list_router.get("/huddles")
async def api_list_huddles(status: Optional[str] = Query(None)):
    """List all huddles, optionally filtered by status."""
    huddles = _load_huddles()
    results = list(huddles.values())
    if status:
        if status not in ("waiting", "active", "ended"):
            raise HTTPException(400, f"Invalid status filter '{status}'. Use waiting|active|ended")
        results = [h for h in results if h.get("status") == status]
    results.sort(key=lambda h: h.get("created_at", ""), reverse=True)
    return {"huddles": results, "total": len(results)}


# ─── Live leaderboard WebSocket ─────────────────────────────────────────────


async def _notify_watchers(huddle_id: str, event: str, payload: dict) -> None:
    """Push a structured event to all dashboard/phone watchers of this huddle."""
    dead = []
    try:
        live = get_huddle_live(huddle_id)
    except KeyError:
        live = None
    msg = json.dumps({"event": event, "payload": payload, "live": live}, default=str)
    for ws in list(_HUDDLE_WATCHERS.get(huddle_id, [])):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns = _HUDDLE_WATCHERS.get(huddle_id, [])
        if ws in conns:
            conns.remove(ws)


@router.websocket("/{huddle_id}/watch")
async def api_watch_huddle(websocket: WebSocket, huddle_id: str):
    """WebSocket stream of leaderboard updates. Sends snapshot every 5s plus on events."""
    try:
        _get_huddle(huddle_id)
    except KeyError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    _HUDDLE_WATCHERS[huddle_id].append(websocket)
    logger.info("huddle watcher connected", extra={"huddle_id": huddle_id})
    try:
        # Immediate snapshot
        try:
            await websocket.send_json({"event": "snapshot", "live": get_huddle_live(huddle_id)})
        except Exception:
            pass
        while True:
            await asyncio.sleep(5)
            try:
                await websocket.send_json({"event": "tick", "live": get_huddle_live(huddle_id)})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        conns = _HUDDLE_WATCHERS.get(huddle_id, [])
        if websocket in conns:
            conns.remove(websocket)


list_router = _list_router
