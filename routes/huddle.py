from __future__ import annotations

"""
Huddle Mode — REST endpoints for group training sessions.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
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
    start_huddle,
)

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


list_router = _list_router
