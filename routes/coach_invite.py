from __future__ import annotations

"""
WhatsApp invite-link generation for coach roster onboarding.

Flow: Coach generates a one-tap link → shares to WhatsApp → athlete taps →
installs app → lands inside coach's roster with no email signup step.
Target: <90 seconds from tap to first session pick.

Endpoints:
  POST /coach/{id}/invite-link          — generate invite token + link
  GET  /invite/{token}                  — validate token, return roster context
  POST /invite/{token}/accept           — athlete accepts, joins roster
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import require_coach_or_admin
from config import settings
from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from routes.coach_roster import _load_rosters, _save_rosters

router = APIRouter(tags=["Coach Invite"])
log = get_logger("routes.coach_invite")

_INVITES_FILE = "coach_invites.json"
_TOKEN_TTL_HOURS = 72


def _load_invites() -> dict:
    return _load_json(_INVITES_FILE)


def _save_invites(data: dict):
    _save_json(_INVITES_FILE, data)


def _base_url() -> str:
    return getattr(settings, "public_base_url", "https://personalhealth.app")


class InviteLinkRequest(BaseModel):
    label: Optional[str] = None  # e.g. "Sprint group April"
    max_uses: int = 10
    ttl_hours: int = 72


class AcceptInviteRequest(BaseModel):
    athlete_id: str  # ID of the athlete accepting


@router.post("/coach/{coach_id}/invite-link")
async def generate_invite_link(
    coach_id: str,
    req: InviteLinkRequest,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """
    Generate a WhatsApp-shareable invite link. Athlete taps the link, installs
    the app, and lands inside the coach's roster without email signup.
    """
    token = secrets.token_urlsafe(20)
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(hours=req.ttl_hours)).isoformat()

    invites = _load_invites()
    invites[token] = {
        "token": token,
        "coach_id": coach_id,
        "label": req.label,
        "max_uses": req.max_uses,
        "uses": 0,
        "created_at": now.isoformat(),
        "expires_at": expires_at,
        "accepted_by": [],
    }
    _save_invites(invites)

    deep_link = f"{_base_url()}/invite/{token}"
    whatsapp_text = (
        f"Train with me on Personal Health — tap to join my coaching roster:\n{deep_link}\n"
        f"(Link valid for {req.ttl_hours} hours)"
    )

    log.info("invite link created", extra={"coach_id": coach_id, "token": token})
    return {
        "token": token,
        "invite_url": deep_link,
        "whatsapp_share_text": whatsapp_text,
        "expires_at": expires_at,
        "max_uses": req.max_uses,
    }


@router.get("/invite/{token}")
async def validate_invite(token: str):
    """
    Validate an invite token. Returns coach context so the app can show a
    branded landing screen before the athlete accepts.
    """
    invites = _load_invites()
    invite = invites.get(token)
    if not invite:
        raise HTTPException(404, "invite link not found or expired")

    now = datetime.now(timezone.utc)
    expires = datetime.fromisoformat(invite["expires_at"])
    if now > expires:
        raise HTTPException(410, "invite link has expired")

    if invite["uses"] >= invite["max_uses"]:
        raise HTTPException(410, "invite link has reached max uses")

    coach_id = invite["coach_id"]
    coach = ATHLETE_DB.get(coach_id) or {}

    return {
        "valid": True,
        "token": token,
        "coach_id": coach_id,
        "coach_name": coach.get("name", coach_id),
        "label": invite.get("label"),
        "expires_at": invite["expires_at"],
        "uses_remaining": invite["max_uses"] - invite["uses"],
    }


@router.post("/invite/{token}/accept")
async def accept_invite(token: str, req: AcceptInviteRequest):
    """
    Athlete accepts invite — adds them to the coach's roster.
    Called right after app install so the athlete skips the roster discovery step.
    """
    invites = _load_invites()
    invite = invites.get(token)
    if not invite:
        raise HTTPException(404, "invite link not found")

    now = datetime.now(timezone.utc)
    expires = datetime.fromisoformat(invite["expires_at"])
    if now > expires:
        raise HTTPException(410, "invite link has expired")

    if invite["uses"] >= invite["max_uses"]:
        raise HTTPException(410, "invite link has reached max uses")

    athlete_id = req.athlete_id
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found — create your profile first")

    coach_id = invite["coach_id"]
    rosters = _load_rosters()
    if coach_id not in rosters:
        rosters[coach_id] = {"athletes": [], "created_at": now.isoformat()}

    roster = rosters[coach_id]["athletes"]
    if athlete_id not in roster:
        if len(roster) >= 50:
            raise HTTPException(400, "coach roster is full")
        roster.append(athlete_id)
        _save_rosters(rosters)

    if athlete_id not in invite["accepted_by"]:
        invite["accepted_by"].append(athlete_id)
        invite["uses"] += 1
        _save_invites(invites)

    log.info(
        "invite accepted",
        extra={"athlete_id": athlete_id, "coach_id": coach_id, "token": token},
    )
    return {
        "status": "joined",
        "athlete_id": athlete_id,
        "coach_id": coach_id,
        "message": f"You've joined {ATHLETE_DB.get(coach_id, {}).get('name', 'your coach')}'s roster.",
    }


@router.get("/coach/{coach_id}/invite-links")
async def list_invite_links(
    coach_id: str,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """List all active invite links for a coach."""
    invites = _load_invites()
    now = datetime.now(timezone.utc)
    mine = [
        {
            "token": v["token"],
            "label": v.get("label"),
            "uses": v["uses"],
            "max_uses": v["max_uses"],
            "expires_at": v["expires_at"],
            "expired": datetime.fromisoformat(v["expires_at"]) < now,
            "invite_url": f"{_base_url()}/invite/{v['token']}",
        }
        for v in invites.values()
        if v["coach_id"] == coach_id
    ]
    mine.sort(key=lambda x: x["expires_at"], reverse=True)
    return {"coach_id": coach_id, "invite_links": mine}
