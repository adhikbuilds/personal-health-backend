from __future__ import annotations

"""
Expo Push Notification token registry + delivery.

Android app registers its Expo push token on startup.
Backend stores it, and send_push() is called whenever a notification
is generated that needs live delivery (PB, priority alert, broadcast).

Endpoints:
  POST /push-token/register     — app registers or refreshes its token
  DELETE /push-token/{athlete_id} — app revokes token (logout / uninstall)

Internal helper:
  send_push(athlete_id, title, body, data={}) — fire-and-forget Expo push
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import _load_json, _save_json
from logging_setup import get_logger

router = APIRouter(prefix="/push-token", tags=["Push Notifications"])
log = get_logger("routes.push_tokens")

_TOKENS_FILE = "push_tokens.json"
_EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    log.warning("httpx not installed — push notifications will be stubbed")


def _load_tokens() -> dict:
    return _load_json(_TOKENS_FILE)


def _save_tokens(data: dict):
    _save_json(_TOKENS_FILE, data)


class TokenRequest(BaseModel):
    athlete_id: str
    expo_push_token: str  # e.g. "ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx]"
    platform: Optional[str] = "android"


@router.post("/register")
async def register_push_token(req: TokenRequest):
    """
    Called by the Android app on launch (and whenever the token changes).
    Overwrites previous token for the same athlete.
    """
    if not req.expo_push_token.startswith("ExponentPushToken["):
        raise HTTPException(400, "invalid Expo push token format")

    tokens = _load_tokens()
    tokens[req.athlete_id] = {
        "athlete_id": req.athlete_id,
        "token": req.expo_push_token,
        "platform": req.platform,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_tokens(tokens)
    log.info("push token registered", extra={"athlete_id": req.athlete_id})
    return {"status": "registered", "athlete_id": req.athlete_id}


@router.delete("/{athlete_id}")
async def revoke_push_token(athlete_id: str):
    """App revokes token on logout or uninstall callback."""
    tokens = _load_tokens()
    if athlete_id in tokens:
        del tokens[athlete_id]
        _save_tokens(tokens)
    return {"status": "revoked", "athlete_id": athlete_id}


# ─── Internal delivery helper (called from other routes) ─────────────────────


async def send_push(
    athlete_id: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
) -> bool:
    """
    Fire-and-forget Expo push to a single athlete.
    Returns True if the push was sent, False if no token or send failed.
    Logs errors but never raises — callers must not depend on delivery.
    """
    tokens = _load_tokens()
    record = tokens.get(athlete_id)
    if not record:
        return False

    token = record["token"]
    payload = {
        "to": token,
        "title": title,
        "body": body,
        "sound": "default",
        "data": data or {},
        "priority": "high",
    }

    if not HTTPX_AVAILABLE:
        log.info(
            "push stub (httpx missing)",
            extra={"athlete_id": athlete_id, "title": title},
        )
        return True

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                _EXPO_PUSH_URL,
                json=payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                log.warning(
                    "expo push non-200",
                    extra={"status": resp.status_code, "athlete_id": athlete_id},
                )
                return False

            result = resp.json()
            ticket = (result.get("data") or [{}])[0]
            if ticket.get("status") == "error":
                details = ticket.get("details", {})
                if details.get("error") == "DeviceNotRegistered":
                    # Token is stale — remove it
                    tokens.pop(athlete_id, None)
                    _save_tokens(tokens)
                log.warning("expo push error", extra={"ticket": ticket, "athlete_id": athlete_id})
                return False

            log.info("push sent", extra={"athlete_id": athlete_id, "title": title})
            return True

    except Exception as e:
        log.error("push send exception", extra={"error": str(e), "athlete_id": athlete_id})
        return False
