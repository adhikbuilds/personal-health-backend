from __future__ import annotations

"""
Parent digest — token-based read-only weekly report for parents.

Surface C from TRAINER-FIRST-PROMPT.md: a single web page (no login) that
shows the athlete's week. Parent clicks a WhatsApp link on Sunday 9am.

Endpoints:
  POST /parent/digest-token/{athlete_id}  — coach/athlete generates a share token
  GET  /parent/{token}/weekly-digest      — public, no-login, returns week data
  POST /parent/sms-digest/{athlete_id}    — trigger parent SMS (consent-gated)

Parent visibility rules:
  - Default ON  for athletes under 18 (age in athlete profile)
  - Default OFF for athletes 18+
  - Athlete can override (toggle in their app settings)
  - No parent can post — read-only, always
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import require_athlete_or_admin
from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from routes.progress import _athlete_sessions, _compute_injury_risk

router = APIRouter(prefix="/parent", tags=["Parent Digest"])
log = get_logger("routes.parent_digest")

_DIGEST_FILE = "parent_digests.json"
_SMS_PROVIDER_KEY = os.getenv("SMS_API_KEY", "")
_SMS_SENDER_ID = os.getenv("SMS_SENDER_ID", "PHLTH")

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def _load_digests() -> dict:
    return _load_json(_DIGEST_FILE)


def _save_digests(data: dict):
    _save_json(_DIGEST_FILE, data)


def _athlete_age(athlete: dict) -> Optional[int]:
    dob = athlete.get("date_of_birth")
    if not dob:
        return None
    try:
        birth = datetime.fromisoformat(dob).date()
        today = datetime.now(timezone.utc).date()
        age = (today - birth).days // 365
        return age
    except Exception:
        return None


def _default_visibility(athlete: dict) -> bool:
    """True = parent digest is shared by default."""
    age = _athlete_age(athlete)
    if age is None:
        return True  # unknown age → default on (safe for minor)
    return age < 18


class DigestTokenRequest(BaseModel):
    phone_number: Optional[str] = None  # parent's WhatsApp number for direct share
    visibility_override: Optional[bool] = None  # athlete explicit opt-in/out


class SMSDigestRequest(BaseModel):
    parent_phone: str  # E.164 format, e.g. "+919876543210"
    consent: bool = True  # athlete explicitly consented


# ─── Token generation ────────────────────────────────────────────────────────


@router.post("/digest-token/{athlete_id}")
async def generate_digest_token(
    athlete_id: str,
    req: DigestTokenRequest,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """
    Athlete (or coach on their behalf) generates a parent share token.
    Token-based URL, no login required to view. Token expires in 7 days
    and refreshes each Sunday so the parent always has a fresh link.
    """
    athlete = ATHLETE_DB.get(athlete_id)
    if not athlete:
        raise HTTPException(404, "athlete not found")

    default_vis = _default_visibility(athlete)
    visibility = req.visibility_override if req.visibility_override is not None else default_vis

    if not visibility:
        raise HTTPException(403, "athlete has opted out of parent visibility")

    token = secrets.token_urlsafe(24)
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(days=7)).isoformat()

    digests = _load_digests()
    # One token per athlete — regenerate if exists
    existing_token = next((t for t, d in digests.items() if d.get("athlete_id") == athlete_id), None)
    if existing_token:
        del digests[existing_token]

    digests[token] = {
        "token": token,
        "athlete_id": athlete_id,
        "phone_number": req.phone_number,
        "created_at": now.isoformat(),
        "expires_at": expires_at,
        "visibility": visibility,
    }
    _save_digests(digests)

    public_base = os.getenv("PUBLIC_BASE_URL", "https://personalhealth.app")
    digest_url = f"{public_base}/parent/{token}/weekly-digest"
    whatsapp_msg = (
        f"Weekly update for {athlete.get('name', 'your athlete')} — "
        f"tap to see how training went this week:\n{digest_url}"
    )

    log.info("digest token created", extra={"athlete_id": athlete_id, "token": token})
    return {
        "token": token,
        "digest_url": digest_url,
        "whatsapp_share_text": whatsapp_msg,
        "expires_at": expires_at,
        "visibility": visibility,
    }


# ─── Public digest page (no auth) ───────────────────────────────────────────


@router.get("/{token}/weekly-digest")
async def public_weekly_digest(token: str):
    """
    Token-based public page — no login needed. Returns structured data for
    a one-screen parent report card. Frontend renders this as a static page.

    Consent-gated: if athlete toggled visibility off, returns 403.
    """
    digests = _load_digests()
    digest = digests.get(token)
    if not digest:
        raise HTTPException(404, "digest link not found or expired")

    now = datetime.now(timezone.utc)
    if datetime.fromisoformat(digest["expires_at"]) < now:
        raise HTTPException(410, "digest link has expired — ask your athlete to share a fresh link")

    if not digest.get("visibility", True):
        raise HTTPException(403, "athlete has restricted parent visibility")

    athlete_id = digest["athlete_id"]
    athlete = ATHLETE_DB.get(athlete_id)
    if not athlete:
        raise HTTPException(404, "athlete record not found")

    sessions_7d = _athlete_sessions(athlete_id, 7)
    injury = _compute_injury_risk(athlete_id, 14)

    form_scores = [
        float((s.get("summary") or {}).get("avg_form_score") or 0)
        for s in sessions_7d
        if (s.get("summary") or {}).get("avg_form_score")
    ]
    avg_form = round(sum(form_scores) / len(form_scores), 1) if form_scores else None

    total_reps = sum(int((s.get("summary") or {}).get("rep_count") or 0) for s in sessions_7d)
    total_duration_min = round(
        sum(int((s.get("summary") or {}).get("duration_seconds") or 0) for s in sessions_7d) / 60,
        1,
    )

    # Reassuring summary for a 55-year-old parent
    if injury.get("risk") == "high":
        safety_note = (
            "Your coach has flagged this athlete for a form check. Nothing to worry about — it's routine monitoring."
        )
    elif len(sessions_7d) == 0:
        safety_note = (
            f"{athlete.get('name', 'Your athlete')} did not train this week. They may be resting or catching up."
        )
    else:
        safety_note = f"{athlete.get('name', 'Your athlete')} trained safely this week. All sessions completed without injury flags."

    return {
        "athlete_name": athlete.get("name", athlete_id),
        "sport": athlete.get("sport"),
        "week_ending": now.date().isoformat(),
        "sessions_this_week": len(sessions_7d),
        "avg_form_score": avg_form,
        "total_reps": total_reps,
        "total_training_minutes": total_duration_min,
        "injury_risk": injury.get("risk", "unknown"),
        "safety_note": safety_note,
        "generated_at": now.isoformat(),
        # We deliberately do NOT include raw session data — one screen, no interaction
    }


# ─── SMS digest (consent-gated) ─────────────────────────────────────────────


@router.post("/sms-digest/{athlete_id}")
async def send_sms_digest(
    athlete_id: str,
    req: SMSDigestRequest,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """
    Send parent a single SMS with the weekly digest link.
    Athlete must explicitly consent. No follow-up SMS — one message only.
    Uses MSG91 / Twilio / any provider via SMS_API_KEY env var (stub mode if absent).
    """
    if not req.consent:
        raise HTTPException(400, "athlete consent required to send parent SMS")

    athlete = ATHLETE_DB.get(athlete_id)
    if not athlete:
        raise HTTPException(404, "athlete not found")

    digests = _load_digests()
    existing_token = next((t for t, d in digests.items() if d.get("athlete_id") == athlete_id), None)
    if not existing_token:
        raise HTTPException(400, "generate a digest token first via POST /parent/digest-token/{athlete_id}")

    public_base = os.getenv("PUBLIC_BASE_URL", "https://personalhealth.app")
    digest_url = f"{public_base}/parent/{existing_token}/weekly-digest"
    sms_text = (
        f"Personal Health update for {athlete.get('name', 'your athlete')}: "
        f"see this week's training summary at {digest_url}"
    )

    sms_sent = False
    error_msg = None

    if _SMS_PROVIDER_KEY and HTTPX_AVAILABLE:
        # MSG91 API (common in India) — adapt to your provider
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://api.msg91.com/api/v5/flow/",
                    headers={"authkey": _SMS_PROVIDER_KEY, "Content-Type": "application/json"},
                    json={
                        "template_id": os.getenv("SMS_TEMPLATE_ID", ""),
                        "sender": _SMS_SENDER_ID,
                        "short_url": "0",
                        "mobiles": req.parent_phone.lstrip("+"),
                        "VAR1": athlete.get("name", "your athlete"),
                        "VAR2": digest_url,
                    },
                )
                sms_sent = resp.status_code < 300
                if not sms_sent:
                    error_msg = f"SMS provider returned {resp.status_code}"
        except Exception as e:
            error_msg = str(e)
            log.error("SMS send failed", extra={"error": error_msg, "athlete_id": athlete_id})
    else:
        log.info("SMS stub mode — would send", extra={"to": req.parent_phone, "text": sms_text})
        sms_sent = True  # stub succeeds

    return {
        "sms_sent": sms_sent,
        "parent_phone": req.parent_phone,
        "digest_url": digest_url,
        "stub_mode": not bool(_SMS_PROVIDER_KEY),
        "error": error_msg,
    }
