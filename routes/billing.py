from __future__ import annotations

"""
Billing — Razorpay subscription and payment webhook.

Coach invoices athletes monthly. UPI/card via Razorpay.
If the trainer has to chase WhatsApp for money, the product failed.

Endpoints:
  POST /billing/subscription            — create a Razorpay subscription for an athlete
  POST /billing/webhook                 — Razorpay payment webhook (signature-verified)
  GET  /billing/coach/{coach_id}        — coach sees who's paid / who's overdue
  GET  /billing/athlete/{athlete_id}    — athlete sees their payment status
  POST /billing/pause/{athlete_id}      — coach pauses access after non-payment (day 15)
  POST /billing/override/{athlete_id}   — coach manually overrides pause

Razorpay SDK is optional — endpoints degrade gracefully if not installed.
Set env vars: RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET
"""

import hashlib
import hmac
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from auth import require_coach_or_admin
from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from routes.coach_roster import _load_rosters

router = APIRouter(prefix="/billing", tags=["Billing"])
log = get_logger("routes.billing")

_BILLING_FILE = "billing.json"
_RAZORPAY_KEY = os.getenv("RAZORPAY_KEY_ID", "")
_RAZORPAY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

try:
    import razorpay

    _rp_client = razorpay.Client(auth=(_RAZORPAY_KEY, _RAZORPAY_SECRET)) if _RAZORPAY_KEY else None
    RAZORPAY_AVAILABLE = bool(_RAZORPAY_KEY)
except ImportError:
    _rp_client = None
    RAZORPAY_AVAILABLE = False
    log.warning("razorpay SDK not installed — billing will use stub mode")


def _load_billing() -> dict:
    return _load_json(_BILLING_FILE)


def _save_billing(data: dict):
    _save_json(_BILLING_FILE, data)


def _billing_key(coach_id: str, athlete_id: str) -> str:
    return f"{coach_id}::{athlete_id}"


class SubscriptionRequest(BaseModel):
    athlete_id: str
    amount_inr: int = 1000  # ₹1000 default
    plan_name: Optional[str] = "Monthly Coaching"


class PauseRequest(BaseModel):
    reason: Optional[str] = "payment_overdue"


# ─── Subscription creation ───────────────────────────────────────────────────


@router.post("/subscription")
async def create_subscription(
    coach_id: str,
    req: SubscriptionRequest,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """
    Coach creates a monthly subscription for one athlete.
    Returns a Razorpay payment link the athlete/parent can tap to pay via UPI.
    Falls back to stub mode if Razorpay is not configured.
    """
    rosters = _load_rosters()
    if req.athlete_id not in rosters.get(coach_id, {}).get("athletes", []):
        raise HTTPException(400, "athlete not in this coach's roster")

    athlete = ATHLETE_DB.get(req.athlete_id)
    if not athlete:
        raise HTTPException(404, "athlete not found")

    billing = _load_billing()
    key = _billing_key(coach_id, req.athlete_id)
    now = datetime.now(timezone.utc)
    due_date = (now + timedelta(days=30)).date().isoformat()

    if RAZORPAY_AVAILABLE and _rp_client:
        try:
            payment_link = _rp_client.payment_link.create(
                {
                    "amount": req.amount_inr * 100,  # paise
                    "currency": "INR",
                    "description": f"{req.plan_name} — {athlete.get('name', req.athlete_id)}",
                    "customer": {
                        "name": athlete.get("name", req.athlete_id),
                    },
                    "notify": {"sms": False, "email": False},
                    "reminder_enable": True,
                    "callback_method": "get",
                }
            )
            payment_url = payment_link.get("short_url", "")
            razorpay_link_id = payment_link.get("id", "")
        except Exception as e:
            log.error("razorpay payment link failed", extra={"error": str(e)})
            raise HTTPException(502, f"Razorpay error: {e}")
    else:
        # Stub mode — UPI deeplink placeholder
        payment_url = f"upi://pay?pa=coach.{coach_id}@upi&pn=PersonalHealth&am={req.amount_inr}&cu=INR"
        razorpay_link_id = None

    record_id = str(uuid.uuid4())
    billing[key] = billing.get(key) or {}
    billing[key].update(
        {
            "record_id": record_id,
            "coach_id": coach_id,
            "athlete_id": req.athlete_id,
            "amount_inr": req.amount_inr,
            "plan_name": req.plan_name,
            "status": "pending",
            "payment_url": payment_url,
            "razorpay_link_id": razorpay_link_id,
            "due_date": due_date,
            "created_at": now.isoformat(),
            "paid_at": None,
            "paused": False,
            "paused_at": None,
        }
    )
    _save_billing(billing)

    whatsapp_msg = (
        f"₹{req.amount_inr} due to your coach for {req.plan_name}.\n"
        f"Pay via UPI: {payment_url}\n"
        f"Due by: {due_date}"
    )

    log.info("subscription created", extra={"key": key, "amount": req.amount_inr})
    return {
        "record_id": record_id,
        "athlete_id": req.athlete_id,
        "amount_inr": req.amount_inr,
        "due_date": due_date,
        "payment_url": payment_url,
        "whatsapp_reminder_text": whatsapp_msg,
        "razorpay_available": RAZORPAY_AVAILABLE,
    }


# ─── Razorpay webhook ────────────────────────────────────────────────────────


@router.post("/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(default=None),
):
    """
    Razorpay payment webhook. Marks subscription paid on payment.captured event.
    Validates HMAC-SHA256 signature when RAZORPAY_WEBHOOK_SECRET is set.
    """
    body = await request.body()

    if _WEBHOOK_SECRET and x_razorpay_signature:
        expected = hmac.new(_WEBHOOK_SECRET.encode(), body, digestmod=hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, x_razorpay_signature):
            log.warning("razorpay webhook signature mismatch")
            raise HTTPException(400, "invalid webhook signature")
    elif _WEBHOOK_SECRET and not x_razorpay_signature:
        raise HTTPException(400, "missing X-Razorpay-Signature header")

    import json

    try:
        payload = json.loads(body)
    except Exception:
        raise HTTPException(400, "invalid JSON payload")

    event = payload.get("event", "")
    if event != "payment.captured":
        return {"status": "ignored", "event": event}

    payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
    razorpay_link_id = payment.get("invoice_id") or payment.get("payment_link_id")

    billing = _load_billing()
    matched_key = None
    for key, rec in billing.items():
        if rec.get("razorpay_link_id") == razorpay_link_id:
            matched_key = key
            break

    if not matched_key:
        log.warning("webhook: no billing record for razorpay_link_id", extra={"link_id": razorpay_link_id})
        return {"status": "no_match"}

    billing[matched_key]["status"] = "paid"
    billing[matched_key]["paid_at"] = datetime.now(timezone.utc).isoformat()
    billing[matched_key]["paused"] = False
    _save_billing(billing)

    log.info("payment marked paid", extra={"key": matched_key})
    return {"status": "ok", "key": matched_key}


# ─── Coach billing dashboard ─────────────────────────────────────────────────


@router.get("/coach/{coach_id}")
async def coach_billing_overview(
    coach_id: str,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """Coach sees who's paid, who's overdue, and who's paused."""
    billing = _load_billing()
    records = [r for r in billing.values() if r.get("coach_id") == coach_id]

    paid = [r for r in records if r["status"] == "paid"]
    pending = [r for r in records if r["status"] == "pending" and not r.get("paused")]
    overdue = []
    now = datetime.now(timezone.utc).date().isoformat()
    for r in pending:
        if r.get("due_date", "9999") < now:
            overdue.append(r)
    paused = [r for r in records if r.get("paused")]

    def _enrich(r: dict) -> dict:
        a = ATHLETE_DB.get(r["athlete_id"]) or {}
        return {**r, "athlete_name": a.get("name", r["athlete_id"])}

    return {
        "coach_id": coach_id,
        "summary": {
            "total": len(records),
            "paid": len(paid),
            "pending": len(pending),
            "overdue": len(overdue),
            "paused": len(paused),
        },
        "paid": [_enrich(r) for r in paid],
        "overdue": [_enrich(r) for r in overdue],
        "paused": [_enrich(r) for r in paused],
    }


@router.get("/athlete/{athlete_id}")
async def athlete_billing_status(athlete_id: str):
    """Athlete checks their own payment status. No auth — shown in-app as a gentle banner."""
    billing = _load_billing()
    records = [r for r in billing.values() if r.get("athlete_id") == athlete_id]
    if not records:
        return {"athlete_id": athlete_id, "status": "no_subscription"}

    latest = sorted(records, key=lambda r: r.get("created_at", ""), reverse=True)[0]
    return {
        "athlete_id": athlete_id,
        "status": latest["status"],
        "amount_inr": latest.get("amount_inr"),
        "due_date": latest.get("due_date"),
        "paused": latest.get("paused", False),
        "payment_url": latest.get("payment_url"),
    }


@router.post("/pause/{athlete_id}")
async def pause_access(
    coach_id: str,
    athlete_id: str,
    req: PauseRequest,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """Coach pauses an athlete's access after 15 days non-payment."""
    billing = _load_billing()
    key = _billing_key(coach_id, athlete_id)
    if key not in billing:
        raise HTTPException(404, "no billing record for this coach/athlete pair")

    billing[key]["paused"] = True
    billing[key]["paused_at"] = datetime.now(timezone.utc).isoformat()
    billing[key]["pause_reason"] = req.reason
    _save_billing(billing)
    return {"status": "paused", "athlete_id": athlete_id, "reason": req.reason}


@router.post("/override/{athlete_id}")
async def override_pause(
    coach_id: str,
    athlete_id: str,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """Coach manually lifts a payment pause (e.g. offline cash collected)."""
    billing = _load_billing()
    key = _billing_key(coach_id, athlete_id)
    if key not in billing:
        raise HTTPException(404, "no billing record for this coach/athlete pair")

    billing[key]["paused"] = False
    billing[key]["paused_at"] = None
    billing[key]["status"] = "paid"
    billing[key]["paid_at"] = datetime.now(timezone.utc).isoformat()
    _save_billing(billing)
    return {"status": "override_applied", "athlete_id": athlete_id}
