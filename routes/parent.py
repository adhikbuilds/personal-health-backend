from __future__ import annotations

import contextlib

"""
Personal Health — Parent safety view (Flow 15).

Consent-gated, privacy-first: parents see only safety-relevant signals.
Raw joint angles, session IDs, and exact scores are never exposed.

Endpoints:
  POST /parent/consent-request                                   — parent submits access request
  POST /athlete/{athlete_id}/parent-consent/{consent_id}/accept  — athlete grants access
  POST /athlete/{athlete_id}/parent-consent/{consent_id}/revoke  — athlete revokes access
  GET  /athlete/{athlete_id}/parent-consents                     — athlete views all requests
  GET  /parent/{consent_id}/safety-summary                       — parent reads safety view (token-gated)
"""

import secrets
import statistics
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth import require_athlete_or_admin
from database import ATHLETE_DB, SESSION_DB, _load_json, _save_json

router = APIRouter(tags=["Parent"])

CONSENTS_FILE = "parent_consents.json"


# ─── Persistence helpers ─────────────────────────────────────────────────────


def _load_consents() -> dict:
    return _load_json(CONSENTS_FILE)


def _save_consents(data: dict) -> None:
    _save_json(CONSENTS_FILE, data)


# ─── Pydantic models ─────────────────────────────────────────────────────────


class ConsentRequest(BaseModel):
    parent_name: str
    parent_email: str
    athlete_id: str


# ─── Internal helpers ────────────────────────────────────────────────────────


def _get_consent(consent_id: str) -> Optional[dict]:
    consents = _load_consents()
    return consents.get(consent_id)


def _athlete_sessions_last_n_days(athlete_id: str, days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id:
            continue
        if s.get("status") != "completed":
            continue
        raw = s.get("started_at")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                out.append(s)
        except ValueError:
            continue
    return out


def _compute_safety_summary(athlete_id: str) -> dict:
    from routes.progress import _compute_injury_risk

    athlete = ATHLETE_DB[athlete_id]
    all_sessions = [
        s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"
    ]

    last_session_date: Optional[str] = None
    form_scores: list[float] = []
    for s in sorted(all_sessions, key=lambda x: x.get("started_at", "")):
        started = s.get("started_at", "")[:10]
        if started and (last_session_date is None or started > last_session_date):
            last_session_date = started
        sm = s.get("summary") or {}
        score = sm.get("avg_form_score") or s.get("avg_form_score") or s.get("form_score")
        if score is not None:
            with contextlib.suppress(TypeError, ValueError):
                form_scores.append(float(score))

    sessions_last_7 = len(_athlete_sessions_last_n_days(athlete_id, 7))

    avg_form = round(statistics.mean(form_scores), 1) if form_scores else None

    if sessions_last_7 < 2:
        training_volume = "light"
    elif sessions_last_7 <= 4:
        training_volume = "moderate"
    else:
        training_volume = "heavy"

    risk_data = _compute_injury_risk(athlete_id, 14)
    # _compute_injury_risk uses the key "risk" with values "low"/"watch"/"high"/"unknown"
    raw_risk = risk_data.get("risk", "unknown")
    # Normalise "watch" → "medium" for the parent-facing API contract
    risk_map = {"low": "low", "watch": "medium", "high": "high", "unknown": "low"}
    injury_risk_level = risk_map.get(raw_risk, "low")
    injury_risk_reason = risk_data.get("reason", "insufficient data to assess risk")

    if injury_risk_level == "low":
        summary_sentence = (
            f"{athlete.get('name', 'Athlete')} is training consistently"
            f" with {sessions_last_7} session{'s' if sessions_last_7 != 1 else ''} this week"
            f" and no elevated injury signals detected."
        )
    elif injury_risk_level == "medium":
        summary_sentence = (
            f"{athlete.get('name', 'Athlete')} has been active this week"
            f" ({sessions_last_7} session{'s' if sessions_last_7 != 1 else ''})."
            f" Mild asymmetry detected — worth checking in after training."
        )
    else:
        summary_sentence = (
            f"{athlete.get('name', 'Athlete')} is showing elevated injury-risk signals."
            f" Consider discussing load with them or their coach."
        )

    return {
        "athlete_name": athlete.get("name"),
        "sport": athlete.get("sport"),
        "last_session_date": last_session_date,
        "sessions_last_7_days": sessions_last_7,
        "avg_form_score": avg_form,
        "injury_risk_level": injury_risk_level,
        "injury_risk_reason": injury_risk_reason,
        "training_volume": training_volume,
        "summary_sentence": summary_sentence,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.post("/parent/consent-request", status_code=201)
async def submit_consent_request(body: ConsentRequest):
    if body.athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    consents = _load_consents()
    consent_id = f"pc_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()

    consents[consent_id] = {
        "consent_id": consent_id,
        "athlete_id": body.athlete_id,
        "parent_name": body.parent_name,
        "parent_email": body.parent_email,
        "status": "pending",
        "parent_token": None,
        "created_at": now,
        "updated_at": now,
    }
    _save_consents(consents)

    return {"consent_id": consent_id, "status": "pending"}


@router.post("/athlete/{athlete_id}/parent-consent/{consent_id}/accept")
async def accept_consent(
    athlete_id: str,
    consent_id: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    consents = _load_consents()
    record = consents.get(consent_id)
    if not record:
        raise HTTPException(404, "consent request not found")
    if record.get("athlete_id") != athlete_id:
        raise HTTPException(403, "this consent request does not belong to your account")
    if record.get("status") == "active":
        return {"consent_id": consent_id, "status": "active"}

    record["status"] = "active"
    record["parent_token"] = secrets.token_urlsafe(32)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_consents(consents)

    return {
        "consent_id": consent_id,
        "status": "active",
        "parent_token": record["parent_token"],
    }


@router.post("/athlete/{athlete_id}/parent-consent/{consent_id}/revoke")
async def revoke_consent(
    athlete_id: str,
    consent_id: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    consents = _load_consents()
    record = consents.get(consent_id)
    if not record:
        raise HTTPException(404, "consent request not found")
    if record.get("athlete_id") != athlete_id:
        raise HTTPException(403, "this consent request does not belong to your account")

    record["status"] = "revoked"
    record["parent_token"] = None
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_consents(consents)

    return {"consent_id": consent_id, "status": "revoked"}


@router.get("/athlete/{athlete_id}/parent-consents")
async def list_consents(
    athlete_id: str,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    consents = _load_consents()
    athlete_consents = [
        {
            "consent_id": r["consent_id"],
            "parent_name": r.get("parent_name"),
            "parent_email": r.get("parent_email"),
            "status": r.get("status"),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
        }
        for r in consents.values()
        if r.get("athlete_id") == athlete_id
    ]
    athlete_consents.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {"athlete_id": athlete_id, "consents": athlete_consents, "total": len(athlete_consents)}


@router.get("/parent/{consent_id}/safety-summary")
async def get_safety_summary(consent_id: str, token: str = Query(...)):
    consents = _load_consents()
    record = consents.get(consent_id)
    if not record:
        raise HTTPException(404, "consent not found")
    if record.get("status") != "active":
        raise HTTPException(403, "access not granted — athlete has not accepted this request")
    if record.get("parent_token") != token:
        raise HTTPException(403, "invalid token")

    athlete_id = record["athlete_id"]
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete no longer exists")

    summary = _compute_safety_summary(athlete_id)
    return summary
