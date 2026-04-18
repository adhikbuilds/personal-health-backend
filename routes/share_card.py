from __future__ import annotations

"""
Post-session share card — Flow 7 from BIOMECHANICS-ARCHITECT.md.

The calibration-good post-session share card:
  - ONE hero number (form score) in #06b6d4 on #0a0e1a
  - ONE delta (vs. previous session OR best-of-last-4)
  - Third line = sport + rep count in #9ca3af
  - NO motivational quote, NO gradient, NO profile photo, NO "challenge friends"

Three variants driven by the data, not by the client:
  - "pb"         = this session produced a personal best form score
  - "streak"     = no PB, but athlete has trained 3+ days this week
  - "show_up"    = neither of the above; still shareable as consistency

The variant controls copy only. Visual treatment stays identical per the
locked design system.

Multiplayer logic map baked into the response:
  - huddle_auto_post:   server will auto-post PBs to the athlete's huddle feed
  - coach_will_see:     server will surface PBs on coach's morning priorities
  - instagram_deeplink: a mobile-share URL the app can hand to the OS share sheet

Endpoint:
  GET /session/{session_id}/share-card
"""

from urllib.parse import quote

from fastapi import APIRouter, HTTPException

from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger

router = APIRouter(tags=["Share Card"])
log = get_logger("routes.share_card")

DESIGN_SYSTEM = {
    "accent": "#06b6d4",
    "background": "#0a0e1a",
    "surface": "#111827",
    "text_primary": "#f9fafb",
    "text_secondary": "#9ca3af",
    "success": "#10b981",
    "warning": "#f59e0b",
}


def _completed_sessions_for(athlete_id: str) -> list[dict]:
    return sorted(
        [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
        key=lambda s: s.get("started_at", ""),
    )


def _days_trained_this_week(athlete_id: str) -> int:
    """Distinct training dates in the last 7 days."""
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
    dates = set()
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id or s.get("status") != "completed":
            continue
        d = s.get("started_at", "")[:10]
        if d >= cutoff:
            dates.add(d)
    return len(dates)


def _pick_variant(athlete_id: str, this_session: dict) -> tuple[str, dict]:
    """Returns (variant, signals) where signals are booleans for downstream use."""
    sessions = _completed_sessions_for(athlete_id)
    latest_peak = float((this_session.get("summary") or {}).get("peak_form_score") or 0)

    is_pb = False
    if latest_peak > 0 and len(sessions) >= 2:
        # ensure the current session is actually at the tail (sessions sorted by started_at)
        prior_peaks = [
            float((s.get("summary") or {}).get("peak_form_score") or 0)
            for s in sessions
            if s.get("id") != this_session.get("id")
        ]
        is_pb = latest_peak > max(prior_peaks, default=0)

    days_this_week = _days_trained_this_week(athlete_id)

    if is_pb:
        return ("pb", {"is_pb": True, "days_this_week": days_this_week})
    if days_this_week >= 3:
        return ("streak", {"is_pb": False, "days_this_week": days_this_week})
    return ("show_up", {"is_pb": False, "days_this_week": days_this_week})


def _delta_vs_last(athlete_id: str, this_session: dict) -> float | None:
    sessions = _completed_sessions_for(athlete_id)
    other = [s for s in sessions if s.get("id") != this_session.get("id")]
    if not other:
        return None
    last = other[-1]
    this_form = float((this_session.get("summary") or {}).get("avg_form_score") or 0)
    last_form = float((last.get("summary") or {}).get("avg_form_score") or 0)
    if this_form <= 0 or last_form <= 0:
        return None
    return round(this_form - last_form, 1)


def _copy_for(variant: str, signals: dict, athlete_name: str, sport: str) -> dict:
    """Return {headline, sub, chip, share_text}. Copy only; no visual variation."""
    sport_label = (sport or "session").replace("_", " ")

    if variant == "pb":
        return {
            "headline_label": "personal best",
            "sub": f"{sport_label} · {signals['days_this_week']} days this week",
            "chip": "PB",
            "chip_colour": DESIGN_SYSTEM["success"],
            "share_text": f"New personal best on {sport_label}. Graded by Personal Health.",
        }
    if variant == "streak":
        return {
            "headline_label": "form score",
            "sub": f"{sport_label} · {signals['days_this_week']} days this week",
            "chip": f"{signals['days_this_week']}-day week",
            "chip_colour": DESIGN_SYSTEM["accent"],
            "share_text": f"{signals['days_this_week']} training days this week. {sport_label}. Personal Health.",
        }
    return {
        "headline_label": "form score",
        "sub": f"{sport_label} · session logged",
        "chip": "showed up",
        "chip_colour": DESIGN_SYSTEM["text_secondary"],
        "share_text": f"Trained {sport_label}. Personal Health.",
    }


@router.get("/session/{session_id}/share-card")
async def get_share_card(session_id: str):
    session = SESSION_DB.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    if session.get("status") != "completed":
        raise HTTPException(400, "session not completed — end it first")

    athlete_id = session.get("athlete_id") or ""
    athlete = ATHLETE_DB.get(athlete_id, {"id": athlete_id, "name": athlete_id})
    athlete_name = athlete.get("name") or athlete_id
    sport = session.get("sport") or (session.get("summary") or {}).get("sport") or "session"

    summary = session.get("summary") or {}
    hero = float(summary.get("avg_form_score") or 0)
    peak = float(summary.get("peak_form_score") or 0)
    rep_count = int(summary.get("rep_count") or 0)

    # session id on summary is sometimes on the outer session; normalize for variant check
    this_session_with_id = {**session, "id": session_id}
    variant, signals = _pick_variant(athlete_id, this_session_with_id)
    copy = _copy_for(variant, signals, athlete_name, sport)
    delta = _delta_vs_last(athlete_id, this_session_with_id)

    # The PNG endpoint already exists at /session/{id}/scorecard.png. Reference it.
    image_url = f"/session/{session_id}/scorecard.png"
    share_url = f"https://personalhealth.app/session/{quote(session_id)}"

    return {
        "session_id": session_id,
        "athlete_id": athlete_id,
        "athlete_name": athlete_name,
        "sport": sport,
        "variant": variant,
        "hero_number": round(hero, 1) if hero > 0 else round(peak, 1),
        "hero_label": copy["headline_label"],
        "delta": delta,
        "delta_direction": ("up" if (delta or 0) > 0 else "down" if (delta or 0) < 0 else "flat"),
        "sub": copy["sub"],
        "chip": copy["chip"],
        "chip_colour": copy["chip_colour"],
        "rep_count": rep_count,
        "share_text": copy["share_text"],
        "share_url": share_url,
        "image_url": image_url,
        "design_system": DESIGN_SYSTEM,
        "multiplayer": {
            "huddle_auto_post": variant == "pb",
            "coach_will_see": variant == "pb",
            "instagram_deeplink": f"instagram://sharesheet?text={quote(copy['share_text'])}",
        },
    }
