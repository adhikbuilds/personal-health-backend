from __future__ import annotations

"""
Personal Health — Progressive Load Model (Layer 2, Item 5).

Given an athlete's recent training history, recommends next-week volume and
intensity targets to keep improving without overreaching.

Model: Acute:Chronic Workload Ratio (ACWR) simplified for form-score-based
sessions. Acute = last 7 days avg load. Chronic = last 28 days avg load.
Ratio drives the recommendation:

  ACWR < 0.8  → under-trained, ramp up
  0.8–1.3     → sweet spot, maintain or nudge up
  1.3–1.5     → caution zone, reduce next week
  > 1.5       → danger zone, de-load

"Load" here = (session_count * avg_form_score * duration_factor). Higher form
score at longer durations = more productive training load.

The output is a recommendation dict that plugs into the training plan context
and the weekly summary.
"""

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from logging_setup import get_logger

log = get_logger("services.progressive_load")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _compute_load(sessions: list[dict]) -> float:
    """Compute a single load number for a set of sessions."""
    if not sessions:
        return 0.0
    total = 0.0
    for s in sessions:
        score = float(s.get("avg_form_score", 0) or s.get("summary", {}).get("avg_form_score", 0) or 0)
        dur = float(s.get("duration_seconds", 0) or s.get("summary", {}).get("duration_seconds", 0) or 0)
        # Duration factor: normalize to a 30-min session = 1.0
        dur_factor = min(dur / 1800, 2.0) if dur > 0 else 0.5
        total += score * dur_factor
    return round(total, 1)


def compute_progressive_load(
    athlete_id: str,
    all_sessions: list[dict],
    sport: str = "vertical_jump",
) -> dict[str, Any]:
    """
    Compute ACWR-based progressive load recommendation.

    Args:
        athlete_id: the athlete
        all_sessions: ALL completed sessions for this athlete (sorted by date)
        sport: the athlete's sport

    Returns:
        Dict with acute/chronic load, ACWR, recommendation, next-week targets.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff_7 = now - timedelta(days=7)
    cutoff_28 = now - timedelta(days=28)

    recent_7: list[dict] = []
    recent_28: list[dict] = []

    for s in all_sessions:
        ts = _parse_dt(s.get("started_at"))
        if not ts:
            continue
        if ts >= cutoff_28:
            recent_28.append(s)
        if ts >= cutoff_7:
            recent_7.append(s)

    acute_load = _compute_load(recent_7)
    chronic_load = _compute_load(recent_28) / 4.0 if recent_28 else 0.0  # weekly average

    if chronic_load > 0:
        acwr = round(acute_load / chronic_load, 2)
    elif acute_load > 0:
        acwr = 2.0  # spike from zero baseline
    else:
        acwr = 0.0

    # Recommendations based on ACWR zones
    if acwr == 0:
        zone = "inactive"
        recommendation = "No recent training detected. Start with 3 low-intensity sessions this week."
        target_sessions = 3
        target_intensity = "low"
        target_rpe = 5
    elif acwr < 0.8:
        zone = "under-trained"
        recommendation = (
            f"ACWR is {acwr:.2f} — you're under your usual load. "
            "Add one more session this week and push intensity up slightly."
        )
        target_sessions = min(len(recent_7) + 2, 6)
        target_intensity = "moderate"
        target_rpe = 7
    elif acwr <= 1.3:
        zone = "optimal"
        recommendation = (
            f"ACWR is {acwr:.2f} — sweet spot. Maintain this volume and "
            "nudge intensity up on 1–2 sessions if form holds above 75."
        )
        target_sessions = max(len(recent_7), 3)
        target_intensity = "moderate-high"
        target_rpe = 7
    elif acwr <= 1.5:
        zone = "caution"
        recommendation = (
            f"ACWR is {acwr:.2f} — approaching overreach. Drop one session "
            "and reduce intensity. Add an extra recovery day."
        )
        target_sessions = max(len(recent_7) - 1, 2)
        target_intensity = "moderate"
        target_rpe = 6
    else:
        zone = "danger"
        recommendation = (
            f"ACWR is {acwr:.2f} — high injury risk from load spike. "
            "De-load this week: max 2 sessions at low intensity. "
            "Prioritize recovery and mobility."
        )
        target_sessions = 2
        target_intensity = "low"
        target_rpe = 4

    # Form-score-based quality targets
    scores_7 = [
        float(s.get("avg_form_score", 0) or s.get("summary", {}).get("avg_form_score", 0) or 0)
        for s in recent_7
        if float(s.get("avg_form_score", 0) or s.get("summary", {}).get("avg_form_score", 0) or 0) > 0
    ]
    avg_form_7 = round(statistics.mean(scores_7), 1) if scores_7 else 0.0

    if avg_form_7 >= 80:
        form_guidance = "Form is excellent — you can push volume without risk."
    elif avg_form_7 >= 65:
        form_guidance = "Form is steady. Maintain volume; add one technique-focused session."
    elif avg_form_7 > 0:
        form_guidance = "Form is rough. Prioritize technique over intensity this week."
    else:
        form_guidance = "Not enough form data. Focus on logging complete sessions."

    return {
        "athlete_id": athlete_id,
        "sport": sport,
        "acute_load_7d": acute_load,
        "chronic_load_28d_avg": round(chronic_load, 1),
        "acwr": acwr,
        "zone": zone,
        "sessions_last_7d": len(recent_7),
        "sessions_last_28d": len(recent_28),
        "avg_form_7d": avg_form_7,
        "recommendation": recommendation,
        "form_guidance": form_guidance,
        "next_week_targets": {
            "sessions": target_sessions,
            "intensity": target_intensity,
            "rpe_cap": target_rpe,
            "focus": "technique" if avg_form_7 < 65 else "intensity" if zone == "under-trained" else "maintenance",
        },
    }
