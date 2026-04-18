from __future__ import annotations

"""
Coach morning triage — "who do I need to talk to today?"

Flow 12 from BIOMECHANICS-ARCHITECT.md: Coach Raj (28-45) opens the roster
at 6am with 20+ athletes. He needs a single screen that tells him which
2-3 athletes need attention today — not a sea of numbers.

Prioritization rules (highest urgency first):
  1. Injury risk "high"                          urgency 100
  2. Form score dropped ≥10% vs. athlete's baseline  urgency 80
  3. No session for 4+ days                      urgency 60
  4. Streak at risk (trained yesterday, not today by 6pm UTC)  urgency 50
  5. Personal best in last session (positive nudge)  urgency 20

Returns at most 5 items. Each item has a one-sentence "why" in coach
voice — what to actually say to the athlete.

Endpoint:
  GET /coach/{coach_id}/priorities
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from database import ATHLETE_DB, SESSION_DB, _load_json
from logging_setup import get_logger
from routes.progress import _compute_injury_risk

router = APIRouter(prefix="/coach", tags=["Coach Morning"])
log = get_logger("routes.coach_morning")

MAX_PRIORITIES = 5
IDLE_DAYS_THRESHOLD = 4
FORM_DROP_PCT = 10.0


def _session_started_date(s: dict) -> str:
    return s.get("started_at", "")[:10]


def _days_since_last_session(athlete_id: str) -> int | None:
    """Return integer days since last completed session, or None if never trained."""
    last_date = None
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id or s.get("status") != "completed":
            continue
        d = _session_started_date(s)
        if d and (last_date is None or d > last_date):
            last_date = d
    if not last_date:
        return None
    try:
        last_dt = datetime.fromisoformat(last_date).date()
    except ValueError:
        return None
    today = datetime.now(timezone.utc).date()
    return (today - last_dt).days


def _latest_form_vs_baseline(athlete_id: str) -> tuple[float, float]:
    """
    Return (latest_avg_form, baseline_avg_form) over last 2 sessions vs.
    sessions 3-10. Zero if insufficient data.
    """
    sessions = sorted(
        [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
        key=lambda s: s.get("started_at", ""),
    )
    if len(sessions) < 4:
        return (0.0, 0.0)

    def avg_form(batch: list[dict]) -> float:
        vals = [float((s.get("summary") or {}).get("avg_form_score") or 0) for s in batch]
        vals = [v for v in vals if v > 0]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    latest = avg_form(sessions[-2:])
    baseline = avg_form(sessions[-10:-2])
    return (latest, baseline)


def _latest_is_pb(athlete_id: str) -> bool:
    """True if the most recent session produced a personal best in form score."""
    sessions = sorted(
        [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
        key=lambda s: s.get("started_at", ""),
    )
    if len(sessions) < 3:
        return False
    latest_peak = float((sessions[-1].get("summary") or {}).get("peak_form_score") or 0)
    if latest_peak <= 0:
        return False
    prior_peaks = [float((s.get("summary") or {}).get("peak_form_score") or 0) for s in sessions[:-1]]
    return latest_peak > max(prior_peaks, default=0)


def _build_priorities(athlete_ids: list[str]) -> list[dict]:
    priorities: list[dict] = []

    for aid in athlete_ids:
        a = ATHLETE_DB.get(aid)
        if not a:
            continue
        name = a.get("name") or aid
        sport = a.get("sport") or "athlete"

        # Rule 1: injury risk high
        risk = _compute_injury_risk(aid, 14)
        if risk.get("risk") == "high":
            priorities.append(
                {
                    "athlete_id": aid,
                    "name": name,
                    "urgency": 100,
                    "reason_code": "injury_risk_high",
                    "reason": f"{name} is flagged high injury risk over the last 2 weeks. Ask about pain, pull them back to recovery work.",
                }
            )
            continue

        # Rule 2: form decay
        latest_form, baseline_form = _latest_form_vs_baseline(aid)
        if baseline_form > 0 and latest_form > 0:
            drop_pct = (baseline_form - latest_form) / baseline_form * 100
            if drop_pct >= FORM_DROP_PCT:
                priorities.append(
                    {
                        "athlete_id": aid,
                        "name": name,
                        "urgency": 80,
                        "reason_code": "form_decay",
                        "reason": f"{name}'s form score has dropped {drop_pct:.0f}% vs. baseline. Worth a form check session.",
                    }
                )
                continue

        # Rule 3: idle
        days_idle = _days_since_last_session(aid)
        if days_idle is not None and days_idle >= IDLE_DAYS_THRESHOLD:
            priorities.append(
                {
                    "athlete_id": aid,
                    "name": name,
                    "urgency": 60,
                    "reason_code": "idle",
                    "reason": f"{name} hasn't trained in {days_idle} days. A quick nudge gets them back.",
                }
            )
            continue

        # Rule 4: PB in latest session — positive signal
        if _latest_is_pb(aid):
            priorities.append(
                {
                    "athlete_id": aid,
                    "name": name,
                    "urgency": 20,
                    "reason_code": "personal_best",
                    "reason": f"{name} hit a personal best in {sport}. A word of recognition keeps the momentum.",
                }
            )

    priorities.sort(key=lambda p: -p["urgency"])
    return priorities[:MAX_PRIORITIES]


@router.get("/{coach_id}/priorities")
async def coach_priorities(coach_id: str):
    """
    Return the top athletes the coach should talk to today, each with a
    one-sentence reason in coach voice. Max 5 items.
    """
    rosters = _load_json("coach_rosters.json")
    if coach_id not in rosters:
        return {
            "coach_id": coach_id,
            "priorities": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "roster_size": 0,
        }

    athlete_ids = rosters[coach_id].get("athletes", [])
    if not athlete_ids:
        raise HTTPException(404, "coach roster is empty — add athletes first")

    priorities = _build_priorities(athlete_ids)
    return {
        "coach_id": coach_id,
        "priorities": priorities,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roster_size": len(athlete_ids),
    }
