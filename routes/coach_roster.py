from __future__ import annotations

"""
Coach roster — coach-athlete relationship management.

VISION.md Step 4: "Target sports coaches at district and state academies.
The coach becomes a hub: they set up the app for their 20 athletes."

Endpoints:
  POST /coach/{coach_id}/athletes       — add athlete to roster
  DELETE /coach/{coach_id}/athletes/{id} — remove athlete from roster
  GET  /coach/{coach_id}/athletes        — list coach's athletes with stats
  GET  /coach/{coach_id}/dashboard       — aggregated dashboard for all athletes
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from auth import require_coach_or_admin
from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from routes.progress import _athlete_sessions, _compute_injury_risk

router = APIRouter(prefix="/coach", tags=["Coach Roster"])
log = get_logger("routes.coach_roster")


def _load_rosters() -> dict:
    return _load_json("coach_rosters.json")


def _save_rosters(data: dict):
    _save_json("coach_rosters.json", data)


@router.post("/{coach_id}/athletes")
async def add_athlete_to_roster(
    coach_id: str,
    athlete_id: str,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    """Coach claims an athlete. Coach ID can be any athlete ID acting as coach."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    rosters = _load_rosters()
    if coach_id not in rosters:
        rosters[coach_id] = {"athletes": [], "created_at": datetime.now(timezone.utc).isoformat()}

    roster = rosters[coach_id]["athletes"]
    if athlete_id in roster:
        return {"status": "already_added", "coach_id": coach_id, "athlete_id": athlete_id}

    if len(roster) >= 25:
        raise HTTPException(400, "max 25 athletes per coach (upgrade to coach plan for more)")

    roster.append(athlete_id)
    _save_rosters(rosters)
    log.info("athlete added to roster", extra={"coach_id": coach_id, "athlete_id": athlete_id})
    return {"status": "added", "coach_id": coach_id, "athlete_id": athlete_id, "roster_size": len(roster)}


@router.delete("/{coach_id}/athletes/{athlete_id}")
async def remove_athlete_from_roster(
    coach_id: str,
    athlete_id: str,
    _: dict = Depends(require_coach_or_admin("coach_id")),
):
    rosters = _load_rosters()
    if coach_id not in rosters:
        raise HTTPException(404, "coach roster not found")

    roster = rosters[coach_id]["athletes"]
    if athlete_id not in roster:
        raise HTTPException(404, "athlete not in this roster")

    roster.remove(athlete_id)
    _save_rosters(rosters)
    return {"status": "removed", "coach_id": coach_id, "athlete_id": athlete_id}


@router.get("/{coach_id}/athletes")
async def list_roster(coach_id: str, _: dict = Depends(require_coach_or_admin("coach_id"))):
    """List all athletes in a coach's roster with basic stats."""
    rosters = _load_rosters()
    if coach_id not in rosters:
        return {"coach_id": coach_id, "athletes": [], "count": 0}

    athlete_ids = rosters[coach_id]["athletes"]
    athletes = []
    for aid in athlete_ids:
        a = ATHLETE_DB.get(aid)
        if not a:
            continue
        recent = _athlete_sessions(aid, 7)
        athletes.append(
            {
                "id": aid,
                "name": a.get("name", "Unknown"),
                "sport": a.get("sport"),
                "tier": a.get("tier"),
                "bpi": a.get("bpi", 0),
                "sessions_this_week": len(recent),
                "last_session": recent[-1].get("started_at") if recent else None,
            }
        )

    return {"coach_id": coach_id, "athletes": athletes, "count": len(athletes)}


@router.get("/{coach_id}/dashboard")
async def coach_dashboard(coach_id: str, _: dict = Depends(require_coach_or_admin("coach_id"))):
    """
    Aggregated view for a coach: how are all my athletes doing?

    VISION Step 4: "Your athletes can train on their own and you can see
    their form data from your phone."
    """
    rosters = _load_rosters()
    if coach_id not in rosters:
        return {"coach_id": coach_id, "athletes": [], "summary": {}}

    athlete_ids = rosters[coach_id]["athletes"]
    athlete_stats = []
    total_sessions_week = 0
    risk_counts = {"low": 0, "watch": 0, "high": 0, "unknown": 0}
    all_form_scores = []

    for aid in athlete_ids:
        a = ATHLETE_DB.get(aid)
        if not a:
            continue
        recent = _athlete_sessions(aid, 7)
        total_sessions_week += len(recent)

        form_scores = []
        for s in recent:
            sm = s.get("summary") or {}
            af = float(sm.get("avg_form_score") or 0)
            if af > 0:
                form_scores.append(af)
                all_form_scores.append(af)

        avg_form = round(sum(form_scores) / len(form_scores), 1) if form_scores else 0.0
        risk = _compute_injury_risk(aid, 14)
        risk_band = risk.get("risk", "unknown")
        risk_counts[risk_band] = risk_counts.get(risk_band, 0) + 1

        athlete_stats.append(
            {
                "id": aid,
                "name": a.get("name", "Unknown"),
                "sport": a.get("sport"),
                "sessions_this_week": len(recent),
                "avg_form_score": avg_form,
                "injury_risk": risk_band,
                "needs_attention": risk_band == "high" or len(recent) == 0,
            }
        )

    # sort: athletes needing attention first, then by sessions this week desc
    athlete_stats.sort(key=lambda x: (not x["needs_attention"], -x["sessions_this_week"]))

    team_avg_form = round(sum(all_form_scores) / len(all_form_scores), 1) if all_form_scores else 0.0

    return {
        "coach_id": coach_id,
        "athletes": athlete_stats,
        "summary": {
            "total_athletes": len(athlete_stats),
            "total_sessions_this_week": total_sessions_week,
            "team_avg_form_score": team_avg_form,
            "injury_risk_breakdown": risk_counts,
            "athletes_needing_attention": sum(1 for a in athlete_stats if a["needs_attention"]),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
