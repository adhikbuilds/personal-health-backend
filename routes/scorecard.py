from __future__ import annotations

"""
Personal Health — Score Card, Weekly Summary, Athlete Comparison.

Three endpoints that close critical product gaps from VISION.md:

  GET /sessions/{session_id}/scorecard
      Post-session shareable score card (go-to-market Step 2).

  GET /athlete/{athlete_id}/weekly-summary
      Structured week-over-week comparison (Milestone 4).

  GET /compare?athletes=id1,id2&days=14
      Side-by-side athlete comparison for coaches (Step 4).
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query

from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from routes.progress import _athlete_sessions, _compute_injury_risk, _compute_weak_joints, _parse_dt

router = APIRouter(tags=["Scorecard"])
log = get_logger("routes.scorecard")


# ─── Valid sports (used for validation) ────────────────────────────────────

VALID_SPORTS = frozenset(
    {
        "vertical_jump",
        "sprint",
        "snatch",
        "javelin",
        "cricket_bat",
        "squat",
        "push_up",
        "pull_up",
    }
)


# ─── Score Card ────────────────────────────────────────────────────────────


@router.get("/sessions/{session_id}/scorecard")
async def session_scorecard(session_id: str):
    """
    Generate a structured score card for sharing after a session.

    VISION.md Step 2: "App generates a score card image (athlete avatar,
    BPI delta since last session, key metric like jump height: 47cm)."

    Returns structured JSON that the mobile app renders as a card image.
    """
    if session_id not in SESSION_DB:
        raise HTTPException(404, "session not found")

    session = SESSION_DB[session_id]
    if session.get("status") != "completed":
        raise HTTPException(400, "session not completed yet — end it first")

    summary = session.get("summary") or {}
    athlete_id = session.get("athlete_id", "")
    athlete = ATHLETE_DB.get(athlete_id, {})

    # Compute BPI delta: current BPI minus BPI before this session's XP
    xp_earned = summary.get("xp_earned", 0)
    current_bpi = athlete.get("bpi", 0)
    bpi_before = current_bpi - xp_earned

    # Find personal best for this sport
    sport = session.get("sport", "vertical_jump")
    best_jump_ever = 0.0
    best_form_ever = 0.0
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id or s.get("sport") != sport:
            continue
        if s.get("status") != "completed":
            continue
        sm = s.get("summary") or {}
        bj = float(sm.get("peak_jump_height_cm") or 0)
        bf = float(sm.get("peak_form_score") or 0)
        if bj > best_jump_ever:
            best_jump_ever = bj
        if bf > best_form_ever:
            best_form_ever = bf

    session_jump = float(summary.get("peak_jump_height_cm") or 0)
    session_form = float(summary.get("peak_form_score") or 0)

    return {
        "session_id": session_id,
        "athlete": {
            "id": athlete_id,
            "name": athlete.get("name", "Athlete"),
            "avatar": athlete.get("avatar", "?"),
            "sport": sport,
            "tier": athlete.get("tier", "Athlete"),
        },
        "headline_stat": {
            "label": "Peak Form Score" if session_form >= session_jump else "Jump Height",
            "value": round(max(session_form, session_jump), 1),
            "unit": "%" if session_form >= session_jump else "cm",
        },
        "stats": {
            "avg_form_score": round(float(summary.get("avg_form_score") or 0), 1),
            "peak_form_score": round(session_form, 1),
            "peak_jump_cm": round(session_jump, 1),
            "total_frames": summary.get("total_frames", 0),
            "valid_frames": summary.get("valid_frames", 0),
            "duration_seconds": summary.get("duration_seconds", 0),
            "quality_distribution": summary.get("quality_distribution", {}),
        },
        "bpi": {
            "current": current_bpi,
            "before": bpi_before,
            "delta": f"+{xp_earned}" if xp_earned > 0 else str(xp_earned),
            "xp_earned": xp_earned,
        },
        "personal_bests": {
            "jump_cm": round(best_jump_ever, 1),
            "form_score": round(best_form_ever, 1),
            "is_pb_jump": session_jump >= best_jump_ever > 0,
            "is_pb_form": session_form >= best_form_ever > 0,
        },
        "share_text": _build_share_text(athlete, sport, summary, xp_earned),
        "session_date": session.get("started_at", ""),
    }


def _build_share_text(athlete: dict, sport: str, summary: dict, xp: int) -> str:
    name = athlete.get("name", "Athlete")
    avg = round(float(summary.get("avg_form_score") or 0), 1)
    jump = round(float(summary.get("peak_jump_height_cm") or 0), 1)
    sport_label = sport.replace("_", " ").title()
    parts = [f"{name} just trained {sport_label} on ActiveBharat"]
    if avg > 0:
        parts.append(f"Form Score: {avg}")
    if jump > 5:
        parts.append(f"Jump: {jump}cm")
    if xp > 0:
        parts.append(f"+{xp} XP")
    return " | ".join(parts)


# ─── Weekly Summary ────────────────────────────────────────────────────────


def _week_stats(sessions: list[dict]) -> dict:
    """Compute aggregate stats for a list of sessions."""
    if not sessions:
        return {
            "session_count": 0,
            "total_frames": 0,
            "avg_form_score": 0.0,
            "peak_form_score": 0.0,
            "peak_jump_cm": 0.0,
            "total_xp": 0,
            "quality_distribution": {"elite": 0, "good": 0, "average": 0, "poor": 0},
        }

    form_scores = []
    peak_form = 0.0
    peak_jump = 0.0
    total_frames = 0
    total_xp = 0
    quality_dist: dict[str, int] = {"elite": 0, "good": 0, "average": 0, "poor": 0}

    for s in sessions:
        sm = s.get("summary") or {}
        avg_f = float(sm.get("avg_form_score") or 0)
        if avg_f > 0:
            form_scores.append(avg_f)
        pf = float(sm.get("peak_form_score") or 0)
        if pf > peak_form:
            peak_form = pf
        pj = float(sm.get("peak_jump_height_cm") or 0)
        if pj > peak_jump:
            peak_jump = pj
        total_frames += int(sm.get("total_frames") or s.get("frame_count") or 0)
        total_xp += int(sm.get("xp_earned") or 0)
        for q, cnt in (sm.get("quality_distribution") or {}).items():
            if q in quality_dist:
                quality_dist[q] += int(cnt)

    return {
        "session_count": len(sessions),
        "total_frames": total_frames,
        "avg_form_score": round(statistics.mean(form_scores), 1) if form_scores else 0.0,
        "peak_form_score": round(peak_form, 1),
        "peak_jump_cm": round(peak_jump, 1),
        "total_xp": total_xp,
        "quality_distribution": quality_dist,
    }


@router.get("/athlete/{athlete_id}/weekly-summary")
async def weekly_summary(athlete_id: str, weeks: int = Query(default=2, ge=1, le=8)):
    """
    Structured week-over-week comparison.

    VISION.md Milestone 4: "Weekly session summary API endpoint built."

    Returns stats for each of the last N weeks plus deltas so the app/dashboard
    can show trends at a glance.
    """
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    all_sessions = _athlete_sessions(athlete_id, weeks * 7)

    # Bucket sessions by week (week 0 = current, week 1 = last week, etc.)
    week_buckets: dict[int, list[dict]] = defaultdict(list)
    for s in all_sessions:
        ts = _parse_dt(s.get("started_at"))
        if not ts:
            continue
        days_ago = (now - ts).days
        week_idx = days_ago // 7
        if week_idx < weeks:
            week_buckets[week_idx].append(s)

    week_summaries = []
    for i in range(weeks):
        sessions = week_buckets.get(i, [])
        stats = _week_stats(sessions)
        week_start = (now - timedelta(days=(i + 1) * 7 - 1)).date().isoformat()
        week_end = (now - timedelta(days=i * 7)).date().isoformat()
        week_summaries.append(
            {
                "week": i,
                "label": "This week" if i == 0 else f"{i} week{'s' if i > 1 else ''} ago",
                "period": f"{week_start} to {week_end}",
                **stats,
            }
        )

    # Compute deltas (this week vs last week)
    deltas = {}
    if len(week_summaries) >= 2:
        curr = week_summaries[0]
        prev = week_summaries[1]
        deltas = {
            "sessions": curr["session_count"] - prev["session_count"],
            "avg_form_score": round(curr["avg_form_score"] - prev["avg_form_score"], 1),
            "peak_form_score": round(curr["peak_form_score"] - prev["peak_form_score"], 1),
            "peak_jump_cm": round(curr["peak_jump_cm"] - prev["peak_jump_cm"], 1),
            "xp": curr["total_xp"] - prev["total_xp"],
        }

    return {
        "athlete_id": athlete_id,
        "athlete_name": athlete.get("name"),
        "sport": athlete.get("sport"),
        "weeks_requested": weeks,
        "week_summaries": week_summaries,
        "deltas": deltas,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Athlete Comparison ───────────────────────────────────────────────────


@router.get("/compare")
async def compare_athletes(
    athletes: str = Query(description="Comma-separated athlete IDs (2-4)"),
    days: int = Query(default=14, ge=1, le=90),
):
    """
    Side-by-side athlete comparison for coaches.

    VISION.md Step 4: coaches need to see how their athletes stack up.
    Compares form scores, volume, injury risk, and weak joints.
    """
    ids = [a.strip() for a in athletes.split(",") if a.strip()]
    if len(ids) < 2:
        raise HTTPException(400, "provide at least 2 athlete IDs separated by commas")
    if len(ids) > 4:
        raise HTTPException(400, "max 4 athletes per comparison")

    results = []
    for aid in ids:
        if aid not in ATHLETE_DB:
            raise HTTPException(404, f"athlete {aid} not found")

        athlete = ATHLETE_DB[aid]
        sessions = _athlete_sessions(aid, days)
        risk = _compute_injury_risk(aid, days)
        weak = _compute_weak_joints(aid, days)

        # Session stats
        form_scores = []
        peak_jump = 0.0
        total_xp = 0
        for s in sessions:
            sm = s.get("summary") or {}
            af = float(sm.get("avg_form_score") or 0)
            if af > 0:
                form_scores.append(af)
            pj = float(sm.get("peak_jump_height_cm") or 0)
            if pj > peak_jump:
                peak_jump = pj
            total_xp += int(sm.get("xp_earned") or 0)

        results.append(
            {
                "athlete_id": aid,
                "name": athlete.get("name", "Unknown"),
                "sport": athlete.get("sport"),
                "tier": athlete.get("tier"),
                "bpi": athlete.get("bpi", 0),
                "window_days": days,
                "session_count": len(sessions),
                "avg_form_score": round(statistics.mean(form_scores), 1) if form_scores else 0.0,
                "peak_jump_cm": round(peak_jump, 1),
                "total_xp": total_xp,
                "injury_risk": risk["risk"],
                "injury_reason": risk.get("reason", ""),
                "top_weak_joints": [{"joint": j["joint"], "deviation_deg": j["deviation_deg"]} for j in weak[:3]],
            }
        )

    # Rank on each metric
    rankings: dict[str, list[str]] = {}
    for metric in ["avg_form_score", "bpi", "session_count", "peak_jump_cm"]:
        ranked = sorted(results, key=lambda r: r.get(metric, 0), reverse=True)
        rankings[metric] = [r["athlete_id"] for r in ranked]

    return {
        "athletes": results,
        "rankings": rankings,
        "window_days": days,
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }
