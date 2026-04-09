from __future__ import annotations

"""
Personal Health — progress / injury-risk / weak-joint analytics.

All computed live from SESSION_DB (existing JSON store). No model training.
Cached for 2 minutes per athlete via in-process LRU.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query

from cache import progress_cache
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger

router = APIRouter(tags=["Progress"])
log = get_logger("routes.progress")


# Sport-ideal angle ranges (degrees) for "weak joint" deviation scoring.
# Conservative defaults — meant for ranking, not clinical diagnosis.
IDEAL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "vertical_jump": {
        "knee_angle": (90, 130),
        "hip_angle": (80, 120),
        "ankle_dorsiflexion": (15, 30),
        "trunk_lean": (0, 15),
        "shoulder_angle": (140, 180),
    },
    "sprint": {
        "knee_angle": (85, 110),
        "hip_angle": (90, 130),
        "trunk_lean": (5, 18),
        "shoulder_angle": (70, 110),
    },
    "snatch": {
        "knee_angle": (90, 140),
        "hip_angle": (60, 100),
        "shoulder_angle": (160, 180),
        "elbow_angle": (160, 180),
        "trunk_lean": (0, 12),
    },
    "javelin": {
        "shoulder_angle": (140, 180),
        "elbow_angle": (90, 160),
        "trunk_lean": (5, 25),
    },
    "cricket_bat": {
        "elbow_angle": (60, 120),
        "shoulder_angle": (60, 120),
        "trunk_lean": (5, 20),
    },
}
DEFAULT_RANGES = IDEAL_RANGES["vertical_jump"]


def _avg_pair(left: float, right: float) -> float:
    return (float(left) + float(right)) / 2.0


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # normalize to naive UTC for safe comparison with the cutoff below
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _athlete_sessions(athlete_id: str, days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    out = []
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id:
            continue
        if s.get("status") != "completed":
            continue
        ts = _parse_dt(s.get("started_at"))
        if ts and ts < cutoff:
            continue
        out.append(s)
    out.sort(key=lambda x: x.get("started_at", ""))
    return out


def _frame_metric(frame: dict, name: str) -> float | None:
    """Pull a paired-angle metric from a frame, averaging L+R when present."""
    if name == "trunk_lean":
        v = frame.get("trunk_lean")
        return float(v) if v is not None else None
    left = frame.get(f"{name}_l")
    right = frame.get(f"{name}_r")
    if left is None and right is None:
        return None
    if left is None:
        return float(right)
    if right is None:
        return float(left)
    return _avg_pair(left, right)


def _compute_progress(athlete_id: str, days: int) -> dict:
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    sessions = _athlete_sessions(athlete_id, days)

    # form score trend per day
    by_day: dict[str, list[float]] = defaultdict(list)
    bpi_curve: list[dict] = []
    total_reps = 0
    best_jump = 0.0
    all_form_scores: list[float] = []

    for s in sessions:
        ts = _parse_dt(s.get("started_at"))
        day = ts.date().isoformat() if ts else "unknown"
        # Bug fix: end_session() and seed_sessions.py store metrics inside
        # s["summary"], not at the session root. Read from both for compat.
        summary = s.get("summary") or {}
        score = float(
            summary.get("avg_form_score") or s.get("avg_form_score") or s.get("form_score") or 0
        )
        if score > 0:
            by_day[day].append(score)
            all_form_scores.append(score)
        total_reps += int(summary.get("rep_count") or s.get("rep_count") or 0)
        bj = float(
            summary.get("peak_jump_height_cm")
            or s.get("peak_jump_height_cm")
            or s.get("best_jump_height_cm")
            or 0
        )
        if bj > best_jump:
            best_jump = bj
        if "bpi_after" in s:
            bpi_curve.append({"date": day, "bpi": int(s["bpi_after"])})

    form_score_trend = [
        {"date": day, "score": round(sum(scores) / len(scores), 1), "samples": len(scores)}
        for day, scores in sorted(by_day.items())
    ]

    avg_form = round(statistics.mean(all_form_scores), 1) if all_form_scores else 0.0

    # 7-day vs prior trend %
    if len(form_score_trend) >= 4:
        recent = [d["score"] for d in form_score_trend[-3:]]
        prior = [d["score"] for d in form_score_trend[-6:-3]] or recent
        recent_mean = sum(recent) / len(recent)
        prior_mean = sum(prior) / len(prior) if prior else recent_mean
        form_trend_pct = ((recent_mean - prior_mean) / prior_mean * 100) if prior_mean else 0.0
    else:
        form_trend_pct = 0.0

    bpi_delta = 0
    if len(bpi_curve) >= 2:
        bpi_delta = int(bpi_curve[-1]["bpi"] - bpi_curve[0]["bpi"])

    return {
        "athlete_id": athlete_id,
        "athlete_name": athlete.get("name"),
        "sport": athlete.get("sport"),
        "window_days": days,
        "session_count": len(sessions),
        "total_reps": total_reps,
        "best_jump_cm": round(best_jump, 1),
        "avg_form_score": avg_form,
        "form_trend_pct": round(form_trend_pct, 2),
        "form_score_trend": form_score_trend,
        "bpi_curve": bpi_curve,
        "bpi_delta": bpi_delta,
        "last_session_at": sessions[-1].get("started_at") if sessions else None,
    }


def _compute_weak_joints(athlete_id: str, days: int) -> list[dict]:
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")
    athlete = ATHLETE_DB[athlete_id]
    sport = athlete.get("sport", "vertical_jump")
    ideal = IDEAL_RANGES.get(sport, DEFAULT_RANGES)

    sessions = _athlete_sessions(athlete_id, days)
    metric_values: dict[str, list[float]] = defaultdict(list)
    for s in sessions:
        for frame in s.get("frames", []) or []:
            for joint in ideal:
                v = _frame_metric(frame, joint)
                if v is not None:
                    metric_values[joint].append(v)

    out: list[dict] = []
    for joint, values in metric_values.items():
        if not values:
            continue
        mean_val = statistics.mean(values)
        lo, hi = ideal[joint]
        if mean_val < lo:
            deviation = lo - mean_val
        elif mean_val > hi:
            deviation = mean_val - hi
        else:
            deviation = 0.0
        out.append(
            {
                "joint": joint,
                "mean_deg": round(mean_val, 1),
                "ideal_min": lo,
                "ideal_max": hi,
                "deviation_deg": round(deviation, 1),
                "samples": len(values),
            }
        )
    out.sort(key=lambda x: x["deviation_deg"], reverse=True)
    return out


def _compute_injury_risk(athlete_id: str, days: int) -> dict:
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")
    sessions = _athlete_sessions(athlete_id, days)
    symmetry_values: list[float] = []
    for s in sessions:
        for frame in s.get("frames", []) or []:
            v = frame.get("limb_symmetry_idx")
            if v is not None:
                symmetry_values.append(float(v))

    if not symmetry_values:
        return {
            "athlete_id": athlete_id,
            "window_days": days,
            "risk": "unknown",
            "reason": "no symmetry data captured in this window",
            "samples": 0,
        }

    mean_sym = statistics.mean(symmetry_values)
    deviation_pct = abs(1.0 - mean_sym) * 100  # 1.0 == perfect symmetry
    if deviation_pct < 5:
        band = "low"
        reason = f"limb symmetry deviation {deviation_pct:.1f}% — within normal range"
    elif deviation_pct < 12:
        band = "watch"
        reason = f"limb symmetry deviation {deviation_pct:.1f}% — mild asymmetry, monitor"
    else:
        band = "high"
        side = "left-dominant" if mean_sym > 1 else "right-dominant"
        reason = f"limb symmetry deviation {deviation_pct:.1f}% — {side}, recommend rest + mobility"

    return {
        "athlete_id": athlete_id,
        "window_days": days,
        "risk": band,
        "deviation_pct": round(deviation_pct, 2),
        "mean_symmetry": round(mean_sym, 3),
        "reason": reason,
        "samples": len(symmetry_values),
    }


# ─── Routes ─────────────────────────────────────────────────────────────────


@router.get("/progress/{athlete_id}")
async def get_progress(athlete_id: str, days: int = Query(default=30, ge=1, le=365)):
    key = f"progress:{athlete_id}:{days}"
    cached = progress_cache.get(key)
    if cached:
        return cached
    payload = _compute_progress(athlete_id, days)
    progress_cache.set(key, payload)
    return payload


@router.get("/injury-risk/{athlete_id}")
async def get_injury_risk(athlete_id: str, days: int = Query(default=14, ge=1, le=90)):
    key = f"risk:{athlete_id}:{days}"
    cached = progress_cache.get(key)
    if cached:
        return cached
    payload = _compute_injury_risk(athlete_id, days)
    progress_cache.set(key, payload)
    return payload


@router.get("/weak-joints/{athlete_id}")
async def get_weak_joints(athlete_id: str, days: int = Query(default=30, ge=1, le=180)):
    key = f"weak:{athlete_id}:{days}"
    cached = progress_cache.get(key)
    if cached:
        return cached
    weak = _compute_weak_joints(athlete_id, days)
    payload = {"athlete_id": athlete_id, "window_days": days, "weak_joints": weak[:5]}
    progress_cache.set(key, payload)
    return payload
