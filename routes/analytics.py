from __future__ import annotations

"""
Personal Health — analytics route.

Two endpoints both called out as next-up in VISION.md:

  GET /sessions/{session_id}/rep-count
        Count reps inside a completed (or active) session by detecting
        local minima in the sport-relevant cyclic joint angle.
        Vision §"Layer 2" item 2 — "Rep counting … Next to build."

  GET /readiness/{athlete_id}
        Composite competition-readiness score over a window:
            form trend (40%) + symmetry (20%) + volume (20%) + HRV (20%)
        Vision §"Layer 3" — Competition readiness score.
"""

import statistics
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from cache import progress_cache
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from routes.progress import (
    _athlete_sessions,
    _compute_injury_risk,
    _compute_progress,
    _frame_metric,
)

router = APIRouter(tags=["Analytics"])
log = get_logger("routes.analytics")


# ─── Rep counting ───────────────────────────────────────────────────────────

# Per-sport cyclic joint and the threshold (deg) below which the joint is
# considered to be at the bottom of a rep. Tuned conservatively against the
# IDEAL_RANGES used elsewhere — meant for counting, not biomechanical scoring.
REP_CYCLE: dict[str, dict] = {
    "vertical_jump": {"joint": "knee_angle", "bottom_below": 130, "top_above": 160},
    "squat": {"joint": "knee_angle", "bottom_below": 110, "top_above": 160},
    "snatch": {"joint": "knee_angle", "bottom_below": 130, "top_above": 165},
    "push_up": {"joint": "elbow_angle", "bottom_below": 100, "top_above": 155},
    "pull_up": {"joint": "elbow_angle", "bottom_below": 100, "top_above": 160},
    "sprint": {"joint": "knee_angle", "bottom_below": 110, "top_above": 150},
    "cricket_bat": {"joint": "shoulder_angle", "bottom_below": 70, "top_above": 120},
}


def _count_reps_from_frames(frames: list[dict], sport: str) -> dict:
    """
    Count reps via a simple state machine over a cyclic joint angle.

    Walks the frame timeline; whenever the angle drops below `bottom_below`
    (state=bottom) and then rises back above `top_above` (state=top), that's
    one completed rep. Robust to noise because both thresholds must be crossed.
    """
    cfg = REP_CYCLE.get(sport)
    if not cfg:
        return {"sport": sport, "supported": False, "rep_count": 0, "samples": 0}

    joint = cfg["joint"]
    bottom = cfg["bottom_below"]
    top = cfg["top_above"]

    state = "top"  # assume athletes start standing/extended
    reps = 0
    samples = 0
    for f in frames:
        v = _frame_metric(f, joint)
        if v is None:
            continue
        samples += 1
        if state == "top" and v < bottom:
            state = "bottom"
        elif state == "bottom" and v > top:
            reps += 1
            state = "top"

    return {
        "sport": sport,
        "supported": True,
        "joint": joint,
        "bottom_below_deg": bottom,
        "top_above_deg": top,
        "rep_count": reps,
        "samples": samples,
    }


@router.get("/sessions/{session_id}/rep-count")
async def rep_count(session_id: str):
    if session_id not in SESSION_DB:
        raise HTTPException(404, "session not found")
    session = SESSION_DB[session_id]
    frames = session.get("frames") or []
    sport = session.get("sport", "vertical_jump")
    result = _count_reps_from_frames(frames, sport)
    return {"session_id": session_id, **result}


# ─── Competition readiness ─────────────────────────────────────────────────


def _volume_score(session_count: int, window_days: int) -> float:
    """≥ 1 session every 2 days in the window = full marks."""
    target = max(1.0, window_days / 2.0)
    return max(0.0, min(1.0, session_count / target))


def _form_score_component(form_trend_pct: float, avg_form: float) -> float:
    """
    Blend absolute form quality with trend direction.
    avg_form is already 0-100; trend bumps/penalises by up to +/- 15 points.
    """
    base = max(0.0, min(100.0, avg_form)) / 100.0
    bump = max(-0.15, min(0.15, form_trend_pct / 100.0))
    return max(0.0, min(1.0, base + bump))


def _symmetry_component(injury_risk: dict) -> float:
    if injury_risk.get("risk") == "unknown":
        return 0.6  # neutral when no data
    dev = float(injury_risk.get("deviation_pct", 0.0))
    # 0% deviation = 1.0, 20%+ deviation = 0.0
    return max(0.0, 1.0 - (dev / 20.0))


def _hrv_component(athlete_id: str, days: int) -> tuple[float, int]:
    """
    Pull HRV samples from frames in window. Returns (component_0_1, sample_count).
    Higher HRV = better recovery. Neutral 0.6 when nothing captured.
    """
    samples: list[float] = []
    for s in _athlete_sessions(athlete_id, days):
        for frame in s.get("frames", []) or []:
            v = frame.get("hrv") or frame.get("hrv_ms")
            if v is not None:
                try:
                    samples.append(float(v))
                except (TypeError, ValueError):
                    continue
    if not samples:
        return 0.6, 0
    mean_hrv = statistics.mean(samples)
    # 20ms = poor, 80ms+ = elite (rPPG-derived rough band)
    score = (mean_hrv - 20.0) / 60.0
    return max(0.0, min(1.0, score)), len(samples)


def _readiness_band(score: float) -> str:
    if score >= 80:
        return "peak"
    if score >= 65:
        return "ready"
    if score >= 45:
        return "training"
    return "recover"


def _compute_readiness(athlete_id: str, days: int) -> dict:
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    progress = _compute_progress(athlete_id, days)
    risk = _compute_injury_risk(athlete_id, days)
    hrv_c, hrv_samples = _hrv_component(athlete_id, days)

    form_c = _form_score_component(progress["form_trend_pct"], progress["avg_form_score"])
    sym_c = _symmetry_component(risk)
    vol_c = _volume_score(progress["session_count"], days)

    score = round((form_c * 0.40 + sym_c * 0.20 + vol_c * 0.20 + hrv_c * 0.20) * 100, 1)
    band = _readiness_band(score)

    return {
        "athlete_id": athlete_id,
        "athlete_name": progress["athlete_name"],
        "sport": progress["sport"],
        "window_days": days,
        "score": score,
        "band": band,
        "components": {
            "form": {"weight": 0.40, "value": round(form_c * 100, 1)},
            "symmetry": {"weight": 0.20, "value": round(sym_c * 100, 1)},
            "volume": {"weight": 0.20, "value": round(vol_c * 100, 1)},
            "hrv": {"weight": 0.20, "value": round(hrv_c * 100, 1), "samples": hrv_samples},
        },
        "session_count": progress["session_count"],
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/readiness/{athlete_id}")
async def readiness(athlete_id: str, days: int = Query(default=14, ge=3, le=90)):
    key = f"readiness:{athlete_id}:{days}"
    cached = progress_cache.get(key)
    if cached:
        return cached
    payload = _compute_readiness(athlete_id, days)
    progress_cache.set(key, payload)
    return payload


@router.get("/athlete/{athlete_id}/advanced-metrics")
async def advanced_metrics(athlete_id: str, days: int = Query(default=60, ge=7, le=180)):
    """Bundle of derived training metrics computed by services.metrics_service.

    Includes ACWR + band, training monotony + strain, form momentum,
    trend %, asymmetry %, intensity of the latest session, fatigue index,
    and a composite readiness score. One endpoint the Android MetricsScreen
    can hit to render every chart.
    """
    from services import metrics_service as ms
    from services.repositories import athletes, sessions as session_repo

    if not athletes.exists(athlete_id):
        raise HTTPException(404, "athlete not found")

    athlete_sessions = [s for s in session_repo.list(athlete_id=athlete_id, status="completed")]
    # Keep only sessions within the window. Normalise parsed timestamps to
    # tz-aware so tz-mixed session data (some legacy sessions stored naive
    # ISO) can still be compared against `now`.
    cutoff_days = days
    now = datetime.now(timezone.utc)
    recent: list[dict] = []
    for s in athlete_sessions:
        try:
            raw = str(s.get("ended_at") or s.get("started_at") or "").replace("Z", "+00:00")
            ts = datetime.fromisoformat(raw)
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (now - ts).days <= cutoff_days:
            recent.append(s)

    # Collect frames across recent sessions for asymmetry
    all_frames: list[dict] = []
    for s in recent[:15]:
        all_frames.extend(s.get("frames", []) or [])

    latest_summary = (recent[0].get("summary") if recent else {}) or {}
    hr_summary = latest_summary.get("heart_rate") if recent else None

    # Daily load for the chart
    load_map = ms.daily_load(recent)
    load_series = [{"date": d, "load": round(v, 1)} for d, v in sorted(load_map.items())]

    # Form trend series (last 14 days, one point per session day)
    trend_series: list[dict] = []
    seen_days: set[str] = set()
    for s in reversed(recent):
        iso = (s.get("ended_at") or s.get("started_at") or "")[:10]
        if not iso or iso in seen_days:
            continue
        summary = s.get("summary") or {}
        avg = summary.get("avg_form_score", 0)
        if avg:
            trend_series.append({"date": iso, "score": round(float(avg), 1)})
            seen_days.add(iso)
    trend_series = trend_series[-14:]

    return {
        "athlete_id": athlete_id,
        "window_days": days,
        "session_count": len(recent),
        "aggregate": ms.aggregate_sessions(recent),
        "trend_pct": ms.form_score_trend_pct(recent),
        "momentum": ms.form_momentum(recent),
        "acwr": ms.acute_chronic_ratio(recent),
        "monotony": ms.training_monotony(recent),
        "asymmetry": ms.asymmetry_index(all_frames),
        "latest_intensity": ms.session_intensity(latest_summary or {}, hr_summary),
        "fatigue": ms.fatigue_index(recent),
        "readiness": ms.readiness_score(recent),
        "load_series": load_series,
        "form_trend_series": trend_series,
        "heart_rate": hr_summary,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
