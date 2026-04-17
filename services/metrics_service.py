from __future__ import annotations

"""
Metrics Service — one place where derived athlete metrics are computed.

Phase 2 goal: replace the ad-hoc avg/peak/trend calculations repeated in
routes/athletes.py, routes/progress.py, routes/coach.py, routes/scorecard.py
with a single set of pure functions that work over session + frame data.

Phase 3 metrics (ACWR, training monotony, momentum, fatigue index, HR zone
distribution) also live here so every surface — scorecard, dashboard,
Android — pulls the same numbers.

Everything in this file is side-effect-free: pass in the data, get the
metric back. Repositories fetch the data, routes pass it through.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional


# ─── Parsing helpers ───────────────────────────────────────────────────────


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _avg(values: Iterable[float]) -> float:
    lst = [float(v) for v in values if v is not None]
    return round(statistics.mean(lst), 2) if lst else 0.0


def _safe_score(session: dict) -> float:
    summary = session.get("summary") or {}
    return float(summary.get("avg_form_score", 0) or 0)


# ─── Session-level aggregates ──────────────────────────────────────────────


def aggregate_sessions(sessions: list[dict]) -> dict:
    """Roll up a list of completed sessions into the core stats.

    Returns avg_form_score, peak_form_score, total_sessions, total_reps,
    best_jump_cm, quality_distribution, total_xp.
    """
    scores: list[float] = []
    peak_scores: list[float] = []
    reps_total = 0
    best_jump = 0.0
    xp_total = 0
    quality: dict[str, int] = {"elite": 0, "good": 0, "average": 0, "poor": 0}

    for s in sessions:
        summary = s.get("summary") or {}
        avg = float(summary.get("avg_form_score", 0) or 0)
        if avg > 0:
            scores.append(avg)
        peak = float(summary.get("peak_form_score", 0) or 0)
        if peak > 0:
            peak_scores.append(peak)
        reps_total += int(summary.get("total_frames", 0) or 0)
        best_jump = max(best_jump, float(summary.get("peak_jump_height_cm", 0) or 0))
        xp_total += int(summary.get("xp_earned", 0) or 0)
        for q, count in (summary.get("quality_distribution") or {}).items():
            if q in quality:
                quality[q] += int(count)

    return {
        "avg_form_score": _avg(scores),
        "peak_form_score": round(max(peak_scores, default=0.0), 1),
        "total_sessions": len(sessions),
        "total_reps": reps_total,
        "best_jump_cm": round(best_jump, 1),
        "total_xp": xp_total,
        "quality_distribution": quality,
    }


# ─── Trend metrics ─────────────────────────────────────────────────────────


def form_score_trend_pct(sessions: list[dict], split: int = 3) -> float:
    """Return the % change in avg form score between the most recent `split`
    sessions and the prior `split`.  Negative when the athlete is regressing.
    """
    if len(sessions) < split * 2:
        return 0.0
    recent = [_safe_score(s) for s in sessions[:split] if _safe_score(s) > 0]
    prior = [_safe_score(s) for s in sessions[split:split * 2] if _safe_score(s) > 0]
    if not recent or not prior:
        return 0.0
    rm, pm = statistics.mean(recent), statistics.mean(prior)
    if pm == 0:
        return 0.0
    return round((rm - pm) / pm * 100, 2)


def form_momentum(sessions: list[dict]) -> float:
    """7-day derivative on avg form score — how fast the trend is accelerating.

    Simple finite-difference on the last 7 sessions: mean(recent 3) minus
    mean(earliest 3) divided by the time span. Positive = improving fast.
    """
    recent = sessions[:7]
    if len(recent) < 6:
        return 0.0
    first_chunk = [_safe_score(s) for s in recent[-3:] if _safe_score(s) > 0]
    last_chunk = [_safe_score(s) for s in recent[:3] if _safe_score(s) > 0]
    if not first_chunk or not last_chunk:
        return 0.0
    return round(statistics.mean(last_chunk) - statistics.mean(first_chunk), 2)


# ─── Training load (ACWR, monotony) ────────────────────────────────────────


def daily_load(sessions: list[dict]) -> dict[str, float]:
    """Sum "load" per ISO date. Load = duration_seconds × avg_form_score/100.

    Rationale: duration alone under-weights hard sessions with bad form.
    Score × duration captures intensity more faithfully.
    """
    totals: dict[str, float] = defaultdict(float)
    for s in sessions:
        ts = _parse_ts(s.get("ended_at") or s.get("started_at"))
        if not ts:
            continue
        summary = s.get("summary") or {}
        duration = float(summary.get("duration_seconds", 0) or 0)
        score = float(summary.get("avg_form_score", 0) or 0)
        load = duration * max(score, 10) / 100
        totals[ts.date().isoformat()] += load
    return dict(totals)


def acute_chronic_ratio(sessions: list[dict]) -> dict[str, float]:
    """Compute ACWR: acute (7d) : chronic (28d) training load.

    >1.5 signals spike (overreaching); <0.8 signals under-loading; 0.8-1.3 is
    the "sweet spot" for adaptation without injury.
    """
    now = datetime.now(timezone.utc)
    acute_sum = 0.0
    chronic_sum = 0.0
    acute_days = chronic_days = 0
    for iso, load in daily_load(sessions).items():
        ts = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        delta = (now - ts).days
        if delta <= 7:
            acute_sum += load
            acute_days += 1
        if delta <= 28:
            chronic_sum += load
            chronic_days += 1
    acute = acute_sum / 7
    chronic = chronic_sum / 28
    ratio = round(acute / chronic, 2) if chronic > 0 else 0.0
    if ratio == 0:
        band = "unknown"
    elif ratio < 0.8:
        band = "under-loaded"
    elif ratio <= 1.3:
        band = "sweet spot"
    elif ratio <= 1.5:
        band = "high"
    else:
        band = "spike — injury risk"
    return {
        "acute_load": round(acute, 1),
        "chronic_load": round(chronic, 1),
        "acwr": ratio,
        "band": band,
    }


def training_monotony(sessions: list[dict]) -> dict[str, float]:
    """Monotony = mean(daily load) / stdev(daily load) over last 7 days.

    High monotony (>2.0) means every day feels identical — recovery days
    missing. Strain = monotony × weekly load.
    """
    loads = list(daily_load(sessions).values())[-7:]
    if len(loads) < 3:
        return {"monotony": 0.0, "strain": 0.0}
    mean = statistics.mean(loads)
    sd = statistics.pstdev(loads) or 1.0
    monotony = round(mean / sd, 2)
    strain = round(sum(loads) * monotony, 1)
    return {"monotony": monotony, "strain": strain}


# ─── Bilateral asymmetry ───────────────────────────────────────────────────


def asymmetry_index(frames: list[dict]) -> dict[str, float]:
    """Compute left/right asymmetry % for knee, hip, shoulder.

    Non-symmetric joint angles are injury predictors. Returns % difference
    averaged across frames. >10% on a single joint = watch; >15% = flag.
    """
    pairs = [("knee_angle_l", "knee_angle_r"), ("hip_angle_l", "hip_angle_r"),
             ("shoulder_angle_l", "shoulder_angle_r")]
    out: dict[str, float] = {}
    for l_key, r_key in pairs:
        diffs: list[float] = []
        for f in frames:
            l = f.get(l_key)
            r = f.get(r_key)
            if l is None or r is None:
                continue
            avg = (l + r) / 2 or 1
            diffs.append(abs(l - r) / avg * 100)
        joint = l_key.split("_angle_")[0]
        out[joint] = round(statistics.mean(diffs), 2) if diffs else 0.0
    max_joint = max(out.values(), default=0.0)
    out["overall"] = round(max_joint, 2)
    if max_joint == 0:
        out["band"] = "insufficient data"
    elif max_joint < 6:
        out["band"] = "symmetrical"
    elif max_joint < 10:
        out["band"] = "minor"
    elif max_joint < 15:
        out["band"] = "watch"
    else:
        out["band"] = "flag — assess dominant side"
    return out


# ─── Readiness score ───────────────────────────────────────────────────────


def readiness_score(
    sessions: list[dict],
    *,
    hrv_ms: float = 0.0,
    resting_bpm: float = 0.0,
    sleep_hours: float = 0.0,
    soreness: int = 0,  # 0-10, 0 = none
) -> dict:
    """Composite 0-100 readiness score combining form momentum, load balance,
    HRV, resting HR, sleep, and subjective soreness.

    Each component contributes 0-20. Missing components contribute 10
    (neutral) so partial data still yields a sensible score.
    """
    components = {}

    # 1. Form momentum (20 pts)
    momentum = form_momentum(sessions)
    components["form_momentum"] = max(0.0, min(20.0, 10 + momentum))

    # 2. Load balance — center around ACWR 1.0 (20 pts)
    acwr = acute_chronic_ratio(sessions).get("acwr", 0) or 0
    if acwr == 0:
        components["load_balance"] = 10.0
    else:
        # Penalty grows as ACWR deviates from 1.0
        components["load_balance"] = max(0.0, 20.0 - abs(acwr - 1.0) * 20)

    # 3. HRV (20 pts) — 35ms+ is elite, <15ms is fatigued
    if hrv_ms > 0:
        components["hrv"] = max(0.0, min(20.0, (hrv_ms - 10) / 25 * 20))
    else:
        components["hrv"] = 10.0

    # 4. Resting HR (20 pts) — 50bpm = 20, 80bpm = 0
    if resting_bpm > 0:
        components["resting_hr"] = max(0.0, min(20.0, (80 - resting_bpm) / 30 * 20))
    else:
        components["resting_hr"] = 10.0

    # 5. Sleep (10 pts) — 8h = full score
    if sleep_hours > 0:
        components["sleep"] = max(0.0, min(10.0, sleep_hours / 8 * 10))
    else:
        components["sleep"] = 5.0

    # 6. Soreness (10 pts) — 0 soreness = full score
    components["soreness"] = max(0.0, 10.0 - soreness)

    total = round(sum(components.values()), 1)
    band = (
        "elite" if total >= 85
        else "ready" if total >= 70
        else "caution" if total >= 50
        else "recover"
    )
    return {"score": total, "band": band, "components": {k: round(v, 1) for k, v in components.items()}}


# ─── Intensity / fatigue indices ───────────────────────────────────────────


def session_intensity(summary: dict, hr_summary: Optional[dict] = None) -> float:
    """Per-session intensity index (0-100).

    Blends form score, asymmetry, and (if available) average heart-rate zone
    index. Surface on scorecard and home screen to give the athlete one
    number for "how hard was that session".
    """
    if not summary:
        return 0.0
    form = float(summary.get("avg_form_score", 0) or 0)
    sym = float(summary.get("avg_symmetry", 1) or 1)
    form_weight = form * 0.55
    sym_weight = (sym * 100) * 0.15
    hr_weight = 0.0
    if hr_summary:
        zone_dist = hr_summary.get("zone_distribution", {}) or {}
        total = sum(zone_dist.values()) or 1
        zone_score = sum(
            {
                "recovery": 20, "endurance": 40, "tempo": 60,
                "threshold": 80, "anaerobic": 100,
            }.get(z, 0) * c
            for z, c in zone_dist.items()
        ) / total
        hr_weight = zone_score * 0.30
    else:
        hr_weight = form_weight * 0.30 / 0.55  # fall back to form-proportional estimate
    return round(min(100.0, form_weight + sym_weight + hr_weight), 1)


def fatigue_index(sessions: list[dict]) -> dict:
    """Infer fatigue from the last 3 sessions vs the prior 3. Rising avg form
    score + steady load = fresh. Dropping scores under rising load = fatigued.
    """
    if len(sessions) < 6:
        return {"index": 0.0, "band": "insufficient data"}
    recent = sessions[:3]
    prior = sessions[3:6]
    r_score = statistics.mean([_safe_score(s) for s in recent] or [0])
    p_score = statistics.mean([_safe_score(s) for s in prior] or [0])
    r_load = sum((s.get("summary") or {}).get("duration_seconds", 0) or 0 for s in recent)
    p_load = sum((s.get("summary") or {}).get("duration_seconds", 0) or 0 for s in prior) or 1
    # Fatigue if scores drop while load rises
    score_delta = p_score - r_score  # positive = got worse
    load_ratio = r_load / p_load
    index = score_delta * load_ratio
    if index < -5:
        band = "fresh"
    elif index < 2:
        band = "neutral"
    elif index < 8:
        band = "accumulating"
    else:
        band = "fatigued"
    return {"index": round(index, 2), "band": band}
