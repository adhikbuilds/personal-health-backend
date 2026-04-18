from __future__ import annotations

"""
Wellness Extended — WN-08 through WN-14
Enhanced wellness logging, scoring, trends, recovery, correlations, and team overview.
Storage: db/wellness.json  { athlete_id: { "YYYY-MM-DD": { ... } } }
"""

import asyncio
import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database import ATHLETE_DB, SESSION_DB, _load_json, _save_json
from logging_setup import get_logger

log = get_logger("routes.wellness_extended")

router = APIRouter(tags=["Wellness"])

_WELLNESS_LOCKS: dict[str, asyncio.Lock] = {}


def _get_lock(athlete_id: str) -> asyncio.Lock:
    if athlete_id not in _WELLNESS_LOCKS:
        _WELLNESS_LOCKS[athlete_id] = asyncio.Lock()
    return _WELLNESS_LOCKS[athlete_id]


# ─── Formulas ─────────────────────────────────────────────────────────────────


def _compute_sleep_score(duration_hrs: float, quality: int, interruptions: int) -> int:
    base = (min(duration_hrs, 8.0) / 8.0) * 40
    quality_pts = max(1, min(5, quality)) * 8
    penalty = interruptions * 5
    return max(0, min(100, round(base + quality_pts - penalty)))


def _parse_duration(bedtime: str, wake_time: str) -> float:
    """HH:MM strings → duration in hours, handling midnight crossing."""
    bh, bm = int(bedtime[:2]), int(bedtime[3:])
    wh, wm = int(wake_time[:2]), int(wake_time[3:])
    bed_mins = bh * 60 + bm
    wake_mins = wh * 60 + wm
    if wake_mins <= bed_mins:
        wake_mins += 24 * 60
    return (wake_mins - bed_mins) / 60.0


def _compute_wellness_score(
    sleep_score: int,
    hydration_pct: float,
    mood: int,
    energy: int,
    stress: int,
    soreness: int,
) -> tuple[int, dict]:
    sc = sleep_score * 0.30
    hc = min(hydration_pct, 100) * 0.15
    mc = (mood / 10) * 100 * 0.15
    ec = (energy / 10) * 100 * 0.15
    stc = ((10 - stress) / 10) * 100 * 0.10
    soc = ((10 - soreness) / 10) * 100 * 0.15
    total = round(sc + hc + mc + ec + stc + soc)
    breakdown = {
        "sleep": round(sc),
        "hydration": round(hc),
        "mood": round(mc),
        "energy": round(ec),
        "stress": round(stc),
        "soreness": round(soc),
    }
    return total, breakdown


def _make_recommendation(breakdown: dict, stress: int, soreness: int, target_ml: int) -> str:
    if breakdown["sleep"] < 18:
        return "Your sleep score is low. Try to get 7-8 hours tonight."
    if breakdown["hydration"] < 10:
        return f"You're under-hydrated. Aim for {target_ml}ml today."
    if stress > 7:
        return "High stress detected. Consider a lighter session or active recovery."
    if soreness > 7:
        return "High soreness. Rest day recommended to prevent injury."
    return "Looking good! You're ready for a solid session."


# ─── Pydantic models ──────────────────────────────────────────────────────────


class SleepInput(BaseModel):
    bedtime: str = Field("23:00", description="HH:MM 24-hour")
    wake_time: str = Field("07:00", description="HH:MM 24-hour")
    quality: int = Field(3, ge=1, le=5)
    interruptions: int = Field(0, ge=0, le=10)


class HydrationInput(BaseModel):
    water_ml: int = Field(2000, ge=0)


class MentalInput(BaseModel):
    mood: int = Field(5, ge=1, le=10)
    stress: int = Field(5, ge=1, le=10)
    energy: int = Field(5, ge=1, le=10)
    journal_note: Optional[str] = Field(None, max_length=200)


class PhysicalInput(BaseModel):
    soreness: int = Field(3, ge=1, le=10)
    body_weight_kg: Optional[float] = Field(None, gt=0, le=300)


class WellnessCheckinEnhanced(BaseModel):
    date: Optional[date] = None
    sleep: SleepInput = Field(default_factory=SleepInput)
    hydration: HydrationInput = Field(default_factory=HydrationInput)
    mental: MentalInput = Field(default_factory=MentalInput)
    physical: PhysicalInput = Field(default_factory=PhysicalInput)


# ─── WN-08: Enhanced check-in ─────────────────────────────────────────────────


@router.post("/athlete/{athlete_id}/wellness")
async def log_wellness_enhanced(athlete_id: str, body: WellnessCheckinEnhanced):
    """Log detailed morning wellness data with sleep analysis and hydration targets. (WN-08)"""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")

    now = datetime.now(timezone.utc)
    date_key = body.date.isoformat() if body.date else now.date().isoformat()

    duration_hrs = _parse_duration(body.sleep.bedtime, body.sleep.wake_time)
    sleep_score = _compute_sleep_score(duration_hrs, body.sleep.quality, body.sleep.interruptions)

    bw = body.physical.body_weight_kg or 70.0
    target_ml = round(bw * 35)
    hydration_pct = round(min(100, (body.hydration.water_ml / max(target_ml, 1)) * 100), 1)

    wellness_score, breakdown = _compute_wellness_score(
        sleep_score,
        hydration_pct,
        body.mental.mood,
        body.mental.energy,
        body.mental.stress,
        body.physical.soreness,
    )
    recovery_ready = wellness_score >= 60

    entry = {
        "sleep": {
            "bedtime": body.sleep.bedtime,
            "wake_time": body.sleep.wake_time,
            "quality": body.sleep.quality,
            "interruptions": body.sleep.interruptions,
            "duration_hrs": round(duration_hrs, 2),
            "sleep_score": sleep_score,
        },
        "hydration": {
            "water_ml": body.hydration.water_ml,
            "target_ml": target_ml,
            "pct": hydration_pct,
        },
        "mental": {
            "mood": body.mental.mood,
            "stress": body.mental.stress,
            "energy": body.mental.energy,
            "journal_note": body.mental.journal_note,
        },
        "physical": {
            "soreness": body.physical.soreness,
            "body_weight_kg": body.physical.body_weight_kg,
        },
        "wellness_score": wellness_score,
        "recovery_ready": recovery_ready,
        "logged_at": now.isoformat(),
    }

    async with _get_lock(athlete_id):
        db = _load_json("wellness.json") or {}
        db.setdefault(athlete_id, {})[date_key] = entry
        _save_json("wellness.json", db)

    log.info("enhanced wellness logged athlete=%s date=%s score=%d", athlete_id, date_key, wellness_score)
    return {
        "athlete_id": athlete_id,
        "date": date_key,
        "logged": True,
        "wellness_score": wellness_score,
        "recovery_ready": recovery_ready,
        "breakdown": breakdown,
    }


# ─── WN-09: History date range ────────────────────────────────────────────────


@router.get("/athlete/{athlete_id}/wellness")
async def get_wellness_history(
    athlete_id: str,
    from_date: Optional[date] = Query(None, alias="from"),
    to_date: Optional[date] = Query(None, alias="to"),
):
    """Return wellness entries for a date range, sorted ascending. (WN-09)"""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")

    today = datetime.now(timezone.utc).date()
    d_from = from_date or (today - timedelta(days=6))
    d_to = to_date or today

    db = _load_json("wellness.json") or {}
    athlete_data = db.get(athlete_id, {})

    entries = []
    for date_key, entry in sorted(athlete_data.items()):
        try:
            d = date.fromisoformat(date_key)
        except ValueError:
            continue
        if d_from <= d <= d_to:
            entries.append(
                {
                    "date": date_key,
                    "wellness_score": entry.get("wellness_score"),
                    "recovery_ready": entry.get("recovery_ready"),
                    "sleep": entry.get("sleep", {}),
                    "hydration": entry.get("hydration", {}),
                    "mental": entry.get("mental", {}),
                    "physical": entry.get("physical", {}),
                }
            )

    return {"entries": entries, "total": len(entries)}


# ─── WN-10: Enhanced wellness score ───────────────────────────────────────────


@router.get("/athlete/{athlete_id}/wellness/score")
async def get_wellness_score_enhanced(athlete_id: str):
    """Return today's composite wellness score with breakdown and recommendation. (WN-10)"""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    db = _load_json("wellness.json") or {}
    athlete_data = db.get(athlete_id, {})

    entry = athlete_data.get(today.isoformat()) or athlete_data.get(yesterday.isoformat())
    if not entry:
        # Fallback: check old-style wellness_log in athletes.json
        old_log = ATHLETE_DB[athlete_id].get("wellness_log", {})
        old_entry = old_log.get(today.isoformat()) or old_log.get(yesterday.isoformat())
        if old_entry:
            return {
                "athlete_id": athlete_id,
                "wellness_score": old_entry.get("wellness_score"),
                "message": "Using legacy wellness data. Log morning check-in to upgrade.",
            }
        return {
            "athlete_id": athlete_id,
            "wellness_score": None,
            "message": "No recent wellness data. Log your morning check-in!",
        }

    date_used = today.isoformat() if today.isoformat() in athlete_data else yesterday.isoformat()
    wellness_score = entry["wellness_score"]
    recovery_ready = entry["recovery_ready"]
    sleep_score = entry.get("sleep", {}).get("sleep_score", 0)
    hydration_pct = entry.get("hydration", {}).get("pct", 0)
    mood = entry.get("mental", {}).get("mood", 5)
    energy = entry.get("mental", {}).get("energy", 5)
    stress = entry.get("mental", {}).get("stress", 5)
    soreness = entry.get("physical", {}).get("soreness", 5)
    target_ml = entry.get("hydration", {}).get("target_ml", 2450)

    _, breakdown = _compute_wellness_score(sleep_score, hydration_pct, mood, energy, stress, soreness)
    recommendation = _make_recommendation(breakdown, stress, soreness, target_ml)

    return {
        "athlete_id": athlete_id,
        "date": date_used,
        "wellness_score": wellness_score,
        "recovery_ready": recovery_ready,
        "breakdown": breakdown,
        "recommendation": recommendation,
        "sleep": entry.get("sleep", {}),
        "hydration": entry.get("hydration", {}),
    }


# ─── WN-11: Wellness trends ────────────────────────────────────────────────────


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _iso_week_label(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


@router.get("/athlete/{athlete_id}/wellness/trends")
async def get_wellness_trends(
    athlete_id: str,
    period: str = Query("weekly", pattern="^(weekly|monthly)$"),
):
    """Return weekly or monthly wellness averages with trend direction. (WN-11)"""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")

    db = _load_json("wellness.json") or {}
    athlete_data = db.get(athlete_id, {})
    today = datetime.now(timezone.utc).date()

    if period == "weekly":
        buckets: dict[str, list] = {}
        for i in range(4):
            ref = today - timedelta(weeks=i)
            label = _iso_week_label(ref)
            buckets.setdefault(label, [])
        for date_key, entry in athlete_data.items():
            try:
                d = date.fromisoformat(date_key)
            except ValueError:
                continue
            label = _iso_week_label(d)
            if label in buckets:
                buckets[label].append(entry)
        labels_ordered = sorted(buckets.keys())
    else:
        buckets = {}
        for i in range(3):
            ref_day = today.replace(day=1)
            for _ in range(i):
                ref_day = (ref_day - timedelta(days=1)).replace(day=1)
            label = ref_day.strftime("%Y-%m")
            buckets.setdefault(label, [])
        for date_key, entry in athlete_data.items():
            try:
                d = date.fromisoformat(date_key)
            except ValueError:
                continue
            label = d.strftime("%Y-%m")
            if label in buckets:
                buckets[label].append(entry)
        labels_ordered = sorted(buckets.keys())

    def _avg(vals: list) -> float | None:
        return round(sum(vals) / len(vals), 1) if vals else None

    trends = []
    for label in labels_ordered:
        entries = buckets[label]
        count = len(entries)
        period_key = "week" if period == "weekly" else "month"
        if count == 0:
            trends.append(
                {
                    period_key: label,
                    "avg_wellness": None,
                    "avg_sleep_score": None,
                    "avg_hydration_pct": None,
                    "avg_mood": None,
                    "avg_stress": None,
                    "entries_count": 0,
                    "insufficient_data": True,
                }
            )
        else:
            ws = [e.get("wellness_score") for e in entries if e.get("wellness_score") is not None]
            ss = [e.get("sleep", {}).get("sleep_score", 0) for e in entries]
            hp = [e.get("hydration", {}).get("pct", 0) for e in entries]
            mo = [e.get("mental", {}).get("mood", 5) for e in entries]
            st = [e.get("mental", {}).get("stress", 5) for e in entries]
            trends.append(
                {
                    period_key: label,
                    "avg_wellness": _avg(ws),
                    "avg_sleep_score": _avg(ss),
                    "avg_hydration_pct": _avg(hp),
                    "avg_mood": _avg(mo),
                    "avg_stress": _avg(st),
                    "entries_count": count,
                    "insufficient_data": count < 3,
                }
            )

    wellness_vals = [t.get("avg_wellness") for t in trends if t.get("avg_wellness") is not None]
    if len(wellness_vals) >= 2:
        slope = _linear_slope([float(v) for v in wellness_vals])
        direction = "improving" if slope > 1 else ("declining" if slope < -1 else "stable")
    else:
        direction = "insufficient_data"

    return {"period": period, "trends": trends, "direction": direction}


# ─── WN-12: Recovery readiness ────────────────────────────────────────────────


@router.get("/athlete/{athlete_id}/wellness/recovery")
async def get_recovery_readiness(athlete_id: str):
    """Return ready/caution/rest status with 3-day trend override. (WN-12)"""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")

    db = _load_json("wellness.json") or {}
    athlete_data = db.get(athlete_id, {})

    if not athlete_data:
        return {
            "status": "unknown",
            "wellness_score": None,
            "factors": [],
            "suggestion": "No wellness data logged yet.",
        }

    sorted_entries = sorted(athlete_data.items(), reverse=True)
    latest_key, latest_entry = sorted_entries[0]
    wellness_score = latest_entry.get("wellness_score", 0)

    if wellness_score >= 70:
        status = "ready"
    elif wellness_score >= 50:
        status = "caution"
    else:
        status = "rest"

    # 3-day consecutive decline → bump down one level
    if len(sorted_entries) >= 3:
        scores = [e.get("wellness_score", 0) for _, e in sorted_entries[:3]]
        if scores[0] < scores[1] < scores[2]:
            if status == "ready":
                status = "caution"
            elif status == "caution":
                status = "rest"

    sleep_score = latest_entry.get("sleep", {}).get("sleep_score", 0)
    hydration_pct = latest_entry.get("hydration", {}).get("pct", 0)
    mood = latest_entry.get("mental", {}).get("mood", 5)
    energy = latest_entry.get("mental", {}).get("energy", 5)
    stress = latest_entry.get("mental", {}).get("stress", 5)
    soreness = latest_entry.get("physical", {}).get("soreness", 5)

    _, breakdown = _compute_wellness_score(sleep_score, hydration_pct, mood, energy, stress, soreness)
    max_contrib = {"sleep": 30, "hydration": 15, "mood": 15, "energy": 15, "stress": 10, "soreness": 15}
    factors = [k for k, v in breakdown.items() if v < max_contrib[k] * 0.5]

    suggestions = {
        "ready": "Ready to push hard today!",
        "caution": "Train, but monitor your fatigue.",
        "rest": "Take a rest or do light recovery work.",
    }

    return {
        "status": status,
        "wellness_score": wellness_score,
        "factors": factors,
        "suggestion": suggestions[status],
        "date": latest_key,
    }


# ─── WN-13: Wellness-performance correlations ─────────────────────────────────


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return round(num / (den_x * den_y), 3)


@router.get("/athlete/{athlete_id}/wellness/correlations")
async def get_correlations(athlete_id: str):
    """Return Pearson r between wellness components and session form scores. (WN-13)"""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "Athlete not found")

    db = _load_json("wellness.json") or {}
    athlete_wellness = db.get(athlete_id, {})

    pairs: list[dict] = []
    for session in SESSION_DB.values():
        if session.get("athlete_id") != athlete_id:
            continue
        session_date = (session.get("started_at") or "")[:10]
        if not session_date or session_date not in athlete_wellness:
            continue
        frame_results = session.get("frame_results", [])
        form_scores = [f.get("form_score", 0) for f in frame_results if f.get("form_score")]
        avg_form = (
            sum(form_scores) / len(form_scores)
            if form_scores
            else (session.get("avg_form_score") or session.get("form_score") or 0)
        )
        if not avg_form:
            continue
        w = athlete_wellness[session_date]
        pairs.append(
            {
                "form": avg_form,
                "sleep_score": w.get("sleep", {}).get("sleep_score", 0),
                "hydration_pct": w.get("hydration", {}).get("pct", 0),
                "mood": w.get("mental", {}).get("mood", 5),
                "energy": w.get("mental", {}).get("energy", 5),
            }
        )

    if len(pairs) < 10:
        return {
            "correlations": None,
            "message": "Need at least 10 sessions with wellness data to compute correlations",
            "sample_size": len(pairs),
        }

    forms = [p["form"] for p in pairs]
    correlations = {
        "sleep_vs_form": _pearson_r([p["sleep_score"] for p in pairs], forms),
        "hydration_vs_form": _pearson_r([p["hydration_pct"] for p in pairs], forms),
        "mood_vs_form": _pearson_r([p["mood"] for p in pairs], forms),
        "energy_vs_form": _pearson_r([p["energy"] for p in pairs], forms),
    }

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    insights = []
    for key, r in sorted_corr[:2]:
        if abs(r) < 0.3:
            continue
        label = key.split("_vs_")[0]
        direction = "improves" if r > 0 else "declines"
        insights.append(f"Your form score {direction} when {label} is higher (r={r:+.2f}).")

    return {"correlations": correlations, "insights": insights, "sample_size": len(pairs)}


# ─── WN-14: Team wellness overview ────────────────────────────────────────────


@router.get("/wellness/team-overview")
async def get_team_wellness_overview():
    """Return all athletes' latest wellness scores for coach dashboard. (WN-14)"""
    db = _load_json("wellness.json") or {}
    today = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=3)

    athletes_out = []
    for athlete_id, athlete in ATHLETE_DB.items():
        athlete_wellness = db.get(athlete_id, {})
        recent = {}
        for k, v in athlete_wellness.items():
            try:
                if date.fromisoformat(k) >= cutoff:
                    recent[k] = v
            except ValueError:
                pass
        if recent:
            latest_key = max(recent.keys())
            entry = recent[latest_key]
            score = entry.get("wellness_score")
            status = (
                ("ready" if score >= 70 else ("caution" if score >= 50 else "rest")) if score is not None else "unknown"
            )
            athletes_out.append(
                {
                    "id": athlete_id,
                    "name": athlete.get("name", "Unknown"),
                    "wellness_score": score,
                    "recovery_status": status,
                    "last_logged": latest_key,
                }
            )
        else:
            athletes_out.append(
                {
                    "id": athlete_id,
                    "name": athlete.get("name", "Unknown"),
                    "wellness_score": None,
                    "recovery_status": "unknown",
                    "last_logged": None,
                }
            )

    athletes_out.sort(key=lambda a: (a["wellness_score"] is None, a["wellness_score"] or 0))

    known = [a for a in athletes_out if a["wellness_score"] is not None]
    avg_score = round(sum(a["wellness_score"] for a in known) / len(known)) if known else None

    return {
        "athletes": athletes_out,
        "summary": {
            "avg_score": avg_score,
            "ready_count": sum(1 for a in known if a["recovery_status"] == "ready"),
            "caution_count": sum(1 for a in known if a["recovery_status"] == "caution"),
            "rest_count": sum(1 for a in known if a["recovery_status"] == "rest"),
            "total_athletes": len(athletes_out),
        },
    }
