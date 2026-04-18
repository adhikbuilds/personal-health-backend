from __future__ import annotations

"""
Athlete Intelligence Report — the product's core monetizable output.

VISION.md: "Almost no one can tell you, in plain English, what to do about it."

This module generates a comprehensive periodic report that aggregates every
signal the platform captures into a single actionable document. This is what
coaches pay ₹999/month for and what athletes share with their training partners.

The report contains:
  1. Performance trajectory — where you were, where you are, where you're heading
  2. Biomechanical profile — your movement signature across all joints
  3. Training load analysis — volume vs recovery, overreach detection
  4. Periodization recommendation — mesocycle phase + next 4-week plan
  5. Peer benchmarks — where you stand in your tier and sport
  6. Competition readiness timeline — projected date to peak readiness
  7. Key risks and action items — the 3 things to fix this week

Endpoint:
  GET /athlete/{id}/intelligence-report?days=30
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from auth import require_athlete_or_admin
from database import ATHLETE_DB
from logging_setup import get_logger
from routes.progress import (
    IDEAL_RANGES,
    _athlete_sessions,
    _compute_injury_risk,
    _compute_weak_joints,
    _frame_metric,
    _parse_dt,
)

router = APIRouter(tags=["Intelligence Report"])
log = get_logger("routes.intelligence_report")


# ─── 1. Performance Trajectory ─────────────────────────────────────────────


def _performance_trajectory(sessions: list[dict], days: int) -> dict:
    """
    Compute form score trajectory over the window, split into 4 equal segments.
    Fit a simple linear trend and project where the athlete will be in 2 weeks.
    """
    if not sessions:
        return {
            "segments": [],
            "trend_direction": "insufficient_data",
            "projected_form_2w": None,
            "velocity": 0.0,
        }

    # bucket sessions into 4 segments
    seg_size = max(1, len(sessions) // 4)
    segments = []
    for i in range(4):
        start = i * seg_size
        end = start + seg_size if i < 3 else len(sessions)
        seg_sessions = sessions[start:end]
        if not seg_sessions:
            continue

        scores = []
        jumps = []
        for s in seg_sessions:
            sm = s.get("summary") or {}
            af = float(sm.get("avg_form_score") or 0)
            if af > 0:
                scores.append(af)
            pj = float(sm.get("peak_jump_height_cm") or 0)
            if pj > 0:
                jumps.append(pj)

        segments.append(
            {
                "segment": i + 1,
                "sessions": len(seg_sessions),
                "avg_form": round(statistics.mean(scores), 1) if scores else 0.0,
                "peak_form": round(max(scores, default=0), 1),
                "avg_jump_cm": round(statistics.mean(jumps), 1) if jumps else 0.0,
            }
        )

    # linear trend from segments
    scored_segs = [s for s in segments if s["avg_form"] > 0]
    if len(scored_segs) >= 2:
        first_avg = scored_segs[0]["avg_form"]
        last_avg = scored_segs[-1]["avg_form"]
        velocity = (last_avg - first_avg) / len(scored_segs)  # points per segment

        if velocity > 1.5:
            direction = "improving"
        elif velocity < -1.5:
            direction = "declining"
        else:
            direction = "stable"

        # project 2 weeks out (roughly 2 more segments)
        projected = round(last_avg + velocity * 2, 1)
        projected = max(0, min(100, projected))
    else:
        velocity = 0.0
        direction = "insufficient_data"
        projected = None

    return {
        "segments": segments,
        "trend_direction": direction,
        "velocity_per_segment": round(velocity, 2),
        "projected_form_2w": projected,
    }


# ─── 2. Biomechanical Profile ─────────────────────────────────────────────


def _biomechanical_profile(sessions: list[dict], sport: str) -> dict:
    """
    Build the athlete's movement signature: average angle for each joint,
    deviation from ideal, consistency (std dev), and left-right balance.
    """
    ideal = IDEAL_RANGES.get(sport, IDEAL_RANGES.get("vertical_jump", {}))
    joint_data: dict[str, list[float]] = defaultdict(list)
    lr_pairs: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"l": [], "r": []})

    for s in sessions:
        for frame in s.get("frames") or []:
            for joint in ideal:
                v = _frame_metric(frame, joint)
                if v is not None:
                    joint_data[joint].append(v)

            # collect L/R separately for balance analysis
            for side_joint in ["knee_angle", "hip_angle", "elbow_angle", "shoulder_angle"]:
                lv = frame.get(f"{side_joint}_l")
                rv = frame.get(f"{side_joint}_r")
                if lv is not None:
                    lr_pairs[side_joint]["l"].append(float(lv))
                if rv is not None:
                    lr_pairs[side_joint]["r"].append(float(rv))

    profile = []
    for joint, values in joint_data.items():
        if not values:
            continue
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        lo, hi = ideal.get(joint, (0, 180))
        if mean_val < lo:
            deviation = lo - mean_val
            status = "below_range"
        elif mean_val > hi:
            deviation = mean_val - hi
            status = "above_range"
        else:
            deviation = 0.0
            status = "in_range"

        profile.append(
            {
                "joint": joint,
                "mean_deg": round(mean_val, 1),
                "std_deg": round(std_val, 1),
                "ideal_min": lo,
                "ideal_max": hi,
                "deviation_deg": round(deviation, 1),
                "status": status,
                "consistency": "stable" if std_val < 8 else "variable" if std_val < 15 else "inconsistent",
                "samples": len(values),
            }
        )

    # left-right balance
    balance = []
    for joint, sides in lr_pairs.items():
        if sides["l"] and sides["r"]:
            mean_l = statistics.mean(sides["l"])
            mean_r = statistics.mean(sides["r"])
            diff = abs(mean_l - mean_r)
            dominant = "left" if mean_l > mean_r else "right" if mean_r > mean_l else "balanced"
            balance.append(
                {
                    "joint": joint,
                    "mean_left": round(mean_l, 1),
                    "mean_right": round(mean_r, 1),
                    "asymmetry_deg": round(diff, 1),
                    "dominant_side": dominant,
                    "balanced": diff < 5,
                }
            )

    profile.sort(key=lambda x: x["deviation_deg"], reverse=True)

    return {
        "joints": profile,
        "balance": balance,
        "strongest_joint": profile[-1]["joint"] if profile else None,
        "weakest_joint": profile[0]["joint"] if profile else None,
    }


# ─── 3. Training Load Analysis ────────────────────────────────────────────


def _training_load_analysis(sessions: list[dict], days: int) -> dict:
    """
    Analyze training volume, density, and detect overreach patterns.

    Uses acute:chronic workload ratio concept (adapted for form scores):
    - acute = last 7 days avg
    - chronic = full window avg
    - ratio > 1.3 = spike (injury risk)
    - ratio < 0.8 = detraining
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    daily_load: dict[str, float] = defaultdict(float)
    daily_sessions: dict[str, int] = defaultdict(int)

    for s in sessions:
        ts = _parse_dt(s.get("started_at"))
        if not ts:
            continue
        day = ts.date().isoformat()
        sm = s.get("summary") or {}
        score = float(sm.get("avg_form_score") or 0)
        frames = int(sm.get("total_frames") or s.get("frame_count") or 0)
        # load proxy = form_score * frames (effort * volume)
        daily_load[day] += score * max(frames, 1)
        daily_sessions[day] += 1

    if not daily_load:
        return {
            "acute_load": 0,
            "chronic_load": 0,
            "acwr": None,
            "acwr_status": "no_data",
            "training_days": 0,
            "rest_days": 0,
            "avg_sessions_per_week": 0,
            "density": "none",
        }

    # acute (last 7 days) vs chronic (full window)
    acute_cutoff = (now - timedelta(days=7)).date().isoformat()
    acute_loads = [v for d, v in daily_load.items() if d >= acute_cutoff]
    chronic_loads = list(daily_load.values())

    acute = statistics.mean(acute_loads) if acute_loads else 0
    chronic = statistics.mean(chronic_loads) if chronic_loads else 0
    acwr = round(acute / chronic, 2) if chronic > 0 else None

    if acwr is None:
        acwr_status = "no_data"
    elif acwr > 1.5:
        acwr_status = "dangerous_spike"
    elif acwr > 1.3:
        acwr_status = "high_risk"
    elif acwr > 0.8:
        acwr_status = "optimal"
    else:
        acwr_status = "detraining"

    training_days = len(daily_load)
    total_days = max(days, 1)
    rest_days = total_days - training_days
    sessions_per_week = round(len(sessions) / max(1, days / 7), 1)

    if sessions_per_week >= 5:
        density = "high"
    elif sessions_per_week >= 3:
        density = "moderate"
    elif sessions_per_week >= 1:
        density = "low"
    else:
        density = "very_low"

    return {
        "acute_load": round(acute, 0),
        "chronic_load": round(chronic, 0),
        "acwr": acwr,
        "acwr_status": acwr_status,
        "training_days": training_days,
        "rest_days": rest_days,
        "total_sessions": len(sessions),
        "avg_sessions_per_week": sessions_per_week,
        "density": density,
    }


# ─── 4. Periodization Recommendation ──────────────────────────────────────


def _periodization(trajectory: dict, load: dict, risk: dict) -> dict:
    """
    Recommend the current mesocycle phase and the next 4-week plan.

    Standard periodization:
      accumulation (build volume) → transmutation (build intensity) →
      realization (peak/taper) → recovery (deload)
    """
    acwr = load.get("acwr")
    direction = trajectory.get("trend_direction", "stable")
    injury = risk.get("risk", "low")
    sessions_pw = load.get("avg_sessions_per_week", 0)

    # determine current phase
    if injury == "high" or (acwr and acwr > 1.5):
        phase = "recovery"
        rationale = "injury risk or training spike detected. deload before progressing"
    elif direction == "declining" or (acwr and acwr > 1.3):
        phase = "recovery"
        rationale = "form declining or acute load too high. back off for 1-2 weeks"
    elif sessions_pw < 2:
        phase = "accumulation"
        rationale = "low training frequency. build base volume before adding intensity"
    elif direction == "improving" and sessions_pw >= 3:
        phase = "transmutation"
        rationale = "form improving with good volume. safe to push intensity higher"
    elif direction == "stable" and sessions_pw >= 4:
        phase = "realization"
        rationale = "form stable at high volume. time to peak — reduce volume, maintain intensity"
    else:
        phase = "accumulation"
        rationale = "building base. focus on consistency and technique"

    # 4-week plan
    week_plans = {
        "accumulation": [
            {
                "week": 1,
                "focus": "technique",
                "volume": "moderate",
                "intensity": "low",
                "note": "establish movement patterns, film every session",
            },
            {
                "week": 2,
                "focus": "volume",
                "volume": "moderate-high",
                "intensity": "low-moderate",
                "note": "add 1 session. keep form score priority",
            },
            {
                "week": 3,
                "focus": "volume",
                "volume": "high",
                "intensity": "moderate",
                "note": "peak volume week. 4-5 sessions if possible",
            },
            {
                "week": 4,
                "focus": "deload",
                "volume": "low",
                "intensity": "low",
                "note": "half the volume. mobility and skill work only",
            },
        ],
        "transmutation": [
            {
                "week": 1,
                "focus": "intensity",
                "volume": "moderate",
                "intensity": "moderate-high",
                "note": "push form quality. target 5% higher avg score",
            },
            {
                "week": 2,
                "focus": "intensity",
                "volume": "moderate",
                "intensity": "high",
                "note": "max effort sessions. fewer reps, higher quality",
            },
            {
                "week": 3,
                "focus": "competition prep",
                "volume": "moderate",
                "intensity": "high",
                "note": "simulate competition conditions",
            },
            {
                "week": 4,
                "focus": "deload",
                "volume": "low",
                "intensity": "moderate",
                "note": "recover before next block",
            },
        ],
        "realization": [
            {
                "week": 1,
                "focus": "sharpen",
                "volume": "moderate",
                "intensity": "very high",
                "note": "peak performance attempts. video review every set",
            },
            {
                "week": 2,
                "focus": "taper",
                "volume": "low",
                "intensity": "high",
                "note": "reduce volume 40%. keep intensity. rest more",
            },
            {
                "week": 3,
                "focus": "compete/test",
                "volume": "minimal",
                "intensity": "max",
                "note": "competition week or max testing. trust the process",
            },
            {
                "week": 4,
                "focus": "transition",
                "volume": "low",
                "intensity": "low",
                "note": "active recovery. plan the next training block",
            },
        ],
        "recovery": [
            {
                "week": 1,
                "focus": "rest",
                "volume": "very low",
                "intensity": "very low",
                "note": "mobility only. fix the weak points. sleep 8+ hours",
            },
            {
                "week": 2,
                "focus": "mobility",
                "volume": "low",
                "intensity": "low",
                "note": "light movement. address asymmetry. no pushing through pain",
            },
            {
                "week": 3,
                "focus": "rebuild",
                "volume": "low-moderate",
                "intensity": "low",
                "note": "start adding sessions back. technique focus",
            },
            {
                "week": 4,
                "focus": "assess",
                "volume": "moderate",
                "intensity": "low-moderate",
                "note": "test where you are. set goals for next block",
            },
        ],
    }

    return {
        "current_phase": phase,
        "rationale": rationale,
        "four_week_plan": week_plans.get(phase, week_plans["accumulation"]),
        "next_phase": {
            "accumulation": "transmutation",
            "transmutation": "realization",
            "realization": "recovery",
            "recovery": "accumulation",
        }.get(phase, "accumulation"),
    }


# ─── 5. Peer Benchmarks ───────────────────────────────────────────────────


def _peer_benchmarks(athlete_id: str, sport: str, tier: str, avg_form: float, bpi: int) -> dict:
    """
    Where does this athlete rank among peers (same sport or same tier)?
    """
    sport_peers = []
    tier_peers = []

    for aid, a in ATHLETE_DB.items():
        if aid == athlete_id:
            continue
        peer_bpi = a.get("bpi", 0)
        if a.get("sport") == sport:
            sport_peers.append(peer_bpi)
        if a.get("tier") == tier:
            tier_peers.append(peer_bpi)

    def _percentile(peers: list[int], value: int) -> float:
        if not peers:
            return 50.0
        below = sum(1 for p in peers if p < value)
        return round(below / len(peers) * 100, 1)

    sport_pct = _percentile(sport_peers, bpi)
    tier_pct = _percentile(tier_peers, bpi)

    return {
        "sport_percentile": sport_pct,
        "sport_rank_label": f"top {100 - sport_pct:.0f}%" if sport_pct > 50 else f"bottom {sport_pct:.0f}%",
        "sport_peers_count": len(sport_peers),
        "tier_percentile": tier_pct,
        "tier_rank_label": f"top {100 - tier_pct:.0f}%" if tier_pct > 50 else f"bottom {tier_pct:.0f}%",
        "tier_peers_count": len(tier_peers),
        "tier_promotion_gap": _tier_gap(tier, bpi),
    }


def _tier_gap(current_tier: str, bpi: int) -> dict | None:
    """How far to the next tier?"""
    tier_thresholds = {
        "Block": ("District", 6001),
        "District": ("State", 9501),
        "State": ("National", 13001),
        "National": ("Elite", 18001),
    }
    if current_tier not in tier_thresholds:
        return None
    next_tier, threshold = tier_thresholds[current_tier]
    gap = max(0, threshold - bpi)
    return {
        "next_tier": next_tier,
        "bpi_needed": gap,
        "current_bpi": bpi,
        "threshold": threshold,
    }


# ─── 6. Competition Readiness Timeline ────────────────────────────────────


def _readiness_timeline(trajectory: dict, risk: dict, load: dict) -> dict:
    """
    Estimate when the athlete will reach peak competition readiness.

    Based on: current form trajectory, injury risk, training consistency.
    """
    direction = trajectory.get("trend_direction", "stable")
    injury = risk.get("risk", "low")
    sessions_pw = load.get("avg_sessions_per_week", 0)

    if injury == "high":
        weeks_to_ready = 6
        reason = "need to resolve injury risk first (2-3 weeks), then rebuild (3-4 weeks)"
    elif direction == "declining":
        weeks_to_ready = 4
        reason = "form declining. need a deload week then 3 weeks to rebuild momentum"
    elif direction == "improving" and sessions_pw >= 3:
        weeks_to_ready = 2
        reason = "already trending up with good volume. 2 weeks to peak with a taper"
    elif direction == "stable" and sessions_pw >= 3:
        weeks_to_ready = 3
        reason = "form is stable. 1 week intensity push + 1 week taper + 1 week sharpen"
    else:
        weeks_to_ready = 5
        reason = "need to build consistency first. 3 weeks volume + 1 week intensity + 1 week taper"

    now = datetime.now(timezone.utc)
    target_date = (now + timedelta(weeks=weeks_to_ready)).date().isoformat()

    return {
        "estimated_weeks": weeks_to_ready,
        "target_date": target_date,
        "confidence": "high" if direction == "improving" else "medium" if direction == "stable" else "low",
        "reason": reason,
        "prerequisites": _readiness_prerequisites(injury, sessions_pw, direction),
    }


def _readiness_prerequisites(injury: str, sessions_pw: float, direction: str) -> list[str]:
    prereqs = []
    if injury == "high":
        prereqs.append("resolve injury risk — see /injury-risk for details")
    if sessions_pw < 2:
        prereqs.append("increase to at least 3 sessions per week")
    if direction == "declining":
        prereqs.append("take a deload week before pushing again")
    if sessions_pw > 5:
        prereqs.append("reduce volume slightly — you may be overreaching")
    if not prereqs:
        prereqs.append("stay consistent and follow the periodization plan")
    return prereqs


# ─── 7. Key Risks and Action Items ────────────────────────────────────────


def _action_items(
    risk: dict,
    weak_joints: list[dict],
    load: dict,
    trajectory: dict,
    sessions: list[dict],
) -> list[dict]:
    """
    The 3-5 most important things to do this week, prioritized.
    This is what athletes actually read.
    """
    items = []

    # injury risk action
    if risk.get("risk") == "high":
        items.append(
            {
                "priority": 1,
                "category": "injury",
                "action": "reduce training load immediately",
                "detail": risk.get("reason", "elevated asymmetry detected"),
                "urgency": "this week",
            }
        )
    elif risk.get("risk") == "watch":
        items.append(
            {
                "priority": 2,
                "category": "injury",
                "action": "add 10 min mobility work targeting weaker side",
                "detail": risk.get("reason", "mild asymmetry"),
                "urgency": "daily",
            }
        )

    # weakest joint action
    if weak_joints:
        wj = weak_joints[0]
        items.append(
            {
                "priority": 2 if risk.get("risk") != "high" else 3,
                "category": "technique",
                "action": f"focus on {wj['joint'].replace('_', ' ')} — {wj['deviation_deg']:.0f} degrees off ideal",
                "detail": f"your {wj['joint'].replace('_', ' ')} averages {wj['mean_deg']:.0f} degrees (ideal: {wj['ideal_min']}-{wj['ideal_max']})",
                "urgency": "every session",
            }
        )

    # training load action
    acwr_status = load.get("acwr_status", "no_data")
    if acwr_status == "dangerous_spike":
        items.append(
            {
                "priority": 1,
                "category": "load",
                "action": "training spike detected — drop volume 40% this week",
                "detail": f"acute:chronic ratio is {load.get('acwr', 0)} (should be 0.8-1.3)",
                "urgency": "immediate",
            }
        )
    elif acwr_status == "detraining":
        items.append(
            {
                "priority": 2,
                "category": "load",
                "action": "training volume too low — add 1-2 sessions this week",
                "detail": f"averaging {load.get('avg_sessions_per_week', 0)} sessions/week",
                "urgency": "this week",
            }
        )

    # form trend action
    if trajectory.get("trend_direction") == "declining":
        items.append(
            {
                "priority": 2,
                "category": "performance",
                "action": "form score declining — slow down and focus on quality reps",
                "detail": f"velocity: {trajectory.get('velocity_per_segment', 0):+.1f} points per period",
                "urgency": "next 3 sessions",
            }
        )
    elif trajectory.get("trend_direction") == "improving":
        items.append(
            {
                "priority": 4,
                "category": "performance",
                "action": "form improving — maintain current approach and keep logging sessions",
                "detail": "consistency is working. don't change what isn't broken",
                "urgency": "ongoing",
            }
        )

    # volume consistency
    if not sessions:
        items.append(
            {
                "priority": 1,
                "category": "consistency",
                "action": "no sessions recorded — start training to generate your report",
                "detail": "the app needs at least 3-5 sessions to give meaningful feedback",
                "urgency": "today",
            }
        )
    elif len(sessions) < 5:
        items.append(
            {
                "priority": 3,
                "category": "consistency",
                "action": f"only {len(sessions)} sessions in this window — log more for better insights",
                "detail": "recommendations get much more accurate with 10+ sessions",
                "urgency": "this week",
            }
        )

    items.sort(key=lambda x: x["priority"])
    return items[:5]


# ─── Orchestrator ──────────────────────────────────────────────────────────


def _generate_report(athlete_id: str, days: int) -> dict:
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    sport = athlete.get("sport", "vertical_jump")
    tier = athlete.get("tier", "Block")
    bpi = athlete.get("bpi", 0)

    sessions = _athlete_sessions(athlete_id, days)
    risk = _compute_injury_risk(athlete_id, days)
    weak = _compute_weak_joints(athlete_id, days)

    # compute all sections
    trajectory = _performance_trajectory(sessions, days)
    profile = _biomechanical_profile(sessions, sport)
    load = _training_load_analysis(sessions, days)
    period = _periodization(trajectory, load, risk)
    peers = _peer_benchmarks(
        athlete_id,
        sport,
        tier,
        trajectory.get("segments", [{}])[-1].get("avg_form", 0) if trajectory.get("segments") else 0,
        bpi,
    )
    timeline = _readiness_timeline(trajectory, risk, load)
    actions = _action_items(risk, weak, load, trajectory, sessions)

    # overall grade
    form_score = trajectory["segments"][-1]["avg_form"] if trajectory.get("segments") else 0
    risk_penalty = {"low": 0, "watch": -10, "high": -25, "unknown": -5}.get(risk.get("risk", "unknown"), 0)
    consistency_bonus = min(10, load.get("avg_sessions_per_week", 0) * 2)
    overall = max(0, min(100, round(form_score + risk_penalty + consistency_bonus)))

    grade_map = [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"), (50, "C+"), (40, "C"), (0, "D")]
    letter = next(letter for threshold, letter in grade_map if overall >= threshold)

    return {
        "athlete_id": athlete_id,
        "athlete_name": athlete.get("name"),
        "sport": sport,
        "tier": tier,
        "window_days": days,
        "overall_grade": {"score": overall, "letter": letter},
        "trajectory": trajectory,
        "biomechanical_profile": profile,
        "training_load": load,
        "periodization": period,
        "peer_benchmarks": peers,
        "competition_timeline": timeline,
        "action_items": actions,
        "injury_risk": {
            "band": risk.get("risk"),
            "reason": risk.get("reason"),
            "consecutive_bad_sessions": risk.get("consecutive_bad_sessions", 0),
            "worsening": risk.get("worsening_week_over_week", False),
        },
        "session_count": len(sessions),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "next_report_at": (datetime.now(timezone.utc) + timedelta(days=7)).date().isoformat(),
    }


# ─── Route ─────────────────────────────────────────────────────────────────


@router.get("/athlete/{athlete_id}/intelligence-report")
async def intelligence_report(
    athlete_id: str,
    days: int = Query(default=30, ge=7, le=180),
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """
    The comprehensive athlete intelligence report.

    This is the product's core output — the thing that makes this platform
    different from every other fitness app. It takes every signal we capture
    (joint angles, form scores, symmetry, volume, HR) and turns it into
    a structured, actionable document that a coach or athlete can read in
    2 minutes and know exactly what to do next.
    """
    return _generate_report(athlete_id, days)
