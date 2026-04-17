from __future__ import annotations

"""
Personal Health — Dynamic Training Plan generator.

Takes an athlete's last 14 days of progress + weak-joint deviations + injury
risk and produces a personalized 7-day training plan. Each day has one of
six session types and a concrete drill block tuned to the athlete's sport
and current weak points.

Logic lives in three layers:

1. `build_plan_context`  — reads progress, weak joints, injury risk, volume.
                           Pure dict in, pure dict out; no I/O.
2. `generate_plan`        — deterministic rule-based plan from the context.
                           Picks day types, intensities, drills, rationale.
                           Never calls external APIs; always works offline.
3. `narrate_plan`         — optional LLM pass that rewrites the per-day
                           `rationale` field in plain-English coach voice.
                           Falls through to deterministic text on any failure.

The plan shape is stable (see `PlanWeek` / `PlanDay` dataclasses) so the
frontend and Android client can render it without caring which layer
produced the text.
"""

import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from logging_setup import get_logger

log = get_logger("services.training_plan")


# ─── Day-type catalog ────────────────────────────────────────────────────────

# Each day type has an intensity tier and a short human label. The generator
# picks day types based on the athlete's week shape (volume, risk, recovery).
DAY_TYPES: dict[str, dict[str, Any]] = {
    "technique": {
        "intensity": "low",
        "label": "Technique",
        "rpe": 6,
        "duration_min": 45,
    },
    "strength": {
        "intensity": "high",
        "label": "Strength",
        "rpe": 8,
        "duration_min": 55,
    },
    "power": {
        "intensity": "high",
        "label": "Power",
        "rpe": 9,
        "duration_min": 50,
    },
    "speed": {
        "intensity": "high",
        "label": "Speed",
        "rpe": 9,
        "duration_min": 40,
    },
    "recovery": {
        "intensity": "low",
        "label": "Active Recovery",
        "rpe": 3,
        "duration_min": 30,
    },
    "rest": {
        "intensity": "rest",
        "label": "Rest",
        "rpe": 0,
        "duration_min": 0,
    },
}


# ─── Sport-specific drill library ────────────────────────────────────────────

# Keyed by (sport, day_type). Each entry is a list of 3–5 drills with a
# target weak joint the drill addresses. The generator picks 3 drills per
# day, biased toward those that hit the athlete's top weak joint.
DRILL_LIBRARY: dict[str, dict[str, list[dict[str, Any]]]] = {
    "vertical_jump": {
        "technique": [
            {"name": "Landing mechanics drill", "sets": "4×5", "joint": "knee_angle", "cue": "Knees track over toes"},
            {
                "name": "Vertical jump ladder (bodyweight)",
                "sets": "5×3",
                "joint": "hip_angle",
                "cue": "Full hip extension",
            },
            {
                "name": "Ankle pogo hops",
                "sets": "3×20",
                "joint": "ankle_dorsiflexion",
                "cue": "Stiff ankle, short contacts",
            },
            {"name": "Wall sit to jump", "sets": "4×6", "joint": "knee_angle", "cue": "Drive through heels"},
        ],
        "strength": [
            {"name": "Back squat", "sets": "4×5 @ 80%", "joint": "hip_angle", "cue": "Break at hips first"},
            {"name": "Romanian deadlift", "sets": "3×8", "joint": "hip_angle", "cue": "Push hips back, flat back"},
            {
                "name": "Bulgarian split squat",
                "sets": "3×8/side",
                "joint": "knee_angle",
                "cue": "Front knee stacks over ankle",
            },
            {"name": "Calf raise, slow eccentric", "sets": "3×12", "joint": "ankle_dorsiflexion", "cue": "3-sec lower"},
        ],
        "power": [
            {"name": "Box jumps", "sets": "5×3", "joint": "hip_angle", "cue": "Soft landing, reset every rep"},
            {"name": "Depth drops", "sets": "4×4", "joint": "knee_angle", "cue": "Absorb, don't collapse"},
            {"name": "Trap bar jump", "sets": "5×3 @ 30%", "joint": "hip_angle", "cue": "Maximum intent"},
            {"name": "Seated box jump", "sets": "4×3", "joint": "hip_angle", "cue": "No counter-movement"},
        ],
        "speed": [
            {"name": "Broad jump repeat", "sets": "5×3", "joint": "hip_angle", "cue": "Arm swing drives range"},
            {"name": "Single-leg bound", "sets": "4×4/side", "joint": "knee_angle", "cue": "Push horizontal"},
        ],
        "recovery": [
            {
                "name": "Banded ankle mobility",
                "sets": "2×10/side",
                "joint": "ankle_dorsiflexion",
                "cue": "Drive knee over toe",
            },
            {"name": "90/90 hip switches", "sets": "2×10", "joint": "hip_angle", "cue": "Control the rotation"},
            {
                "name": "Foam roll quads + calves",
                "sets": "5 min",
                "joint": "knee_angle",
                "cue": "Slow, find tender spots",
            },
        ],
    },
    "sprint": {
        "technique": [
            {"name": "A-march", "sets": "3×20m", "joint": "hip_angle", "cue": "Knee to hip height"},
            {"name": "A-skip", "sets": "3×20m", "joint": "hip_angle", "cue": "Pop off the ground"},
            {"name": "Wall drive hold", "sets": "4×10s/side", "joint": "trunk_lean", "cue": "45° body angle"},
        ],
        "strength": [
            {"name": "Hip thrust", "sets": "4×6", "joint": "hip_angle", "cue": "Ribs down, squeeze glutes"},
            {"name": "Front squat", "sets": "4×5", "joint": "knee_angle", "cue": "Elbows high, torso tall"},
            {"name": "Split squat", "sets": "3×6/side", "joint": "hip_angle", "cue": "Vertical back shin"},
        ],
        "power": [
            {
                "name": "Medicine ball chest throw",
                "sets": "4×5",
                "joint": "trunk_lean",
                "cue": "Explode through the ball",
            },
            {"name": "Bounding", "sets": "5×20m", "joint": "knee_angle", "cue": "Cover ground, minimize contact"},
        ],
        "speed": [
            {"name": "Flying 20m", "sets": "5×20m", "joint": "knee_angle", "cue": "Max velocity, relaxed face"},
            {"name": "Resisted sled push", "sets": "4×15m", "joint": "trunk_lean", "cue": "Low angle, long steps"},
            {
                "name": "Acceleration starts",
                "sets": "6×10m",
                "joint": "trunk_lean",
                "cue": "Punch, don't rise too fast",
            },
        ],
        "recovery": [
            {"name": "Hamstring walks", "sets": "2×15m", "joint": "hip_angle", "cue": "Controlled reach"},
            {"name": "Foam roll glutes + calves", "sets": "5 min", "joint": "hip_angle", "cue": "Find tender spots"},
        ],
    },
    "snatch": {
        "technique": [
            {
                "name": "Overhead squat",
                "sets": "5×3 @ 40%",
                "joint": "shoulder_angle",
                "cue": "Active shoulders, bar over heels",
            },
            {"name": "Snatch pull from knee", "sets": "4×3", "joint": "hip_angle", "cue": "Patient off the floor"},
            {"name": "Snatch balance", "sets": "4×3", "joint": "shoulder_angle", "cue": "Drop fast, lock tight"},
        ],
        "strength": [
            {"name": "Back squat", "sets": "4×5 @ 80%", "joint": "knee_angle", "cue": "Break at hips first"},
            {"name": "Snatch deadlift", "sets": "4×3", "joint": "hip_angle", "cue": "Vertical bar path"},
            {"name": "Press in snatch", "sets": "3×5", "joint": "shoulder_angle", "cue": "Push the ceiling"},
        ],
        "power": [
            {"name": "Power snatch", "sets": "5×2 @ 70%", "joint": "hip_angle", "cue": "Triple extension"},
            {"name": "High pull", "sets": "4×3", "joint": "elbow_angle", "cue": "Elbows high and to the sides"},
        ],
        "speed": [],
        "recovery": [
            {"name": "Wrist mobility flow", "sets": "5 min", "joint": "elbow_angle", "cue": "Slow, find range"},
            {
                "name": "Thoracic opener on bench",
                "sets": "3×10 breaths",
                "joint": "shoulder_angle",
                "cue": "Breathe into the stretch",
            },
        ],
    },
    "javelin": {
        "technique": [
            {"name": "Standing release drill", "sets": "4×5", "joint": "elbow_angle", "cue": "Elbow above shoulder"},
            {"name": "Med-ball rotational throw", "sets": "3×6/side", "joint": "trunk_lean", "cue": "Lead with hip"},
            {
                "name": "Step-step-throw sequence",
                "sets": "4×4",
                "joint": "shoulder_angle",
                "cue": "Plant wide, block the hip",
            },
        ],
        "strength": [
            {"name": "Overhead press", "sets": "4×5", "joint": "shoulder_angle", "cue": "Ribs stacked"},
            {
                "name": "Single-arm landmine press",
                "sets": "3×8/side",
                "joint": "shoulder_angle",
                "cue": "Punch up and out",
            },
            {"name": "Pallof press", "sets": "3×10/side", "joint": "trunk_lean", "cue": "Resist rotation"},
        ],
        "power": [
            {
                "name": "Med-ball overhead slam",
                "sets": "5×5",
                "joint": "shoulder_angle",
                "cue": "Reach tall, drive down",
            },
            {"name": "Cable woodchop", "sets": "4×6/side", "joint": "trunk_lean", "cue": "Hips lead, arms follow"},
        ],
        "speed": [],
        "recovery": [
            {
                "name": "Rotator cuff band series",
                "sets": "3×10 each",
                "joint": "shoulder_angle",
                "cue": "Light band, full range",
            },
            {"name": "T-spine rotations", "sets": "2×10/side", "joint": "trunk_lean", "cue": "Exhale into rotation"},
        ],
    },
    "cricket_bat": {
        "technique": [
            {"name": "Shadow cover drive", "sets": "4×8", "joint": "elbow_angle", "cue": "Head over front knee"},
            {"name": "Tee batting, front foot", "sets": "4×10", "joint": "elbow_angle", "cue": "Full face of the bat"},
            {"name": "Throw-down off-spin", "sets": "3×10", "joint": "trunk_lean", "cue": "Read from the hand"},
        ],
        "strength": [
            {"name": "Goblet squat", "sets": "4×8", "joint": "knee_angle", "cue": "Chest up, deep enough"},
            {"name": "Single-arm dumbbell row", "sets": "3×8/side", "joint": "shoulder_angle", "cue": "Squeeze at top"},
            {"name": "Pallof hold", "sets": "3×20s/side", "joint": "trunk_lean", "cue": "Resist the pull"},
        ],
        "power": [
            {
                "name": "Med-ball rotational throw",
                "sets": "4×5/side",
                "joint": "trunk_lean",
                "cue": "Hips drive the shot",
            },
        ],
        "speed": [],
        "recovery": [
            {"name": "Wrist circles + grip flow", "sets": "2×10", "joint": "elbow_angle", "cue": "Loose and easy"},
            {"name": "T-spine rotations", "sets": "2×10/side", "joint": "trunk_lean", "cue": "Breathe into rotation"},
        ],
    },
}


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class PlanDay:
    """One day of the training plan."""

    date: str  # YYYY-MM-DD
    day_name: str  # Mon, Tue, ...
    type: str  # key into DAY_TYPES
    label: str
    intensity: str
    rpe: int
    duration_min: int
    drills: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    completed: bool = False
    completed_at: str | None = None


@dataclass
class PlanWeek:
    """The full 7-day plan with metadata."""

    athlete_id: str
    sport: str
    week_start: str  # Monday, YYYY-MM-DD
    week_end: str  # Sunday
    generated_at: str
    source: str  # 'deterministic' | 'anthropic' | 'anthropic-fallback'
    summary: str
    adherence_pct: float
    days: list[PlanDay]
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ─── Context builder ─────────────────────────────────────────────────────────


def build_plan_context(athlete_id: str, days: int = 14) -> dict[str, Any]:
    """
    Gather the inputs the planner needs. Pulls from the existing progress /
    weak-joint / injury-risk computers so the plan stays consistent with
    what the athlete sees on /progress.
    """
    # Local import to avoid circulars — routes/progress imports cache too.
    from database import ATHLETE_DB
    from routes.progress import _compute_injury_risk, _compute_progress, _compute_weak_joints

    if athlete_id not in ATHLETE_DB:
        raise KeyError(f"athlete not found: {athlete_id}")

    athlete = ATHLETE_DB[athlete_id]
    progress = _compute_progress(athlete_id, days)
    risk = _compute_injury_risk(athlete_id, days)
    weak = _compute_weak_joints(athlete_id, days)

    # Under- / over-reaching heuristic: sessions this week vs prior week.
    recent_sessions = progress.get("session_count", 0)
    if recent_sessions == 0:
        volume_state = "inactive"
    elif recent_sessions < 2:
        volume_state = "under"
    elif recent_sessions > 6:
        volume_state = "over"
    else:
        volume_state = "balanced"

    # Form quality bucket
    avg_form = float(progress.get("avg_form_score", 0) or 0)
    if avg_form >= 80:
        form_bucket = "strong"
    elif avg_form >= 65:
        form_bucket = "steady"
    elif avg_form > 0:
        form_bucket = "rough"
    else:
        form_bucket = "unknown"

    return {
        "athlete_id": athlete_id,
        "athlete_name": athlete.get("name", "Athlete"),
        "sport": athlete.get("sport", "vertical_jump"),
        "tier": athlete.get("tier", "Block"),
        "window_days": days,
        "session_count": recent_sessions,
        "avg_form_score": avg_form,
        "form_trend_pct": float(progress.get("form_trend_pct", 0) or 0),
        "form_bucket": form_bucket,
        "bpi_delta": int(progress.get("bpi_delta", 0) or 0),
        "injury_risk": risk.get("risk", "unknown"),
        "injury_reason": risk.get("reason", ""),
        "weak_joints": weak[:3],
        "volume_state": volume_state,
    }


# ─── Deterministic plan generator ────────────────────────────────────────────


# Week templates by volume state + injury risk. Each template is an ordered
# list of 7 day types for Mon–Sun.
_WEEK_TEMPLATES: dict[tuple[str, str], list[str]] = {
    # (volume_state, injury_risk) -> Mon..Sun day types
    ("balanced", "low"): ["strength", "technique", "power", "recovery", "strength", "speed", "rest"],
    ("balanced", "watch"): ["strength", "technique", "recovery", "power", "technique", "recovery", "rest"],
    ("balanced", "high"): ["recovery", "technique", "rest", "recovery", "technique", "recovery", "rest"],
    ("under", "low"): ["technique", "strength", "technique", "power", "recovery", "strength", "rest"],
    ("under", "watch"): ["technique", "strength", "recovery", "technique", "power", "recovery", "rest"],
    ("under", "high"): ["technique", "recovery", "rest", "technique", "recovery", "rest", "rest"],
    ("over", "low"): ["recovery", "technique", "strength", "recovery", "power", "rest", "rest"],
    ("over", "watch"): ["recovery", "technique", "recovery", "strength", "rest", "recovery", "rest"],
    ("over", "high"): ["rest", "recovery", "rest", "recovery", "technique", "rest", "rest"],
    ("inactive", "low"): ["technique", "strength", "technique", "recovery", "strength", "technique", "rest"],
    ("inactive", "watch"): ["technique", "recovery", "technique", "strength", "recovery", "technique", "rest"],
    ("inactive", "high"): ["recovery", "technique", "rest", "recovery", "technique", "recovery", "rest"],
    ("inactive", "unknown"): ["technique", "strength", "technique", "recovery", "strength", "technique", "rest"],
    ("balanced", "unknown"): ["strength", "technique", "power", "recovery", "strength", "speed", "rest"],
    ("under", "unknown"): ["technique", "strength", "technique", "power", "recovery", "strength", "rest"],
    ("over", "unknown"): ["recovery", "technique", "strength", "recovery", "power", "rest", "rest"],
}


def _monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _pick_drills(sport: str, day_type: str, weak_joint: str | None, count: int = 3) -> list[dict[str, Any]]:
    """Pick drills for a day, biased toward the athlete's top weak joint."""
    library = DRILL_LIBRARY.get(sport, DRILL_LIBRARY["vertical_jump"])
    pool = list(library.get(day_type, []))
    if not pool:
        # Fall back to technique drills if the sport has no entries for the
        # requested day type (e.g. no "speed" drills for snatch).
        pool = list(library.get("technique", []))
    if not pool:
        return []

    # Bias: drills targeting the weak joint go first.
    if weak_joint:
        targeted = [d for d in pool if d.get("joint") == weak_joint]
        others = [d for d in pool if d.get("joint") != weak_joint]
        pool = targeted + others

    return pool[:count]


def _deterministic_rationale(ctx: dict[str, Any], day_type: str, weak_joint: str | None) -> str:
    """Plain-English reason for the day choice. Used as fallback + LLM seed."""
    risk = ctx.get("injury_risk", "unknown")
    form_bucket = ctx.get("form_bucket", "unknown")
    weak_txt = weak_joint.replace("_", " ") if weak_joint else None

    if day_type == "rest":
        return "Full rest. Recovery is when adaptation actually happens — take it."
    if day_type == "recovery":
        if risk == "high":
            return "Active recovery to let the asymmetry settle before any load."
        return "Light movement to flush yesterday and keep you loose."
    if day_type == "technique":
        if weak_txt:
            return f"Slow work on {weak_txt} — no point adding load until the angle cleans up."
        return "Skill block — the goal is clean reps, not hard reps."
    if day_type == "strength":
        if form_bucket == "rough":
            return "Baseline strength. Keep weight honest and don't chase numbers."
        return "Load day — push intensity now that your form is holding."
    if day_type == "power":
        if risk == "high":
            return "Power block, but drop to 80% volume. Watch for the weak side."
        return "Rate of force development. Full rest between sets, maximum intent."
    if day_type == "speed":
        return "Velocity work. Keep reps short, stay fresh across the set."
    return "Session."


def generate_plan(ctx: dict[str, Any], week_start: date | None = None) -> PlanWeek:
    """
    Build a 7-day plan from the context dict. Pure function — no I/O, no
    network. Deterministic for a given (ctx, week_start).
    """
    week_start = week_start or _monday_of_week(datetime.now(timezone.utc).date())
    template_key = (ctx.get("volume_state", "balanced"), ctx.get("injury_risk", "unknown"))
    template = _WEEK_TEMPLATES.get(template_key, _WEEK_TEMPLATES[("balanced", "low")])

    weak_joint = None
    if ctx.get("weak_joints"):
        weak_joint = ctx["weak_joints"][0].get("joint")

    sport = ctx.get("sport", "vertical_jump")
    days: list[PlanDay] = []
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for i, day_type in enumerate(template):
        d = week_start + timedelta(days=i)
        meta = DAY_TYPES[day_type]
        drills = _pick_drills(sport, day_type, weak_joint) if day_type not in ("rest",) else []
        rationale = _deterministic_rationale(ctx, day_type, weak_joint)
        days.append(
            PlanDay(
                date=d.isoformat(),
                day_name=day_names[i],
                type=day_type,
                label=meta["label"],
                intensity=meta["intensity"],
                rpe=meta["rpe"],
                duration_min=meta["duration_min"],
                drills=drills,
                rationale=rationale,
            )
        )

    # Composite summary sentence.
    risk = ctx.get("injury_risk", "unknown")
    vol = ctx.get("volume_state", "balanced")
    if risk == "high":
        summary = "Asymmetry is elevated — this week is skewed toward recovery and technique."
    elif vol == "over":
        summary = "You're over your usual volume — cut intensity and bias recovery."
    elif vol == "under":
        summary = "Light week behind you — build volume gradually, lead with technique."
    elif vol == "inactive":
        summary = "No recent sessions — start the week easy to get your groove back."
    elif ctx.get("form_bucket") == "strong":
        summary = "Form is strong — push power and speed with full-volume recovery."
    else:
        summary = "Balanced week: strength, technique, power, with recovery built in."

    return PlanWeek(
        athlete_id=ctx["athlete_id"],
        sport=sport,
        week_start=week_start.isoformat(),
        week_end=(week_start + timedelta(days=6)).isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        source="deterministic",
        summary=summary,
        adherence_pct=0.0,
        days=days,
        context={
            "volume_state": vol,
            "injury_risk": risk,
            "form_bucket": ctx.get("form_bucket"),
            "avg_form_score": ctx.get("avg_form_score"),
            "session_count": ctx.get("session_count"),
            "weak_joint": weak_joint,
        },
    )


# ─── Optional LLM narration ──────────────────────────────────────────────────

# Reuses the ai_coach circuit breaker pattern so a planner failure doesn't
# take down the whole backend.

try:
    from anthropic import Anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


_NARRATE_SYSTEM_PROMPT = (
    "You are a supportive sports biomechanics coach writing short daily "
    "notes for an athlete's 7-day plan. For each day you receive the day "
    "type, drills, and a seed rationale. Rewrite ONLY the rationale in at "
    "most 20 words, plain English, no emojis, no markdown, no day-of-week "
    "prefix. Keep the meaning of the seed rationale intact. Return the "
    "rewritten rationales as a JSON array of exactly 7 strings in the same "
    "order you received them, and nothing else."
)


def _build_narrate_prompt(plan: PlanWeek, ctx: dict[str, Any]) -> str:
    lines = [
        f"Athlete: {ctx.get('athlete_name', 'Athlete')}",
        f"Sport: {plan.sport}",
        f"Volume state: {ctx.get('volume_state')}",
        f"Injury risk: {ctx.get('injury_risk')}",
        f"Avg form score: {ctx.get('avg_form_score'):.1f}",
        "",
        "Days (index, type, seed rationale):",
    ]
    for i, day in enumerate(plan.days):
        lines.append(f"  {i}: {day.type} — {day.rationale}")
    lines.append("")
    lines.append("Return a JSON array of 7 rewritten rationales now.")
    return "\n".join(lines)


def narrate_plan(plan: PlanWeek, ctx: dict[str, Any]) -> PlanWeek:
    """
    Optional LLM rewrite of per-day rationales. If anything fails, returns
    the plan unchanged with source='deterministic'. Never raises.
    """
    from config import settings

    if not (_ANTHROPIC_AVAILABLE and settings.anthropic_api_key):
        return plan

    try:
        client = Anthropic(api_key=settings.anthropic_api_key, timeout=8.0)
        start = time.time()
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=600,
            system=_NARRATE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_narrate_prompt(plan, ctx)}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text").strip()

        # Robust-ish JSON extraction — the model sometimes wraps in prose.
        import json as _json
        import re as _re

        match = _re.search(r"\[.*\]", text, _re.DOTALL)
        if not match:
            raise ValueError("no JSON array in model output")
        rewritten = _json.loads(match.group(0))
        if not isinstance(rewritten, list) or len(rewritten) != 7:
            raise ValueError(f"expected 7 rationales, got {rewritten!r}")

        for day, new_rationale in zip(plan.days, rewritten):
            if isinstance(new_rationale, str) and new_rationale.strip():
                day.rationale = new_rationale.strip()
        plan.source = "anthropic"
        log.info(
            "training_plan narration ok",
            extra={"athlete_id": plan.athlete_id, "latency_ms": int((time.time() - start) * 1000)},
        )
        return plan
    except Exception as e:
        log.warning("training_plan narration failed, using fallback", extra={"error": str(e)})
        plan.source = "anthropic-fallback"
        return plan


# ─── Adherence helpers ───────────────────────────────────────────────────────


def compute_adherence(plan: PlanWeek) -> float:
    """
    % of non-rest days that are marked completed. Rest days don't count
    against adherence — skipping a rest day is fine.
    """
    workable = [d for d in plan.days if d.type != "rest"]
    if not workable:
        return 0.0
    done = sum(1 for d in workable if d.completed)
    return round(done / len(workable) * 100, 1)


def mark_day_complete(plan: PlanWeek, day_date: str) -> PlanDay | None:
    """Mark the day with the given ISO date as completed. Returns the day or None."""
    for day in plan.days:
        if day.date == day_date:
            day.completed = True
            day.completed_at = datetime.now(timezone.utc).isoformat()
            return day
    return None
