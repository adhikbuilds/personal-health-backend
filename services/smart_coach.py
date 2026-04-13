from __future__ import annotations

"""
Personal Health — Smart Coaching Engine.

Goes beyond simple threshold-based feedback. Analyzes patterns across
frames within a session and across sessions over time to give contextual,
specific coaching that feels human.

Three tiers of coaching:
1. Frame-level: immediate cue based on current joint angles
2. Session-level: pattern analysis across all frames in a session
3. Longitudinal: trend analysis across sessions over weeks

This replaces generic feedback like "Adjust Knee Bend" with specific,
actionable coaching like "Your left knee is caving inward 12° more than
your right on the descent phase — add banded clamshells to your warmup."
"""

from collections import defaultdict
from typing import Any


# ─── Sport-specific coaching knowledge base ──────────────────────────────────

COACHING_RULES: dict[str, list[dict[str, Any]]] = {
    "vertical_jump": [
        {
            "condition": lambda f: f.get("knee_angle_l", 180) < 80,
            "cue": "Knees are collapsing too deep. Stop descent at 90°.",
            "joint": "knee_angle",
            "severity": "high",
        },
        {
            "condition": lambda f: abs(f.get("knee_angle_l", 0) - f.get("knee_angle_r", 0)) > 15,
            "cue": "Left-right knee imbalance detected. Check for ankle mobility on the weaker side.",
            "joint": "knee_asymmetry",
            "severity": "high",
        },
        {
            "condition": lambda f: f.get("trunk_lean", 0) > 25,
            "cue": "Excessive forward lean. Drive your chest up on takeoff.",
            "joint": "trunk_lean",
            "severity": "medium",
        },
        {
            "condition": lambda f: f.get("hip_angle_l", 180) > 120 and f.get("phase") == "descent",
            "cue": "Hips aren't loading. Sit back more — imagine sitting into a chair.",
            "joint": "hip_angle",
            "severity": "medium",
        },
        {
            "condition": lambda f: f.get("limb_symmetry_idx", 1) < 0.85,
            "cue": "Significant asymmetry. Bias single-leg work on your weaker side.",
            "joint": "symmetry",
            "severity": "high",
        },
    ],
    "squat": [
        {
            "condition": lambda f: f.get("knee_angle_l", 180) < 70,
            "cue": "Going too deep for your current mobility. Aim for parallel (thigh horizontal).",
            "joint": "knee_angle",
            "severity": "medium",
        },
        {
            "condition": lambda f: f.get("trunk_lean", 0) > 30,
            "cue": "Too much forward lean. Widen your stance or elevate your heels.",
            "joint": "trunk_lean",
            "severity": "high",
        },
        {
            "condition": lambda f: abs(f.get("knee_angle_l", 0) - f.get("knee_angle_r", 0)) > 12,
            "cue": "One knee is tracking differently. Film from the front to check knee-over-toe alignment.",
            "joint": "knee_asymmetry",
            "severity": "high",
        },
    ],
    "sprint": [
        {
            "condition": lambda f: f.get("trunk_lean", 0) < 8 and f.get("phase") == "drive",
            "cue": "Not enough forward lean in the drive phase. Push into the ground at 45°.",
            "joint": "trunk_lean",
            "severity": "medium",
        },
        {
            "condition": lambda f: f.get("hip_angle_l", 0) > 80,
            "cue": "Hip extension is limited. Focus on driving your knee up to hip height.",
            "joint": "hip_angle",
            "severity": "medium",
        },
    ],
}

# Default rules for sports without specific coaching
DEFAULT_RULES = [
    {
        "condition": lambda f: f.get("limb_symmetry_idx", 1) < 0.85,
        "cue": "Significant left-right asymmetry. Prioritize single-side work.",
        "joint": "symmetry",
        "severity": "high",
    },
    {
        "condition": lambda f: f.get("form_score", 100) < 50,
        "cue": "Form needs attention. Slow down your movement and focus on control.",
        "joint": "form",
        "severity": "high",
    },
]


# ─── Frame-level coaching ────────────────────────────────────────────────────

def coach_frame(frame: dict, sport: str = "vertical_jump") -> list[dict]:
    """Analyze a single frame and return relevant coaching cues."""
    rules = COACHING_RULES.get(sport, DEFAULT_RULES)
    cues = []
    for rule in rules:
        try:
            if rule["condition"](frame):
                cues.append({
                    "cue": rule["cue"],
                    "joint": rule["joint"],
                    "severity": rule["severity"],
                })
        except Exception:
            continue
    return cues


# ─── Session-level analysis ──────────────────────────────────────────────────

def analyze_session_patterns(frames: list[dict], sport: str = "vertical_jump") -> dict:
    """Analyze patterns across all frames in a session."""
    if not frames:
        return {"patterns": [], "summary": "No frame data to analyze."}

    # Collect angle distributions
    angles = defaultdict(list)
    phases = defaultdict(int)
    scores = []

    for f in frames:
        for key in ["knee_angle_l", "knee_angle_r", "hip_angle_l", "hip_angle_r",
                     "trunk_lean", "limb_symmetry_idx"]:
            v = f.get(key)
            if v is not None and float(v) > 0:
                angles[key].append(float(v))
        phase = f.get("phase", "setup")
        phases[phase] += 1
        sc = f.get("form_score", 0)
        if sc > 0:
            scores.append(float(sc))

    patterns = []

    # Check for fatigue (scores declining over time)
    if len(scores) >= 6:
        first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        if second_half < first_half - 5:
            patterns.append({
                "type": "fatigue",
                "message": f"Form declined from {first_half:.0f} to {second_half:.0f} over the session. Consider shorter sets with rest between.",
                "severity": "medium",
            })

    # Check for consistency
    if scores:
        import statistics
        if len(scores) >= 4:
            stdev = statistics.stdev(scores)
            if stdev > 15:
                patterns.append({
                    "type": "inconsistency",
                    "message": f"Form score varied a lot (±{stdev:.0f}). Focus on repeating the same movement pattern each rep.",
                    "severity": "medium",
                })
            elif stdev < 5 and sum(scores)/len(scores) > 70:
                patterns.append({
                    "type": "consistency",
                    "message": "Very consistent form throughout. You're ready to increase intensity.",
                    "severity": "positive",
                })

    # Check for persistent asymmetry
    sym_vals = angles.get("limb_symmetry_idx", [])
    if sym_vals:
        avg_sym = sum(sym_vals) / len(sym_vals)
        if avg_sym < 0.88:
            # Determine which side is weaker
            l_knee = sum(angles.get("knee_angle_l", [0])) / max(len(angles.get("knee_angle_l", [1])), 1)
            r_knee = sum(angles.get("knee_angle_r", [0])) / max(len(angles.get("knee_angle_r", [1])), 1)
            weaker = "left" if l_knee < r_knee else "right"
            patterns.append({
                "type": "asymmetry",
                "message": f"Your {weaker} side is consistently weaker (symmetry {avg_sym:.0%}). Add 2 extra sets of single-leg work on the {weaker} side.",
                "severity": "high",
            })

    # Check which phase the athlete spends most time in
    if phases:
        dominant_phase = max(phases, key=phases.get)
        if dominant_phase == "setup" and len(frames) > 10:
            patterns.append({
                "type": "timing",
                "message": "You're spending too long in the setup position. Once aligned, commit to the movement.",
                "severity": "low",
            })

    summary = "Good session." if not patterns else " ".join(p["message"] for p in patterns[:2])

    return {
        "patterns": patterns,
        "summary": summary,
        "frame_count": len(frames),
        "avg_score": round(sum(scores)/len(scores), 1) if scores else 0,
        "score_range": [round(min(scores), 1), round(max(scores), 1)] if scores else [0, 0],
        "dominant_phase": max(phases, key=phases.get) if phases else "setup",
    }


# ─── Auto-label frames for training data ────────────────────────────────────

def auto_label_frame(frame: dict, session_avg_score: float = 0) -> str:
    """
    Assign a quality label to a frame for model training purposes.
    Uses per-frame score when available, falls back to session average.
    More nuanced than labeling all frames in a session with one label.
    """
    score = float(frame.get("form_score", 0) or 0)
    if score == 0:
        score = session_avg_score

    if score >= 85:
        return "elite"
    elif score >= 70:
        return "good"
    elif score >= 50:
        return "average"
    else:
        return "poor"
