from __future__ import annotations

"""
Personal Health — Session Comparison endpoint.

GET /sessions/compare?a={session_id}&b={session_id}

Returns side-by-side metrics for two completed sessions belonging to the
same athlete. Useful for showing improvement between an early session and
a recent one in the Android app.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from auth import current_user
from database import SESSION_DB
from logging_setup import get_logger

router = APIRouter(tags=["Analytics"])
log = get_logger("routes.compare")

_JOINT_KEYS = [
    "knee_angle",
    "hip_angle",
    "elbow_angle",
    "shoulder_angle",
    "trunk_lean",
    "ankle_dorsiflexion",
]


def _avg_joint(frames: list[dict], joint: str) -> float | None:
    if joint == "trunk_lean":
        vals = [float(f["trunk_lean"]) for f in frames if f.get("trunk_lean") is not None]
    else:
        vals = []
        for f in frames:
            left_val, right_val = f.get(f"{joint}_l"), f.get(f"{joint}_r")
            if left_val is not None and right_val is not None:
                vals.append((float(left_val) + float(right_val)) / 2)
            elif left_val is not None:
                vals.append(float(left_val))
            elif right_val is not None:
                vals.append(float(right_val))
    return round(sum(vals) / len(vals), 1) if vals else None


def _session_snapshot(session: dict) -> dict:
    frames = session.get("frames") or []
    summary = session.get("summary") or {}
    joints = {k: _avg_joint(frames, k) for k in _JOINT_KEYS}
    return {
        "session_id": session["session_id"],
        "started_at": session.get("started_at"),
        "sport": session.get("sport"),
        "avg_form_score": float(session.get("avg_form_score") or summary.get("avg_form_score") or 0),
        "rep_count": int(session.get("rep_count") or summary.get("rep_count") or 0),
        "xp_earned": int(session.get("xp_earned") or summary.get("xp_earned") or 0),
        "peak_jump_height_cm": float(session.get("peak_jump_height_cm") or 0),
        "duration_seconds": int(session.get("duration_seconds") or summary.get("duration_seconds") or 0),
        "joint_averages": {k: v for k, v in joints.items() if v is not None},
        "frame_count": len(frames),
    }


@router.get("/sessions/compare", dependencies=[Depends(current_user)])
async def compare_sessions(
    a: str = Query(..., description="First session ID"),
    b: str = Query(..., description="Second session ID"),
):
    """Compare two completed sessions side-by-side with delta metrics."""
    sess_a = SESSION_DB.get(a)
    sess_b = SESSION_DB.get(b)

    if not sess_a:
        raise HTTPException(404, f"session '{a}' not found")
    if not sess_b:
        raise HTTPException(404, f"session '{b}' not found")
    if sess_a.get("status") != "completed" or sess_b.get("status") != "completed":
        raise HTTPException(400, "both sessions must be completed")
    if sess_a.get("athlete_id") != sess_b.get("athlete_id"):
        raise HTTPException(400, "sessions must belong to the same athlete")

    snap_a = _session_snapshot(sess_a)
    snap_b = _session_snapshot(sess_b)

    # Compute deltas (b - a)
    delta_form = round(snap_b["avg_form_score"] - snap_a["avg_form_score"], 1)
    delta_reps = snap_b["rep_count"] - snap_a["rep_count"]
    delta_jump = round(snap_b["peak_jump_height_cm"] - snap_a["peak_jump_height_cm"], 1)

    joint_deltas = {}
    for joint in _JOINT_KEYS:
        va = snap_a["joint_averages"].get(joint)
        vb = snap_b["joint_averages"].get(joint)
        if va is not None and vb is not None:
            joint_deltas[joint] = round(vb - va, 1)

    improved = delta_form > 0
    summary_msg = f"Form {'improved' if improved else 'dropped'} {abs(delta_form):.1f}pts between these sessions."

    return {
        "session_a": snap_a,
        "session_b": snap_b,
        "delta": {
            "form_score": delta_form,
            "rep_count": delta_reps,
            "jump_height_cm": delta_jump,
            "joints": joint_deltas,
        },
        "improved": improved,
        "summary": summary_msg,
    }
