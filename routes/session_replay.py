from __future__ import annotations

"""
Session replay — frame-by-frame detail for reviewing form.

VISION.md priorities #1 and #2:
  "The session flow" and "The feedback loop — the form score and coaching
  text need to feel correct to an athlete who knows their sport."

Athletes and coaches need to review sessions after the fact. This provides
structured frame data with phase transitions, form score timeline, and
per-frame joint angles so the app/dashboard can render a replay.

Endpoints:
  GET /sessions/{id}/replay          — full frame timeline with phases
  GET /sessions/{id}/highlights      — key moments (PBs, phase transitions, worst frames)
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from auth import require_session_owner_or_admin
from database import FRAME_BUFFER, SESSION_DB
from logging_setup import get_logger

router = APIRouter(tags=["Session Replay"])
log = get_logger("routes.session_replay")


@router.get("/sessions/{session_id}/replay")
async def session_replay(
    session_id: str,
    downsample: int = Query(default=1, ge=1, le=10, description="take every Nth frame"),
    _: dict = Depends(require_session_owner_or_admin("session_id")),
):
    """
    Full frame timeline for session replay.

    Returns all frames (or downsampled) with joint angles, form scores,
    phases, and timestamps so the app can scrub through the session.
    """
    if session_id not in SESSION_DB:
        raise HTTPException(404, "session not found")

    session = SESSION_DB[session_id]
    frames = FRAME_BUFFER.get(session_id) or session.get("frames") or []

    if not frames:
        return {
            "session_id": session_id,
            "status": session.get("status"),
            "frame_count": 0,
            "frames": [],
            "phases": [],
        }

    # downsample if requested
    sampled = frames[::downsample]

    # extract clean frame data
    timeline = []
    for i, f in enumerate(sampled):
        timeline.append(
            {
                "idx": i,
                "frame_num": f.get("frame_num", i),
                "timestamp": f.get("timestamp"),
                "form_score": f.get("form_score", 0),
                "form_quality": f.get("form_quality", "unknown"),
                "phase": f.get("phase", f.get("phase_label", "unknown")),
                "knee_l": f.get("knee_angle_l"),
                "knee_r": f.get("knee_angle_r"),
                "hip_l": f.get("hip_angle_l"),
                "hip_r": f.get("hip_angle_r"),
                "elbow_l": f.get("elbow_angle_l"),
                "elbow_r": f.get("elbow_angle_r"),
                "trunk_lean": f.get("trunk_lean"),
                "symmetry": f.get("limb_symmetry_idx"),
                "jump_height": f.get("estimated_jump_height"),
                "feedback": f.get("primary_feedback", ""),
            }
        )

    # detect phase transitions
    phases = []
    current_phase = None
    for t in timeline:
        if t["phase"] != current_phase:
            phases.append(
                {
                    "phase": t["phase"],
                    "starts_at_frame": t["frame_num"],
                    "starts_at_idx": t["idx"],
                }
            )
            current_phase = t["phase"]

    summary = session.get("summary") or {}

    return {
        "session_id": session_id,
        "sport": session.get("sport"),
        "athlete_id": session.get("athlete_id"),
        "status": session.get("status"),
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "total_frames": len(frames),
        "returned_frames": len(timeline),
        "downsample": downsample,
        "avg_form_score": summary.get("avg_form_score"),
        "frames": timeline,
        "phases": phases,
    }


@router.get("/sessions/{session_id}/highlights")
async def session_highlights(
    session_id: str,
    _: dict = Depends(require_session_owner_or_admin("session_id")),
):
    """
    Key moments from a session: best frame, worst frame, phase transitions,
    personal bests, biggest form drops.

    Useful for coaches doing quick reviews without scrubbing every frame.
    """
    if session_id not in SESSION_DB:
        raise HTTPException(404, "session not found")

    session = SESSION_DB[session_id]
    frames = FRAME_BUFFER.get(session_id) or session.get("frames") or []

    if not frames:
        return {"session_id": session_id, "highlights": [], "frame_count": 0}

    scored = [f for f in frames if f.get("form_score", 0) > 0]
    highlights = []

    # best frame
    if scored:
        best = max(scored, key=lambda f: f.get("form_score", 0))
        highlights.append(
            {
                "type": "best_form",
                "label": "best form score",
                "frame_num": best.get("frame_num"),
                "form_score": best.get("form_score"),
                "quality": best.get("form_quality"),
                "feedback": best.get("primary_feedback", ""),
            }
        )

        # worst frame
        worst = min(scored, key=lambda f: f.get("form_score", 0))
        highlights.append(
            {
                "type": "worst_form",
                "label": "needs work",
                "frame_num": worst.get("frame_num"),
                "form_score": worst.get("form_score"),
                "quality": worst.get("form_quality"),
                "feedback": worst.get("primary_feedback", ""),
            }
        )

    # biggest single-frame form drop
    for i in range(1, len(scored)):
        prev_score = scored[i - 1].get("form_score", 0)
        curr_score = scored[i].get("form_score", 0)
        drop = prev_score - curr_score
        if drop > 15:
            highlights.append(
                {
                    "type": "form_drop",
                    "label": f"form dropped {drop:.0f} points",
                    "frame_num": scored[i].get("frame_num"),
                    "from_score": prev_score,
                    "to_score": curr_score,
                    "feedback": scored[i].get("primary_feedback", ""),
                }
            )
            break  # just the biggest one

    # best jump
    jumped = [f for f in frames if (f.get("estimated_jump_height") or 0) > 5]
    if jumped:
        best_jump = max(jumped, key=lambda f: f.get("estimated_jump_height", 0))
        highlights.append(
            {
                "type": "best_jump",
                "label": "highest jump",
                "frame_num": best_jump.get("frame_num"),
                "jump_height_cm": round(best_jump.get("estimated_jump_height", 0), 1),
            }
        )

    # worst symmetry moment
    sym_frames = [f for f in frames if f.get("limb_symmetry_idx") is not None]
    if sym_frames:
        worst_sym = max(sym_frames, key=lambda f: abs(1.0 - f.get("limb_symmetry_idx", 1.0)))
        dev = abs(1.0 - worst_sym.get("limb_symmetry_idx", 1.0))
        if dev > 0.15:
            side = "left heavy" if worst_sym.get("limb_symmetry_idx", 1.0) > 1.0 else "right heavy"
            highlights.append(
                {
                    "type": "asymmetry",
                    "label": f"worst asymmetry ({side})",
                    "frame_num": worst_sym.get("frame_num"),
                    "symmetry_idx": round(worst_sym.get("limb_symmetry_idx", 1.0), 3),
                    "deviation_pct": round(dev * 100, 1),
                }
            )

    return {
        "session_id": session_id,
        "sport": session.get("sport"),
        "frame_count": len(frames),
        "highlights": highlights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
