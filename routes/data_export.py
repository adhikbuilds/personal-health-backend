from __future__ import annotations

"""
Personal Health — Session Data Export for model retraining pipeline.

VISION.md Data Flywheel: export real session data in the same schema as
generate_dataset.py produces, so the retraining pipeline can mix real
and synthetic data seamlessly.

  GET /admin/export/sessions          — all sessions as CSV or JSON
  GET /athlete/{id}/export            — single athlete's data
  GET /admin/export/stats             — dataset summary (no PII)

The CSV columns match pipeline/generate_dataset.py headers exactly.
"""

import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from auth import require_athlete_or_admin
from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger

router = APIRouter(tags=["Export"])
log = get_logger("routes.data_export")

# These columns mirror generate_dataset.py output — keep in sync.
_CSV_HEADERS = [
    "session_id",
    "athlete_id",
    "sport",
    "frame_num",
    "knee_angle_l",
    "knee_angle_r",
    "hip_angle_l",
    "hip_angle_r",
    "ankle_dorsiflexion_l",
    "ankle_dorsiflexion_r",
    "elbow_angle_l",
    "elbow_angle_r",
    "shoulder_angle_l",
    "shoulder_angle_r",
    "trunk_lean",
    "spine_deviation",
    "shoulder_hip_sep",
    "limb_symmetry_idx",
    "estimated_jump_height",
    "form_score",
    "form_quality",
    "phase_label",
    "rep_count",
    "timestamp",
]


def _frame_to_row(session: dict, frame: dict) -> dict:
    """Convert a stored frame dict into a CSV-export row."""
    return {
        "session_id": session.get("session_id", ""),
        "athlete_id": session.get("athlete_id", ""),
        "sport": session.get("sport", ""),
        "frame_num": frame.get("frame_num", ""),
        "knee_angle_l": frame.get("knee_angle_l", ""),
        "knee_angle_r": frame.get("knee_angle_r", ""),
        "hip_angle_l": frame.get("hip_angle_l", ""),
        "hip_angle_r": frame.get("hip_angle_r", ""),
        "ankle_dorsiflexion_l": frame.get("ankle_dorsiflexion_l", ""),
        "ankle_dorsiflexion_r": frame.get("ankle_dorsiflexion_r", ""),
        "elbow_angle_l": frame.get("elbow_angle_l", ""),
        "elbow_angle_r": frame.get("elbow_angle_r", ""),
        "shoulder_angle_l": frame.get("shoulder_angle_l", ""),
        "shoulder_angle_r": frame.get("shoulder_angle_r", ""),
        "trunk_lean": frame.get("trunk_lean", ""),
        "spine_deviation": frame.get("spine_deviation", ""),
        "shoulder_hip_sep": frame.get("shoulder_hip_sep", ""),
        "limb_symmetry_idx": frame.get("limb_symmetry_idx", ""),
        "estimated_jump_height": frame.get("estimated_jump_height", ""),
        "form_score": frame.get("form_score", ""),
        "form_quality": frame.get("form_quality", ""),
        "phase_label": frame.get("phase_label", frame.get("phase", "")),
        "rep_count": frame.get("rep_count", ""),
        "timestamp": frame.get("timestamp", ""),
    }


def _export_sessions_csv(sessions: list[dict]) -> str:
    """Generate CSV string from a list of session dicts."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS, extrasaction="ignore")
    writer.writeheader()
    for s in sessions:
        frames = s.get("frames", []) or []
        for frame in frames:
            writer.writerow(_frame_to_row(s, frame))
    return buf.getvalue()


def _export_sessions_json(sessions: list[dict]) -> list[dict]:
    """Generate JSON-exportable list of frame rows."""
    rows = []
    for s in sessions:
        frames = s.get("frames", []) or []
        for frame in frames:
            rows.append(_frame_to_row(s, frame))
    return rows


def _completed_sessions(athlete_id: str | None = None) -> list[dict]:
    """Get completed sessions, optionally filtered by athlete."""
    sessions = [s for s in SESSION_DB.values() if s.get("status") == "completed"]
    if athlete_id:
        sessions = [s for s in sessions if s.get("athlete_id") == athlete_id]
    sessions.sort(key=lambda x: x.get("started_at", ""))
    return sessions


# ─── Routes ─────────────────────────────────────────────────────────────────


@router.get("/admin/export/sessions")
async def export_all_sessions(
    format: str = Query(default="csv", pattern="^(csv|json)$"),
    sport: str | None = None,
    min_frames: int = Query(default=0, ge=0),
):
    """Export all completed session frame data for model retraining."""
    sessions = _completed_sessions()
    if sport:
        sessions = [s for s in sessions if s.get("sport") == sport]
    if min_frames > 0:
        sessions = [s for s in sessions if len(s.get("frames", []) or []) >= min_frames]

    log.info(
        "data export",
        extra={"format": format, "sessions": len(sessions), "sport": sport or "all"},
    )

    if format == "json":
        rows = _export_sessions_json(sessions)
        return {
            "format": "json",
            "sessions": len(sessions),
            "frames": len(rows),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "data": rows,
        }

    csv_content = _export_sessions_csv(sessions)
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=session_export.csv"},
    )


@router.get("/athlete/{athlete_id}/export")
async def export_athlete_data(
    athlete_id: str,
    format: str = Query(default="csv", pattern="^(csv|json)$"),
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    """Export a single athlete's session frame data."""
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    sessions = _completed_sessions(athlete_id)
    athlete = ATHLETE_DB[athlete_id]

    if format == "json":
        rows = _export_sessions_json(sessions)
        return {
            "athlete_id": athlete_id,
            "athlete_name": athlete.get("name"),
            "sport": athlete.get("sport"),
            "format": "json",
            "sessions": len(sessions),
            "frames": len(rows),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "data": rows,
        }

    csv_content = _export_sessions_csv(sessions)
    safe_name = athlete_id.replace("/", "_")
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={safe_name}_export.csv"},
    )


@router.get("/admin/export/stats")
async def export_stats():
    """Dataset summary — no PII, useful for monitoring data flywheel growth."""
    sessions = _completed_sessions()
    total_frames = sum(len(s.get("frames", []) or []) for s in sessions)

    sport_dist: dict[str, int] = {}
    tier_dist: dict[str, int] = {}
    athlete_ids: set[str] = set()

    for s in sessions:
        sp = s.get("sport", "unknown")
        sport_dist[sp] = sport_dist.get(sp, 0) + 1
        aid = s.get("athlete_id", "")
        athlete_ids.add(aid)

    for aid in athlete_ids:
        a = ATHLETE_DB.get(aid, {})
        t = a.get("tier", "Unknown")
        tier_dist[t] = tier_dist.get(t, 0) + 1

    return {
        "total_sessions": len(sessions),
        "total_frames": total_frames,
        "unique_athletes": len(athlete_ids),
        "sessions_by_sport": sport_dist,
        "athletes_by_tier": tier_dist,
        "ready_for_retrain": total_frames >= 5000,
        "retrain_threshold": 5000,
        "frames_to_threshold": max(0, 5000 - total_frames),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
