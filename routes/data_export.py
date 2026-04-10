from __future__ import annotations

"""
Data export pipeline for model retraining.

VISION.md data flywheel: "The moment we have 500 real sessions from real
athletes, we retrain the model and it gets noticeably better."

VISION.md Milestone 3: "generate_dataset.py supplemented with real session
exports. Model retrained on real + synthetic mix."

Endpoints:
  GET  /data/export/sessions    — export real sessions as CSV for retraining
  GET  /data/export/stats       — dataset health: how close to 500 sessions
  POST /data/export/snapshot    — save a point-in-time training snapshot to disk
"""

import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from database import DB_PATH, SESSION_DB
from logging_setup import get_logger

router = APIRouter(prefix="/data/export", tags=["Data Export"])
log = get_logger("routes.data_export")

# the feature columns the model expects (same as pipeline/generate_dataset.py)
FEATURE_COLS = [
    "sport",
    "knee_angle_l",
    "knee_angle_r",
    "hip_angle_l",
    "hip_angle_r",
    "elbow_angle_l",
    "elbow_angle_r",
    "ankle_dorsiflexion_l",
    "ankle_dorsiflexion_r",
    "trunk_lean",
    "spine_deviation",
    "shoulder_hip_sep",
    "head_forward_pos",
    "com_height_norm",
    "limb_symmetry_idx",
    "estimated_jump_height",
    "shoulder_angle_l",
    "shoulder_angle_r",
]

LABEL_COL = "form_quality"
SCORE_COL = "form_score"


def _extract_training_rows(min_score: float = 0.0) -> list[dict]:
    """Extract labelled frame rows from all completed sessions."""
    rows = []
    for s in SESSION_DB.values():
        if s.get("status") != "completed":
            continue
        sport = s.get("sport", "vertical_jump")
        athlete_id = s.get("athlete_id", "")
        session_id = s.get("session_id", "")
        for frame in s.get("frames") or []:
            score = float(frame.get("form_score") or 0)
            quality = frame.get("form_quality", "unknown")
            if score < min_score or quality == "unknown":
                continue
            row = {
                "session_id": session_id,
                "athlete_id": athlete_id,
                "sport": sport,
                "form_score": score,
                "form_quality": quality,
            }
            for col in FEATURE_COLS[1:]:  # skip sport, already added
                row[col] = frame.get(col, 0.0)
            rows.append(row)
    return rows


@router.get("/sessions")
async def export_sessions_csv(
    min_score: float = Query(default=0.0, ge=0, description="min form_score to include"),
    sport: str | None = Query(default=None, description="filter by sport"),
):
    """
    Export real session frames as CSV in the same format as training_data.csv.

    This is what you feed into pipeline/mix_datasets.py to combine with
    synthetic data before retraining.
    """
    rows = _extract_training_rows(min_score)
    if sport:
        rows = [r for r in rows if r["sport"] == sport]

    if not rows:
        raise HTTPException(404, "no exportable frames found. train more sessions first")

    output = io.StringIO()
    cols = ["session_id", "athlete_id", "sport", "form_score", "form_quality", *FEATURE_COLS[1:]]
    writer = csv.DictWriter(output, fieldnames=cols)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    output.seek(0)
    filename = f"real_sessions_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/stats")
async def export_stats():
    """
    How close are we to the 500 session retrain threshold?

    VISION: "The moment we have 500 real sessions from real athletes,
    we retrain the model and it gets noticeably better."
    """
    completed = [s for s in SESSION_DB.values() if s.get("status") == "completed"]
    with_frames = [s for s in completed if s.get("frames")]

    total_frames = sum(len(s.get("frames") or []) for s in with_frames)
    labelled_frames = 0
    sport_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {"elite": 0, "good": 0, "average": 0, "poor": 0}

    for s in with_frames:
        sport = s.get("sport", "unknown")
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
        for f in s.get("frames") or []:
            q = f.get("form_quality", "unknown")
            if q != "unknown" and float(f.get("form_score") or 0) > 0:
                labelled_frames += 1
                if q in quality_counts:
                    quality_counts[q] += 1

    target = 500
    pct = round(len(with_frames) / target * 100, 1) if target else 0

    return {
        "total_sessions": len(completed),
        "sessions_with_frames": len(with_frames),
        "retrain_target": target,
        "progress_pct": min(pct, 100.0),
        "ready_to_retrain": len(with_frames) >= target,
        "total_frames": total_frames,
        "labelled_frames": labelled_frames,
        "sport_breakdown": sport_counts,
        "quality_breakdown": quality_counts,
        "unique_athletes": len({s.get("athlete_id") for s in with_frames}),
        "message": (
            f"{len(with_frames)}/{target} sessions collected ({pct}%). "
            + ("ready to retrain!" if pct >= 100 else f"need {target - len(with_frames)} more.")
        ),
    }


@router.post("/snapshot")
async def save_snapshot():
    """
    Save a point-in-time export of real session data to disk for retraining.
    Writes to dataset/real_export_YYYYMMDD.csv alongside the synthetic data.
    """
    rows = _extract_training_rows(min_score=10.0)
    if not rows:
        raise HTTPException(404, "no exportable frames. need more completed sessions with analysis")

    dataset_dir = DB_PATH.parent / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    filename = f"real_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = dataset_dir / filename

    cols = ["session_id", "athlete_id", "sport", "form_score", "form_quality", *FEATURE_COLS[1:]]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log.info("snapshot saved", extra={"path": str(filepath), "rows": len(rows)})
    return {
        "status": "saved",
        "path": str(filepath),
        "rows": len(rows),
        "message": f"exported {len(rows)} frames to {filename}. run pipeline/mix_datasets.py next",
    }
