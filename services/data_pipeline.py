from __future__ import annotations

"""
Personal Health — Unified Data Pipeline.

Single entry point for all data flowing through the system.
Handles: frame storage, quality gating, auto-labeling, export,
and retrain readiness checks.

Architecture:
  Phone → frame → quality_gate → store → accumulate
  accumulate → threshold check → export → mix → retrain → deploy

This replaces the scattered data handling across fitness.py,
export_real_data.py, and frame_quality.py with one clean pipeline.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database import DB_PATH, SESSION_DB
from logging_setup import get_logger
from services.frame_quality import score_frame_quality

log = get_logger("services.data_pipeline")

# ─── Frame Storage ───────────────────────────────────────────────────────────


def store_frame(session_id: str, frame: dict, sport: str = "general") -> dict:
    """
    Process and store a single analyzed frame.
    1. Score quality
    2. Auto-label for training
    3. Persist to disk
    4. Return quality metadata
    """
    # Quality gate
    quality = score_frame_quality(frame, sport)

    # Auto-label based on form score
    score = float(frame.get("form_score", 0) or 0)
    if score >= 85:
        label = "elite"
    elif score >= 70:
        label = "good"
    elif score >= 50:
        label = "average"
    elif score > 0:
        label = "poor"
    else:
        label = "unlabeled"

    frame["_quality_score"] = quality["quality_score"]
    frame["_train_ready"] = quality["train_ready"]
    frame["_auto_label"] = label

    # Persist to disk (JSONL per session)
    try:
        frames_dir = DB_PATH / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        with open(frames_dir / f"{session_id}.jsonl", "a") as f:
            # Don't store image data — only metrics
            row = {k: v for k, v in frame.items() if k != "image_b64"}
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:
        log.warning("frame persist failed", extra={"error": str(e)})

    return quality


# ─── Dataset Statistics ──────────────────────────────────────────────────────


def get_dataset_stats() -> dict[str, Any]:
    """Get current state of the training data pipeline."""
    total_sessions = 0
    total_frames = 0
    train_ready_frames = 0
    frames_by_sport: dict[str, int] = {}
    frames_by_quality: dict[str, int] = {}

    for sid, session in SESSION_DB.items():
        if session.get("status") != "completed":
            continue
        total_sessions += 1
        frames = session.get("frames", []) or []
        for f in frames:
            total_frames += 1
            sport = session.get("sport", "unknown")
            frames_by_sport[sport] = frames_by_sport.get(sport, 0) + 1

            qs = float(f.get("_quality_score", 0) or 0)
            if qs >= 0.6:
                train_ready_frames += 1

            label = f.get("_auto_label", "unlabeled")
            frames_by_quality[label] = frames_by_quality.get(label, 0) + 1

    # Also count persisted frames on disk
    disk_frames = 0
    frames_dir = DB_PATH / "frames"
    if frames_dir.exists():
        for f in frames_dir.glob("*.jsonl"):
            with open(f) as fp:
                disk_frames += sum(1 for _ in fp)

    threshold = 500
    return {
        "total_sessions": total_sessions,
        "total_frames_memory": total_frames,
        "total_frames_disk": disk_frames,
        "train_ready_frames": train_ready_frames,
        "frames_by_sport": frames_by_sport,
        "frames_by_quality": frames_by_quality,
        "retrain_threshold": threshold,
        "retrain_ready": max(total_frames, disk_frames) >= threshold,
        "frames_to_threshold": max(0, threshold - max(total_frames, disk_frames)),
        "pipeline_health": "ready" if total_sessions > 0 else "no_data",
    }


# ─── Session Summary Enrichment ─────────────────────────────────────────────


def enrich_session_summary(session_id: str, summary: dict) -> dict:
    """
    Add intelligence to a session summary after it ends.
    Called by routes/fitness.py end_session.
    """
    frames = SESSION_DB.get(session_id, {}).get("frames", []) or []
    sport = SESSION_DB.get(session_id, {}).get("sport", "general")

    # Smart coaching
    try:
        from services.smart_coach import analyze_session_patterns

        coaching = analyze_session_patterns(frames, sport)
        summary["coaching"] = coaching
    except Exception as e:
        summary["coaching"] = {"patterns": [], "summary": str(e)}

    # Data quality stats
    quality_scores = [float(f.get("_quality_score", 0) or 0) for f in frames]
    train_ready = sum(1 for q in quality_scores if q >= 0.6)
    summary["data_quality"] = {
        "total_frames": len(frames),
        "train_ready": train_ready,
        "avg_quality": round(sum(quality_scores) / max(len(quality_scores), 1), 3),
        "contributed_to_flywheel": train_ready > 0,
    }

    return summary
