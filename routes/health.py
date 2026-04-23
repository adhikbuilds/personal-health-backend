from __future__ import annotations

"""
Health & Meta endpoints — /, /health, /banner
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from database import ATHLETE_DB, DATASET_PATH, SESSION_DB

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    return {
        "service": "Personal Health Sports Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "sessions": "/sessions",
            "start_session": "POST /session/start",
            "add_frame": "POST /session/{id}/frame",
            "end_session": "POST /session/{id}/end",
            "athletes": "/athletes",
            "leaderboard": "/leaderboard",
            "live_stream": "ws://HOST:8082/metrics/live/{session_id}",
            "dataset": "/dataset/export",
            "docs": "/docs",
        },
    }


@router.get("/health")
async def health():
    models_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "models"
    tflite_path = models_dir / "pose_classifier.tflite"
    keras_path = models_dir / "pose_classifier.h5"
    norm_path = models_dir / "norm_params.json"
    face_path = models_dir / "face_detector.tflite"

    # Discover which sport-specific models are on disk so clients can surface
    # "sport X will fall back to rule-based" without server logs.
    sport_specific = {}
    if models_dir.exists():
        for fp in models_dir.glob("pose_classifier_*.tflite"):
            sport = fp.stem.replace("pose_classifier_", "")
            sport_specific[sport] = True

    # Pull active model version from the registry if available
    active_version = None
    try:
        from services.model_registry import get_active_version
        active_version = get_active_version()
    except Exception:
        pass

    return {
        "status": "ok",
        "sessions_count": len(SESSION_DB),
        "athletes_count": len(ATHLETE_DB),
        "dataset_ready": (DATASET_PATH / "training_data.csv").exists(),
        "model_ready": tflite_path.exists() or keras_path.exists(),
        "models": {
            "tflite": tflite_path.exists(),
            "keras": keras_path.exists(),
            "norm_params": norm_path.exists(),
            "face_detector": face_path.exists(),
            "active_version": active_version,
            "sport_specific": sport_specific,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/banner")
async def banner():
    return {
        "connected": True,
        "server": "Personal Health API v2.0",
        "athletes": len(ATHLETE_DB),
        "sessions_today": sum(
            1 for s in SESSION_DB.values() if s.get("started_at", "")[:10] == datetime.now(timezone.utc).date().isoformat()
        ),
    }
