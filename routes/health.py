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
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
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
    return {
        "status": "ok",
        "sessions_count": len(SESSION_DB),
        "athletes_count": len(ATHLETE_DB),
        "dataset_ready": (DATASET_PATH / "training_data.csv").exists(),
        "model_ready": (
            Path(os.path.dirname(os.path.abspath(__file__))).parent / "models" / "pose_classifier.tflite"
        ).exists(),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
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
