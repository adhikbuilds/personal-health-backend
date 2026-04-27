from __future__ import annotations

"""
Personal Health — admin / ops endpoints: /livez, /readyz, /metrics, /audit.
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

import database
from auth import create_api_key, require_role
from logging_setup import get_logger
from metrics import render, set_gauge
from sqlite_store import recent_audit

router = APIRouter(tags=["Admin"])
log = get_logger("routes.admin")


@router.get("/livez")
async def livez():
    """Liveness — process is up. Always 200 unless interpreter is stuck."""
    return {"status": "alive"}


@router.get("/readyz")
async def readyz():
    """Readiness — DB loaded, analysis queue ready, queue not saturated, SQLite OK."""
    queue_ready = database.ANALYSIS_QUEUE is not None
    queue_depth = database.ANALYSIS_QUEUE.qsize() if queue_ready else -1
    queue_max = database.ANALYSIS_QUEUE.maxsize if queue_ready else 0
    saturated = queue_ready and queue_max and (queue_depth / queue_max) > 0.9
    db_loaded = len(database.ATHLETE_DB) > 0

    sqlite_ok = False
    try:
        from sqlite_store import cursor

        with cursor() as cur:
            row = cur.execute("PRAGMA integrity_check(1)").fetchone()
            sqlite_ok = row and row[0] == "ok"
    except Exception:
        pass

    ok = queue_ready and db_loaded and not saturated and sqlite_ok
    return {
        "status": "ready" if ok else "not_ready",
        "queue_ready": queue_ready,
        "queue_depth": queue_depth,
        "queue_max": queue_max,
        "db_loaded": db_loaded,
        "saturated": saturated,
        "sqlite_ok": sqlite_ok,
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # Refresh dynamic gauges before rendering
    if database.ANALYSIS_QUEUE is not None:
        set_gauge("analysis_queue_depth", float(database.ANALYSIS_QUEUE.qsize()))
    set_gauge("active_sessions", float(sum(1 for s in database.SESSION_DB.values() if s.get("status") == "active")))
    set_gauge("ws_connections", float(sum(len(v) for v in database.WS_CONNECTIONS.values())))
    return render()


@router.get("/audit", dependencies=[Depends(require_role("admin"))])
async def audit_log(limit: int = Query(default=100, ge=1, le=1000)):
    return {"entries": recent_audit(limit)}


@router.post("/admin/api-keys", dependencies=[Depends(require_role("admin"))])
async def mint_api_key(label: str = Query(min_length=1, max_length=80)):
    """Create a new service-to-service API key. The raw token is returned ONCE."""
    raw = create_api_key(label)
    return {"label": label, "api_key": raw, "warning": "store now — cannot be retrieved later"}


# ─── Model Registry + Retrain ──────────────────────────────────────────────


@router.get("/admin/model", dependencies=[Depends(require_role("admin"))])
async def model_status():
    """Current model version, accuracy, and registry history."""
    from services.model_registry import get_registry, predict_quality

    reg = get_registry()
    # Quick inference test to verify model is loaded
    test_pred = predict_quality(
        {"hip_angle_l": 100, "knee_angle_l": 120, "limb_symmetry_idx": 0.95},
        sport="vertical_jump",
    )
    return {
        "active_version": reg.get("active_version"),
        "total_versions": len(reg.get("versions", [])),
        "versions": reg.get("versions", []),
        "model_loaded": test_pred is not None,
        "model_type": test_pred.get("model_type") if test_pred else None,
    }


@router.get("/admin/retrain-status", dependencies=[Depends(require_role("admin"))])
async def retrain_status():
    """Check whether a retrain is warranted based on available data."""
    from services.model_registry import get_registry

    sessions_file = database.DB_PATH / "sessions.json"
    total_frames = 0
    completed_sessions = 0
    if sessions_file.exists():
        import json as _json

        with open(sessions_file) as f:
            sessions = _json.load(f)
        for s in sessions.values():
            if s.get("status") == "completed":
                completed_sessions += 1
                total_frames += len(s.get("frames", []) or [])

    reg = get_registry()
    current_acc = 0.0
    if reg.get("versions"):
        current_acc = reg["versions"][-1].get("accuracy", 0)

    threshold = 500
    return {
        "total_real_frames": total_frames,
        "completed_sessions": completed_sessions,
        "retrain_threshold": threshold,
        "frames_to_threshold": max(0, threshold - total_frames),
        "retrain_ready": total_frames >= threshold,
        "current_model_version": reg.get("active_version"),
        "current_model_accuracy": current_acc,
    }


@router.post("/admin/model/reload", dependencies=[Depends(require_role("admin"))])
async def reload_model():
    """Hot-reload the model from disk (after manual retrain)."""
    from services.model_registry import reload_model as _reload

    success = _reload()
    return {"reloaded": success}
