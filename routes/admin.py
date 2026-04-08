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
    """Readiness — DB loaded, analysis queue ready, queue not saturated."""
    queue_ready = database.ANALYSIS_QUEUE is not None
    queue_depth = database.ANALYSIS_QUEUE.qsize() if queue_ready else -1
    queue_max = database.ANALYSIS_QUEUE.maxsize if queue_ready else 0
    saturated = queue_ready and queue_max and (queue_depth / queue_max) > 0.9
    db_loaded = len(database.ATHLETE_DB) > 0
    ok = queue_ready and db_loaded and not saturated
    return {
        "status": "ready" if ok else "not_ready",
        "queue_ready": queue_ready,
        "queue_depth": queue_depth,
        "queue_max": queue_max,
        "db_loaded": db_loaded,
        "saturated": saturated,
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
