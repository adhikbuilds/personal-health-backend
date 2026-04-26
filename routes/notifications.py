from __future__ import annotations

"""
Athlete notification system — streak risk, PBs, milestones, injury warnings,
re-engagement nudges.

  GET  /athlete/{id}/notifications
  POST /athlete/{id}/notifications/read
  POST /notifications/generate/{id}

Notifications are stored in SQLite. The generate endpoint runs server-side
checks and inserts new notifications; the frontend polls GET.
"""

import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from database import ATHLETE_DB, SESSION_DB
from logging_setup import get_logger
from routes.progress import _athlete_sessions, _compute_injury_risk
from sqlite_store import cursor

router = APIRouter(tags=["Notifications"])
log = get_logger("routes.notifications")


def _ensure_table() -> None:
    with cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
              id TEXT PRIMARY KEY,
              athlete_id TEXT NOT NULL,
              type TEXT NOT NULL,
              title TEXT NOT NULL,
              body TEXT,
              read INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_notif_athlete ON notifications(athlete_id, created_at DESC)")


_ensure_table()


def _insert_notif(athlete_id: str, ntype: str, title: str, body: str) -> dict:
    nid = uuid4().hex[:12]
    now = time.time()
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notifications(id, athlete_id, type, title, body, read, created_at) VALUES(?,?,?,?,?,0,?)",
            (nid, athlete_id, ntype, title, body, now),
        )
    return {
        "id": nid,
        "athlete_id": athlete_id,
        "type": ntype,
        "title": title,
        "body": body,
        "read": False,
        "created_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
    }


def _recent_notif_types(athlete_id: str, hours: int = 24) -> set[str]:
    cutoff = time.time() - hours * 3600
    with cursor() as cur:
        rows = cur.execute(
            "SELECT type FROM notifications WHERE athlete_id = ? AND created_at > ?",
            (athlete_id, cutoff),
        ).fetchall()
        return {r["type"] for r in rows}


@router.get("/athlete/{athlete_id}/notifications")
async def list_notifications(
    athlete_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM notifications WHERE athlete_id = ? ORDER BY created_at DESC LIMIT ?",
            (athlete_id, limit),
        ).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        ts = d.pop("created_at", None)
        if isinstance(ts, (int, float)):
            d["created_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        d["read"] = bool(d.get("read", 0))
        items.append(d)

    unread = sum(1 for i in items if not i["read"])
    return {"athlete_id": athlete_id, "notifications": items, "total": len(items), "unread": unread}


@router.post("/athlete/{athlete_id}/notifications/read")
async def mark_all_read(athlete_id: str):
    with cursor() as cur:
        cur.execute(
            "UPDATE notifications SET read = 1 WHERE athlete_id = ? AND read = 0",
            (athlete_id,),
        )
        affected = cur.execute("SELECT changes()").fetchone()
    count = int(affected[0]) if affected else 0
    return {"ok": True, "marked": count}


@router.post("/notifications/generate/{athlete_id}")
async def generate_notifications(athlete_id: str):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    sport = (athlete.get("sport") or "training").replace("_", " ")
    recent_types = _recent_notif_types(athlete_id, hours=24)
    created = []

    sessions_7d = _athlete_sessions(athlete_id, 7)
    sessions_28d = _athlete_sessions(athlete_id, 28)
    risk = _compute_injury_risk(athlete_id, 7)

    # Streak risk: trained 3+ days in row previously but 0 sessions in last 2 days
    if "streak_risk" not in recent_types:
        sessions_2d = _athlete_sessions(athlete_id, 2)
        sessions_5d = _athlete_sessions(athlete_id, 5)
        if len(sessions_5d) >= 3 and len(sessions_2d) == 0:
            created.append(
                _insert_notif(
                    athlete_id,
                    "streak_risk",
                    "Your training streak is at risk",
                    f"You trained {len(sessions_5d)} times in the last 5 days but nothing in the last 2. A short session keeps the streak alive.",
                )
            )

    # Personal best: check if latest session has a PB
    if "personal_best" not in recent_types and sessions_7d:
        latest = max(sessions_7d, key=lambda s: s.get("started_at", ""))
        latest_score = latest.get("avg_form_score") or latest.get("summary", {}).get("avg_form_score", 0) or 0
        if latest_score > 0 and sessions_28d:
            prev_best = max(
                (
                    s.get("avg_form_score") or s.get("summary", {}).get("avg_form_score", 0) or 0
                    for s in sessions_28d
                    if s.get("session_id") != latest.get("session_id")
                ),
                default=0,
            )
            if latest_score > prev_best and prev_best > 0:
                created.append(
                    _insert_notif(
                        athlete_id,
                        "personal_best",
                        f"New personal best: {latest_score:.1f}",
                        f"Your latest {sport} session scored {latest_score:.1f}, beating your previous best of {prev_best:.1f}.",
                    )
                )

    # Milestone: session count hits 10, 25, 50, 100
    if "milestone" not in recent_types:
        total_sessions = sum(
            1 for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"
        )
        for threshold in [10, 25, 50, 100, 200, 500]:
            if total_sessions >= threshold and total_sessions < threshold + 3:
                created.append(
                    _insert_notif(
                        athlete_id,
                        "milestone",
                        f"{threshold} sessions completed",
                        f"You've completed {total_sessions} {sport} sessions. Consistency builds champions.",
                    )
                )
                break

    # Injury warning: elevated risk
    if "injury_warning" not in recent_types:
        risk_level = risk.get("risk", "low")
        if risk_level in ("medium", "high"):
            created.append(
                _insert_notif(
                    athlete_id,
                    "injury_warning",
                    f"Injury risk: {risk_level}",
                    risk.get(
                        "reason", "Biomechanical patterns suggest elevated injury risk. Consider a recovery session."
                    ),
                )
            )

    # Re-engagement: no sessions in 7+ days
    if "reengage" not in recent_types and len(sessions_7d) == 0:
        created.append(
            _insert_notif(
                athlete_id,
                "reengage",
                "Time to get back to training",
                f"It's been over a week since your last {sport} session. Even a light session keeps you progressing.",
            )
        )

    return {"athlete_id": athlete_id, "generated": len(created), "notifications": created}
