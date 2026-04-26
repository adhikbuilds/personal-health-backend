from __future__ import annotations

"""
Coach ↔ Athlete 1:1 messaging.

Serves the coach-inbox.ejs "Messages" tab and the athlete-inbox.ejs reply flow.

  GET  /coach/{coach_id}/messages           — thread list with previews
  GET  /messages/{coach_id}/{athlete_id}    — messages in a thread
  POST /messages/{coach_id}/{athlete_id}    — send a message
"""

import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from database import ATHLETE_DB
from logging_setup import get_logger
from sqlite_store import cursor

router = APIRouter(tags=["Messaging"])
log = get_logger("routes.messaging")


def _ensure_table() -> None:
    with cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS coach_messages (
              id TEXT PRIMARY KEY,
              coach_id TEXT NOT NULL,
              athlete_id TEXT NOT NULL,
              sender TEXT NOT NULL,
              body TEXT NOT NULL,
              created_at REAL NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_thread ON coach_messages(coach_id, athlete_id, created_at DESC)")


_ensure_table()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@router.get("/coach/{coach_id}/messages")
async def list_threads(coach_id: str):
    with cursor() as cur:
        rows = cur.execute(
            """
            SELECT coach_id, athlete_id,
                   MAX(created_at) AS last_at,
                   COUNT(*) AS msg_count
            FROM coach_messages
            WHERE coach_id = ?
            GROUP BY athlete_id
            ORDER BY last_at DESC
            """,
            (coach_id,),
        ).fetchall()

    threads = []
    for r in rows:
        aid = r["athlete_id"]
        athlete = ATHLETE_DB.get(aid, {})
        with cursor() as cur:
            last = cur.execute(
                "SELECT body, sender FROM coach_messages WHERE coach_id = ? AND athlete_id = ? ORDER BY created_at DESC LIMIT 1",
                (coach_id, aid),
            ).fetchone()
        preview = (last["body"][:60] if last else "") if last else ""
        threads.append({
            "athlete_id": aid,
            "athlete_name": athlete.get("name", aid),
            "last_msg_preview": preview,
            "last_msg_at": _iso(r["last_at"]),
            "msg_count": r["msg_count"],
        })

    return {"coach_id": coach_id, "threads": threads}


@router.get("/messages/{coach_id}/{athlete_id}")
async def get_thread(
    coach_id: str,
    athlete_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    with cursor() as cur:
        rows = cur.execute(
            """
            SELECT * FROM coach_messages
            WHERE coach_id = ? AND athlete_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (coach_id, athlete_id, limit),
        ).fetchall()

    msgs = []
    for r in rows:
        msgs.append({
            "id": r["id"],
            "sender": r["sender"],
            "text": r["body"],
            "sent_at": _iso(r["created_at"]),
        })

    return {"coach_id": coach_id, "athlete_id": athlete_id, "msgs": msgs}


class SendMessage(BaseModel):
    sender: str = "coach"
    text: str


@router.post("/messages/{coach_id}/{athlete_id}")
async def send_message(coach_id: str, athlete_id: str, body: SendMessage):
    if not body.text.strip():
        raise HTTPException(400, "message body is required")

    msg_id = uuid4().hex[:12]
    now = time.time()
    with cursor() as cur:
        cur.execute(
            "INSERT INTO coach_messages(id, coach_id, athlete_id, sender, body, created_at) VALUES(?,?,?,?,?,?)",
            (msg_id, coach_id, athlete_id, body.sender, body.text.strip(), now),
        )

    return {
        "id": msg_id,
        "coach_id": coach_id,
        "athlete_id": athlete_id,
        "sender": body.sender,
        "text": body.text.strip(),
        "sent_at": _iso(now),
    }
