from __future__ import annotations

# Module-level task set prevents fire-and-forget tasks from being GC'd
_background_tasks: set = set()

"""
Notification system — in-app notifications for athletes.

Drives engagement by surfacing timely, relevant alerts:
  - streak at risk (trained yesterday but not today)
  - new coaching note available
  - personal best broken
  - injury risk elevated
  - weekly summary ready
  - achievement unlocked

Endpoints:
  GET  /athlete/{id}/notifications         — get unread notifications
  POST /athlete/{id}/notifications/read    — mark all as read
  POST /notifications/generate/{athlete_id} — generate pending notifications (called by cron or end_session)
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from auth import require_athlete_or_admin
from database import ATHLETE_DB, SESSION_DB, _load_json, _save_json
from logging_setup import get_logger


# Imported lazily inside async context to avoid circular imports
async def _push(athlete_id: str, title: str, body: str, data: dict | None = None) -> None:
    try:
        from routes.push_tokens import send_push

        await send_push(athlete_id, title, body, data)
    except Exception:
        pass


router = APIRouter(tags=["Notifications"])
log = get_logger("routes.notifications")

NOTIF_FILE = "notifications.json"


def _load_notifs() -> dict:
    return _load_json(NOTIF_FILE)


def _save_notifs(data: dict):
    _save_json(NOTIF_FILE, data)


def _add_notif(athlete_id: str, notif_type: str, title: str, body: str, data: dict | None = None):
    """Add a notification for an athlete. Deduplicates by type+title within 24h."""
    notifs = _load_notifs()
    if athlete_id not in notifs:
        notifs[athlete_id] = []

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # dedup: dont send same type+title within 24h
    cutoff = (now - timedelta(hours=24)).isoformat()
    for existing in notifs[athlete_id]:
        if (
            existing.get("type") == notif_type
            and existing.get("title") == title
            and existing.get("created_at", "") > cutoff
        ):
            return  # already sent recently

    notifs[athlete_id].append(
        {
            "type": notif_type,
            "title": title,
            "body": body,
            "data": data or {},
            "read": False,
            "created_at": now_iso,
        }
    )

    # keep only last 50 notifications per athlete
    notifs[athlete_id] = notifs[athlete_id][-50:]
    _save_notifs(notifs)
    # NOTE: push delivery is the caller's responsibility — see the explicit
    # `await _push(...)` calls in generate_notifications. Keeping transport
    # out of this sync helper avoids double-firing and lets seed scripts +
    # tests call _add_notif without hitting an external push endpoint.


@router.get("/athlete/{athlete_id}/notifications")
async def get_notifications(
    athlete_id: str,
    unread_only: bool = False,
    _: dict = Depends(require_athlete_or_admin("athlete_id")),
):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    notifs = _load_notifs()
    athlete_notifs = notifs.get(athlete_id, [])

    if unread_only:
        athlete_notifs = [n for n in athlete_notifs if not n.get("read")]

    # most recent first
    athlete_notifs = list(reversed(athlete_notifs))

    return {
        "athlete_id": athlete_id,
        "notifications": athlete_notifs,
        "unread_count": sum(1 for n in athlete_notifs if not n.get("read")),
        "total": len(athlete_notifs),
    }


@router.post("/athlete/{athlete_id}/notifications/read")
async def mark_all_read(athlete_id: str, _: dict = Depends(require_athlete_or_admin("athlete_id"))):
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    notifs = _load_notifs()
    if athlete_id not in notifs:
        return {"status": "ok", "marked": 0}

    count = 0
    for n in notifs[athlete_id]:
        if not n.get("read"):
            n["read"] = True
            count += 1

    _save_notifs(notifs)
    return {"status": "ok", "marked": count}


@router.post("/notifications/generate/{athlete_id}")
async def generate_notifications(athlete_id: str, _: dict = Depends(require_athlete_or_admin("athlete_id"))):
    """
    Check conditions and create notifications for an athlete.
    Call this after end_session or on a daily cron.
    """
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, "athlete not found")

    athlete = ATHLETE_DB[athlete_id]
    generated = []

    # 1. streak at risk
    now = datetime.now(timezone.utc).date()
    yesterday = (now - timedelta(days=1)).isoformat()
    today_str = now.isoformat()
    trained_yesterday = False
    trained_today = False
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id or s.get("status") != "completed":
            continue
        started = s.get("started_at", "")[:10]
        if started == yesterday:
            trained_yesterday = True
        if started == today_str:
            trained_today = True

    if trained_yesterday and not trained_today:
        _add_notif(
            athlete_id,
            "streak_risk",
            "dont lose your streak",
            "you trained yesterday but not today. one session keeps it alive",
        )
        generated.append("streak_risk")

    # 2. check for personal best in most recent session
    athlete_sessions = sorted(
        [s for s in SESSION_DB.values() if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
        key=lambda s: s.get("started_at", ""),
    )
    if len(athlete_sessions) >= 2:
        latest = athlete_sessions[-1]
        latest_summary = latest.get("summary") or {}
        latest_form = float(latest_summary.get("peak_form_score") or 0)
        latest_jump = float(latest_summary.get("peak_jump_height_cm") or 0)

        prev_best_form = 0.0
        prev_best_jump = 0.0
        for s in athlete_sessions[:-1]:
            sm = s.get("summary") or {}
            pf = float(sm.get("peak_form_score") or 0)
            pj = float(sm.get("peak_jump_height_cm") or 0)
            if pf > prev_best_form:
                prev_best_form = pf
            if pj > prev_best_jump:
                prev_best_jump = pj

        if latest_form > prev_best_form and latest_form > 50:
            pb_body = f"form score {latest_form:.0f} beats your previous best of {prev_best_form:.0f}"
            _add_notif(
                athlete_id,
                "personal_best",
                "new personal best",
                pb_body,
                {"metric": "form_score", "value": latest_form},
            )
            await _push(athlete_id, "Personal Best!", pb_body, {"screen": "CoachInbox"})
            generated.append("pb_form")

        if latest_jump > prev_best_jump and latest_jump > 10:
            jump_body = f"{latest_jump:.1f}cm beats your previous best of {prev_best_jump:.1f}cm"
            _add_notif(
                athlete_id,
                "personal_best",
                "new jump record",
                jump_body,
                {"metric": "jump_height", "value": latest_jump},
            )
            await _push(athlete_id, "New Jump Record!", jump_body, {"screen": "CoachInbox"})
            generated.append("pb_jump")

    # 3. check if they hit a session milestone
    session_count = int(athlete.get("sessions", 0))
    milestones = [10, 25, 50, 100, 200, 500]
    for m in milestones:
        if session_count == m:
            _add_notif(
                athlete_id,
                "milestone",
                f"{m} sessions",
                f"you just completed your {m}th session. every one makes the model better for everyone",
            )
            generated.append(f"milestone_{m}")

    # 4. injury risk warning
    try:
        from routes.progress import _compute_injury_risk

        risk = _compute_injury_risk(athlete_id, 14)
        if risk.get("risk") == "high":
            _add_notif(
                athlete_id,
                "injury_warning",
                "injury risk elevated",
                risk.get("reason", "asymmetry detected. consider reducing volume"),
            )
            generated.append("injury_warning")
    except Exception:
        pass

    # 5. re-engagement — Flow 14 from BIOMECHANICS-ARCHITECT.md
    # Lapsed athletes: push ONCE between day 6 and day 13. Silent after day 14
    # (accept the churn; spam damages trust more than one lost user).
    # Copy is drill-specific, not guilt-trippy. "We miss you" is banned.
    last_trained = None
    for s in SESSION_DB.values():
        if s.get("athlete_id") != athlete_id or s.get("status") != "completed":
            continue
        started = s.get("started_at", "")[:10]
        if started and (last_trained is None or started > last_trained):
            last_trained = started

    if last_trained:
        try:
            last_dt = datetime.fromisoformat(last_trained).date()
            idle_days = (now - last_dt).days
        except ValueError:
            idle_days = 0

        if 6 <= idle_days <= 13:
            # Only push once per lapse window.
            existing = [
                n
                for n in _load_notifs().get(athlete_id, [])
                if n.get("type") == "reengage" and (n.get("data") or {}).get("lapse_anchor") == last_trained
            ]
            if not existing:
                sport = athlete.get("sport", "your sport")
                _add_notif(
                    athlete_id,
                    "reengage",
                    f"{idle_days} days since your last rep",
                    f"your {sport} drill from last week is queued up. 15 minutes, one drill, pick it up when you can.",
                    {"idle_days": idle_days, "lapse_anchor": last_trained},
                )
                generated.append("reengage")

    return {
        "athlete_id": athlete_id,
        "generated": generated,
        "count": len(generated),
    }
