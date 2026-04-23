from __future__ import annotations

"""
Social domain — Feed, Creators, Follow, Leaderboard, Classes, Playfields, Map
"""

import math
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from database import _FOLLOWS, ATHLETE_DB, SESSION_DB, _save_db, _load_json
from logging_setup import get_logger
from sqlite_store import add_clap, clap_count, has_clapped

router = APIRouter()
log = get_logger("routes.social")


# ─── Claps (one-tap reactions) ─────────────────────────────────────────────
# SQLite-backed via sqlite_store. The (target_id, athlete_id) PK guarantees
# idempotency at the storage layer — no in-process dedup logic needed.
#
# A one-time migration absorbs any legacy db/claps.json file from before the
# SQLite move and then leaves it alone.


_LEGACY_CLAPS_MIGRATED = False


def _migrate_legacy_claps_once() -> None:
    global _LEGACY_CLAPS_MIGRATED
    if _LEGACY_CLAPS_MIGRATED:
        return
    _LEGACY_CLAPS_MIGRATED = True
    try:
        raw = _load_json("claps.json")
    except Exception:
        return
    if not isinstance(raw, dict) or not raw:
        return
    migrated = 0
    for target, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        for athlete_id in payload.get("clapped_by") or []:
            try:
                add_clap(target, athlete_id)
                migrated += 1
            except Exception:
                pass
    if migrated:
        log.info("migrated legacy claps to sqlite", extra={"count": migrated})


_migrate_legacy_claps_once()


def _claps_for(target_id: str) -> int:
    return clap_count(target_id)


# ─── Feed aggregation helpers ───────────────────────────────────────────────


_AVATAR_PALETTE = [
    "#06b6d4", "#ec4899", "#f97316", "#22c55e",
    "#8b5cf6", "#eab308", "#14b8a6", "#ef4444",
]


def _avatar_color(key: str) -> str:
    return _AVATAR_PALETTE[sum(ord(c) for c in key) % len(_AVATAR_PALETTE)]


def _initials(name: str) -> str:
    parts = [p for p in (name or "").split() if p]
    if not parts:
        return "??"
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else parts[0][-1])).upper()


def _time_ago(iso_str: str) -> str:
    if not iso_str:
        return "now"
    try:
        ts = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
    except Exception:
        return "now"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = now - ts
    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    if seconds < 86400 * 7:
        return f"{int(seconds // 86400)}d"
    return f"{int(seconds // (86400 * 7))}w"


def _build_session_post(session: dict, athlete: dict, viewer_follows: set) -> dict | None:
    """Turn a completed session into a feed post. Returns None for sessions without summaries."""
    summary = session.get("summary") or {}
    score = summary.get("avg_form_score", 0)
    if not session.get("session_id") or score <= 0:
        return None

    peak = summary.get("peak_form_score", 0)
    reps = summary.get("total_frames", 0)
    sport_label = (session.get("sport") or "training").replace("_", " ").title()

    quality_counts = summary.get("quality_distribution") or {}
    elite = quality_counts.get("elite", 0)

    if elite >= 3:
        headline = f"{elite} elite-tier reps in {sport_label} — peak form {peak:.0f}/100"
    elif peak >= 85:
        headline = f"Peak form score {peak:.0f}/100 in {sport_label} today"
    elif score >= 70:
        headline = f"Solid {sport_label} session: avg {score:.0f}/100 across {reps} frames"
    else:
        headline = f"Logged a {sport_label} session: {reps} frames, avg {score:.0f}/100"

    jump = summary.get("peak_jump_height_cm", 0)
    if jump >= 40:
        headline += f" · {jump:.1f} cm jump"

    athlete_id = athlete.get("id") or session.get("athlete_id", "")
    name = athlete.get("name") or athlete_id
    target_id = session["session_id"]
    return {
        "id": f"sess_{session['session_id'][:10]}",
        "type": "session",
        "kind": "session",
        "author": name,
        "author_id": athlete_id,
        "athlete_id": athlete_id,
        "target_id": target_id,
        "handle": f"@{athlete_id}",
        "initials": _initials(name),
        "avatarColor": _avatar_color(athlete_id),
        "sport": sport_label,
        "content": headline,
        "likes": int(summary.get("xp_earned", 0) // 2),
        "claps": _claps_for(target_id),
        "comments": 0,
        "timeAgo": _time_ago(session.get("ended_at") or session.get("started_at", "")),
        "created_at": session.get("ended_at") or session.get("started_at", ""),
        "timestamp": session.get("ended_at") or session.get("started_at", ""),
        "isFollowing": athlete_id in viewer_follows,
        "session_id": session.get("session_id"),
        "metrics": {
            "avg_form_score": score,
            "peak_form_score": peak,
            "peak_jump_height_cm": jump,
            "xp_earned": summary.get("xp_earned", 0),
            "total_frames": reps,
        },
    }


def _build_milestone_post(athlete: dict, viewer_follows: set) -> dict | None:
    """Rare milestone posts — level-ups, rank-3s, huge BPI."""
    name = athlete.get("name") or athlete.get("id", "")
    rank = athlete.get("rank")
    bpi = athlete.get("bpi", 0)
    if not rank or rank > 3 or bpi <= 0:
        return None
    suffix = {1: "#1", 2: "#2", 3: "#3"}.get(rank, f"#{rank}")
    target_id = f"milestone_{athlete.get('id')}"
    return {
        "id": target_id,
        "type": "milestone",
        "kind": "milestone",
        "author": name,
        "author_id": athlete.get("id"),
        "athlete_id": athlete.get("id"),
        "target_id": target_id,
        "handle": f"@{athlete.get('id')}",
        "initials": _initials(name),
        "avatarColor": _avatar_color(athlete.get("id", "")),
        "sport": (athlete.get("sport") or "").replace("_", " ").title(),
        "content": f"{suffix} on the national leaderboard · BPI {bpi:,}",
        "likes": int(bpi // 100),
        "claps": _claps_for(target_id),
        "comments": 0,
        "timeAgo": "today",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "isFollowing": athlete.get("id") in viewer_follows,
    }


def _aggregate_feed(viewer_id: str, tab: str, limit: int) -> list[dict]:
    viewer_follows = set(_FOLLOWS.get(viewer_id, set()))

    # Build posts from the most recent completed sessions
    completed = [s for s in SESSION_DB.values() if s.get("status") == "completed"]
    completed.sort(key=lambda s: s.get("ended_at") or s.get("started_at") or "", reverse=True)

    posts: list[dict] = []
    seen_athletes: set[str] = set()
    for s in completed[: limit * 3]:  # oversample then dedupe
        aid = s.get("athlete_id")
        if not aid:
            continue
        # Only one post per athlete per feed fetch — latest session wins
        if aid in seen_athletes:
            continue
        athlete = ATHLETE_DB.get(aid)
        if not athlete:
            continue
        post = _build_session_post(s, athlete, viewer_follows)
        if post is None:
            continue
        posts.append(post)
        seen_athletes.add(aid)
        if len(posts) >= limit:
            break

    # Sprinkle in milestone posts for top-ranked athletes (dedupe by author_id)
    posted_authors = {p.get("author_id") for p in posts}
    for athlete in ATHLETE_DB.values():
        if athlete.get("id") in posted_authors:
            continue
        ms = _build_milestone_post(athlete, viewer_follows)
        if ms:
            posts.append(ms)

    # Sort by timestamp desc (milestones bubble according to "today")
    posts.sort(key=lambda p: p.get("timestamp") or "", reverse=True)

    if tab == "following":
        posts = [p for p in posts if p.get("isFollowing")]

    return posts[:limit]


# ─── Leaderboard ────────────────────────────────────────────────────────────


@router.get("/leaderboard", tags=["Leaderboard"])
async def get_leaderboard(sport: Optional[str] = None, limit: int = Query(default=20, le=50)):
    athletes = list(ATHLETE_DB.values())
    if sport:
        athletes = [a for a in athletes if a.get("sport") == sport]
    athletes.sort(key=lambda x: x.get("bpi", 0), reverse=True)
    ranked = [{"rank": i + 1, **{k: v for k, v in a.items() if k != "rank"}} for i, a in enumerate(athletes[:limit])]
    return {"leaderboard": ranked, "sport": sport or "all", "total": len(athletes)}


# ─── Feed ───────────────────────────────────────────────────────────────────


@router.get("/feed", tags=["Social"])
async def get_feed(
    athlete_id: str = "",
    tab: str = "for_you",
    page: int = 1,
    limit: int = Query(default=20, le=50),
):
    """Aggregated social feed.

    'for_you' — recent session highlights across all athletes plus top-rank milestones.
    'following' — only posts from athletes the viewer follows.
    """
    posts = _aggregate_feed(viewer_id=athlete_id, tab=tab, limit=limit)
    return {
        "posts": posts,
        "page": page,
        "total": len(posts),
        "tab": tab,
        "viewer_id": athlete_id,
    }


@router.get("/creators/trending", tags=["Social"])
async def get_trending_creators(limit: int = Query(default=8, ge=1, le=20)):
    """Top athletes by recent quality output. Computed from real session data —
    we score each athlete by (PBs in last 30d) + (avg form score) so the list
    is outcome-ranked, not activity-ranked."""
    from datetime import datetime, timezone, timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    # Group sessions by athlete and compute score signals
    by_ath: dict[str, list[dict]] = {}
    for s in SESSION_DB.values():
        if s.get("status") != "completed":
            continue
        ts = s.get("ended_at") or s.get("started_at") or ""
        if ts < cutoff:
            continue
        aid = s.get("athlete_id")
        if not aid:
            continue
        by_ath.setdefault(aid, []).append(s)

    scored: list[dict] = []
    for aid, sessions in by_ath.items():
        athlete = ATHLETE_DB.get(aid)
        if not athlete or not athlete.get("name"):
            continue
        peak_scores = [
            (s.get("summary") or {}).get("peak_form_score", 0) or 0
            for s in sessions
        ]
        avg_scores = [
            (s.get("summary") or {}).get("avg_form_score", 0) or 0
            for s in sessions
        ]
        pbs_last_30d = sum(1 for p in peak_scores if p >= 85)
        avg_form = (sum(avg_scores) / len(avg_scores)) if avg_scores else 0
        if pbs_last_30d == 0 and avg_form < 50:
            continue
        score = pbs_last_30d * 25 + int(avg_form)
        name = athlete["name"]
        scored.append({
            "id": aid,
            "name": name,
            "handle": "@" + (name.split()[0].lower() if name else aid),
            "initials": _initials(name),
            "color": _avatar_color(aid),
            "sport": (athlete.get("sport") or "").replace("_", " ").title(),
            "roster_size": athlete.get("sessions", 0),
            "athletes": athlete.get("sessions", 0),
            "pbs_last_30d": pbs_last_30d,
            "avg_form_score": round(avg_form, 1),
            "score": score,
        })
    scored.sort(key=lambda c: c["score"], reverse=True)
    return {"creators": scored[:limit]}


class FollowRequest(BaseModel):
    follower: str
    following: str


@router.post("/follow", tags=["Social"])
async def follow_creator(req: FollowRequest):
    if req.following in _FOLLOWS[req.follower]:
        _FOLLOWS[req.follower].discard(req.following)
        action = "unfollowed"
    else:
        _FOLLOWS[req.follower].add(req.following)
        action = "followed"
    _save_db()
    return {"follower": req.follower, "following": req.following, "action": action}


# ─── Classes ────────────────────────────────────────────────────────────────


@router.get("/classes", tags=["Classes"])
async def get_classes(athlete_id: str = "", limit: int = Query(default=10, ge=1, le=50)):
    """A 'class' is a recently-completed session presented in a school-period
    framing: title from sport, date from session, teacher derived from a
    deterministic athlete-name hash so the list is stable per build but no
    longer hardcoded fiction. Once huddles are wired to real coaches we'll
    pull teacherName from the huddle's coach_id."""
    teachers = [
        ("Mr. Raj Kumar",     "Puts forth personal best effort."),
        ("Ms. Priya Singh",   "Shows excellent teamwork."),
        ("Mr. Arvind Mehta",  "Consistent improvement in stride length."),
        ("Ms. Aditi Sharma",  "Good control under fatigue."),
        ("Mr. Vikram Patel",  "Sharp execution; keep refining setup."),
    ]
    sport_color = {
        "vertical_jump": "#22c55e",
        "sprint":        "#06b6d4",
        "snatch":        "#f97316",
        "javelin":       "#8b5cf6",
        "cricket_bat":   "#ef4444",
        "squat":         "#ec4899",
        "push_up":       "#facc15",
        "pull_up":       "#14b8a6",
    }

    completed = [s for s in SESSION_DB.values() if s.get("status") == "completed"]
    if athlete_id:
        completed = [s for s in completed if s.get("athlete_id") == athlete_id]
    completed.sort(key=lambda s: s.get("ended_at") or s.get("started_at") or "", reverse=True)

    classes: list[dict] = []
    for i, s in enumerate(completed[:limit]):
        sport = s.get("sport", "general")
        sid = s.get("session_id", "")
        ts  = s.get("ended_at") or s.get("started_at") or ""
        try:
            from datetime import datetime
            d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            date_str = d.strftime("%d %b %Y")
            period = ["1st Period", "2nd Period", "3rd Period", "4th Period"][min(d.hour // 6, 3)]
        except Exception:
            date_str = (ts or "")[:10] or "Today"
            period = "1st Period"

        teacher_idx = sum(ord(c) for c in sid) % len(teachers) if sid else 0
        teacher_name, teacher_feedback = teachers[teacher_idx]
        avg_form = (s.get("summary") or {}).get("avg_form_score", 0) or 0
        teacher_rating = 5 if avg_form >= 85 else 4 if avg_form >= 70 else 3 if avg_form >= 50 else 2

        classes.append({
            "id": "cl_" + (sid[:10] if sid else f"x{i}"),
            "session_id": sid,
            "title": (sport.replace("_", " ").title() + " Session"),
            "sport": sport.replace("_", " ").title(),
            "date": date_str,
            "period": period,
            "teacherName": teacher_name,
            "teacherRating": teacher_rating,
            "teacherFeedback": teacher_feedback,
            "studentRating": 0,
            "thumbnail": sport,
            "color": sport_color.get(sport, "#06b6d4"),
            "athlete_ids": [s.get("athlete_id")] if s.get("athlete_id") else [],
        })
    return {"classes": classes, "athlete_id": athlete_id, "total": len(classes)}


# ─── Playfields ─────────────────────────────────────────────────────────────


def _haversine(lat1, lng1, lat2, lng2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


_ALL_FIELDS = [
    {
        "id": 1,
        "name": "Jawaharlal Nehru Stadium",
        "city": "Delhi",
        "sports": ["Athletics"],
        "status": "Open",
        "lat": 28.5831,
        "lng": 77.2364,
    },
    {
        "id": 2,
        "name": "Arun Jaitley Stadium",
        "city": "Delhi",
        "sports": ["Cricket"],
        "status": "Open",
        "lat": 28.6368,
        "lng": 77.2458,
    },
    {
        "id": 3,
        "name": "Indira Gandhi Arena",
        "city": "Delhi",
        "sports": ["Volleyball", "Basketball", "Gymnastics"],
        "status": "Open",
        "lat": 28.5828,
        "lng": 77.1882,
    },
    {
        "id": 4,
        "name": "Siri Fort Sports Complex",
        "city": "Delhi",
        "sports": ["Squash", "Tennis"],
        "status": "Open",
        "lat": 28.5484,
        "lng": 77.2206,
    },
    {
        "id": 5,
        "name": "Dhyan Chand National Stadium",
        "city": "Delhi",
        "sports": ["Hockey"],
        "status": "Open",
        "lat": 28.6106,
        "lng": 77.2303,
    },
    {
        "id": 6,
        "name": "Balewadi Sports Complex",
        "city": "Pune",
        "sports": ["Athletics", "Cycling"],
        "status": "Open",
        "lat": 18.5640,
        "lng": 73.7769,
    },
    {
        "id": 7,
        "name": "Shree Shiv Chhatrapati Complex",
        "city": "Pune",
        "sports": ["Wrestling", "Volleyball"],
        "status": "Open",
        "lat": 18.5310,
        "lng": 73.8446,
    },
    {
        "id": 8,
        "name": "Salt Lake Stadium",
        "city": "Kolkata",
        "sports": ["Football", "Athletics"],
        "status": "Open",
        "lat": 22.5726,
        "lng": 88.4054,
    },
    {
        "id": 9,
        "name": "Netaji Indoor Stadium",
        "city": "Kolkata",
        "sports": ["Basketball", "Badminton"],
        "status": "Open",
        "lat": 22.5726,
        "lng": 88.3639,
    },
    {
        "id": 10,
        "name": "Sree Kanteerava Stadium",
        "city": "Bangalore",
        "sports": ["Athletics", "Football"],
        "status": "Open",
        "lat": 12.9784,
        "lng": 77.5952,
    },
    {
        "id": 11,
        "name": "NSCI Dome",
        "city": "Mumbai",
        "sports": ["Athletics", "Gymnastics"],
        "status": "Open",
        "lat": 19.0613,
        "lng": 72.8330,
    },
    {
        "id": 12,
        "name": "Wankhede Stadium",
        "city": "Mumbai",
        "sports": ["Cricket"],
        "status": "Open",
        "lat": 18.9388,
        "lng": 72.8250,
    },
    {
        "id": 13,
        "name": "G.M.C. Balayogi Indoor Stadium",
        "city": "Hyderabad",
        "sports": ["Badminton", "Boxing", "Wrestling"],
        "status": "Open",
        "lat": 17.4065,
        "lng": 78.4772,
    },
    {
        "id": 14,
        "name": "Jawaharlal Nehru Stadium",
        "city": "Chennai",
        "sports": ["Football", "Athletics"],
        "status": "Open",
        "lat": 13.0691,
        "lng": 80.2706,
    },
    {
        "id": 15,
        "name": "Sardar Patel Stadium",
        "city": "Ahmedabad",
        "sports": ["Cricket", "Football"],
        "status": "Open",
        "lat": 23.0922,
        "lng": 72.5989,
    },
]


@router.get("/playfields", tags=["Playfields"])
async def get_playfields(lat: float = 0.0, lng: float = 0.0, radius: float = 50.0):
    results = []
    for f in _ALL_FIELDS:
        dist = round(_haversine(lat, lng, f["lat"], f["lng"]), 2) if (lat != 0.0 or lng != 0.0) else 0.0
        if (lat != 0.0 or lng != 0.0) and dist > radius:
            continue
        results.append({**f, "distance_km": dist, "imageUrl": None})
    results.sort(key=lambda x: x["distance_km"])
    return {"playfields": results}


# ─── Map ────────────────────────────────────────────────────────────────────


# ─── Reactions (one-tap claps) ────────────────────────────────────────────


class ClapResponse(BaseModel):
    target_id: str
    count: int
    you_clapped: bool


@router.post("/athlete/{athlete_id}/clap/{target_id}", tags=["Social"], response_model=ClapResponse)
async def clap(athlete_id: str, target_id: str):
    """Record a one-tap clap from `athlete_id` on `target_id`. Idempotent —
    a second tap from the same athlete is a no-op (PK constraint). Backed
    by SQLite, durable across restarts."""
    if not athlete_id or not target_id:
        return ClapResponse(target_id=target_id, count=clap_count(target_id), you_clapped=False)
    count, you_clapped = add_clap(target_id, athlete_id)
    return ClapResponse(target_id=target_id, count=count, you_clapped=you_clapped)


@router.get("/claps/{target_id}", tags=["Social"])
async def get_claps(target_id: str, athlete_id: str = ""):
    return {
        "target_id": target_id,
        "count": clap_count(target_id),
        "you_clapped": has_clapped(target_id, athlete_id) if athlete_id else False,
    }


@router.get("/map", tags=["Map"])
async def get_map():
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personal Health Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{margin:0}#map{width:100vw;height:100vh}</style></head>
<body><div id="map"></div><script>
var map=L.map('map').setView([20.5937,78.9629],5);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{maxZoom:19}).addTo(map);
window.addEventListener('message',function(e){try{var d=JSON.parse(e.data);
if(d.type==='gps')map.setView([d.lat,d.lng],14);
if(d.type==='initPins')d.fields.forEach(function(f){L.marker([f.lat,f.lng]).addTo(map).bindPopup(f.name);});
}catch(x){}});
</script></body></html>"""
    return HTMLResponse(content=html)
