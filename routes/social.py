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

from database import _FOLLOWS, ATHLETE_DB, SESSION_DB, _save_db
from logging_setup import get_logger

router = APIRouter()
log = get_logger("routes.social")


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
    return {
        "id": f"sess_{session['session_id'][:10]}",
        "kind": "session",
        "author": name,
        "author_id": athlete_id,
        "handle": f"@{athlete_id}",
        "initials": _initials(name),
        "avatarColor": _avatar_color(athlete_id),
        "sport": sport_label,
        "content": headline,
        "likes": int(summary.get("xp_earned", 0) // 2),
        "comments": 0,
        "timeAgo": _time_ago(session.get("ended_at") or session.get("started_at", "")),
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
    return {
        "id": f"milestone_{athlete.get('id')}",
        "kind": "milestone",
        "author": name,
        "author_id": athlete.get("id"),
        "handle": f"@{athlete.get('id')}",
        "initials": _initials(name),
        "avatarColor": _avatar_color(athlete.get("id", "")),
        "sport": (athlete.get("sport") or "").replace("_", " ").title(),
        "content": f"{suffix} on the national leaderboard · BPI {bpi:,}",
        "likes": int(bpi // 100),
        "comments": 0,
        "timeAgo": "today",
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
async def get_trending_creators():
    return {
        "creators": [
            {"id": "c1", "name": "Fit India Icons", "handle": "@FitIndiaIcons", "initials": "FI", "color": "#f97316"},
            {
                "id": "c2",
                "name": "Fit India Champions",
                "handle": "@FitChampions",
                "initials": "FC",
                "color": "#22c55e",
            },
            {
                "id": "c3",
                "name": "Fit India Ambassadors",
                "handle": "@FitAmbassadors",
                "initials": "FA",
                "color": "#8b5cf6",
            },
            {"id": "c4", "name": "Rishi Arora", "handle": "@RishiArora", "initials": "RA", "color": "#06b6d4"},
            {"id": "c5", "name": "Aditi Dixit", "handle": "@AditiDixit", "initials": "AD", "color": "#ec4899"},
        ]
    }


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
async def get_classes(athlete_id: str = ""):
    all_classes = [
        {
            "id": "cl1",
            "title": "3 V 3 Bounce Ball",
            "sport": "Basketball",
            "date": "19 May 2024",
            "period": "3rd Period",
            "teacherName": "Mr. Raj Kumar",
            "teacherRating": 5,
            "teacherFeedback": "Puts forth personal best effort.",
            "studentRating": 0,
            "thumbnail": "basketball",
            "color": "#f97316",
            "athlete_ids": [],
        },
        {
            "id": "cl2",
            "title": "Kabaddi Fundamentals",
            "sport": "Kabaddi",
            "date": "15 May 2024",
            "period": "2nd Period",
            "teacherName": "Ms. Priya Singh",
            "teacherRating": 4,
            "teacherFeedback": "Shows excellent teamwork.",
            "studentRating": 4,
            "thumbnail": "kabaddi",
            "color": "#ef4444",
            "athlete_ids": [],
        },
        {
            "id": "cl3",
            "title": "100m Sprint Drills",
            "sport": "Athletics",
            "date": "12 May 2024",
            "period": "1st Period",
            "teacherName": "Mr. Arvind Mehta",
            "teacherRating": 5,
            "teacherFeedback": "Consistent improvement in stride length.",
            "studentRating": 5,
            "thumbnail": "athletics",
            "color": "#22c55e",
            "athlete_ids": [],
        },
    ]
    return {"classes": all_classes, "athlete_id": athlete_id}


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
