from __future__ import annotations

"""
Social domain — Feed, Creators, Follow, Leaderboard, Classes, Playfields, Map
"""

import math
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from database import _FOLLOWS, ATHLETE_DB, _save_db

router = APIRouter()


# ─── Leaderboard ────────────────────────────────────────────────────────────


@router.get("/leaderboard", tags=["Leaderboard"])
async def get_leaderboard(
    sport: Optional[str] = None,
    limit: int = Query(default=20, le=50),
    athlete_id: Optional[str] = None,
    bucket_size: int = Query(default=8, ge=4, le=20),
):
    """
    Public leaderboard. When athlete_id is provided, returns the athlete's
    bucketed view — their nearby peers only (TRAINER-FIRST-PROMPT anti-pattern:
    athlete #47 of 50 must NOT see the full list; they see their bucket of 8).
    """
    athletes = list(ATHLETE_DB.values())
    if sport:
        athletes = [a for a in athletes if a.get("sport") == sport]
    athletes.sort(key=lambda x: x.get("bpi", 0), reverse=True)

    full_ranked = [{"rank": i + 1, **{k: v for k, v in a.items() if k != "rank"}} for i, a in enumerate(athletes)]

    if athlete_id:
        # Find athlete's position, then return a window of bucket_size centred on them
        athlete_pos = next((i for i, a in enumerate(full_ranked) if a.get("id") == athlete_id), None)
        if athlete_pos is not None:
            half = bucket_size // 2
            start = max(0, athlete_pos - half)
            end = min(len(full_ranked), start + bucket_size)
            start = max(0, end - bucket_size)  # re-anchor if near bottom
            bucket = full_ranked[start:end]
            return {
                "leaderboard": bucket,
                "sport": sport or "all",
                "total": len(athletes),
                "athlete_rank": athlete_pos + 1,
                "bucket_start": start + 1,
                "bucket_end": end,
                "bucketed": True,
            }

    return {
        "leaderboard": full_ranked[:limit],
        "sport": sport or "all",
        "total": len(athletes),
        "bucketed": False,
    }


# ─── Feed ───────────────────────────────────────────────────────────────────


@router.get("/feed", tags=["Social"])
async def get_feed(athlete_id: str = "", tab: str = "for_you", page: int = 1):
    posts = [
        {
            "id": "p1",
            "author": "Rishi Arora",
            "handle": "@RishiArora",
            "initials": "RA",
            "avatarColor": "#06b6d4",
            "sport": "Athletics",
            "content": "Just hit a new PB in the 400m! Hard work finally paying off",
            "likes": 142,
            "comments": 18,
            "timeAgo": "2h",
            "isFollowing": False,
        },
        {
            "id": "p2",
            "author": "Aditi Dixit",
            "handle": "@AditiDixit",
            "initials": "AD",
            "avatarColor": "#ec4899",
            "sport": "Yoga",
            "content": "Morning session complete. Pranayama + 45 min flow.",
            "likes": 287,
            "comments": 34,
            "timeAgo": "4h",
            "isFollowing": True,
        },
        {
            "id": "p3",
            "author": "Moh. Usman",
            "handle": "@MohUsman",
            "initials": "MU",
            "avatarColor": "#f97316",
            "sport": "Kabaddi",
            "content": "District championships next week! Training twice a day.",
            "likes": 98,
            "comments": 22,
            "timeAgo": "6h",
            "isFollowing": False,
        },
        {
            "id": "p4",
            "author": "Fit India Icons",
            "handle": "@FitIndiaIcons",
            "initials": "FI",
            "avatarColor": "#22c55e",
            "sport": "National Program",
            "content": "Congratulations to all athletes who completed the FitIndiaSchoolWeek! 10,000+ schools participated.",
            "likes": 1450,
            "comments": 203,
            "timeAgo": "1d",
            "isFollowing": True,
        },
    ]
    if tab == "following":
        followed_ids = _FOLLOWS.get(athlete_id, set())
        _creator_handles = {
            c["id"]: c["handle"]
            for c in [
                {"id": "c1", "handle": "@FitIndiaIcons"},
                {"id": "c2", "handle": "@FitChampions"},
                {"id": "c3", "handle": "@FitAmbassadors"},
                {"id": "c4", "handle": "@RishiArora"},
                {"id": "c5", "handle": "@AditiDixit"},
            ]
        }
        followed_handles = {_creator_handles[cid] for cid in followed_ids if cid in _creator_handles}
        posts = [p for p in posts if p.get("isFollowing") or p.get("handle") in followed_handles]
    return {"posts": posts, "page": page, "total": len(posts)}


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
