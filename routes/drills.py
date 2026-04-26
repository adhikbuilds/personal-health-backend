"""
Drill library catalog — form videos and cues for pre-session education.

Endpoints:
  GET /drills/catalog  — returns all drills with video URLs and cues
"""

from fastapi import APIRouter

from database import _load_json

router = APIRouter(tags=["Drills"])


@router.get("/drills/catalog")
async def get_drills_catalog():
    """
    Returns the complete drill catalog with video URLs, thumbnails, and form cues.
    Used by Android and web to populate drill pickers and onboarding flow.
    """
    data = _load_json("db/drills.json")
    return {"drills": data.get("drills", [])}
