from __future__ import annotations

"""
Personal Health — Auth routes (/auth/*).
"""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, EmailStr, Field

from auth import (
    current_user,
    issue_token_pair,
    login_user,
    register_user,
    revoke,
    rotate_refresh,
)
from logging_setup import get_logger
from sqlite_store import audit

router = APIRouter(prefix="/auth", tags=["Auth"])
log = get_logger("routes.auth")


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=8, max_length=128)
    athlete_id: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


def _ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rid(request: Request) -> str:
    return request.headers.get("X-Request-ID", "-")


@router.post("/register", status_code=201)
async def register(req: RegisterRequest, request: Request):
    user = register_user(req.email, req.password, req.name, req.athlete_id)

    # Auto-provision an Athlete record so the new user has a profile to read
    # from on first launch (HomeScreen, ProfileScreen, etc all expect an
    # athlete_id). If the caller passed an existing athlete_id we link to it
    # instead of creating a new one.
    if not user.get("athlete_id"):
        from database import ATHLETE_DB, _save_db
        from sqlite_store import update_user_athlete_id

        existing = sorted(
            int(k.split("_", 1)[1]) for k in ATHLETE_DB.keys()
            if k.startswith("athlete_") and k.split("_", 1)[1].isdigit()
        )
        next_n = (existing[-1] + 1) if existing else 1
        new_athlete_id = f"athlete_{next_n:02d}"
        initials = "".join(p[0].upper() for p in req.name.split() if p)[:2] or "AT"
        ATHLETE_DB[new_athlete_id] = {
            "id": new_athlete_id,
            "name": req.name,
            "avatar": initials,
            "sport": "vertical_jump",
            "tier": "Block",
            "bpi": 0,
            "sessions": 0,
            "rank": next_n + 1000,
        }
        _save_db()
        update_user_athlete_id(user["id"], new_athlete_id)
        user["athlete_id"] = new_athlete_id

    audit("user.register", user_id=user["id"], ip=_ip(request), request_id=_rid(request))
    tokens = issue_token_pair(user["id"], user["role"])
    return {
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "athlete_id": user.get("athlete_id"),
        },
        **tokens,
    }


@router.post("/login")
async def login(req: LoginRequest, request: Request):
    user = login_user(req.email, req.password)
    audit("user.login", user_id=user["id"], ip=_ip(request), request_id=_rid(request))
    tokens = issue_token_pair(user["id"], user["role"])
    return {
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "athlete_id": user.get("athlete_id"),
        },
        **tokens,
    }


@router.post("/refresh")
async def refresh(req: RefreshRequest):
    return rotate_refresh(req.refresh_token)


@router.post("/logout", status_code=204)
async def logout(req: RefreshRequest, request: Request):
    revoke(req.refresh_token)
    audit("user.logout", ip=_ip(request), request_id=_rid(request))
    return None


@router.get("/me")
async def me(user: dict = Depends(current_user)):
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "athlete_id": user.get("athlete_id"),
    }
