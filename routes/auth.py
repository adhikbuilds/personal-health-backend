from __future__ import annotations

"""
Personal Health — Auth routes (/auth/*).
"""

from typing import Optional

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
from sqlite_store import audit, delete_user, set_user_consent

router = APIRouter(prefix="/auth", tags=["Auth"])
log = get_logger("routes.auth")


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=8, max_length=128)
    athlete_id: Optional[str] = None


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


class DisclaimerRequest(BaseModel):
    key: str = Field(pattern=r"^[a-z_]{3,40}$", description="e.g. injury_disclaimer, dpdp_notice")
    accepted: bool = True


@router.post("/accept-disclaimer", status_code=200)
async def accept_disclaimer(req: DisclaimerRequest, request: Request, user: dict = Depends(current_user)):
    """
    Record that the user accepted a named disclaimer/consent.
    Used for injury-liability acknowledgement and DPDP privacy-notice acceptance.
    """
    set_user_consent(user["id"], req.key, req.accepted)
    audit(
        "user.consent",
        user_id=user["id"],
        key=req.key,
        accepted=req.accepted,
        ip=_ip(request),
        request_id=_rid(request),
    )
    return {"ok": True, "key": req.key, "accepted": req.accepted}


@router.delete("/account", status_code=200)
async def delete_account(request: Request, user: dict = Depends(current_user)):
    """
    Hard-delete the caller's account. DPDP right-to-delete.

    Removes the user row + all refresh tokens immediately. Linked athlete_id
    record in ATHLETE_DB is NOT deleted here (it may be coach-owned or have
    referential integrity implications) — surfaced for follow-up in an async
    purge job. The user can no longer authenticate after this endpoint returns.
    """
    user_id = user["id"]
    delete_user(user_id)
    audit("user.account_deleted", user_id=user_id, ip=_ip(request), request_id=_rid(request))
    log.info("account deleted user=%s", user_id)
    return {"ok": True, "deleted_user_id": user_id}
