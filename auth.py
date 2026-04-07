from __future__ import annotations

"""
Personal Health — auth helpers (bcrypt + JWT) and FastAPI dependencies.
"""

import time
import uuid
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from config import settings
from logging_setup import get_logger
from sqlite_store import (
    get_user_by_email, get_user_by_id, insert_user, is_refresh_valid,
    revoke_refresh, store_refresh, touch_login,
)

log = get_logger("auth")

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    log.warning("bcrypt not installed — auth disabled until requirements installed")

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    log.warning("pyjwt not installed — auth disabled until requirements installed")


REFRESH_TTL_SECONDS = 30 * 24 * 3600


# ─── Password hashing ───────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    if not BCRYPT_AVAILABLE:
        raise RuntimeError("bcrypt not installed")
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not BCRYPT_AVAILABLE:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ─── JWT ────────────────────────────────────────────────────────────────────


def _encode(payload: dict) -> str:
    if not JWT_AVAILABLE:
        raise RuntimeError("pyjwt not installed")
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode(token: str) -> dict:
    if not JWT_AVAILABLE:
        raise RuntimeError("pyjwt not installed")
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def issue_access_token(user_id: str, role: str) -> str:
    now = int(time.time())
    return _encode({
        "sub": user_id,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + settings.jwt_ttl_minutes * 60,
    })


def issue_refresh_token(user_id: str) -> tuple[str, str]:
    jti = uuid.uuid4().hex
    now = int(time.time())
    token = _encode({
        "sub": user_id,
        "jti": jti,
        "type": "refresh",
        "iat": now,
        "exp": now + REFRESH_TTL_SECONDS,
    })
    store_refresh(jti, user_id, REFRESH_TTL_SECONDS)
    return token, jti


def issue_token_pair(user_id: str, role: str) -> dict:
    access = issue_access_token(user_id, role)
    refresh, _jti = issue_refresh_token(user_id)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": settings.jwt_ttl_minutes * 60,
    }


def rotate_refresh(refresh_token: str) -> dict:
    try:
        payload = _decode(refresh_token)
    except Exception as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid refresh token: {e}") from e
    if payload.get("type") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong token type")
    jti = payload.get("jti")
    if not jti or not is_refresh_valid(jti):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "refresh revoked or expired")
    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")
    revoke_refresh(jti)  # single-use rotation
    return issue_token_pair(user["id"], user["role"])


def revoke(refresh_token: str) -> None:
    try:
        payload = _decode(refresh_token)
        if payload.get("jti"):
            revoke_refresh(payload["jti"])
    except Exception:
        pass


# ─── Registration / login ──────────────────────────────────────────────────


def register_user(email: str, password: str, name: str, athlete_id: Optional[str] = None) -> dict:
    email = email.lower().strip()
    if get_user_by_email(email):
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    if len(password) < 8:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "password must be ≥8 chars")
    user = {
        "id": f"u_{uuid.uuid4().hex[:12]}",
        "email": email,
        "name": name,
        "password_hash": hash_password(password),
        "role": "athlete",
        "athlete_id": athlete_id,
        "created_at": time.time(),
    }
    insert_user(user)
    log.info("user registered", extra={"user_id": user["id"], "email": email})
    return user


def login_user(email: str, password: str) -> dict:
    user = get_user_by_email(email.lower().strip())
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    touch_login(user["id"])
    return user


# ─── Dependencies ──────────────────────────────────────────────────────────


def _extract_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    token = _extract_token(authorization)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        payload = _decode(token)
    except Exception as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}") from e
    if payload.get("type") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong token type")
    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")
    return user


def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    token = _extract_token(authorization)
    if not token:
        return None
    try:
        payload = _decode(token)
        return get_user_by_id(payload["sub"])
    except Exception:
        return None


def require_role(*roles: str):
    def _dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient role")
        return user
    return _dep
