from __future__ import annotations

"""
Shared pytest fixtures for the Personal Health backend.

We pin the SQLite users db to a per-test-session temp file via env vars
*before* importing the app, so we never touch the dev database.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Use an isolated DB dir for the test session.
_tmp = tempfile.mkdtemp(prefix="ph-test-")
os.environ["JWT_SECRET"] = "test-secret-do-not-use-in-prod-must-be-32+bytes-long-xx"
os.environ["RATE_LIMIT_PER_MINUTE"] = "6000"  # don't trip limiter in tests
os.environ["RATE_LIMIT_BURST"] = "6000"
os.environ["ENV"] = "dev"
os.environ.pop("ANTHROPIC_API_KEY", None)  # force fallback path

# Point sqlite at temp dir by monkey-patching settings *after* import
import config

object.__setattr__(config.settings, "db_path", Path(_tmp))


@pytest.fixture(scope="session")
def app():
    # Import here so config patching above takes effect first.
    import api_server

    return api_server.app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def authed(client):
    """Returns ({athlete_id, user_id, access_token, headers}, client) — a
    fresh user registered + signed in. Tests that hit ownership-gated
    endpoints can use `authed['headers']` for the Authorization header
    or `authed['athlete_id']` to identify the right resource owner."""
    import uuid

    email = f"t_{uuid.uuid4().hex[:10]}@example.com"
    r = client.post(
        "/auth/register",
        json={"email": email, "name": "Test User", "password": "Sup3rsecret!"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return {
        "email": email,
        "user_id": body["user"]["id"],
        "athlete_id": body["user"]["athlete_id"],
        "access_token": body["access_token"],
        "refresh_token": body["refresh_token"],
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


@pytest.fixture()
def admin_client(client):
    """Returns ({headers, ...}) for an admin-role user. We mint by
    inserting directly into the DB then minting tokens."""
    import time
    import uuid as _uuid

    from auth import hash_password, issue_token_pair
    from sqlite_store import insert_user

    user_id = "admin_" + _uuid.uuid4().hex[:8]
    email = f"adm_{_uuid.uuid4().hex[:8]}@example.com"
    insert_user(
        {
            "id": user_id,
            "email": email,
            "name": "Test Admin",
            "password_hash": hash_password("Sup3rsecret!"),
            "role": "admin",
            "athlete_id": None,
            "created_at": time.time(),
        }
    )
    tokens = issue_token_pair(user_id, "admin")
    return {
        "user_id": user_id,
        "headers": {"Authorization": f"Bearer {tokens['access_token']}"},
    }
