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
def _fastapi_app():
    # Import here so config patching above takes effect first.
    import api_server

    return api_server.app


@pytest.fixture()
def anonymous_client(_fastapi_app):
    """TestClient with no auth headers. Use when asserting 401s."""
    from fastapi.testclient import TestClient

    with TestClient(_fastapi_app) as c:
        yield c


@pytest.fixture()
def client(_fastapi_app):
    """
    Default test client — authenticated as an admin so existing tests that
    predate auth enforcement continue to work. For anonymous-only tests use
    the `anonymous_client` fixture.
    """
    import sqlite_store
    from auth import issue_access_token
    from fastapi.testclient import TestClient

    with TestClient(_fastapi_app) as c:
        # insert admin AFTER lifespan startup has run init_db()
        try:
            sqlite_store.insert_user(
                {
                    "id": "test-admin-id",
                    "email": "admin@test.local",
                    "name": "Test Admin",
                    "password_hash": "x",
                    "role": "admin",
                    "athlete_id": None,
                }
            )
        except Exception:
            pass  # already exists from an earlier test in the session
        token = issue_access_token("test-admin-id", "admin")
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c


@pytest.fixture()
def admin_token(_fastapi_app):
    """Mint an admin JWT for tests that need to bypass self-only auth checks."""
    import sqlite_store
    from auth import issue_access_token

    try:
        sqlite_store.insert_user(
            {
                "id": "test-admin-id",
                "email": "admin@test.local",
                "name": "Test Admin",
                "password_hash": "x",
                "role": "admin",
                "athlete_id": None,
            }
        )
    except Exception:
        pass
    return issue_access_token("test-admin-id", "admin")


@pytest.fixture()
def admin_client(_fastapi_app, admin_token):
    """A TestClient pre-configured with an admin bearer token."""
    from fastapi.testclient import TestClient

    with TestClient(_fastapi_app) as c:
        c.headers.update({"Authorization": f"Bearer {admin_token}"})
        yield c


@pytest.fixture()
def athlete_client(_fastapi_app):
    """
    A TestClient authenticated as a specific athlete. Use when tests need
    to assert *self-scope* behavior (e.g. "athlete A cannot read athlete B").

    Returns a factory that takes an athlete_id and gives back a TestClient
    whose token asserts that athlete_id.
    """
    from auth import issue_access_token
    from fastapi.testclient import TestClient

    def _make(athlete_id: str):
        # Inject a synthetic user row so current_user's DB lookup succeeds
        import sqlite_store

        try:
            sqlite_store.insert_user(
                {
                    "id": f"u-{athlete_id}",
                    "email": f"{athlete_id}@test.local",
                    "name": athlete_id,
                    "password_hash": "x",
                    "role": "athlete",
                    "athlete_id": athlete_id,
                }
            )
        except Exception:
            pass  # already exists from a previous test
        token = issue_access_token(f"u-{athlete_id}", "athlete")
        c = TestClient(_fastapi_app)
        c.headers.update({"Authorization": f"Bearer {token}"})
        return c

    return _make
