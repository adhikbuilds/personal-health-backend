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
import config  # noqa: E402

object.__setattr__(config.settings, "db_path", Path(_tmp))


@pytest.fixture(scope="session")
def app():
    # Import here so config patching above takes effect first.
    import api_server  # noqa: WPS433
    return api_server.app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
