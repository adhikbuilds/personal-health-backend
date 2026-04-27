from __future__ import annotations

"""
Tests for the new analytics endpoints (rep counting + competition readiness).
"""

import uuid

import pytest

from routes.analytics import _count_reps_from_frames, _readiness_band

# ─── Pure-function unit tests (no app fixture needed) ──────────────────────


def _frame(knee_l: float, knee_r: float | None = None) -> dict:
    return {"knee_angle_l": knee_l, "knee_angle_r": knee_r if knee_r is not None else knee_l}


def test_rep_counter_counts_three_full_squat_cycles():
    # 3 down/up cycles for vertical_jump (knee_angle joint)
    frames = []
    for _ in range(3):
        frames += [_frame(170), _frame(140), _frame(110), _frame(140), _frame(170)]
    out = _count_reps_from_frames(frames, "vertical_jump")
    assert out["supported"] is True
    assert out["rep_count"] == 3
    assert out["samples"] == 15


def test_rep_counter_ignores_partial_dip():
    # Dips below 130 but never crosses back above 160 → not a rep
    frames = [_frame(170), _frame(125), _frame(155)]
    out = _count_reps_from_frames(frames, "vertical_jump")
    assert out["rep_count"] == 0


def test_rep_counter_unsupported_sport():
    out = _count_reps_from_frames([_frame(170)], "javelin")
    assert out["supported"] is False
    assert out["rep_count"] == 0


def test_rep_counter_skips_missing_metric():
    frames = [{"foo": "bar"}, _frame(170), _frame(110), _frame(170)]
    out = _count_reps_from_frames(frames, "vertical_jump")
    assert out["rep_count"] == 1
    assert out["samples"] == 3


def test_readiness_band_thresholds():
    assert _readiness_band(85) == "peak"
    assert _readiness_band(70) == "ready"
    assert _readiness_band(50) == "training"
    assert _readiness_band(20) == "recover"


# ─── HTTP integration tests ────────────────────────────────────────────────


def test_rep_count_404_for_unknown_session(client, authed):
    r = client.get(f"/sessions/{uuid.uuid4()}/rep-count", headers=authed["headers"])
    assert r.status_code == 404


def test_rep_count_on_real_session(client, authed):
    from database import SESSION_DB

    sid = f"test_sess_{uuid.uuid4().hex[:8]}"
    SESSION_DB[sid] = {
        "session_id": sid,
        "athlete_id": "athlete_01",
        "sport": "vertical_jump",
        "status": "completed",
        "frames": [
            {"knee_angle_l": 170, "knee_angle_r": 170},
            {"knee_angle_l": 110, "knee_angle_r": 110},
            {"knee_angle_l": 170, "knee_angle_r": 170},
            {"knee_angle_l": 110, "knee_angle_r": 110},
            {"knee_angle_l": 170, "knee_angle_r": 170},
        ],
    }
    try:
        r = client.get(f"/sessions/{sid}/rep-count", headers=authed["headers"])
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert body["rep_count"] == 2
        assert body["supported"] is True
    finally:
        SESSION_DB.pop(sid, None)


def test_readiness_404_for_unknown_athlete(client, admin_client):
    r = client.get("/readiness/does_not_exist", headers=admin_client["headers"])
    assert r.status_code == 404


def test_readiness_returns_score_and_components(client, admin_client):
    r = client.get("/readiness/athlete_01?days=14", headers=admin_client["headers"])
    if r.status_code == 404:
        pytest.skip("athlete_01 not seeded in this run")
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 100
    assert body["band"] in {"peak", "ready", "training", "recover"}
    comps = body["components"]
    for key in ("form", "symmetry", "volume", "hrv"):
        assert key in comps
        assert 0 <= comps[key]["value"] <= 100
    # weights sum to 1.0
    assert round(sum(c["weight"] for c in comps.values()), 2) == 1.0
