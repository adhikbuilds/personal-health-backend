"""
Tests for the new platform-upgrade pieces:
  - rPPG signal processing with a synthetic cardiac-rate input.
  - Advanced-metrics endpoint round-trips without schema surprises.
  - sports_catalog registry is internally consistent with model_registry.
  - metrics_service handles an empty athlete gracefully.
"""

import math

import pytest


def _synthetic_rgb_stream(fs=30.0, duration=10.0, hr_bpm=72.0):
    """Generate a fake RGB signal where the green channel has a clear
    cardiac oscillation at hr_bpm. Returns a list of (r, g, b, ts) tuples."""
    n = int(fs * duration)
    samples = []
    freq = hr_bpm / 60.0
    for i in range(n):
        t = i / fs
        phase = 2 * math.pi * freq * t
        # Clean pulse on green, quieter on red/blue — matches real rPPG physics.
        g = 128 + 2.0 * math.sin(phase)
        r = 140 + 0.6 * math.sin(phase + 0.2)
        b = 110 + 0.4 * math.sin(phase + 0.4)
        samples.append((r, g, b, t))
    return samples


def test_rppg_detects_synthetic_heart_rate():
    from services.rppg_processor import RPPGProcessor

    proc = RPPGProcessor()
    target_bpm = 72.0
    for r, g, b, ts in _synthetic_rgb_stream(hr_bpm=target_bpm):
        proc.add_rgb(r, g, b, ts)

    result = proc.compute()
    assert result["status"] == "ok"
    # With a clean signal and 10s window the estimator should be within 8 bpm
    # even with its heavy temporal smoothing.
    assert abs(result["bpm"] - target_bpm) < 8, f"expected ~{target_bpm}, got {result['bpm']}"


def test_rppg_invalid_rgb_returns_signal_warning():
    from services.rppg_processor import RPPGProcessor

    proc = RPPGProcessor()
    # Saturated white frame — should be flagged
    result = proc.add_rgb(250, 250, 250, 0.0)
    assert result["signal_quality"] == "invalid"


def test_rppg_warmup_below_min_frames():
    from services.rppg_processor import RPPGProcessor

    proc = RPPGProcessor()
    for i in range(5):
        proc.add_rgb(140, 128, 110, i / 30.0)
    result = proc.compute()
    assert result["status"] == "warmup"


def test_sports_catalog_index_matches_model_registry():
    from services import sports_catalog
    from services.model_registry import SPORT_INDEX

    # SPORT_INDEX is populated from the catalog; must round-trip
    for key, idx in SPORT_INDEX.items():
        assert sports_catalog.sport_index(key) == idx


def test_sports_catalog_rep_transition_has_all_sports():
    from services import sports_catalog

    for key in sports_catalog.keys():
        frm, to = sports_catalog.rep_transition(key)
        assert frm and to
        assert isinstance(frm, str) and isinstance(to, str)


def test_metrics_service_handles_empty_sessions():
    from services import metrics_service as ms

    assert ms.aggregate_sessions([])["total_sessions"] == 0
    assert ms.form_score_trend_pct([]) == 0.0
    assert ms.form_momentum([]) == 0.0
    assert ms.acute_chronic_ratio([])["band"] == "unknown"
    assert ms.training_monotony([])["monotony"] == 0.0
    assert ms.asymmetry_index([])["band"] == "insufficient data"
    assert ms.fatigue_index([])["band"] == "insufficient data"


def test_metrics_service_aggregate_counts_quality():
    from services import metrics_service as ms

    sessions = [
        {
            "summary": {
                "avg_form_score": 80.0,
                "peak_form_score": 90.0,
                "total_frames": 100,
                "duration_seconds": 600,
                "peak_jump_height_cm": 50.0,
                "xp_earned": 100,
                "quality_distribution": {"elite": 2, "good": 4, "average": 1, "poor": 0},
            }
        },
        {
            "summary": {
                "avg_form_score": 70.0,
                "peak_form_score": 80.0,
                "total_frames": 80,
                "duration_seconds": 500,
                "peak_jump_height_cm": 45.0,
                "xp_earned": 80,
                "quality_distribution": {"elite": 1, "good": 3, "average": 2, "poor": 1},
            }
        },
    ]
    agg = ms.aggregate_sessions(sessions)
    assert agg["total_sessions"] == 2
    assert agg["avg_form_score"] == 75.0
    assert agg["peak_form_score"] == 90.0
    assert agg["quality_distribution"] == {"elite": 3, "good": 7, "average": 3, "poor": 1}
    assert agg["total_reps"] == 180


def test_advanced_metrics_endpoint_returns_bundle(client):
    # Use any seeded athlete
    resp = client.get("/athletes")
    athletes = resp.json().get("athletes", resp.json() if isinstance(resp.json(), list) else [])
    if not athletes:
        pytest.skip("no athletes seeded")
    athlete_id = athletes[0]["id"]

    r = client.get(f"/athlete/{athlete_id}/advanced-metrics?days=60")
    assert r.status_code == 200
    body = r.json()
    for key in ("aggregate", "trend_pct", "momentum", "acwr", "monotony", "asymmetry",
                "latest_intensity", "fatigue", "readiness", "load_series", "form_trend_series"):
        assert key in body
    assert body["acwr"]["band"] in (
        "unknown", "under-loaded", "sweet spot", "high", "spike — injury risk"
    )


def test_huddle_create_and_leave(client):
    # This also smoke-tests the new leave endpoint wired in routes/huddle.py.
    resp = client.get("/athletes")
    athletes = resp.json().get("athletes") or []
    if not athletes:
        pytest.skip("no athletes seeded")
    aid = athletes[0]["id"]

    create = client.post("/huddle/create", json={"name": "Test Huddle", "sport": "vertical_jump"})
    assert create.status_code == 200
    hid = create.json()["huddle"]["huddle_id"]

    join = client.post(f"/huddle/{hid}/join", json={"athlete_id": aid})
    assert join.status_code == 200

    leave = client.post(f"/huddle/{hid}/leave", json={"athlete_id": aid})
    assert leave.status_code == 200
    assert leave.json()["remaining"] == 0


def test_health_endpoint_surfaces_model_info(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    for key in ("tflite", "keras", "norm_params", "face_detector"):
        assert key in body["models"]


def test_social_feed_now_aggregates_real_data(client):
    r = client.get("/feed?athlete_id=athlete_01&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert "posts" in body
    assert "tab" in body
    # Posts must be structured — at minimum the fields the app reads
    for post in body["posts"]:
        assert "id" in post and "author" in post and "content" in post
        assert post["id"].startswith(("sess_", "milestone_"))
