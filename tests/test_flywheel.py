from __future__ import annotations

"""Tests for the data flywheel architecture: model registry, frame quality, model inference."""


# ─── Frame Quality Gate ──────────────────────────────────────────────────────


def test_quality_gate_good_frame():
    from services.frame_quality import score_frame_quality

    frame = {
        "hip_angle_l": 100,
        "hip_angle_r": 102,
        "knee_angle_l": 130,
        "knee_angle_r": 128,
        "shoulder_angle_l": 90,
        "shoulder_angle_r": 88,
        "elbow_angle_l": 140,
        "elbow_angle_r": 138,
        "ankle_dorsiflexion_l": 80,
        "ankle_dorsiflexion_r": 82,
        "trunk_lean": 12,
        "limb_symmetry_idx": 0.95,
        "pose_detected": True,
        "form_score": 75,
    }
    result = score_frame_quality(frame, sport="vertical_jump")
    assert result["quality_score"] > 0.6
    assert result["train_ready"] is True
    assert len(result["flags"]) == 0


def test_quality_gate_no_pose():
    from services.frame_quality import score_frame_quality

    frame = {"pose_detected": False, "form_score": 0}
    result = score_frame_quality(frame)
    assert result["quality_score"] == 0.0
    assert result["train_ready"] is False


def test_quality_gate_out_of_range():
    from services.frame_quality import score_frame_quality

    frame = {
        "knee_angle_l": 999,  # impossible
        "knee_angle_r": 130,
        "hip_angle_l": 100,
        "hip_angle_r": 100,
        "limb_symmetry_idx": 0.9,
    }
    result = score_frame_quality(frame, sport="vertical_jump")
    assert result["factors"]["reasonableness"] < 1.0
    assert any("out of range" in f for f in result["flags"])


def test_quality_gate_temporal_spike():
    from services.frame_quality import score_frame_quality

    prev = {"knee_angle_l": 130, "limb_symmetry_idx": 0.95}
    curr = {"knee_angle_l": 30, "limb_symmetry_idx": 0.95}  # 100° jump = spike
    result = score_frame_quality(curr, sport="vertical_jump", prev_frame=prev)
    assert result["factors"]["consistency"] < 1.0
    assert any("temporal spike" in f for f in result["flags"])


def test_filter_training_frames():
    from services.frame_quality import filter_training_frames

    frames = [
        {
            "knee_angle_l": 130,
            "knee_angle_r": 128,
            "hip_angle_l": 100,
            "hip_angle_r": 102,
            "limb_symmetry_idx": 0.95,
            "pose_detected": True,
        },
        {"pose_detected": False, "form_score": 0},
        {
            "knee_angle_l": 130,
            "knee_angle_r": 128,
            "hip_angle_l": 100,
            "hip_angle_r": 102,
            "limb_symmetry_idx": 0.95,
            "pose_detected": True,
        },
    ]
    _accepted, stats = filter_training_frames(frames, "vertical_jump")
    assert stats["total_frames"] == 3
    assert stats["accepted"] >= 1
    assert stats["rejected"] >= 1
    assert len(_accepted) == stats["accepted"]


# ─── Model Registry ─────────────────────────────────────────────────────────


def test_model_registry_version():
    from services.model_registry import QUALITY_CLASSES, QUALITY_SCORE_MAP

    assert len(QUALITY_CLASSES) == 4
    assert all(c in QUALITY_SCORE_MAP for c in QUALITY_CLASSES)
    assert QUALITY_SCORE_MAP["elite"] > QUALITY_SCORE_MAP["poor"]


def test_model_prediction_returns_result_or_none():
    """Model may or may not be loadable (TF may not be installed)."""
    from services.model_registry import predict_quality

    result = predict_quality(
        {"hip_angle_l": 100, "knee_angle_l": 120, "limb_symmetry_idx": 0.95},
        sport="vertical_jump",
    )
    # Either a valid prediction or None (no TF)
    if result is not None:
        assert result["quality"] in ["poor", "average", "good", "elite"]
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["form_score"] <= 100
        assert result["source"] == "model"


# ─── Model Integration in PoseAnalyzer ───────────────────────────────────────


def test_compute_form_score_still_works():
    """Ensure form scoring works regardless of model availability."""
    from services.pose_analyzer import BiomechanicalFrame, compute_form_score

    frame = BiomechanicalFrame(
        knee_angle_l=100,
        knee_angle_r=98,
        hip_angle_l=95,
        hip_angle_r=93,
        trunk_lean=10,
        ankle_dorsiflexion_l=85,
        ankle_dorsiflexion_r=83,
        limb_symmetry_idx=0.96,
    )
    score, quality, feedback = compute_form_score(frame, "vertical_jump")
    assert 0 <= score <= 100
    assert quality in ["poor", "average", "good", "elite"]
    assert isinstance(feedback, str)


# ─── Admin Endpoints ─────────────────────────────────────────────────────────


def test_model_status_endpoint(client, admin_client):
    r = client.get("/admin/model", headers=admin_client["headers"])
    assert r.status_code == 200
    body = r.json()
    assert "active_version" in body
    assert "model_loaded" in body


def test_retrain_status_endpoint(client, admin_client):
    r = client.get("/admin/retrain-status", headers=admin_client["headers"])
    assert r.status_code == 200
    body = r.json()
    assert "total_real_frames" in body
    assert "retrain_ready" in body
    assert "retrain_threshold" in body
