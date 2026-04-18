from __future__ import annotations

"""
Personal Health — Frame Quality Gate.

Scores each captured frame on a 0-1 quality scale. Only frames above a
configurable threshold (default 0.6) should enter the model retraining
pipeline. This prevents garbage-in-garbage-out in the data flywheel.

Quality factors:
  1. Landmark visibility — are the key joints detected with high confidence?
  2. Angle reasonableness — are joint angles in physically possible ranges?
  3. Temporal consistency — no wild jumps from the previous frame (noise)?
  4. Pose completeness — are all required joints present for the sport?

Called by:
  - export_real_data.py: filters frames before writing to training CSV
  - routes/fitness.py: tags each frame with quality_score at capture time
"""

from typing import Any

# Physically possible joint angle ranges (degrees). Anything outside
# these is either a detection error or a non-human pose.
_ANGLE_BOUNDS: dict[str, tuple[float, float]] = {
    "hip_angle_l": (20, 200),
    "hip_angle_r": (20, 200),
    "knee_angle_l": (20, 200),
    "knee_angle_r": (20, 200),
    "shoulder_angle_l": (0, 200),
    "shoulder_angle_r": (0, 200),
    "elbow_angle_l": (10, 200),
    "elbow_angle_r": (10, 200),
    "ankle_dorsiflexion_l": (30, 180),
    "ankle_dorsiflexion_r": (30, 180),
    "trunk_lean": (0, 90),
}

# Required angles per sport — a frame must have non-zero values for
# these joints to be considered complete for that sport.
_SPORT_REQUIRED: dict[str, list[str]] = {
    "vertical_jump": ["knee_angle_l", "knee_angle_r", "hip_angle_l", "hip_angle_r"],
    "snatch": ["knee_angle_l", "shoulder_angle_l", "hip_angle_l"],
    "sprint": ["knee_angle_l", "hip_angle_l", "trunk_lean"],
    "javelin": ["shoulder_angle_l", "elbow_angle_l", "trunk_lean"],
    "cricket_bat": ["elbow_angle_l", "shoulder_angle_l", "trunk_lean"],
    "squat": ["knee_angle_l", "knee_angle_r", "hip_angle_l", "hip_angle_r"],
    "push_up": ["elbow_angle_l", "elbow_angle_r", "trunk_lean"],
    "pull_up": ["elbow_angle_l", "elbow_angle_r", "shoulder_angle_l"],
}

# Maximum allowed angle change between consecutive frames (degrees per frame).
# At 5 FPS, even explosive movements shouldn't exceed ~60°/frame.
_MAX_DELTA_PER_FRAME = 60.0

# Default quality threshold — frames below this should NOT enter training data.
QUALITY_THRESHOLD = 0.6


def score_frame_quality(
    frame: dict[str, Any],
    sport: str = "vertical_jump",
    prev_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Score a single frame's quality for training-data eligibility.

    Args:
        frame: dict with joint angle keys (hip_angle_l, knee_angle_l, etc.)
        sport: the sport context (determines which joints are required)
        prev_frame: the previous frame (for temporal consistency check)

    Returns:
        {
            "quality_score": 0.0-1.0,
            "train_ready": bool,
            "factors": {
                "visibility": 0.0-1.0,
                "reasonableness": 0.0-1.0,
                "completeness": 0.0-1.0,
                "consistency": 0.0-1.0,
            },
            "flags": ["angle out of range: knee_angle_l", ...]
        }
    """
    flags: list[str] = []

    # 1. Visibility — check if pose was detected at all
    if not frame.get("pose_detected", True) and frame.get("form_score", 0) == 0:
        return {
            "quality_score": 0.0,
            "train_ready": False,
            "factors": {"visibility": 0, "reasonableness": 0, "completeness": 0, "consistency": 0},
            "flags": ["no pose detected"],
        }

    # 2. Angle reasonableness — are values physically possible?
    angles_checked = 0
    angles_ok = 0
    for angle_name, (lo, hi) in _ANGLE_BOUNDS.items():
        val = frame.get(angle_name)
        if val is None or val == 0:
            continue
        angles_checked += 1
        val = float(val)
        if lo <= val <= hi:
            angles_ok += 1
        else:
            flags.append(f"angle out of range: {angle_name}={val:.1f} (expected {lo}-{hi})")

    reasonableness = angles_ok / max(angles_checked, 1)

    # 3. Completeness — does the frame have all required joints for this sport?
    required = _SPORT_REQUIRED.get(sport, _SPORT_REQUIRED["vertical_jump"])
    present = 0
    for joint in required:
        val = frame.get(joint)
        if val is not None and float(val) > 0:
            present += 1
        else:
            flags.append(f"missing required joint: {joint}")
    completeness = present / max(len(required), 1)

    # 4. Temporal consistency — no wild jumps from previous frame
    consistency = 1.0
    if prev_frame is not None:
        deltas = []
        for angle_name in _ANGLE_BOUNDS:
            curr = frame.get(angle_name)
            prev = prev_frame.get(angle_name)
            if curr is not None and prev is not None and float(curr) > 0 and float(prev) > 0:
                delta = abs(float(curr) - float(prev))
                deltas.append(delta)
                if delta > _MAX_DELTA_PER_FRAME:
                    flags.append(f"temporal spike: {angle_name} delta={delta:.1f}")

        if deltas:
            max_delta = max(deltas)
            if max_delta > _MAX_DELTA_PER_FRAME:
                consistency = max(0.0, 1.0 - (max_delta - _MAX_DELTA_PER_FRAME) / _MAX_DELTA_PER_FRAME)

    # Visibility score — use limb_symmetry_idx as a proxy (0 = no detection)
    sym = float(frame.get("limb_symmetry_idx", 0) or 0)
    visibility = 1.0 if sym > 0.5 else (0.5 if sym > 0 else 0.0)

    # Weighted composite
    quality_score = round(
        visibility * 0.20 + reasonableness * 0.30 + completeness * 0.30 + consistency * 0.20,
        3,
    )

    return {
        "quality_score": quality_score,
        "train_ready": quality_score >= QUALITY_THRESHOLD,
        "factors": {
            "visibility": round(visibility, 3),
            "reasonableness": round(reasonableness, 3),
            "completeness": round(completeness, 3),
            "consistency": round(consistency, 3),
        },
        "flags": flags,
    }


def filter_training_frames(
    frames: list[dict],
    sport: str = "vertical_jump",
    threshold: float = QUALITY_THRESHOLD,
) -> tuple[list[dict], dict]:
    """
    Filter a list of frames to only those suitable for model training.

    Returns:
        (filtered_frames, stats_dict)
    """
    accepted = []
    rejected = 0
    total_score = 0.0
    prev = None

    for frame in frames:
        result = score_frame_quality(frame, sport, prev)
        total_score += result["quality_score"]
        if result["quality_score"] >= threshold:
            frame["_quality_score"] = result["quality_score"]
            accepted.append(frame)
        else:
            rejected += 1
        prev = frame

    stats = {
        "total_frames": len(frames),
        "accepted": len(accepted),
        "rejected": rejected,
        "avg_quality": round(total_score / max(len(frames), 1), 3),
        "acceptance_rate": round(len(accepted) / max(len(frames), 1), 3),
        "threshold": threshold,
    }
    return accepted, stats
