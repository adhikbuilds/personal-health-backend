"""
=============================================================================
ActiveBharat — Feature Extractor
=============================================================================
Converts BiomechanicalFrame dataclasses into flat feature vectors suitable
for training ML classifiers (TFLite model) and for dataset export.

Feature Vector (23 dimensions):
 [0]  hip_angle_l
 [1]  hip_angle_r
 [2]  hip_angle_avg
 [3]  knee_angle_l
 [4]  knee_angle_r
 [5]  knee_angle_avg
 [6]  shoulder_angle_l
 [7]  shoulder_angle_r
 [8]  elbow_angle_l
 [9]  elbow_angle_r
 [10] ankle_dorsiflexion_l
 [11] ankle_dorsiflexion_r
 [12] trunk_lean
 [13] spine_deviation
 [14] shoulder_hip_sep
 [15] head_forward_pos
 [16] com_height_norm
 [17] limb_symmetry_idx
 [18] estimated_jump_height
 [19] hip_knee_ratio       (derived: hip/knee balance indicator)
 [20] upper_lower_ratio    (derived: shoulder/hip ratio)
 [21] bilateral_deviation  (derived: max L-R deviation)
 [22] extension_index      (derived: avg extension of all joints)
=============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pose_analyzer import BiomechanicalFrame

# Quality label encoding
QUALITY_LABELS = {"elite": 3, "good": 2, "average": 1, "poor": 0}
PHASE_LABELS = {"setup": 0, "descent": 1, "takeoff": 2, "flight": 3, "landing": 4}

# Sport encoding
SPORT_LABELS = {
    "vertical_jump": 0,
    "snatch": 1,
    "sprint": 2,
    "javelin": 3,
    "cricket_bat": 4
}


class FeatureExtractor:
    """
    Converts BiomechanicalFrame to ML feature vectors.
    Handles normalization, derived features, and sequence windowing.
    """

    FEATURE_NAMES = [
        "hip_angle_l", "hip_angle_r", "hip_angle_avg",
        "knee_angle_l", "knee_angle_r", "knee_angle_avg",
        "shoulder_angle_l", "shoulder_angle_r",
        "elbow_angle_l", "elbow_angle_r",
        "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
        "trunk_lean", "spine_deviation", "shoulder_hip_sep",
        "head_forward_pos", "com_height_norm", "limb_symmetry_idx",
        "estimated_jump_height",
        # Derived
        "hip_knee_ratio", "upper_lower_ratio",
        "bilateral_deviation", "extension_index"
    ]

    # Normalization ranges (min, max) for each feature
    NORM_RANGES = {
        "hip_angle_l": (0, 180), "hip_angle_r": (0, 180), "hip_angle_avg": (0, 180),
        "knee_angle_l": (0, 180), "knee_angle_r": (0, 180), "knee_angle_avg": (0, 180),
        "shoulder_angle_l": (0, 180), "shoulder_angle_r": (0, 180),
        "elbow_angle_l": (0, 180), "elbow_angle_r": (0, 180),
        "ankle_dorsiflexion_l": (0, 180), "ankle_dorsiflexion_r": (0, 180),
        "trunk_lean": (0, 90), "spine_deviation": (0, 50),
        "shoulder_hip_sep": (0, 90), "head_forward_pos": (-30, 30),
        "com_height_norm": (0, 1), "limb_symmetry_idx": (0, 1),
        "estimated_jump_height": (0, 100),
        "hip_knee_ratio": (0, 5), "upper_lower_ratio": (0, 5),
        "bilateral_deviation": (0, 90), "extension_index": (0, 180),
    }

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def extract(self, frame: BiomechanicalFrame, sport: str = "vertical_jump") -> np.ndarray:
        """Extract 23-dimensional feature vector from a single frame."""
        hip_avg = (frame.hip_angle_l + frame.hip_angle_r) / 2
        knee_avg = (frame.knee_angle_l + frame.knee_angle_r) / 2

        # Derived features
        hip_knee_ratio = hip_avg / max(knee_avg, 1.0)
        upper_avg = (frame.shoulder_angle_l + frame.shoulder_angle_r) / 2
        lower_avg = (frame.hip_angle_l + frame.hip_angle_r + frame.knee_angle_l + frame.knee_angle_r) / 4
        upper_lower_ratio = upper_avg / max(lower_avg, 1.0)

        bilateral_deviation = max(
            abs(frame.hip_angle_l - frame.hip_angle_r),
            abs(frame.knee_angle_l - frame.knee_angle_r),
            abs(frame.ankle_dorsiflexion_l - frame.ankle_dorsiflexion_r)
        )

        all_joints = [
            frame.hip_angle_l, frame.hip_angle_r,
            frame.knee_angle_l, frame.knee_angle_r,
            frame.elbow_angle_l, frame.elbow_angle_r,
        ]
        extension_index = sum(all_joints) / len(all_joints)

        raw = {
            "hip_angle_l": frame.hip_angle_l,
            "hip_angle_r": frame.hip_angle_r,
            "hip_angle_avg": hip_avg,
            "knee_angle_l": frame.knee_angle_l,
            "knee_angle_r": frame.knee_angle_r,
            "knee_angle_avg": knee_avg,
            "shoulder_angle_l": frame.shoulder_angle_l,
            "shoulder_angle_r": frame.shoulder_angle_r,
            "elbow_angle_l": frame.elbow_angle_l,
            "elbow_angle_r": frame.elbow_angle_r,
            "ankle_dorsiflexion_l": frame.ankle_dorsiflexion_l,
            "ankle_dorsiflexion_r": frame.ankle_dorsiflexion_r,
            "trunk_lean": frame.trunk_lean,
            "spine_deviation": frame.spine_deviation,
            "shoulder_hip_sep": frame.shoulder_hip_sep,
            "head_forward_pos": frame.head_forward_pos,
            "com_height_norm": frame.com_height_norm,
            "limb_symmetry_idx": frame.limb_symmetry_idx,
            "estimated_jump_height": frame.estimated_jump_height,
            "hip_knee_ratio": hip_knee_ratio,
            "upper_lower_ratio": upper_lower_ratio,
            "bilateral_deviation": bilateral_deviation,
            "extension_index": extension_index,
        }

        vector = np.array([raw[name] for name in self.FEATURE_NAMES], dtype=np.float32)

        if self.normalize:
            vector = self._normalize(vector)

        return vector

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Min-max normalize each feature to [0, 1]."""
        normalized = np.zeros_like(vector)
        for i, name in enumerate(self.FEATURE_NAMES):
            lo, hi = self.NORM_RANGES.get(name, (0, 180))
            normalized[i] = np.clip((vector[i] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return normalized

    def extract_sequence(
        self, frames: List[BiomechanicalFrame], sport: str = "vertical_jump",
        window_size: int = 30
    ) -> np.ndarray:
        """
        Extract a temporal sequence feature matrix (window_size × 23).
        Pads or truncates to exactly window_size frames.
        """
        vectors = [self.extract(f, sport) for f in frames]

        if len(vectors) >= window_size:
            # Use the most recent window_size frames
            matrix = np.array(vectors[-window_size:])
        else:
            # Pad with zeros at the start
            pad = np.zeros((window_size - len(vectors), len(self.FEATURE_NAMES)), dtype=np.float32)
            matrix = np.vstack([pad, np.array(vectors)])

        return matrix  # shape: (window_size, 23)

    def frame_to_record(
        self, frame: BiomechanicalFrame, sport: str,
        session_id: str, frame_num: int, athlete_id: str = "unknown"
    ) -> Dict:
        """Convert frame to a flat dictionary for CSV export / API response."""
        return {
            "session_id": session_id,
            "athlete_id": athlete_id,
            "frame_num": frame_num,
            "sport": sport,
            # Joint angles
            "hip_angle_l": round(frame.hip_angle_l, 2),
            "hip_angle_r": round(frame.hip_angle_r, 2),
            "knee_angle_l": round(frame.knee_angle_l, 2),
            "knee_angle_r": round(frame.knee_angle_r, 2),
            "shoulder_angle_l": round(frame.shoulder_angle_l, 2),
            "shoulder_angle_r": round(frame.shoulder_angle_r, 2),
            "elbow_angle_l": round(frame.elbow_angle_l, 2),
            "elbow_angle_r": round(frame.elbow_angle_r, 2),
            "ankle_dorsiflexion_l": round(frame.ankle_dorsiflexion_l, 2),
            "ankle_dorsiflexion_r": round(frame.ankle_dorsiflexion_r, 2),
            # Posture
            "trunk_lean": round(frame.trunk_lean, 2),
            "spine_deviation": round(frame.spine_deviation, 2),
            "shoulder_hip_sep": round(frame.shoulder_hip_sep, 2),
            "head_forward_pos": round(frame.head_forward_pos, 2),
            # Dynamics
            "com_height_norm": round(frame.com_height_norm, 3),
            "estimated_jump_height": round(frame.estimated_jump_height, 1),
            # Aggregate
            "limb_symmetry_idx": round(frame.limb_symmetry_idx, 3),
            "form_score": round(frame.form_score, 1),
            # Labels
            "phase_label": frame.phase,
            "quality_label": frame.form_quality,
            "feedback_tag": frame.primary_feedback,
        }

    def get_csv_headers(self) -> List[str]:
        """Return CSV column headers in order."""
        return [
            "session_id", "athlete_id", "frame_num", "sport",
            "hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
            "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
            "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
            "trunk_lean", "spine_deviation", "shoulder_hip_sep", "head_forward_pos",
            "com_height_norm", "estimated_jump_height",
            "limb_symmetry_idx", "form_score",
            "phase_label", "quality_label", "feedback_tag"
        ]
