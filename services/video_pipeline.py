from __future__ import annotations

"""
Personal Health — Video Pipeline Service.

Processes real-time landmark streams from on-device MediaPipe.
Handles: form scoring, rep counting, phase detection, temporal smoothing,
injury flag accumulation, and session state management.

Architecture:
  Phone (30fps landmarks) → VideoSession → PoseAnalyzer → FormResult
  FormResult → broadcast to dashboard + store sampled frames

This is the production path. The photo-every-3s approach is the
development fallback.
"""

import time
import types
from dataclasses import dataclass, field
from typing import Any

from services import sports_catalog
from services.pose_analyzer import PoseAnalyzer
from services.smart_coach import coach_frame
from logging_setup import get_logger

log = get_logger("services.video_pipeline")


@dataclass
class FormResult:
    """Single frame analysis result."""

    form_score: float = 0.0
    form_quality: str = "unknown"
    primary_feedback: str = ""
    phase: str = "setup"
    joint_angles: dict = field(default_factory=dict)
    symmetry: float = 1.0
    rep_count: int = 0
    frame_num: int = 0
    coaching_cues: list = field(default_factory=list)
    timestamp: float = 0.0


class VideoSession:
    """
    Manages a real-time video analysis session.

    One instance per active session. Maintains:
    - PoseAnalyzer with temporal smoothing
    - Rep counter state machine
    - Frame sampling for data storage
    - Running statistics
    """

    STORE_EVERY_N = 10  # store 1 in 10 frames for training data
    BROADCAST_EVERY_N = 5  # send 1 in 5 frames to dashboard

    # Module-level table kept for external callers still importing it.
    REP_TRANSITIONS = sports_catalog.build_rep_transitions_map()

    def __init__(self, sport: str = "general", body_height_cm: float = 170):
        self.sport = sport
        self.analyzer = PoseAnalyzer(sport=sport, body_height_cm=body_height_cm)
        self.frame_count = 0
        self.rep_count = 0
        self.last_phase = None
        self.scores: list[float] = []
        self.start_time = time.time()
        self._rep_from, self._rep_to = sports_catalog.rep_transition(sport)

    def process_landmarks(self, landmarks: list[list[float]], timestamp: float = None) -> FormResult:
        """
        Process a single frame's 33 landmarks.

        Args:
            landmarks: [[x, y, z, visibility], ...] — 33 landmarks from MediaPipe
            timestamp: frame timestamp (defaults to now)

        Returns:
            FormResult with scores, feedback, rep count
        """
        ts = timestamp or time.time()
        self.frame_count += 1

        if not landmarks or len(landmarks) < 33:
            return FormResult(timestamp=ts, frame_num=self.frame_count)

        # Convert to PoseAnalyzer format
        lms = [
            types.SimpleNamespace(x=lm[0], y=lm[1], z=lm[2], visibility=lm[3] if len(lm) > 3 else 0.9)
            for lm in landmarks
        ]
        mock = types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=lms))

        bio = self.analyzer.analyze(mock)

        # Rep counting
        if bio.phase and self.last_phase == self._rep_from and bio.phase == self._rep_to:
            self.rep_count += 1
        self.last_phase = bio.phase

        # Track scores
        if bio.form_score > 0:
            self.scores.append(bio.form_score)

        # Frame-level coaching
        frame_dict = {
            "knee_angle_l": bio.knee_angle_l,
            "knee_angle_r": bio.knee_angle_r,
            "hip_angle_l": bio.hip_angle_l,
            "hip_angle_r": bio.hip_angle_r,
            "trunk_lean": bio.trunk_lean,
            "limb_symmetry_idx": bio.limb_symmetry_idx,
            "form_score": bio.form_score,
            "phase": bio.phase,
        }
        cues = coach_frame(frame_dict, self.sport) if self.frame_count % 5 == 0 else []

        return FormResult(
            form_score=bio.form_score,
            form_quality=bio.form_quality,
            primary_feedback=bio.primary_feedback,
            phase=bio.phase,
            joint_angles={
                "KNEE_L": round(bio.knee_angle_l, 1),
                "KNEE_R": round(bio.knee_angle_r, 1),
                "HIP_L": round(bio.hip_angle_l, 1),
                "HIP_R": round(bio.hip_angle_r, 1),
                "TRUNK": round(bio.trunk_lean, 1),
                "ELBOW_L": round(bio.elbow_angle_l, 1),
                "ELBOW_R": round(bio.elbow_angle_r, 1),
            },
            symmetry=round(bio.limb_symmetry_idx, 3),
            rep_count=self.rep_count,
            frame_num=self.frame_count,
            coaching_cues=cues,
            timestamp=ts,
        )

    def should_store(self) -> bool:
        """Whether the current frame should be stored for training data."""
        return self.frame_count % self.STORE_EVERY_N == 0

    def should_broadcast(self) -> bool:
        """Whether the current frame should be broadcast to dashboard."""
        return self.frame_count % self.BROADCAST_EVERY_N == 0

    def get_session_stats(self) -> dict[str, Any]:
        """Current session statistics."""
        elapsed = time.time() - self.start_time
        return {
            "frames_processed": self.frame_count,
            "reps_counted": self.rep_count,
            "elapsed_seconds": round(elapsed, 1),
            "avg_fps": round(self.frame_count / max(elapsed, 0.1), 1),
            "avg_score": round(sum(self.scores) / max(len(self.scores), 1), 1),
            "peak_score": round(max(self.scores, default=0), 1),
            "score_count": len(self.scores),
        }


# ─── Session Manager ─────────────────────────────────────────────────────────

_SESSIONS: dict[str, VideoSession] = {}


def get_or_create_session(session_id: str, sport: str = "general", height: float = 170) -> VideoSession:
    """Get existing session or create new one."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = VideoSession(sport=sport, body_height_cm=height)
        log.info("video session created", extra={"session_id": session_id[:8], "sport": sport})
    return _SESSIONS[session_id]


def end_session(session_id: str) -> dict[str, Any] | None:
    """End a video session and return final stats."""
    session = _SESSIONS.pop(session_id, None)
    if session:
        stats = session.get_session_stats()
        log.info("video session ended", extra={"session_id": session_id[:8], **stats})
        return stats
    return None
