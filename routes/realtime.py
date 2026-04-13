from __future__ import annotations

"""
Personal Health — Real-time Landmark Stream (Video Pipeline).

WebSocket endpoint for receiving pose landmarks at 30-60fps from
on-device MediaPipe running on the phone. Instead of sending full
JPEG frames (slow, 0.3fps), the phone sends only landmark coordinates
(~1KB per frame, 30x per second).

Architecture:
  Phone (MediaPipe on-device) → landmarks via WebSocket → server scores form

  ws://.../realtime/{session_id}/landmarks
    Client sends: { "landmarks": [[x,y,z,vis], ...], "ts": 1234.5 }
    Server sends: { "form_score": 78.5, "quality": "good", "feedback": "...", "phase": "descent" }

This is the production architecture described in VISION.md Layer 2 #6
(on-device inference). It replaces the photo-every-3s approach.
"""

import json
import time
import types
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from database import ATHLETE_DB, FRAME_BUFFER, SESSION_DB, WS_CONNECTIONS, _save_db
from logging_setup import get_logger
from services.pose_analyzer import BiomechanicalFrame, PoseAnalyzer, compute_form_score

router = APIRouter(tags=["Realtime"])
log = get_logger("routes.realtime")

# Per-session analyzers (reused across frames for temporal smoothing)
_ANALYZERS: dict[str, PoseAnalyzer] = {}


def _get_analyzer(session_id: str) -> PoseAnalyzer:
    if session_id not in _ANALYZERS:
        session = SESSION_DB.get(session_id, {})
        sport = session.get("sport", "vertical_jump")
        height = float(ATHLETE_DB.get(session.get("athlete_id", ""), {}).get("height_cm", 170))
        _ANALYZERS[session_id] = PoseAnalyzer(sport=sport, body_height_cm=height)
    return _ANALYZERS[session_id]


@router.websocket("/realtime/{session_id}/landmarks")
async def realtime_landmarks(websocket: WebSocket, session_id: str):
    """
    Receive pose landmarks at 30-60fps and return real-time form scores.

    Client sends JSON per frame:
    {
        "landmarks": [[x, y, z, visibility], ...],  // 33 landmarks
        "ts": 1234567.89  // timestamp
    }

    Server responds with:
    {
        "form_score": 78.5,
        "form_quality": "good",
        "primary_feedback": "Control trunk lean",
        "phase": "descent",
        "joint_angles": { "KNEE_L": 95.2, ... },
        "rep_count": 3,
        "ts": 1234567.89
    }
    """
    await websocket.accept()
    log.info("realtime stream started", extra={"session_id": session_id[:8]})

    analyzer = _get_analyzer(session_id)
    frame_count = 0
    rep_count = 0
    last_phase = None
    sport = SESSION_DB.get(session_id, {}).get("sport", "vertical_jump")

    # Rep counting transitions
    REP_TRANSITIONS = {
        "vertical_jump": ("descent", "takeoff"),
        "squat": ("descent", "setup"),
        "push_up": ("descent", "setup"),
        "pull_up": ("descent", "setup"),
        "snatch": ("descent", "catch"),
        "general": ("descent", "setup"),
    }
    rep_from, rep_to = REP_TRANSITIONS.get(sport, ("descent", "setup"))

    try:
        while True:
            data = await websocket.receive_json()
            landmarks = data.get("landmarks", [])
            ts = data.get("ts", time.time())

            if not landmarks or len(landmarks) < 33:
                await websocket.send_json({"error": "need 33 landmarks", "ts": ts})
                continue

            # Convert to the format PoseAnalyzer.analyze() expects
            lms = [
                types.SimpleNamespace(x=lm[0], y=lm[1], z=lm[2], visibility=lm[3] if len(lm) > 3 else 0.9)
                for lm in landmarks
            ]
            mock_result = types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=lms))

            bio = analyzer.analyze(mock_result)
            frame_count += 1

            # Rep counting
            if bio.phase and last_phase == rep_from and bio.phase == rep_to:
                rep_count += 1
            last_phase = bio.phase

            # Store frame data (sampled — not all 60fps)
            if frame_count % 10 == 0:  # store every 10th frame
                frame_dict = {
                    "frame_num": frame_count,
                    "timestamp": ts,
                    "form_score": bio.form_score,
                    "form_quality": bio.form_quality,
                    "phase": bio.phase,
                    "knee_angle_l": bio.knee_angle_l,
                    "knee_angle_r": bio.knee_angle_r,
                    "hip_angle_l": bio.hip_angle_l,
                    "hip_angle_r": bio.hip_angle_r,
                    "trunk_lean": bio.trunk_lean,
                    "limb_symmetry_idx": bio.limb_symmetry_idx,
                    "estimated_jump_height": bio.estimated_jump_height,
                    "pose_detected": bio.visibility_ok,
                }
                FRAME_BUFFER.setdefault(session_id, []).append(frame_dict)

            # Respond with real-time feedback
            response = {
                "form_score": bio.form_score,
                "form_quality": bio.form_quality,
                "primary_feedback": bio.primary_feedback,
                "phase": bio.phase,
                "joint_angles": {
                    "KNEE_L": round(bio.knee_angle_l, 1),
                    "KNEE_R": round(bio.knee_angle_r, 1),
                    "HIP_L": round(bio.hip_angle_l, 1),
                    "HIP_R": round(bio.hip_angle_r, 1),
                    "TRUNK": round(bio.trunk_lean, 1),
                },
                "symmetry": round(bio.limb_symmetry_idx, 3),
                "rep_count": rep_count,
                "frame_num": frame_count,
                "ts": ts,
            }

            # Also broadcast to dashboard WebSocket listeners
            if frame_count % 5 == 0:  # broadcast every 5th frame to reduce load
                text = json.dumps({"type": "frame", **response}, default=str)
                for ws in list(WS_CONNECTIONS.get(session_id, [])):
                    try:
                        await ws.send_text(text)
                    except Exception:
                        pass

            await websocket.send_json(response)

    except WebSocketDisconnect:
        log.info(
            "realtime stream ended",
            extra={
                "session_id": session_id[:8],
                "frames": frame_count,
                "reps": rep_count,
            },
        )
    except Exception as e:
        log.error("realtime error", extra={"error": str(e), "session_id": session_id[:8]})
    finally:
        _ANALYZERS.pop(session_id, None)
