from __future__ import annotations

"""
Personal Health — Real-time Video Pipeline endpoint.

WebSocket: ws://.../realtime/{session_id}/landmarks

Receives 33 pose landmarks at 30-60fps from on-device MediaPipe.
Returns real-time form scores, coaching cues, rep counts.
Stores sampled frames for the training data flywheel.
Broadcasts to dashboard WebSocket listeners.
"""

import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from database import ATHLETE_DB, FRAME_BUFFER, SESSION_DB, WS_CONNECTIONS
from logging_setup import get_logger
from services.data_pipeline import store_frame
from services.video_pipeline import end_session, get_or_create_session

router = APIRouter(tags=["Realtime"])
log = get_logger("routes.realtime")


# DEPRECATED: No active consumers as of 2026-04-26. Candidate for removal.
# Was intended for: high-frequency landmark pipeline from on-device MediaPipe to server-side scoring
@router.websocket("/realtime/{session_id}/landmarks")
async def realtime_landmarks(websocket: WebSocket, session_id: str):
    """
    Client sends per frame:
      { "landmarks": [[x,y,z,vis], ...], "ts": 1234.5 }

    Server responds:
      { "form_score": 78.5, "form_quality": "good", "feedback": "...",
        "phase": "descent", "angles": {...}, "reps": 3, "cues": [...] }
    """
    await websocket.accept()

    session = SESSION_DB.get(session_id, {})
    sport = session.get("sport", "general")
    height = float(ATHLETE_DB.get(session.get("athlete_id", ""), {}).get("height_cm", 170))

    video = get_or_create_session(session_id, sport, height)

    try:
        while True:
            data = await websocket.receive_json()
            landmarks = data.get("landmarks", [])
            ts = data.get("ts", time.time())

            result = video.process_landmarks(landmarks, ts)

            # Store sampled frames for training data
            if video.should_store() and result.form_score > 0:
                frame_dict = {
                    "frame_num": result.frame_num,
                    "timestamp": ts,
                    "form_score": result.form_score,
                    "form_quality": result.form_quality,
                    "phase": result.phase,
                    **result.joint_angles,
                    "limb_symmetry_idx": result.symmetry,
                    "pose_detected": True,
                }
                FRAME_BUFFER.setdefault(session_id, []).append(frame_dict)
                store_frame(session_id, frame_dict, sport)

            # Broadcast to dashboard
            if video.should_broadcast():
                text = json.dumps(
                    {
                        "type": "frame",
                        "form_score": result.form_score,
                        "form_quality": result.form_quality,
                        "phase": result.phase,
                        "rep_count": result.rep_count,
                        "frame_num": result.frame_num,
                    },
                    default=str,
                )
                for ws in list(WS_CONNECTIONS.get(session_id, [])):
                    try:
                        await ws.send_text(text)
                    except Exception:
                        pass

            # Respond to client
            response = {
                "form_score": result.form_score,
                "form_quality": result.form_quality,
                "feedback": result.primary_feedback,
                "phase": result.phase,
                "angles": result.joint_angles,
                "symmetry": result.symmetry,
                "reps": result.rep_count,
                "frame": result.frame_num,
                "cues": [{"cue": c["cue"], "severity": c["severity"]} for c in result.coaching_cues],
                "ts": ts,
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        stats = end_session(session_id)
        log.info("realtime ended", extra={"session_id": session_id[:8], **(stats or {})})
    except Exception as e:
        log.error("realtime error", extra={"error": str(e), "session_id": session_id[:8]})
        end_session(session_id)
