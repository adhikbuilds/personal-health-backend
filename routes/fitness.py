from __future__ import annotations

"""
Fitness domain — Sessions, Pose Analysis, rPPG, Dataset, Fitness Test
All biomechanics-related endpoints live here.
"""

import asyncio
import csv
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from database import (
    _POSE_ANALYZERS,
    _RATE_LIMITS,
    ANALYSIS_QUEUE,
    ATHLETE_DB,
    DATASET_PATH,
    FRAME_BUFFER,
    RESULT_STORE,
    RPPG_STORE,
    SESSION_DB,
    WS_CONNECTIONS,
    _compute_xp,
    _save_db,
)

router = APIRouter()


# ─── Pydantic Models ────────────────────────────────────────────────────────


VALID_SPORTS = frozenset(
    {
        "vertical_jump",
        "sprint",
        "snatch",
        "javelin",
        "cricket_bat",
        "squat",
        "push_up",
        "pull_up",
    }
)


class StartSessionRequest(BaseModel):
    athlete_id: str = "athlete_01"
    sport: str = "vertical_jump"
    model_config = {"json_schema_extra": {"example": {"athlete_id": "athlete_01", "sport": "vertical_jump"}}}


class FrameData(BaseModel):
    hip_angle_l: float = 0.0
    hip_angle_r: float = 0.0
    knee_angle_l: float = 0.0
    knee_angle_r: float = 0.0
    shoulder_angle_l: float = 0.0
    shoulder_angle_r: float = 0.0
    elbow_angle_l: float = 0.0
    elbow_angle_r: float = 0.0
    ankle_dorsiflexion_l: float = 0.0
    ankle_dorsiflexion_r: float = 0.0
    trunk_lean: float = 0.0
    spine_deviation: float = 0.0
    shoulder_hip_sep: float = 0.0
    head_forward_pos: float = 0.0
    com_height_norm: float = 0.5
    estimated_jump_height: float = 0.0
    limb_symmetry_idx: float = 1.0
    form_score: float = 0.0
    form_quality: str = "unknown"
    primary_feedback: str = ""
    phase: str = "setup"
    image_b64: Optional[str] = None


class FitnessTestRequest(BaseModel):
    athlete_id: str
    score: int
    level: int
    bmi: Optional[float] = None
    sit_reach_cm: Optional[float] = None
    run_600_seconds: Optional[float] = None
    age_group: str = "Adult"


# ─── Helpers ────────────────────────────────────────────────────────────────


async def _broadcast(session_id: str, payload: dict):
    dead = []
    text = json.dumps(payload, default=str)
    for ws in list(WS_CONNECTIONS.get(session_id, [])):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns = WS_CONNECTIONS.get(session_id, [])
        if ws in conns:
            conns.remove(ws)


def _resolve_sport_model_path(sport: str) -> Optional[str]:
    """PF-10: pick a sport-specific .tflite if it exists, else fall back to the generic one.
    Returns the resolved path as a string, or None if no model is available.
    pose_analyzer is rule-based today, so this is attached to the analyzer for
    observability / eventual on-device deployment — it does not change scoring."""
    from pathlib import Path as _P

    models_dir = _P(__file__).resolve().parent.parent / "models"
    specific = models_dir / f"pose_classifier_{sport}.tflite"
    generic = models_dir / "pose_classifier.tflite"
    if specific.exists():
        return str(specific)
    if generic.exists():
        return str(generic)
    return None


async def analysis_worker():
    while True:
        try:
            item = await ANALYSIS_QUEUE.get()
            session_id, image_b64, sport, frame_dict = item
            try:
                from services.pose_analyzer import PoseAnalyzer

                if session_id not in _POSE_ANALYZERS:
                    # PF-03: use athlete's actual height for jump-height calc
                    _ath_id = SESSION_DB.get(session_id, {}).get("athlete_id")
                    _height = float(ATHLETE_DB.get(_ath_id, {}).get("height_cm", 170))
                    analyzer = PoseAnalyzer(sport=sport, body_height_cm=_height)
                    # PF-10: attach sport-specific model path (fallback to generic)
                    analyzer.model_path = _resolve_sport_model_path(sport)
                    _POSE_ANALYZERS[session_id] = analyzer
                analyzer = _POSE_ANALYZERS[session_id]
                result = analyzer.analyze_base64_image(image_b64, sport)

                if result.get("pose_detected"):
                    angles = result.get("joint_angles", {})
                    update = {
                        "form_score": result["form_score"],
                        "form_quality": result["form_quality"],
                        "primary_feedback": result["primary_feedback"],
                        "phase": result["phase"],
                        "limb_symmetry_idx": result["symmetry_score"],
                        "trunk_lean": result["trunk_lean"],
                        "estimated_jump_height": result.get("estimated_jump_height", 0.0),
                        "knee_angle_l": angles.get("KNEE_L", 0.0),
                        "knee_angle_r": angles.get("KNEE_R", 0.0),
                        "hip_angle_l": angles.get("HIP_L", 0.0),
                        "hip_angle_r": angles.get("HIP_R", 0.0),
                        "elbow_angle_l": angles.get("ELBOW_L", 0.0),
                        "elbow_angle_r": angles.get("ELBOW_R", 0.0),
                        "pose_detected": True,
                    }
                    frame_dict.update(update)
                    result_entry = {
                        **update,
                        "frame_num": frame_dict["frame_num"],
                        "sport": sport,
                        "analyzed_at": time.time(),
                        "data_source": "real",
                        "keypoints": result.get("keypoints", []),
                    }
                    RESULT_STORE[session_id] = result_entry
                    if FRAME_BUFFER.get(session_id):
                        FRAME_BUFFER[session_id][-1].update(update)

                    # PF-08: Log prediction to db/predictions.jsonl for drift detection
                    try:
                        from database import DB_PATH

                        log_entry = {
                            "session_id": session_id,
                            "frame_num": frame_dict["frame_num"],
                            "timestamp": time.time(),
                            "sport": sport,
                            "form_score": result["form_score"],
                            "form_quality": result["form_quality"],
                            "phase": result["phase"],
                            "symmetry_idx": result["symmetry_score"],
                        }
                        with open(DB_PATH / "predictions.jsonl", "a") as plog:
                            plog.write(json.dumps(log_entry) + "\n")
                    except Exception as plog_err:
                        print(f"[PRED LOG] {plog_err}")

                    print(
                        f"[AI] session={session_id[:8]} score={result['form_score']:.0f} "
                        f"quality={result['form_quality']} phase={result['phase']}"
                    )
                    await _broadcast(session_id, {"type": "frame", **result_entry})
                else:
                    print(f"[AI] No pose in frame (session {session_id[:8]})")
                    await _broadcast(
                        session_id,
                        {
                            "type": "frame",
                            "frame_num": frame_dict["frame_num"],
                            "pose_detected": False,
                            "analyzed_at": time.time(),
                        },
                    )
            except Exception as e:
                print(f"[WORKER ERROR] {e}")
            finally:
                ANALYSIS_QUEUE.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WORKER FATAL] {e}")
            await asyncio.sleep(1)


async def session_cleanup_worker():
    while True:
        try:
            await asyncio.sleep(1800)
            now = time.time()

            # PF-12: Delete persisted frame files for sessions that ended > 24h ago.
            try:
                from database import DB_PATH

                frames_dir = DB_PATH / "frames"
                if frames_dir.exists():
                    removed = 0
                    for fp in frames_dir.glob("*.jsonl"):
                        sid = fp.stem
                        session = SESSION_DB.get(sid)
                        if not session or session.get("status") != "completed":
                            continue
                        ended_at = session.get("ended_at")
                        if not ended_at:
                            continue
                        try:
                            ended_ts = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00")).timestamp()
                        except Exception:
                            continue
                        if now - ended_ts > 86400:  # 24 hours
                            try:
                                fp.unlink()
                                removed += 1
                            except Exception as rm_err:
                                print(f"[PF-12] Failed to remove {fp}: {rm_err}")
                    if removed:
                        print(f"[PF-12] Cleaned up {removed} expired frame files")
            except Exception as cleanup_err:
                print(f"[PF-12] Frame file cleanup error: {cleanup_err}")

            for sid, session in list(SESSION_DB.items()):
                if session.get("status") != "active":
                    continue
                frames = session.get("frames", [])
                if frames:
                    ts_str = frames[-1].get("timestamp")
                    try:
                        last_ts = datetime.fromisoformat(str(ts_str)).timestamp() if ts_str else now
                    except Exception:
                        last_ts = now
                else:
                    started = session.get("started_at", "")
                    try:
                        last_ts = datetime.fromisoformat(str(started)).timestamp()
                    except Exception:
                        last_ts = now
                if now - last_ts > 7200:
                    session["status"] = "completed"
                    session["ended_at"] = datetime.utcnow().isoformat()
                    session["auto_ended"] = True
                    print(f"[CLEANUP] Auto-ended stale session {sid[:8]}")
            _save_db()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")
            await asyncio.sleep(60)


# ─── Session Endpoints ──────────────────────────────────────────────────────


@router.post("/session/start", tags=["Sessions"])
async def start_session(req: StartSessionRequest):
    if req.sport not in VALID_SPORTS:
        raise HTTPException(
            400,
            f"unknown sport '{req.sport}' — valid: {', '.join(sorted(VALID_SPORTS))}",
        )
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "athlete_id": req.athlete_id,
        "sport": req.sport,
        "status": "active",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "ended_at": None,
        "frame_count": 0,
        "summary": None,
    }
    SESSION_DB[session_id] = session
    FRAME_BUFFER[session_id] = []
    return {"session_id": session_id, "sport": req.sport, "athlete_id": req.athlete_id, "message": "Session started"}


@router.post("/session/{session_id}/frame", tags=["Sessions"])
async def add_frame(session_id: str, frame: FrameData):
    if session_id not in SESSION_DB:
        raise HTTPException(404, "Session not found")
    if SESSION_DB[session_id]["status"] != "active":
        raise HTTPException(400, "Session not active")

    # PF-04: Rate limiting — max 10 frames/second per session
    now = time.time()
    last_frame_time = _RATE_LIMITS.get(session_id, 0)
    if now - last_frame_time < 0.1:
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded", "max_fps": 10})
    _RATE_LIMITS[session_id] = now

    sport = SESSION_DB[session_id].get("sport", "vertical_jump")
    frame_dict = frame.model_dump()

    # PF-12: Persist frame to disk so server restarts don't lose session data
    try:
        from database import DB_PATH

        frames_dir = DB_PATH / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_log = {k: v for k, v in frame_dict.items() if k != "image_b64"}
        with open(frames_dir / f"{session_id}.jsonl", "a") as fp:
            fp.write(json.dumps(frame_log, default=str) + "\n")
    except Exception as fperr:
        print(f"[FRAME PERSIST] {fperr}")
    image_b64 = frame_dict.pop("image_b64", None)
    frame_dict["frame_num"] = len(FRAME_BUFFER[session_id])
    frame_dict["timestamp"] = time.time()
    frame_dict["pose_detected"] = None
    FRAME_BUFFER[session_id].append(frame_dict)
    SESSION_DB[session_id]["frame_count"] += 1
    if image_b64 and ANALYSIS_QUEUE is not None:
        try:
            ANALYSIS_QUEUE.put_nowait((session_id, image_b64, sport, frame_dict))
        except asyncio.QueueFull:
            print(f"[WARN] Analysis queue full, dropping frame for {session_id[:8]}")
    latest = RESULT_STORE.get(session_id, {})
    return {
        "frame_num": frame_dict["frame_num"],
        "status": "queued" if image_b64 else "stored",
        "queued_for_analysis": bool(image_b64),
        "last_form_score": latest.get("form_score"),
        "last_feedback": latest.get("primary_feedback"),
        "last_quality": latest.get("form_quality"),
    }


@router.get("/session/{session_id}/latest-result", tags=["Sessions"])
async def latest_result(session_id: str):
    if session_id not in SESSION_DB:
        raise HTTPException(404, "Session not found")
    result = RESULT_STORE.get(session_id)
    if not result:
        return {
            "session_id": session_id,
            "data_source": "none",
            "message": "No analysis yet — send frames with image_b64",
        }
    return {"session_id": session_id, **result}


@router.post("/session/calibrate", tags=["Sessions"])
async def calibrate_pose(frame: FrameData, sport: str = Query(default="vertical_jump")):
    if not frame.image_b64:
        raise HTTPException(400, "image_b64 required for calibration")
    try:
        from services.pose_analyzer import PoseAnalyzer

        analyzer = PoseAnalyzer(sport=sport)
        result = analyzer.analyze_base64_image(frame.image_b64, sport)
        if not result.get("pose_detected"):
            return {
                "pose_detected": False,
                "form_score": 0,
                "keypoints": [],
                "deviations": {},
                "primary_feedback": "Move into frame — stand 1.5–2m from camera",
            }
        angles = result.get("joint_angles", {})
        IDEAL = {"KNEE_L": 170, "KNEE_R": 170, "HIP_L": 170, "HIP_R": 170, "ELBOW_L": 160, "ELBOW_R": 160}
        deviations = {}
        for joint, ideal_angle in IDEAL.items():
            actual = angles.get(joint, ideal_angle)
            diff = abs(actual - ideal_angle)
            deviations[joint] = "good" if diff < 15 else "warning" if diff < 35 else "critical"
        return {
            "pose_detected": True,
            "form_score": result["form_score"],
            "form_quality": result["form_quality"],
            "primary_feedback": result["primary_feedback"],
            "keypoints": result.get("keypoints", []),
            "joint_angles": angles,
            "deviations": deviations,
            "symmetry_score": result["symmetry_score"],
        }
    except Exception as e:
        print(f"[CALIBRATE ERROR] {e}")
        return {
            "pose_detected": False,
            "form_score": 0,
            "keypoints": [],
            "deviations": {},
            "primary_feedback": "Calibration unavailable — check server logs",
        }


@router.post("/session/{session_id}/end", tags=["Sessions"])
async def end_session(session_id: str):
    if session_id not in SESSION_DB:
        raise HTTPException(404, "Session not found")
    if SESSION_DB[session_id]["status"] != "active":
        raise HTTPException(400, "Session already ended")
    frames = FRAME_BUFFER.get(session_id, [])
    # PF-12: If memory buffer is empty but persisted frames exist on disk, load them.
    if not frames:
        try:
            from database import DB_PATH

            disk_path = DB_PATH / "frames" / f"{session_id}.jsonl"
            if disk_path.exists():
                recovered = []
                with open(disk_path, encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            recovered.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                if recovered:
                    frames = recovered
                    FRAME_BUFFER[session_id] = recovered
                    print(f"[PF-12] Recovered {len(recovered)} frames from disk for {session_id[:8]}")
        except Exception as recover_err:
            print(f"[PF-12] Recovery failed: {recover_err}")
    if not frames:
        summary = {
            "session_id": session_id,
            "athlete_id": SESSION_DB[session_id]["athlete_id"],
            "sport": SESSION_DB[session_id]["sport"],
            "total_frames": 0,
            "valid_frames": 0,
            "avg_form_score": 0,
            "peak_form_score": 0,
            "peak_jump_height_cm": 0,
            "avg_jump_height_cm": 0,
            "avg_symmetry": 0,
            "xp_earned": 50,
            "quality_distribution": {"elite": 0, "good": 0, "average": 0, "poor": 0},
        }
    else:
        valid = [f for f in frames if f.get("form_score", 0) > 0]
        scores = [f["form_score"] for f in valid]
        jump_heights = [f["estimated_jump_height"] for f in frames if f.get("estimated_jump_height", 0) > 5]
        symmetries = [f["limb_symmetry_idx"] for f in valid]
        quality_counts = {"elite": 0, "good": 0, "average": 0, "poor": 0}
        for f in frames:
            q = f.get("form_quality", "unknown")
            if q in quality_counts:
                quality_counts[q] += 1
        duration = frames[-1]["timestamp"] - frames[0]["timestamp"] if len(frames) > 1 else 0
        summary = {
            "session_id": session_id,
            "athlete_id": SESSION_DB[session_id]["athlete_id"],
            "sport": SESSION_DB[session_id]["sport"],
            "total_frames": len(frames),
            "valid_frames": len(valid),
            "duration_seconds": round(duration, 1),
            "avg_form_score": round(sum(scores) / max(len(scores), 1), 1),
            "peak_form_score": round(max(scores, default=0), 1),
            "peak_jump_height_cm": round(max(jump_heights, default=0), 1),
            "avg_jump_height_cm": round(sum(jump_heights) / max(len(jump_heights), 1), 1),
            "avg_symmetry": round(sum(symmetries) / max(len(symmetries), 1), 3),
            "quality_distribution": quality_counts,
            "xp_earned": _compute_xp(scores, jump_heights),
        }
    SESSION_DB[session_id]["status"] = "completed"
    SESSION_DB[session_id]["ended_at"] = datetime.now(timezone.utc).isoformat()
    SESSION_DB[session_id]["summary"] = summary
    SESSION_DB[session_id]["frames"] = frames
    athlete_id = SESSION_DB[session_id]["athlete_id"]
    if athlete_id in ATHLETE_DB:
        ATHLETE_DB[athlete_id]["sessions"] = ATHLETE_DB[athlete_id].get("sessions", 0) + 1
        ATHLETE_DB[athlete_id]["bpi"] = ATHLETE_DB[athlete_id].get("bpi", 0) + summary["xp_earned"]
    _RATE_LIMITS.pop(session_id, None)  # PF-04: cleanup rate limit tracker
    _save_db()
    return summary


@router.get("/session/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    if session_id not in SESSION_DB:
        raise HTTPException(404, "Session not found")
    return SESSION_DB[session_id]


@router.get("/sessions", tags=["Sessions"])
async def list_sessions(
    athlete_id: Optional[str] = None,
    sport: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
):
    sessions = list(SESSION_DB.values())
    if athlete_id:
        sessions = [s for s in sessions if s.get("athlete_id") == athlete_id]
    if sport:
        sessions = [s for s in sessions if s.get("sport") == sport]
    if status:
        sessions = [s for s in sessions if s.get("status") == status]
    sessions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return {"total": len(sessions), "offset": offset, "limit": limit, "sessions": sessions[offset : offset + limit]}


@router.get("/sessions/active", tags=["Sessions"])
async def get_active_sessions():
    active = [
        {
            "session_id": sid,
            "athlete_id": s.get("athlete_id"),
            "sport": s.get("sport"),
            "started_at": s.get("started_at"),
            "frame_count": len(FRAME_BUFFER.get(sid, [])),
            "latest_score": RESULT_STORE.get(sid, {}).get("form_score"),
        }
        for sid, s in SESSION_DB.items()
        if s.get("status") == "active"
    ]
    return {"active_sessions": active, "count": len(active)}


# ─── rPPG WebSocket ─────────────────────────────────────────────────────────


@router.websocket("/rppg/live-stream/{session_id}")
async def rppg_live_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[WS-RPPG] Client connected: {session_id[:8]}")
    try:
        from services.rppg_processor import RPPGProcessor

        if session_id not in RPPG_STORE:
            RPPG_STORE[session_id] = RPPGProcessor()
        proc = RPPG_STORE[session_id]
        while True:
            data = await websocket.receive_json()
            if data.get("face_found") is False:
                await websocket.send_json(
                    {
                        "status": "warmup",
                        "signal_quality": "no_face",
                        "message": "Center your face",
                        "bpm": 0,
                        "hrv_ms": 0,
                        "waveform": [],
                    }
                )
                continue
            if data.get("image_b64"):
                try:
                    import base64 as _b64
                    from io import BytesIO as _BytesIO

                    import numpy as _np
                    from PIL import Image as _Image

                    _img_bytes = _b64.b64decode(data["image_b64"])
                    _img = _Image.open(_BytesIO(_img_bytes)).convert("RGB").resize((16, 16))
                    _arr = _np.array(_img, dtype=_np.float32)
                    r, g, b = float(_arr[:, :, 0].mean()), float(_arr[:, :, 1].mean()), float(_arr[:, :, 2].mean())
                except Exception as _e:
                    print(f"[RPPG] image_b64 decode error: {_e}")
                    continue
            else:
                r, g, b = data.get("r", 0.0), data.get("g", 0.0), data.get("b", 0.0)
            t = data.get("ts", time.time())
            proc.add_rgb(r, g, b, t)
            result = proc.compute()
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print(f"[WS-RPPG] Client disconnected: {session_id[:8]}")
    except Exception as e:
        print(f"[WS-RPPG] Error: {e}")


@router.get("/rppg/result/{session_id}", tags=["rPPG"])
async def rppg_get_result(session_id: str):
    proc = RPPG_STORE.get(session_id)
    if proc is None:
        return {"status": "no_data", "bpm": 0, "hrv_ms": 0, "waveform": []}
    return proc.compute()


# ─── Biomechanics WebSockets ────────────────────────────────────────────────


@router.websocket("/metrics/live/{session_id}")
async def websocket_live(websocket: WebSocket, session_id: str):
    await websocket.accept()
    WS_CONNECTIONS[session_id].append(websocket)
    print(f"[WS] Client connected to session {session_id[:8]}")
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping", "ts": time.time()}))
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected from session {session_id[:8]}")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        conns = WS_CONNECTIONS.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)


@router.websocket("/session/{session_id}/live-stream")
async def websocket_metadata_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[WS-STREAM] Native phone streaming for {session_id[:8]}")
    if session_id not in SESSION_DB:
        SESSION_DB[session_id] = {"athlete_id": "test", "sport": "vertical_jump", "status": "active"}
        FRAME_BUFFER[session_id] = []
    try:
        import types

        from services.pose_analyzer import PoseAnalyzer

        if session_id not in _POSE_ANALYZERS:
            _sport = SESSION_DB[session_id].get("sport", "vertical_jump")
            _ath_id = SESSION_DB[session_id].get("athlete_id")
            _height = float(ATHLETE_DB.get(_ath_id, {}).get("height_cm", 170))  # PF-03
            _ana = PoseAnalyzer(sport=_sport, body_height_cm=_height)
            _ana.model_path = _resolve_sport_model_path(_sport)  # PF-10
            _POSE_ANALYZERS[session_id] = _ana
        analyzer = _POSE_ANALYZERS[session_id]
        while True:
            data = await websocket.receive_json()
            points = data.get("points", [])
            if not points or len(points) < 33:
                continue
            lms = [
                types.SimpleNamespace(x=p.get("x", 0), y=p.get("y", 0), z=p.get("z", 0), visibility=p.get("v", 1.0))
                for p in points
            ]
            res = types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=lms))
            bio = analyzer.analyze(res)
            await websocket.send_json(
                {
                    "form_score": bio.form_score,
                    "phase_space_dm": getattr(bio, "phase_space_dm", 0),
                    "torsion_error": getattr(bio, "torsion_error", 0),
                    "dimensionless_jerk": getattr(bio, "dimensionless_jerk", 0),
                    "phase": getattr(bio, "phase", "setup"),
                    "primary_feedback": bio.primary_feedback,
                }
            )
    except WebSocketDisconnect:
        print(f"[WS-STREAM] Native device disconnected {session_id[:8]}")
    except Exception as e:
        print(f"[WS-STREAM] Error: {e}")


# ─── Dataset ────────────────────────────────────────────────────────────────


@router.get("/dataset/export", tags=["Dataset"])
async def export_dataset(format: str = Query(default="csv")):
    csv_path = DATASET_PATH / "training_data.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Dataset not found. Run: python generate_dataset.py")
    if format == "json":
        rows = []
        str_fields = {"session_id", "athlete_id", "sport", "phase_label", "quality_label", "feedback_tag"}
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({k: v if k in str_fields else float(v) for k, v in row.items()})
        return JSONResponse({"data": rows, "count": len(rows)})
    return FileResponse(csv_path, media_type="text/csv", filename="personal_health_dataset.csv")


@router.get("/dataset/stats", tags=["Dataset"])
async def dataset_stats():
    stats_path = DATASET_PATH / "sample_stats.json"
    if not stats_path.exists():
        return {"error": "Run: python generate_dataset.py"}
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


# PF-08: Model prediction stats endpoint
@router.get("/model/stats", tags=["Model"])
async def model_stats():
    """Read prediction log and return aggregate stats for drift detection."""
    from database import DB_PATH

    log_path = DB_PATH / "predictions.jsonl"
    if not log_path.exists():
        return {"total_predictions": 0, "message": "No predictions logged yet"}

    today = datetime.utcnow().date().isoformat()
    today_count = 0
    score_sum = 0.0
    quality_dist: dict = {"poor": 0, "average": 0, "good": 0, "elite": 0, "unknown": 0}
    sport_dist: dict = {}
    total = 0

    try:
        with open(log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                total += 1
                ts = entry.get("timestamp", 0)
                entry_date = datetime.utcfromtimestamp(ts).date().isoformat() if ts else ""
                if entry_date == today:
                    today_count += 1
                    score_sum += float(entry.get("form_score", 0))
                    q = entry.get("form_quality", "unknown")
                    if q in quality_dist:
                        quality_dist[q] += 1
                    sport = entry.get("sport", "unknown")
                    sport_dist[sport] = sport_dist.get(sport, 0) + 1
    except Exception as e:
        return {"error": f"Could not read prediction log: {e}"}

    avg_score = round(score_sum / today_count, 1) if today_count > 0 else 0
    return {
        "total_predictions": total,
        "today_predictions": today_count,
        "today_avg_form_score": avg_score,
        "quality_distribution_today": quality_dist,
        "predictions_by_sport_today": sport_dist,
    }


# ─── Fitness Test ───────────────────────────────────────────────────────────


@router.post("/fitness-test", tags=["Fitness Test"])
async def save_fitness_test(req: FitnessTestRequest):
    athlete = ATHLETE_DB.get(req.athlete_id)
    if not athlete:
        athlete = {"id": req.athlete_id, "name": req.athlete_id, "fitness_tests": []}
        ATHLETE_DB[req.athlete_id] = athlete
    if "fitness_tests" not in athlete:
        athlete["fitness_tests"] = []
    record = {
        "score": req.score,
        "level": req.level,
        "bmi": req.bmi,
        "sit_reach_cm": req.sit_reach_cm,
        "run_600_seconds": req.run_600_seconds,
        "age_group": req.age_group,
        "timestamp": datetime.utcnow().isoformat(),
    }
    athlete["fitness_tests"].insert(0, record)
    athlete["fitness_tests"] = athlete["fitness_tests"][:10]
    _save_db()
    return {"athlete_id": req.athlete_id, "score": req.score, "level": req.level, "timestamp": record["timestamp"]}


@router.get("/fitness-test/history/{athlete_id}", tags=["Fitness Test"])
async def get_fitness_test_history(athlete_id: str):
    athlete = ATHLETE_DB.get(athlete_id)
    if not athlete:
        return {"athlete_id": athlete_id, "history": []}
    return {"athlete_id": athlete_id, "history": athlete.get("fitness_tests", [])[:5]}
