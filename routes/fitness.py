from __future__ import annotations

"""
Fitness domain — Sessions, Pose Analysis, rPPG, Dataset, Fitness Test
All biomechanics-related endpoints live here.
"""

import asyncio
import base64
import csv
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import database
from database import (
    _POSE_ANALYZERS,
    _RATE_LIMITS,
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
from logging_setup import get_logger

router = APIRouter()
log = get_logger("routes.fitness")


def _hr_zone(bpm: float, resting_hr: int = 60, max_hr: int = 190) -> dict:
    """Classify heart rate into training zones.

    Uses HR reserve (Karvonen) so zones respect the athlete's resting baseline.
    Returns zone name, 1-5 index, and intensity % of HR reserve.
    """
    if bpm < 30:
        return {"zone": "unknown", "index": 0, "intensity_pct": 0.0, "color": "#64748b"}
    reserve = max(max_hr - resting_hr, 1)
    pct = max(0.0, min(1.0, (bpm - resting_hr) / reserve))
    if pct < 0.50:
        return {"zone": "recovery", "index": 1, "intensity_pct": round(pct * 100, 1), "color": "#38bdf8"}
    if pct < 0.60:
        return {"zone": "endurance", "index": 2, "intensity_pct": round(pct * 100, 1), "color": "#22c55e"}
    if pct < 0.70:
        return {"zone": "tempo", "index": 3, "intensity_pct": round(pct * 100, 1), "color": "#facc15"}
    if pct < 0.85:
        return {"zone": "threshold", "index": 4, "intensity_pct": round(pct * 100, 1), "color": "#f97316"}
    return {"zone": "anaerobic", "index": 5, "intensity_pct": round(pct * 100, 1), "color": "#ef4444"}


def _hr_summary_for_session(session_id: str) -> Optional[dict]:
    """Build an HR summary from the rPPG processor's full readings if one ran for this session."""
    proc = RPPG_STORE.get(session_id)
    if proc is None or not getattr(proc, "last_bpm", 0):
        return None
    bpms: list[float] = list(getattr(proc, "bpm_history", []) or [])
    if not bpms:
        bpms = [float(proc.last_bpm)]
    # Zone distribution in seconds (approx — we don't store per-sample timestamps for bpm_history)
    zone_counts: dict[str, int] = {}
    for bpm in bpms:
        z = _hr_zone(bpm).get("zone", "unknown")
        zone_counts[z] = zone_counts.get(z, 0) + 1
    return {
        "avg_bpm": round(sum(bpms) / len(bpms), 1),
        "peak_bpm": round(max(bpms), 1),
        "min_bpm": round(min(bpms), 1),
        "last_bpm": round(float(proc.last_bpm), 1),
        "last_hrv_ms": round(float(getattr(proc, "last_hrv", 0) or 0), 1),
        "last_zone": _hr_zone(float(proc.last_bpm)),
        "zone_distribution": zone_counts,
        "sample_count": len(bpms),
    }


# ─── Pydantic Models ────────────────────────────────────────────────────────


class StartSessionRequest(BaseModel):
    athlete_id: str = "athlete_01"
    sport: str = "vertical_jump"
    huddle_id: Optional[str] = None
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


_RESOLVED_MODEL_LOGGED: set[str] = set()


def _resolve_sport_model_path(sport: str) -> Optional[str]:
    """PF-10: pick a sport-specific .tflite if it exists, else fall back to the generic one.
    Returns the resolved path as a string, or None if no model is available.
    Logs once per (sport, outcome) pair so operators can see when a sport silently
    falls back to rule-based scoring.
    """
    from pathlib import Path as _P

    models_dir = _P(__file__).resolve().parent.parent / "models"
    specific = models_dir / f"pose_classifier_{sport}.tflite"
    generic = models_dir / "pose_classifier.tflite"

    if specific.exists():
        key = f"{sport}:specific"
        if key not in _RESOLVED_MODEL_LOGGED:
            log.info("sport model resolved", extra={"sport": sport, "kind": "specific"})
            _RESOLVED_MODEL_LOGGED.add(key)
        return str(specific)

    if generic.exists():
        key = f"{sport}:generic"
        if key not in _RESOLVED_MODEL_LOGGED:
            log.info("sport model resolved", extra={"sport": sport, "kind": "generic"})
            _RESOLVED_MODEL_LOGGED.add(key)
        return str(generic)

    key = f"{sport}:none"
    if key not in _RESOLVED_MODEL_LOGGED:
        log.warning("no pose model available — falling back to rule-based", extra={"sport": sport})
        _RESOLVED_MODEL_LOGGED.add(key)
    return None


async def analysis_worker():
    while True:
        try:
            item = await database.ANALYSIS_QUEUE.get()
            session_id, image_b64, sport, frame_dict = item
            try:
                from services.pose_analyzer import PoseAnalyzer

                if session_id not in _POSE_ANALYZERS:
                    # PF-03: use athlete's actual height for jump-height calc
                    _ath_id = SESSION_DB.get(session_id, {}).get("athlete_id")
                    _height = float(ATHLETE_DB.get(_ath_id, {}).get("height_cm", 170))
                    analyzer = PoseAnalyzer(sport=sport, body_height_cm=_height)
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
                        "shoulder_angle_l": angles.get("SHOULDER_L", 0.0),
                        "shoulder_angle_r": angles.get("SHOULDER_R", 0.0),
                        "ankle_dorsiflexion_l": angles.get("ANKLE_L", 0.0),
                        "ankle_dorsiflexion_r": angles.get("ANKLE_R", 0.0),
                        "spine_deviation": result.get("spine_deviation", 0.0),
                        "shoulder_hip_sep": result.get("shoulder_hip_sep", 0.0),
                        "head_forward_pos": result.get("head_forward_pos", 0.0),
                        "com_height_norm": result.get("com_height_norm", 0.5),
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
                        log.warning("prediction log write failed", extra={"error": str(plog_err)})

                    log.info(
                        "frame analyzed",
                        extra={
                            "session_id": session_id[:8],
                            "form_score": round(result["form_score"], 1),
                            "form_quality": result["form_quality"],
                            "phase": result["phase"],
                        },
                    )
                    await _broadcast(session_id, {"type": "frame", **result_entry})
                else:
                    log.debug("no pose in frame", extra={"session_id": session_id[:8]})
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
                log.error("analysis worker error", extra={"error": str(e), "session_id": session_id[:8]})
            finally:
                database.ANALYSIS_QUEUE.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("analysis worker fatal", extra={"error": str(e)})
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
                                log.warning("frame file unlink failed", extra={"path": str(fp), "error": str(rm_err)})
                    if removed:
                        log.info("frame files cleaned up", extra={"removed": removed})
            except Exception as cleanup_err:
                log.error("frame cleanup error", extra={"error": str(cleanup_err)})

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
                    session["ended_at"] = datetime.now(timezone.utc).isoformat()
                    session["auto_ended"] = True
                    log.info("auto-ended stale session", extra={"session_id": sid[:8]})
            _save_db()
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("cleanup worker error", extra={"error": str(e)})
            await asyncio.sleep(60)


# ─── Session Endpoints ──────────────────────────────────────────────────────


@router.post("/session/start", tags=["Sessions"])
async def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "athlete_id": req.athlete_id,
        "sport": req.sport,
        "status": "active",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "frame_count": 0,
        "summary": None,
        "huddle_id": req.huddle_id,
    }
    SESSION_DB[session_id] = session
    FRAME_BUFFER[session_id] = []
    _save_db()  # durable handshake — crash before first frame still preserves the session

    if req.huddle_id:
        try:
            from services.huddle import bind_session_to_huddle

            bind_session_to_huddle(req.huddle_id, req.athlete_id, session_id)
        except Exception as bind_err:
            log.warning(
                "huddle bind failed",
                extra={"session_id": session_id[:8], "huddle_id": req.huddle_id, "error": str(bind_err)},
            )

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
        log.warning("frame persist failed", extra={"session_id": session_id[:8], "error": str(fperr)})
    image_b64 = frame_dict.pop("image_b64", None)
    frame_dict["frame_num"] = len(FRAME_BUFFER[session_id])
    frame_dict["timestamp"] = time.time()
    frame_dict["pose_detected"] = None
    FRAME_BUFFER[session_id].append(frame_dict)
    SESSION_DB[session_id]["frame_count"] += 1
    if image_b64 and database.ANALYSIS_QUEUE is not None:
        try:
            database.ANALYSIS_QUEUE.put_nowait((session_id, image_b64, sport, frame_dict))
        except asyncio.QueueFull:
            log.warning("analysis queue full, dropped frame", extra={"session_id": session_id[:8]})
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
        log.error("calibration error", extra={"error": str(e)})
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
                    log.info(
                        "recovered frames from disk",
                        extra={"session_id": session_id[:8], "count": len(recovered)},
                    )
        except Exception as recover_err:
            log.warning("frame recovery failed", extra={"session_id": session_id[:8], "error": str(recover_err)})
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
    # Enrich summary with coaching + data quality stats
    try:
        from services.data_pipeline import enrich_session_summary

        summary = enrich_session_summary(session_id, summary)
    except Exception as enrich_err:
        summary["coaching"] = {"patterns": [], "summary": f"Analysis unavailable: {enrich_err}"}

    # Fold in heart-rate summary if an rPPG stream ran during the session
    hr_summary = _hr_summary_for_session(session_id)
    if hr_summary:
        summary["heart_rate"] = hr_summary

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
    log.info("rppg client connected", extra={"session_id": session_id[:8]})
    try:
        from services.rppg_processor import RPPGProcessor

        if session_id not in RPPG_STORE:
            RPPG_STORE[session_id] = RPPGProcessor()
        proc = RPPG_STORE[session_id]
        # Track bpm history so end_session can summarise zones
        if not hasattr(proc, "bpm_history"):
            proc.bpm_history = []

        # Pull resting/max HR from athlete profile when available
        _ath_id = SESSION_DB.get(session_id, {}).get("athlete_id", "")
        _athlete = ATHLETE_DB.get(_ath_id, {}) if _ath_id else {}
        resting_hr = int(_athlete.get("resting_hr", 60))
        max_hr = int(_athlete.get("max_hr", 190))

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
                        "zone": _hr_zone(0),
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
                    _img = _Image.open(_BytesIO(_img_bytes)).convert("RGB")
                    _img_np = _np.array(_img)
                    ih, iw = _img_np.shape[:2]

                    # Face detection: run every 10th frame, cache bbox
                    if not hasattr(proc, "_face_bbox"):
                        proc._face_bbox = None
                        proc._face_det_counter = 0
                        # Init face detector once
                        try:
                            from pathlib import Path

                            import mediapipe as _mp
                            from mediapipe.tasks.python import BaseOptions as _BO
                            from mediapipe.tasks.python.vision import FaceDetector as _FD
                            from mediapipe.tasks.python.vision import FaceDetectorOptions as _FDO

                            _model = Path(__file__).parent.parent / "models" / "face_detector.tflite"
                            if _model.exists():
                                proc._face_det = _FD.create_from_options(
                                    _FDO(base_options=_BO(model_asset_path=str(_model)))
                                )
                            else:
                                proc._face_det = None
                        except Exception:
                            proc._face_det = None

                    # Run face detection every 10 frames (expensive), cache result
                    proc._face_det_counter = getattr(proc, "_face_det_counter", 0) + 1
                    if proc._face_det and (proc._face_bbox is None or proc._face_det_counter % 10 == 0):
                        try:
                            import mediapipe as _mp

                            mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=_img_np)
                            det = proc._face_det.detect(mp_img)
                            if det.detections:
                                bb = det.detections[0].bounding_box
                                proc._face_bbox = (bb.origin_x, bb.origin_y, bb.width, bb.height)
                            else:
                                proc._face_bbox = None
                        except Exception:
                            pass

                    # Extract RGB from face ROI (or center fallback)
                    if proc._face_bbox:
                        bx, by, bw, bh = proc._face_bbox
                        # Forehead/cheek region: top 50%, center 60%
                        fx = max(0, bx + int(bw * 0.2))
                        fy = max(0, by + int(bh * 0.1))
                        fw = min(iw - fx, int(bw * 0.6))
                        fh = min(ih - fy, int(bh * 0.5))
                        roi = _img_np[fy : fy + fh, fx : fx + fw] if fw > 0 and fh > 0 else None
                    else:
                        roi = None

                    if roi is not None and roi.size > 0:
                        r, g, b = float(roi[:, :, 0].mean()), float(roi[:, :, 1].mean()), float(roi[:, :, 2].mean())
                    else:
                        # No face — center crop fallback
                        cy, cx = ih // 2, iw // 2
                        ch, cw = ih // 5, iw // 5
                        crop = _img_np[max(0, cy - ch) : cy + ch, max(0, cx - cw) : cx + cw]
                        r, g, b = float(crop[:, :, 0].mean()), float(crop[:, :, 1].mean()), float(crop[:, :, 2].mean())

                    # Send face status back to client
                    result_extra = {"face_detected": proc._face_bbox is not None}
                except Exception as _e:
                    log.warning("rppg image decode failed", extra={"session_id": session_id[:8], "error": str(_e)})
                    continue
            else:
                r, g, b = data.get("r", 0.0), data.get("g", 0.0), data.get("b", 0.0)
                finger_on = r > 100 and (r > g * 1.3)
                result_extra = {"face_detected": finger_on}
            t = data.get("ts", time.time())
            proc.add_rgb(r, g, b, t)
            result = proc.compute()
            result.update(result_extra)

            # Enrich with training zone and record history for session summary
            bpm_val = float(result.get("bpm", 0) or 0)
            if bpm_val > 30 and result.get("status") == "ok":
                result["zone"] = _hr_zone(bpm_val, resting_hr, max_hr)
                # Keep last 10 minutes at ~2 Hz = 1200 samples
                proc.bpm_history.append(bpm_val)
                if len(proc.bpm_history) > 1200:
                    proc.bpm_history = proc.bpm_history[-1200:]
            else:
                result["zone"] = _hr_zone(0)

            # Broadcast to dashboard listeners so biomech + HR land on one stream
            await _broadcast(
                session_id,
                {
                    "type": "hr",
                    "bpm": result.get("bpm"),
                    "hrv_ms": result.get("hrv_ms"),
                    "zone": result["zone"],
                    "signal_quality": result.get("signal_quality"),
                    "ts": t,
                },
            )

            await websocket.send_json(result)
    except WebSocketDisconnect:
        log.info("rppg client disconnected", extra={"session_id": session_id[:8]})
    except Exception as e:
        log.error("rppg stream error", extra={"session_id": session_id[:8], "error": str(e)})


@router.get("/rppg/result/{session_id}", tags=["rPPG"])
async def rppg_get_result(session_id: str):
    proc = RPPG_STORE.get(session_id)
    if proc is None:
        return {"status": "no_data", "bpm": 0, "hrv_ms": 0, "waveform": []}
    return proc.compute()


# ─── Biomechanics WebSockets ────────────────────────────────────────────────


# DEPRECATED: No active consumers as of 2026-04-26. Candidate for removal.
# Was intended for: dashboard live metrics feed — relays form scores to web observers
@router.websocket("/metrics/live/{session_id}")
async def websocket_live(websocket: WebSocket, session_id: str):
    await websocket.accept()
    WS_CONNECTIONS[session_id].append(websocket)
    log.info("metrics ws connected", extra={"session_id": session_id[:8]})
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping", "ts": time.time()}))
    except WebSocketDisconnect:
        log.info("metrics ws disconnected", extra={"session_id": session_id[:8]})
    except Exception as e:
        log.error("metrics ws error", extra={"session_id": session_id[:8], "error": str(e)})
    finally:
        conns = WS_CONNECTIONS.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)


# DEPRECATED: No active consumers as of 2026-04-26. Candidate for removal.
# Was intended for: Android native landmark stream — api.connectLiveStream() exists but no screen calls it
@router.websocket("/session/{session_id}/live-stream")
async def websocket_metadata_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    log.info("native landmark stream connected", extra={"session_id": session_id[:8]})
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
        log.info("native landmark stream disconnected", extra={"session_id": session_id[:8]})
    except Exception as e:
        log.error("native landmark stream error", extra={"session_id": session_id[:8], "error": str(e)})


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

    today = datetime.now(timezone.utc).date().isoformat()
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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


# ─── Phase 2: WebSocket JPEG frame ingest ──────────────────────────────────
# Replaces the per-frame HTTP POST with a single long-lived socket so the
# client can stream at higher fps without paying TCP+HTTP overhead per frame.
# Accepts JSON messages of the shape:
#   { "image_b64": "<jpeg-base64>", "ts": <ms> }
# Emits, after each frame is analyzed:
#   { "type": "result", "frame_num": N, "form_score": ..., "form_quality": ...,
#     "primary_feedback": "...", "phase": "..." }
# Plus periodic { "type": "ping" } when idle.


@router.websocket("/ws/session/{session_id}/frames-jpeg")
async def websocket_jpeg_frames(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in SESSION_DB:
        await websocket.send_json({"type": "error", "code": "session_not_found"})
        await websocket.close(code=4404)
        return
    if SESSION_DB[session_id].get("status") != "active":
        await websocket.send_json({"type": "error", "code": "session_not_active"})
        await websocket.close(code=4400)
        return

    sport = SESSION_DB[session_id].get("sport", "vertical_jump")
    log.info("ws-frames-jpeg connected", extra={"session_id": session_id[:8], "sport": sport})

    last_pushed_ts = [0.0]
    _exclude_keys = {"keypoints", "data_source"}

    async def result_pusher():
        while True:
            await asyncio.sleep(0.1)
            latest = RESULT_STORE.get(session_id)
            if latest and latest.get("analyzed_at", 0) > last_pushed_ts[0]:
                last_pushed_ts[0] = latest["analyzed_at"]
                payload = {"type": "result", **{k: v for k, v in latest.items() if k not in _exclude_keys}}
                try:
                    await websocket.send_json(payload)
                except Exception:
                    return

    pusher_task = asyncio.create_task(result_pusher())
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping", "ts": time.time()})
                continue

            if raw.get("type") == "websocket.disconnect":
                break

            image_b64 = None
            if raw.get("bytes"):
                image_b64 = base64.b64encode(raw["bytes"]).decode("ascii")
            elif raw.get("text"):
                try:
                    msg = json.loads(raw["text"])
                except (json.JSONDecodeError, TypeError):
                    continue
                image_b64 = (msg or {}).get("image_b64")
            else:
                continue

            if not image_b64:
                continue

            now = time.time()
            if now - _RATE_LIMITS.get(session_id, 0) < 0.05:
                continue
            _RATE_LIMITS[session_id] = now

            frame_num = SESSION_DB[session_id].get("frame_count", 0)
            frame_dict = {
                "frame_num": frame_num,
                "timestamp": now,
                "pose_detected": None,
                "via": "ws",
            }
            FRAME_BUFFER[session_id].append(frame_dict)
            SESSION_DB[session_id]["frame_count"] = frame_num + 1

            if database.ANALYSIS_QUEUE is not None:
                try:
                    database.ANALYSIS_QUEUE.put_nowait((session_id, image_b64, sport, frame_dict))
                except asyncio.QueueFull:
                    log.warning("analysis queue full (ws), dropped frame", extra={"session_id": session_id[:8]})
    except WebSocketDisconnect:
        log.info("ws-frames-jpeg disconnected", extra={"session_id": session_id[:8]})
    except Exception as e:
        log.warning("ws-frames-jpeg error", extra={"session_id": session_id[:8], "error": str(e)})
    finally:
        pusher_task.cancel()
