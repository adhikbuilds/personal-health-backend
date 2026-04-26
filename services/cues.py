from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from database import SESSION_DB

GLOBAL_LANDING_THRESHOLD = 1.2


@dataclass
class CueRule:
    trigger_fn: Callable[["CueState", dict], bool]
    message: str
    cooldown_s: float
    urgency: str = "normal"


@dataclass
class CueState:
    session_id: str
    athlete_id: str
    sport: str
    rep_index: int = 0
    last_phase: str | None = None
    cooldown_until: float = 0.0
    knee_drive_bad_reps: deque = field(default_factory=lambda: deque(maxlen=3))
    stride_ratio_bad_reps: deque = field(default_factory=lambda: deque(maxlen=3))
    landing_bad_reps: deque = field(default_factory=lambda: deque(maxlen=2))
    phase_times: dict = field(default_factory=dict)
    baseline_landing_impact: float = GLOBAL_LANDING_THRESHOLD
    last_warning_ts: float = -9999.0
    current_drive_bad: bool | None = None


_STATE: dict[str, CueState] = {}


def _compute_stride_ratio(state: CueState, frame: dict) -> float | None:
    ts = float(frame.get("timestamp") or 0)
    phase = frame.get("phase")
    if not ts or not phase:
        return None

    if phase == "drive":
        state.phase_times["drive"] = ts
    elif phase == "flight":
        state.phase_times["flight"] = ts
    elif phase == "landing":
        state.phase_times["landing"] = ts

    drive_ts = state.phase_times.get("drive")
    flight_ts = state.phase_times.get("flight")
    landing_ts = state.phase_times.get("landing")
    if drive_ts and flight_ts and landing_ts and drive_ts < flight_ts < landing_ts:
        ground_contact = max(flight_ts - drive_ts, 0.001)
        stride_time = max(landing_ts - drive_ts, 0.001)
        return round(stride_time / ground_contact, 3)
    return None


def _landing_impact_proxy(frame: dict) -> float:
    knee_l = float(frame.get("knee_angle_l") or 180)
    knee_r = float(frame.get("knee_angle_r") or 180)
    trunk = float(frame.get("trunk_lean") or 0)
    asym = abs(knee_l - knee_r) / 90.0
    knee_stiffness = max(0.0, (190 - min(knee_l, knee_r)) / 100.0)
    return round(knee_stiffness + min(trunk / 45.0, 1.0) + asym, 3)


def _load_landing_baseline(athlete_id: str) -> float:
    if not athlete_id:
        return GLOBAL_LANDING_THRESHOLD
    values: list[float] = []
    for session in SESSION_DB.values():
        if session.get("athlete_id") != athlete_id or session.get("status") != "completed":
            continue
        for frame in session.get("frames", [])[-40:]:
            if frame.get("landing_impact_proxy") is not None:
                values.append(float(frame["landing_impact_proxy"]))
    if len(values) >= 5:
        recent = values[-50:]
        return round(sum(recent) / len(recent), 3)
    return GLOBAL_LANDING_THRESHOLD


def _trigger_knee_drive(state: CueState, frame: dict, rep_completed: bool) -> bool:
    phase = frame.get("phase")
    if phase == "drive":
        knee = min(float(frame.get("knee_angle_l") or 180), float(frame.get("knee_angle_r") or 180))
        state.current_drive_bad = knee < 85
        return False
    if rep_completed:
        state.rep_index += 1
        state.knee_drive_bad_reps.append(bool(state.current_drive_bad))
        state.current_drive_bad = None
    return len(state.knee_drive_bad_reps) == 3 and all(state.knee_drive_bad_reps)


def _trigger_land_softer(state: CueState, frame: dict, _rep_completed: bool) -> bool:
    if frame.get("phase") != "landing":
        return False
    impact = _landing_impact_proxy(frame)
    frame["landing_impact_proxy"] = impact
    state.landing_bad_reps.append(impact > state.baseline_landing_impact * 1.2)
    return len(state.landing_bad_reps) == 2 and all(state.landing_bad_reps)


def _trigger_longer_stride(state: CueState, _frame: dict, rep_completed: bool, ratio: float | None) -> bool:
    if ratio is None:
        return False
    if rep_completed:
        state.stride_ratio_bad_reps.append(ratio < 1.1)
    return len(state.stride_ratio_bad_reps) == 3 and all(state.stride_ratio_bad_reps)


SPRINT_RULES = [
    CueRule(_trigger_knee_drive, "drive your knee higher", 4.0, "normal"),
    CueRule(_trigger_land_softer, "land softer", 4.0, "normal"),
    CueRule(_trigger_longer_stride, "longer stride", 4.0, "normal"),
]


def _warning_cue(frame: dict) -> dict | None:
    if frame.get("phase") != "landing":
        return None
    knee_l = float(frame.get("knee_angle_l") or 180)
    knee_r = float(frame.get("knee_angle_r") or 180)
    if min(knee_l, knee_r) < 65:
        return {"type": "cue", "text": "pause — check your landing", "urgency": "warning"}
    return None


def _state_for(session_id: str, athlete_id: str, sport: str) -> CueState:
    state = _STATE.get(session_id)
    if state is None:
        state = CueState(
            session_id=session_id,
            athlete_id=athlete_id,
            sport=sport,
            baseline_landing_impact=_load_landing_baseline(athlete_id),
        )
        _STATE[session_id] = state
    return state


def evaluate_cues(session_id: str, athlete_id: str, sport: str, frame: dict) -> list[dict]:
    if sport != "sprint":
        return []

    state = _state_for(session_id, athlete_id, sport)
    ts = float(frame.get("timestamp") or 0)
    phase = frame.get("phase")
    rep_completed = state.last_phase == "drive" and phase == "flight"
    ratio = _compute_stride_ratio(state, frame)
    warning = _warning_cue(frame)
    state.last_phase = phase
    if warning and ts >= state.last_warning_ts + 4.0:
        state.last_warning_ts = ts
        return [warning]

    if ts < state.cooldown_until:
        return []

    if _trigger_knee_drive(state, frame, rep_completed):
        state.cooldown_until = ts + 4.0
        return [{"type": "cue", "text": "drive your knee higher", "urgency": "normal"}]
    if _trigger_land_softer(state, frame, rep_completed):
        state.cooldown_until = ts + 4.0
        return [{"type": "cue", "text": "land softer", "urgency": "normal"}]
    if _trigger_longer_stride(state, frame, rep_completed, ratio):
        state.cooldown_until = ts + 4.0
        return [{"type": "cue", "text": "longer stride", "urgency": "normal"}]
    return []


def clear_cue_state(session_id: str) -> None:
    _STATE.pop(session_id, None)
