from __future__ import annotations

"""
Personal Health — Recovery Engine

Computes a daily recovery readiness score (0-100) and a training
recommendation (rest / active_recovery / light / normal / hard) based on:

  - Acute:Chronic Workload Ratio (ACWR) from session loads
  - Wellness composite: sleep quality, soreness, energy, mood
  - Session frequency in the past 48 h

The ACWR model uses a simple rolling-window approximation:
  acute  = mean daily load over the last 7 days
  chronic = mean daily load over the last 28 days

Load per session is a composite of form_score * rep_count * intensity_factor.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from logging_setup import get_logger

log = get_logger("services.recovery_engine")

# ─── Intensity factors per sport ─────────────────────────────────────────────

_INTENSITY: dict[str, float] = {
    "snatch": 1.4,
    "vertical_jump": 1.2,
    "sprint": 1.3,
    "javelin": 1.1,
    "cricket_bat": 0.9,
    "squat": 1.1,
    "push_up": 0.8,
    "pull_up": 0.9,
}
_DEFAULT_INTENSITY = 1.0


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _session_load(session: dict) -> float:
    """Compute a numeric load value for a single completed session."""
    score = float(session.get("avg_form_score") or session.get("form_score") or 50)
    reps = int(session.get("rep_count") or 1)
    sport = session.get("sport", "")
    intensity = _INTENSITY.get(sport, _DEFAULT_INTENSITY)
    return round(score * reps * intensity, 2)


def _daily_loads(sessions: list[dict], window_days: int) -> list[float]:
    """Return a list of total load per day (oldest→newest) over window_days."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=window_days)
    by_day: dict[str, float] = {}
    for s in sessions:
        if s.get("status") != "completed":
            continue
        ts = _parse_dt(s.get("started_at"))
        if not ts or ts < cutoff:
            continue
        day = ts.date().isoformat()
        by_day[day] = by_day.get(day, 0.0) + _session_load(s)
    # Fill gaps with 0 so averages are honest
    filled: list[float] = []
    for i in range(window_days):
        d = (now - timedelta(days=window_days - 1 - i)).date().isoformat()
        filled.append(by_day.get(d, 0.0))
    return filled


def _acwr(sessions: list[dict]) -> float:
    """Acute:Chronic Workload Ratio, clamped to [0.0, 3.0]."""
    acute_loads = _daily_loads(sessions, 7)
    chronic_loads = _daily_loads(sessions, 28)
    acute = sum(acute_loads) / 7
    chronic = sum(chronic_loads) / 28
    if chronic < 1e-6:
        return 0.0
    return min(acute / chronic, 3.0)


def _sessions_past_48h(sessions: list[dict]) -> int:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(hours=48)
    count = 0
    for s in sessions:
        if s.get("status") != "completed":
            continue
        ts = _parse_dt(s.get("started_at"))
        if ts and ts >= cutoff:
            count += 1
    return count


# ─── Public API ──────────────────────────────────────────────────────────────


def compute_recovery_score(
    sessions: list[dict],
    wellness_today: dict | None = None,
) -> dict[str, Any]:
    """
    Returns:
      {
        score: int (0-100),
        acwr: float,
        load_status: 'undertraining'|'optimal'|'high'|'overreaching',
        sessions_48h: int,
        wellness_penalty: int,
        recommendation: 'rest'|'active_recovery'|'light'|'normal'|'hard',
        reasons: list[str],
      }
    """
    acwr = _acwr(sessions)
    sessions_48h = _sessions_past_48h(sessions)

    reasons: list[str] = []

    # Base score from ACWR
    if acwr < 0.3:
        load_status = "undertraining"
        base_score = 90
        reasons.append("low recent load — body is rested")
    elif acwr <= 1.3:
        load_status = "optimal"
        base_score = 80
        reasons.append("workload ratio in optimal range")
    elif acwr <= 1.8:
        load_status = "high"
        base_score = 55
        reasons.append(f"acute load is {acwr:.2f}x chronic — elevated fatigue risk")
    else:
        load_status = "overreaching"
        base_score = 30
        reasons.append(f"ACWR {acwr:.2f} — overreaching zone, rest required")

    # 48-h session frequency penalty
    freq_penalty = 0
    if sessions_48h >= 3:
        freq_penalty = 20
        reasons.append(f"{sessions_48h} sessions in last 48 h — high frequency")
    elif sessions_48h == 2:
        freq_penalty = 8
        reasons.append("2 sessions in last 48 h")

    # Wellness penalty from morning check-in
    wellness_penalty = 0
    if wellness_today:
        soreness = int(wellness_today.get("soreness") or 3)
        energy = int(wellness_today.get("energy") or 5)
        sleep_q = int(wellness_today.get("sleep_quality") or 3)

        if soreness >= 4:
            wellness_penalty += 10
            reasons.append("high soreness reported")
        if energy <= 3:
            wellness_penalty += 8
            reasons.append("low energy reported")
        if sleep_q <= 2:
            wellness_penalty += 10
            reasons.append("poor sleep quality reported")

    score = max(0, min(100, base_score - freq_penalty - wellness_penalty))

    # Map score → recommendation
    if score >= 80:
        rec = "hard"
    elif score >= 65:
        rec = "normal"
    elif score >= 50:
        rec = "light"
    elif score >= 30:
        rec = "active_recovery"
    else:
        rec = "rest"

    return {
        "score": score,
        "acwr": round(acwr, 3),
        "load_status": load_status,
        "sessions_48h": sessions_48h,
        "wellness_penalty": wellness_penalty,
        "recommendation": rec,
        "reasons": reasons,
    }


_REC_DETAIL: dict[str, dict[str, str]] = {
    "hard": {
        "label": "Full Training",
        "description": "Body is well-rested. Push intensity today.",
        "tip": "Prime window for high-load power or speed work.",
    },
    "normal": {
        "label": "Normal Session",
        "description": "Ready for your regular training session.",
        "tip": "Good day for technique + moderate strength work.",
    },
    "light": {
        "label": "Light Session",
        "description": "Some accumulated fatigue — keep it moderate.",
        "tip": "Focus on quality reps, not volume. Stop before failure.",
    },
    "active_recovery": {
        "label": "Active Recovery",
        "description": "High fatigue detected. Keep intensity very low.",
        "tip": "20-30 min light mobility, walking, or swimming only.",
    },
    "rest": {
        "label": "Rest Day",
        "description": "Overreaching or high wellness penalties — rest today.",
        "tip": "Sleep, hydrate, and do 5-10 min gentle stretching.",
    },
}


def enrich_recommendation(rec: str) -> dict[str, str]:
    return _REC_DETAIL.get(rec, _REC_DETAIL["normal"])
