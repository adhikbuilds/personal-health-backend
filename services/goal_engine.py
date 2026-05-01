from __future__ import annotations

"""
Personal Health — Goal Engine

Athletes can set weekly goals and the engine tracks progress against them.
Goals are stored in the athlete record under `goals` key.

Supported goal types:
  - sessions_per_week  : integer target (e.g. 3)
  - form_score_target  : minimum avg_form_score (e.g. 75.0)
  - streak_days        : consecutive active days target (e.g. 7)
  - weekly_reps        : total reps this week (e.g. 100)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


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


def _week_start() -> datetime:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now - timedelta(days=now.weekday())


def _sessions_this_week(sessions: list[dict]) -> list[dict]:
    cutoff = _week_start().replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        s
        for s in sessions
        if s.get("status") == "completed" and (_parse_dt(s.get("started_at")) or datetime.min) >= cutoff
    ]


# ─── Default goal presets by sport ───────────────────────────────────────────

_SPORT_DEFAULTS: dict[str, dict[str, Any]] = {
    "vertical_jump": {"sessions_per_week": 4, "form_score_target": 70.0, "streak_days": 5, "weekly_reps": 80},
    "sprint": {"sessions_per_week": 5, "form_score_target": 72.0, "streak_days": 5, "weekly_reps": 60},
    "snatch": {"sessions_per_week": 3, "form_score_target": 68.0, "streak_days": 4, "weekly_reps": 50},
    "cricket_bat": {"sessions_per_week": 4, "form_score_target": 65.0, "streak_days": 4, "weekly_reps": 100},
    "javelin": {"sessions_per_week": 4, "form_score_target": 68.0, "streak_days": 4, "weekly_reps": 40},
    "squat": {"sessions_per_week": 3, "form_score_target": 70.0, "streak_days": 4, "weekly_reps": 60},
    "push_up": {"sessions_per_week": 4, "form_score_target": 65.0, "streak_days": 5, "weekly_reps": 120},
    "pull_up": {"sessions_per_week": 3, "form_score_target": 65.0, "streak_days": 4, "weekly_reps": 60},
}
_DEFAULT_GOALS = {"sessions_per_week": 3, "form_score_target": 65.0, "streak_days": 4, "weekly_reps": 60}


def default_goals(sport: str) -> dict[str, Any]:
    return dict(_SPORT_DEFAULTS.get(sport, _DEFAULT_GOALS))


def evaluate_goals(
    athlete: dict,
    sessions: list[dict],
    current_streak: int,
) -> dict[str, Any]:
    """
    Evaluate current week's progress against the athlete's goals.

    Returns a list of goal objects each with:
      { type, target, current, pct, status: 'on_track'|'behind'|'achieved' }
    """
    sport = athlete.get("sport", "vertical_jump")
    stored_goals: dict = athlete.get("goals") or default_goals(sport)
    week_sessions = _sessions_this_week(sessions)

    results: list[dict] = []

    # sessions_per_week
    if "sessions_per_week" in stored_goals:
        target = int(stored_goals["sessions_per_week"])
        current = len(week_sessions)
        pct = min(100, round(current / target * 100)) if target else 100
        results.append(
            {
                "type": "sessions_per_week",
                "label": "Sessions this week",
                "target": target,
                "current": current,
                "unit": "sessions",
                "pct": pct,
                "status": "achieved" if current >= target else ("on_track" if pct >= 60 else "behind"),
            }
        )

    # form_score_target
    if "form_score_target" in stored_goals:
        target = float(stored_goals["form_score_target"])
        scores = [
            float(s.get("avg_form_score") or s.get("form_score") or 0)
            for s in week_sessions
            if (s.get("avg_form_score") or s.get("form_score"))
        ]
        current_avg = round(sum(scores) / len(scores), 1) if scores else 0.0
        pct = min(100, round(current_avg / target * 100)) if target else 100
        results.append(
            {
                "type": "form_score_target",
                "label": "Avg form score",
                "target": target,
                "current": current_avg,
                "unit": "pts",
                "pct": pct,
                "status": "achieved" if current_avg >= target else ("on_track" if pct >= 80 else "behind"),
            }
        )

    # streak_days
    if "streak_days" in stored_goals:
        target = int(stored_goals["streak_days"])
        pct = min(100, round(current_streak / target * 100)) if target else 100
        results.append(
            {
                "type": "streak_days",
                "label": "Training streak",
                "target": target,
                "current": current_streak,
                "unit": "days",
                "pct": pct,
                "status": "achieved" if current_streak >= target else ("on_track" if pct >= 50 else "behind"),
            }
        )

    # weekly_reps
    if "weekly_reps" in stored_goals:
        target = int(stored_goals["weekly_reps"])
        current_reps = sum(int(s.get("rep_count") or 0) for s in week_sessions)
        pct = min(100, round(current_reps / target * 100)) if target else 100
        results.append(
            {
                "type": "weekly_reps",
                "label": "Reps this week",
                "target": target,
                "current": current_reps,
                "unit": "reps",
                "pct": pct,
                "status": "achieved" if current_reps >= target else ("on_track" if pct >= 60 else "behind"),
            }
        )

    overall_pct = round(sum(g["pct"] for g in results) / len(results)) if results else 0
    achieved = sum(1 for g in results if g["status"] == "achieved")

    return {
        "goals": results,
        "overall_pct": overall_pct,
        "achieved_count": achieved,
        "total_count": len(results),
        "week_start": _week_start().date().isoformat(),
        "message": _goal_message(achieved, len(results), overall_pct),
    }


def _goal_message(achieved: int, total: int, pct: int) -> str:
    if achieved == total:
        return "All goals achieved this week! Outstanding consistency."
    if pct >= 75:
        return f"{achieved}/{total} goals done. Strong week — finish strong."
    if pct >= 40:
        return f"Halfway there on {achieved}/{total} goals. Push through."
    return f"Early in the week — {total - achieved} goal(s) to hit. Let's go."
