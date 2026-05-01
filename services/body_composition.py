from __future__ import annotations

"""
Personal Health — Body Composition Tracker

Reads body_weight entries from wellness logs and computes:
  - Weight trend over a window
  - BMI (from height in athlete profile)
  - Estimated lean/fat split using a simple sport-adjusted formula
  - Weekly weight change rate

All calculations are estimates — appropriate for a fitness MVP with no DEXA.
"""

from __future__ import annotations

import statistics
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


# Sport-adjusted typical body fat % ranges for "athletic" build (mid-point used)
_SPORT_BF_ESTIMATE: dict[str, float] = {
    "sprint": 9.0,
    "vertical_jump": 12.0,
    "snatch": 13.0,
    "javelin": 14.0,
    "cricket_bat": 15.0,
    "squat": 15.0,
    "push_up": 14.0,
    "pull_up": 13.0,
}
_DEFAULT_BF = 15.0


def _bmi_band(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25.0:
        return "normal"
    if bmi < 30.0:
        return "overweight"
    return "obese"


def compute_body_composition(
    athlete: dict,
    days: int = 90,
) -> dict[str, Any]:
    """
    Extracts body_weight_kg entries from wellness_entries and returns trend data.
    """
    height_cm = float(athlete.get("height_cm") or 170)
    sport = athlete.get("sport", "vertical_jump")
    entries: list[dict] = athlete.get("wellness_entries") or []

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=days)

    weight_series: list[dict] = []
    for entry in entries:
        ts = _parse_dt(entry.get("logged_at"))
        if not ts or ts < cutoff:
            continue
        bw = (entry.get("physical") or {}).get("body_weight_kg")
        if bw is not None:
            try:
                weight_series.append({"date": ts.date().isoformat(), "weight_kg": float(bw)})
            except (TypeError, ValueError):
                continue

    weight_series.sort(key=lambda x: x["date"])

    if not weight_series:
        return {
            "athlete_id": athlete.get("id"),
            "has_data": False,
            "message": "No body weight entries found. Log weight in morning check-in.",
            "weight_series": [],
        }

    weights = [w["weight_kg"] for w in weight_series]
    latest_weight = weights[-1]
    first_weight = weights[0]

    # BMI
    height_m = height_cm / 100
    bmi = round(latest_weight / (height_m**2), 1)
    bmi_band = _bmi_band(bmi)

    # Weekly change rate
    days_span = max(
        1, (datetime.fromisoformat(weight_series[-1]["date"]) - datetime.fromisoformat(weight_series[0]["date"])).days
    )
    total_change = latest_weight - first_weight
    weekly_change = round((total_change / days_span) * 7, 2) if days_span > 0 else 0.0

    # Estimated body fat & lean mass
    bf_pct = _SPORT_BF_ESTIMATE.get(sport, _DEFAULT_BF)
    lean_mass = round(latest_weight * (1 - bf_pct / 100), 1)
    fat_mass = round(latest_weight * (bf_pct / 100), 1)

    trend = "gaining" if weekly_change > 0.1 else ("losing" if weekly_change < -0.1 else "stable")

    return {
        "athlete_id": athlete.get("id"),
        "has_data": True,
        "latest_weight_kg": latest_weight,
        "first_weight_kg": first_weight,
        "total_change_kg": round(total_change, 2),
        "weekly_change_kg": weekly_change,
        "trend": trend,
        "bmi": bmi,
        "bmi_band": bmi_band,
        "height_cm": height_cm,
        "estimated_bf_pct": bf_pct,
        "estimated_lean_mass_kg": lean_mass,
        "estimated_fat_mass_kg": fat_mass,
        "data_points": len(weight_series),
        "window_days": days,
        "weight_series": weight_series[-30:],
    }
