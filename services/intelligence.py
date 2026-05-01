"""
ActiveBharat — Intelligence Layer
──────────────────────────────────────────────────────────────────────────────
Components:
  TrendAnalyzer      — detects improving/declining form over sessions
  AnomalyDetector    — flags sudden drops (fatigue / injury signals)
  CoachRecommender   — generates 3 targeted drill suggestions per sport
  ProgressPredictor  — estimates weeks to next athlete tier
  StreakTracker      — counts consecutive active training days per athlete

Phase 1 (NOW):   rule-based on session summaries from SESSION_DB
Phase 2 (NEXT):  replace rules with ML model trained on real session data
Phase 3 (LATER): personalized model per athlete
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# ─── Sport-specific coaching rules ───────────────────────────────────────────

SPORT_DRILLS: dict[str, list[dict[str, str]]] = {
    "vertical_jump": [
        {"drill": "Box Jumps (3×8)", "focus": "Explosive power", "cue": "Drive arms up at peak"},
        {"drill": "Hip Hinge Holds", "focus": "Hip loading", "cue": "Feel the stretch in glutes"},
        {"drill": "Single-Leg Calf Raises (3×15)", "focus": "Ankle stiffness", "cue": "Control the descent"},
    ],
    "snatch": [
        {"drill": "Overhead Squat (light)", "focus": "Mobility", "cue": "Wrists turned out, bar over heels"},
        {"drill": "Hang Pulls", "focus": "Bar path", "cue": "Keep the bar close to body"},
        {"drill": "Pause Squats", "focus": "Leg drive", "cue": "3-second pause at bottom"},
    ],
    "sprint": [
        {"drill": "A-March", "focus": "Knee drive", "cue": "Drive knee to hip height"},
        {"drill": "Wall Drives", "focus": "Acceleration", "cue": "45° body angle"},
        {"drill": "Bounding Strides", "focus": "Ground contact time", "cue": "Minimise ground contact"},
    ],
    "javelin": [
        {"drill": "Rotation Throws (medicine ball)", "focus": "Hip rotation", "cue": "Lead with elbow"},
        {"drill": "Standing Release Drill", "focus": "Release angle", "cue": "45° wrist snap"},
        {"drill": "Step-Step-Throw Sequence", "focus": "Run-up rhythm", "cue": "Plant foot wide"},
    ],
    "cricket_bat": [
        {"drill": "Shadow Cover Drive", "focus": "Footwork", "cue": "Front foot to pitch of ball"},
        {"drill": "Tee Batting (front foot)", "focus": "Head position", "cue": "Head level, watch the tee"},
        {"drill": "Throw-Down Off-Spin", "focus": "Reading spin", "cue": "Read from the hand"},
    ],
    "squat": [
        {"drill": "Goblet Squat (light KB)", "focus": "Depth & upright torso", "cue": "Elbows push knees out"},
        {"drill": "Pause Squat (3s hold)", "focus": "Bottom position stability", "cue": "Brace core, stay tall"},
        {"drill": "Box Squat (parallel box)", "focus": "Hip hinge pattern", "cue": "Sit back, not down"},
    ],
    "push_up": [
        {"drill": "Scapular Push-Ups", "focus": "Scap protraction", "cue": "Round upper back at top"},
        {"drill": "Tempo Push-Ups (3-1-1)", "focus": "Chest stretch depth", "cue": "Lower slow, push fast"},
        {"drill": "Archer Push-Ups", "focus": "Unilateral load", "cue": "Keep hips square throughout"},
    ],
    "pull_up": [
        {"drill": "Dead Hang (30–60 s)", "focus": "Grip & shoulder mobility", "cue": "Depress scapula, breathe"},
        {"drill": "Scapular Pull-Ups", "focus": "Lat activation", "cue": "Pull shoulder blades down first"},
        {"drill": "Negatives (5 s descent)", "focus": "Eccentric strength", "cue": "Resist gravity the whole way"},
    ],
}

TIER_BPI_THRESHOLDS = {
    "Block": 0,
    "District": 5000,
    "State": 15000,
    "National": 30000,
}


# ─── TrendAnalyzer ────────────────────────────────────────────────────────────


class TrendAnalyzer:
    """Detects form score trend over the last N sessions."""

    MIN_SESSIONS = 3

    def analyze(self, session_summaries: list[dict]) -> dict:
        """
        Args:
            session_summaries: list of completed session summary dicts,
                               sorted oldest-first.
        Returns:
            { trend: 'improving'|'declining'|'stable'|'insufficient_data',
              slope: float,  # change per session
              peak: float,
              recent_avg: float }
        """
        scores = [s.get("avg_form_score", 0) for s in session_summaries if s.get("avg_form_score")]

        if len(scores) < self.MIN_SESSIONS:
            return {
                "trend": "insufficient_data",
                "slope": 0.0,
                "peak": max(scores) if scores else 0,
                "recent_avg": sum(scores) / len(scores) if scores else 0,
            }

        # Simple linear regression slope
        n = len(scores)
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(scores) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, scores))
        denom = sum((x - mx) ** 2 for x in xs) or 1
        slope = num / denom

        if slope > 1.5:
            trend = "improving"
        elif slope < -1.5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 2),
            "peak": round(max(scores), 1),
            "recent_avg": round(sum(scores[-3:]) / min(len(scores), 3), 1),
        }


# ─── AnomalyDetector ─────────────────────────────────────────────────────────


class AnomalyDetector:
    """Flags sudden drops in form score that may signal fatigue or injury."""

    DROP_THRESHOLD = 15.0  # drop of >15 points from prev session = anomaly

    def detect(self, session_summaries: list[dict]) -> list[dict]:
        """
        Returns list of anomaly events, each:
            { session_id, score, prev_score, drop, flag: 'fatigue'|'injury_risk' }
        """
        anomalies = []
        prev_score = None
        for s in session_summaries:
            score = s.get("avg_form_score", 0)
            if prev_score is not None and (prev_score - score) >= self.DROP_THRESHOLD:
                drop = prev_score - score
                anomalies.append(
                    {
                        "session_id": s.get("session_id", ""),
                        "score": round(score, 1),
                        "prev_score": round(prev_score, 1),
                        "drop": round(drop, 1),
                        "flag": "injury_risk" if drop >= 25 else "fatigue",
                        "message": (
                            f"⚠️ Form dropped {drop:.0f}pts — consider rest or lighter training"
                            if drop >= 25
                            else f"😤 Slight dip ({drop:.0f}pts) — may be fatigue"
                        ),
                    }
                )
            prev_score = score
        return anomalies


# ─── CoachRecommender ─────────────────────────────────────────────────────────


class CoachRecommender:
    """Generates targeted drill recommendations from sport + joint angle data."""

    def recommend(self, sport: str, avg_joint_angles: dict, trend: str) -> list[dict]:
        """
        Args:
            sport: sport key
            avg_joint_angles: averaged joint angles from sessions (KNEE_L, HIP_L, etc.)
            trend: 'improving'|'declining'|'stable'|'insufficient_data'
        Returns:
            List of 3 drill dicts with { drill, focus, cue, priority }
        """
        drills = list(SPORT_DRILLS.get(sport, SPORT_DRILLS["vertical_jump"]))

        # Sort by relevance based on weakest joint angles
        knee_avg = (avg_joint_angles.get("KNEE_L", 170) + avg_joint_angles.get("KNEE_R", 170)) / 2
        hip_avg = (avg_joint_angles.get("HIP_L", 170) + avg_joint_angles.get("HIP_R", 170)) / 2

        # Bump hip drill priority if hip angle is off
        if hip_avg < 150 and len(drills) > 1:
            drills = [drills[1], drills[0]] + drills[2:]

        # If declining, push fundamentals
        if trend == "declining":
            message = "🔴 Focus on fundamentals — form is declining."
        elif trend == "improving":
            message = "🟢 Great progress! Push intensity on these drills."
        else:
            message = "🟡 Maintain consistency — small gains add up."

        return {
            "drills": [dict(d, priority=("high" if i == 0 else "medium")) for i, d in enumerate(drills[:3])],
            "message": message,
        }


# ─── ProgressPredictor ───────────────────────────────────────────────────────


class ProgressPredictor:
    """Estimates weeks to reach next athlete tier based on BPI growth rate."""

    def predict(self, current_bpi: int, sessions: list[dict]) -> dict:
        """
        Args:
            current_bpi: current BPI value
            sessions: list of completed session summaries

        Returns:
            { current_tier, next_tier, weeks_to_next, sessions_needed, bpi_gap }
        """
        # Find current and next tier
        current_tier = "Block"
        next_tier = "District"
        for tier, threshold in reversed(list(TIER_BPI_THRESHOLDS.items())):
            if current_bpi >= threshold:
                current_tier = tier
                break

        tiers = list(TIER_BPI_THRESHOLDS.keys())
        tier_idx = tiers.index(current_tier)
        next_tier = tiers[tier_idx + 1] if tier_idx + 1 < len(tiers) else None

        if not next_tier:
            return {
                "current_tier": current_tier,
                "next_tier": None,
                "weeks_to_next": None,
                "sessions_needed": None,
                "bpi_gap": 0,
                "message": "🏆 Maximum tier reached. Focus on national championships.",
            }

        bpi_gap = TIER_BPI_THRESHOLDS[next_tier] - current_bpi

        # Estimate average BPI gain per session
        if len(sessions) >= 2:
            xp_gains = [s.get("xp_earned", 100) for s in sessions]
            avg_gain_per_session = sum(xp_gains) / len(xp_gains) * 0.5  # XP → BPI factor
        else:
            avg_gain_per_session = 100  # default assumption

        sessions_needed = max(1, round(bpi_gap / avg_gain_per_session))
        # Assume 3 sessions per week
        weeks_to_next = max(1, round(sessions_needed / 3))

        return {
            "current_tier": current_tier,
            "next_tier": next_tier,
            "weeks_to_next": weeks_to_next,
            "sessions_needed": sessions_needed,
            "bpi_gap": bpi_gap,
            "message": (
                f"📈 {bpi_gap:,} BPI to reach {next_tier} tier → "
                f"~{sessions_needed} sessions (~{weeks_to_next} weeks at 3×/week)"
            ),
        }


# ─── StreakTracker ────────────────────────────────────────────────────────────


class StreakTracker:
    """Counts consecutive active training days for an athlete."""

    def _parse_dt(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    def compute(self, sessions: list[dict]) -> dict:
        """
        Returns:
          { current_streak, longest_streak, last_active_date, active_today }
        """
        completed = [s for s in sessions if s.get("status") == "completed"]
        active_days: set[str] = set()
        for s in completed:
            ts = self._parse_dt(s.get("started_at"))
            if ts:
                active_days.add(ts.date().isoformat())

        if not active_days:
            return {"current_streak": 0, "longest_streak": 0, "last_active_date": None, "active_today": False}

        today = datetime.now(timezone.utc).replace(tzinfo=None).date()
        today_str = today.isoformat()
        active_today = today_str in active_days

        # Walk backwards from today (or yesterday) to find current streak
        current_streak = 0
        check = today if active_today else today - timedelta(days=1)
        while check.isoformat() in active_days:
            current_streak += 1
            check -= timedelta(days=1)

        # Compute longest streak
        sorted_days = sorted(active_days)
        longest = 1
        run = 1
        for i in range(1, len(sorted_days)):
            prev = datetime.fromisoformat(sorted_days[i - 1]).date()
            curr = datetime.fromisoformat(sorted_days[i]).date()
            if (curr - prev).days == 1:
                run += 1
                longest = max(longest, run)
            else:
                run = 1

        last_active = sorted_days[-1]
        return {
            "current_streak": current_streak,
            "longest_streak": longest,
            "last_active_date": last_active,
            "active_today": active_today,
        }


# ─── Combined Insights ────────────────────────────────────────────────────────


def generate_insights(
    athlete: dict,
    session_summaries: list[dict],
) -> dict:
    """
    Generates the full intelligence payload for /athlete/{id}/insights.

    Args:
        athlete:          ATHLETE_DB entry
        session_summaries: sorted-by-date list of completed session dicts
    Returns:
        Full insights dict
    """
    sport = athlete.get("sport", "vertical_jump")
    bpi = athlete.get("bpi", 0)

    trend_data = TrendAnalyzer().analyze(session_summaries)
    anomalies = AnomalyDetector().detect(session_summaries)

    # Average joint angles across sessions (from FRAME_BUFFER would be ideal,
    # but we work with what's in summaries for now)
    avg_angles: dict[str, float] = {}  # populated from real session frames in future

    coach_data = CoachRecommender().recommend(sport, avg_angles, trend_data["trend"])
    prediction = ProgressPredictor().predict(bpi, session_summaries)

    # Data source signal
    is_real_data = len(session_summaries) > 3
    data_source = "real" if is_real_data else "mock" if len(session_summaries) == 0 else "hybrid"

    return {
        "athlete_id": athlete.get("id"),
        "sport": sport,
        "data_source": data_source,
        "trend": trend_data,
        "anomalies": anomalies,
        "coaching": coach_data,
        "prediction": prediction,
        "summary": (f"{trend_data['trend'].title()} form. {prediction['message']}"),
    }
