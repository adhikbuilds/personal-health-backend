"""
Personal Health — Session History Seed Script
==============================================
Generates realistic completed session records for every athlete so the
dashboard "Recent Sessions" and stats row show meaningful data.

For each athlete:
  - Creates N sessions (proportional to athlete.sessions count)
  - Each session has a summary with realistic form scores, jump heights, XP
  - Scores trend upward over time (simulates learning/improvement)
  - Timestamps spread over the past 90 days

Run:
  python seed_sessions.py

Prereq: Run seed_athletes.py first (needs db/athletes.json)
Output: db/sessions.json (loaded automatically by api_server.py on next start)
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

SEED = 99
random.seed(SEED)

DB_PATH = Path(__file__).parent / "db"
DB_PATH.mkdir(parents=True, exist_ok=True)

# Form score starting ranges by tier (athletes improve each session by small delta)
TIER_BASE_SCORE = {
    "Block": (38, 55),
    "District": (52, 68),
    "State": (65, 78),
    "National": (74, 88),
    "Elite": (85, 96),
}

SPORT_JUMP_HEIGHTS = {
    "vertical_jump": (18, 62),
    "sprint": (0, 0),
    "squat": (0, 0),
    "push_up": (0, 0),
    "pull_up": (0, 0),
    "snatch": (0, 0),
    "javelin": (0, 0),
    "cricket_bat": (0, 0),
}

NOW = datetime.utcnow()


def _gauss_clamped(mean, std, lo, hi):
    return max(lo, min(hi, random.gauss(mean, std)))


def generate_session(athlete: dict, session_index: int, total_sessions: int) -> dict:
    sport = athlete["sport"]
    tier = athlete["tier"]

    # Progress factor: 0.0 (first session) → 1.0 (last session)
    progress = session_index / max(total_sessions - 1, 1)

    base_lo, base_hi = TIER_BASE_SCORE[tier]
    base_score = base_lo + (base_hi - base_lo) * progress
    form_score = round(_gauss_clamped(base_score, 4, 20, 99), 1)
    peak_score = round(min(99, form_score + random.uniform(2, 8)), 1)

    frame_count = random.randint(80, 320)

    # Jump height only relevant for vertical jump
    jh_lo, jh_hi = SPORT_JUMP_HEIGHTS.get(sport, (0, 0))
    if jh_lo > 0:
        peak_jump = round(_gauss_clamped(jh_lo + (jh_hi - jh_lo) * progress * 0.6, 4, jh_lo, jh_hi), 1)
    else:
        peak_jump = 0.0

    symmetry = round(_gauss_clamped(0.78 + progress * 0.12, 0.04, 0.5, 1.0), 3)

    # XP scales with form_score
    xp = int(50 + form_score * 1.8 + (peak_jump * 1.2 if peak_jump else 0))

    # Spread sessions over last 90 days, earlier sessions older
    days_ago = int(90 * (1 - progress)) + random.randint(0, 5)
    started_at = (NOW - timedelta(days=days_ago, hours=random.randint(6, 21))).isoformat()

    session_id = str(uuid.uuid4())

    return {
        "session_id": session_id,
        "athlete_id": athlete["id"],
        "sport": sport,
        "status": "completed",
        "started_at": started_at,
        "frame_count": frame_count,
        "xp_earned": xp,
        "summary": {
            "avg_form_score": form_score,
            "peak_form_score": peak_score,
            "peak_jump_height_cm": peak_jump,
            "symmetry_index": symmetry,
            "total_frames": frame_count,
            "feedback_summary": _feedback(form_score, sport),
        },
    }


def _feedback(score: float, sport: str) -> str:
    if score >= 88:
        return "Excellent session — near-elite biomechanics."
    if score >= 74:
        return "Good form overall. Focus on symmetry consistency."
    if score >= 58:
        return "Average form. Work on depth and trunk stability."
    return "Below-average form detected. Review technique basics."


def main():
    athletes_path = DB_PATH / "athletes.json"
    if not athletes_path.exists():
        print("[ERROR] db/athletes.json not found. Run seed_athletes.py first.")
        return

    with open(athletes_path, encoding="utf-8") as f:
        athletes = json.load(f)

    sessions = {}
    total = 0

    for athlete in athletes.values():
        n = max(1, athlete.get("sessions", 5))
        for i in range(n):
            s = generate_session(athlete, i, n)
            sessions[s["session_id"]] = s
            total += 1

    out = DB_PATH / "sessions.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, default=str)

    print(f"[SEED] {total} sessions written to {out}")
    print(f"[SEED] Covered {len(athletes)} athletes over last 90 days")


if __name__ == "__main__":
    main()
