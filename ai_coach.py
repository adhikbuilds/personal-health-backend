from __future__ import annotations

"""
Personal Health — AI Coach (Anthropic + deterministic fallback).

Skill applied: `claude-api`.

Behavior:
  - If ANTHROPIC_API_KEY is set AND the circuit breaker is closed, call
    Anthropic claude-haiku-4-5 with a structured prompt and return the model
    text. Up to 2 retries with exponential backoff. 8s timeout.
  - On any failure (no key, network, breaker open, parse fail) we fall
    through to a deterministic template note built from the same stats.
    The endpoint is therefore *never* broken in dev.
  - Response includes `source: "anthropic" | "fallback"` for transparency.
"""

import threading
import time
from typing import Any

from config import settings
from logging_setup import get_logger

log = get_logger("ai_coach")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    log.info("anthropic SDK not installed — coach will use fallback")


# ─── Circuit breaker ────────────────────────────────────────────────────────


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_seconds: float = 60.0):
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.opened_at: float | None = None
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            if self.opened_at is None:
                return True
            if time.time() - self.opened_at > self.reset_seconds:
                self.opened_at = None
                self.failures = 0
                return True
            return False

    def record_success(self) -> None:
        with self.lock:
            self.failures = 0
            self.opened_at = None

    def record_failure(self) -> None:
        with self.lock:
            self.failures += 1
            if self.failures >= self.fail_threshold:
                self.opened_at = time.time()
                log.warning("ai_coach circuit opened", extra={"failures": self.failures})


_BREAKER = CircuitBreaker()


# ─── Prompt builder ────────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are a precise, supportive sports biomechanics coach. "
    "You write short coaching notes for athletes who train alone with a phone camera. "
    "You receive structured weekly stats and respond in EXACTLY four bullet points, "
    "each ≤ 22 words, in plain English, no emojis, no markdown headings. "
    "The four bullets must be, in order: "
    "(1) WHAT IMPROVED — call out the specific metric. "
    "(2) WHAT REGRESSED or what to watch — be honest, not sugar-coated. "
    "(3) ONE drill to do this week — concrete, equipment-light. "
    "(4) ONE warning sign to monitor — injury-risk-aware. "
    "If the athlete has too few sessions to judge, say so in bullet 1 and still provide 2-4."
)


def build_user_prompt(athlete_name: str, sport: str, stats: dict) -> str:
    lines = [
        f"Athlete: {athlete_name}",
        f"Sport: {sport}",
        f"Sessions in window: {stats.get('session_count', 0)}",
        f"Avg form score: {stats.get('avg_form_score', 0):.1f}",
        f"Form score 7d trend: {stats.get('form_trend_pct', 0):+.1f}%",
        f"BPI delta: {stats.get('bpi_delta', 0):+d}",
        f"Best jump (cm): {stats.get('best_jump_cm', 0):.1f}",
        f"Injury risk band: {stats.get('injury_risk', 'unknown')}",
        f"Injury reason: {stats.get('injury_reason', '-')}",
        "Top weak joints (joint, deviation_deg):",
    ]
    for j in stats.get("weak_joints", [])[:3]:
        lines.append(f"  - {j.get('joint')}: {j.get('deviation_deg', 0):.1f}")
    lines.append("")
    lines.append("Write the four-bullet coaching note now.")
    return "\n".join(lines)


# ─── Deterministic fallback ─────────────────────────────────────────────────


def fallback_note(athlete_name: str, sport: str, stats: dict) -> list[str]:
    sessions = stats.get("session_count", 0)
    trend = stats.get("form_trend_pct", 0)
    avg = stats.get("avg_form_score", 0)
    bpi_delta = stats.get("bpi_delta", 0)
    weak = stats.get("weak_joints", [])
    risk = stats.get("injury_risk", "low")

    if sessions < 3:
        improved = f"Only {sessions} session(s) this week — log a few more so we can spot trends."
    elif trend > 0:
        improved = f"Form score is up {trend:+.1f}% to an average of {avg:.0f}. BPI {bpi_delta:+d}."
    else:
        improved = f"Form is steady at {avg:.0f}; consistency is its own win — stack {max(3, sessions)} more this week."

    if trend < -3:
        regressed = f"Form dropped {trend:.1f}% — likely fatigue or rushed reps. Slow tempo on the next session."
    elif weak:
        j = weak[0]
        regressed = f"{j.get('joint','joint').replace('_',' ').title()} is drifting {j.get('deviation_deg',0):.0f}° from ideal — fix angle before adding load."
    else:
        regressed = "Nothing red yet — keep the warmup honest and don't skip mobility."

    drill_map = {
        "vertical_jump": "3×5 box jumps with a 2-second pause on landing — check knee tracking on every rep.",
        "sprint": "4×30m wall drives focusing on a 90° front-side knee — film one rep.",
        "snatch": "5×3 high-hang muscle snatches at 40% — pause at the catch.",
        "javelin": "3×6 single-arm med-ball throws against a wall — keep the elbow above the shoulder.",
        "cricket_bat": "3×8 shadow drives in front of a mirror — check head over front knee.",
    }
    drill = drill_map.get(sport, "3×8 tempo reps of your sport's main move at 60% effort, filmed from the side.")

    if risk == "high":
        warning = "Limb symmetry is off — if anything pinches, stop. Bias mobility on the weaker side this week."
    elif risk == "watch":
        warning = "Mild asymmetry showing up — log how you feel after each session and bias the weaker side."
    else:
        warning = "Watch for any sharp joint pain on landings — log it immediately, don't push through."

    return [improved, regressed, drill, warning]


# ─── Orchestrator ───────────────────────────────────────────────────────────


def _call_anthropic(athlete_name: str, sport: str, stats: dict) -> list[str]:
    """Call Anthropic with retry. Returns list of bullets or raises."""
    client = Anthropic(api_key=settings.anthropic_api_key, timeout=8.0)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(athlete_name, sport, stats)}],
            )
            text = "".join(
                block.text for block in msg.content if getattr(block, "type", None) == "text"
            ).strip()
            bullets = [
                line.lstrip("-*• 0123456789.)").strip()
                for line in text.splitlines()
                if line.strip()
            ]
            bullets = [b for b in bullets if b]
            if len(bullets) < 2:
                raise ValueError(f"model returned too few bullets: {bullets}")
            return bullets[:4]
        except Exception as e:  # noqa: BLE001
            last_exc = e
            wait = 0.5 * (2 ** attempt)
            log.warning(
                "ai_coach attempt failed",
                extra={"attempt": attempt + 1, "error": str(e), "backoff_s": wait},
            )
            time.sleep(wait)
    raise RuntimeError(f"anthropic failed after retries: {last_exc}")


def generate_coach_note(athlete_name: str, sport: str, stats: dict) -> dict[str, Any]:
    """Public entrypoint. Always returns a dict; never raises to the caller."""
    use_anthropic = (
        ANTHROPIC_AVAILABLE
        and bool(settings.anthropic_api_key)
        and _BREAKER.allow()
    )
    if use_anthropic:
        try:
            bullets = _call_anthropic(athlete_name, sport, stats)
            _BREAKER.record_success()
            return {
                "source": "anthropic",
                "model": settings.anthropic_model,
                "bullets": bullets,
                "stats_used": stats,
            }
        except Exception as e:  # noqa: BLE001
            _BREAKER.record_failure()
            log.error("ai_coach falling back", extra={"error": str(e)})
    return {
        "source": "fallback",
        "model": "deterministic-template",
        "bullets": fallback_note(athlete_name, sport, stats),
        "stats_used": stats,
    }
