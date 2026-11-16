from __future__ import annotations

"""
Personal Health — /plan/* endpoints.

Exposes the Dynamic Training Plan feature:

  GET  /plan/{athlete_id}/weekly          → fetch or lazily-build this week's plan
  POST /plan/{athlete_id}/regenerate      → force regenerate (invalidates cache)
  POST /plan/{athlete_id}/day/{date}/complete → mark a day as completed
  GET  /plan/{athlete_id}/history         → list past weeks (most-recent first)

All plans are persisted in db/plans.json, keyed by athlete_id → week_start →
plan payload. We use a per-athlete asyncio lock to make concurrent
regeneration + completion safe, plus the standard progress_cache for reads.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from auth import require_athlete_owner
from cache import progress_cache
from database import ATHLETE_DB, _load_json, _save_json
from logging_setup import get_logger
from services.training_plan import (
    PlanDay,
    PlanWeek,
    build_plan_context,
    compute_adherence,
    generate_plan,
    mark_day_complete,
    narrate_plan,
)

router = APIRouter(prefix="/plan", tags=["Plan"])
log = get_logger("routes.plan")


# One async lock per athlete, lazily created. Protects the read-modify-write
# cycle on db/plans.json from races between regenerate + day-complete.
_ATHLETE_LOCKS: dict[str, asyncio.Lock] = {}


def _lock_for(athlete_id: str) -> asyncio.Lock:
    lock = _ATHLETE_LOCKS.get(athlete_id)
    if lock is None:
        lock = asyncio.Lock()
        _ATHLETE_LOCKS[athlete_id] = lock
    return lock


# ─── Persistence layer ──────────────────────────────────────────────────────


_PLANS_FILE = "plans.json"


def _load_all_plans() -> dict[str, dict[str, dict]]:
    """Shape: {athlete_id: {week_start: plan_dict}}."""
    raw = _load_json(_PLANS_FILE)
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_all_plans(data: dict[str, dict[str, dict]]) -> None:
    _save_json(_PLANS_FILE, data)


def _load_plan(athlete_id: str, week_start: str) -> PlanWeek | None:
    all_plans = _load_all_plans()
    athlete_plans = all_plans.get(athlete_id, {})
    raw = athlete_plans.get(week_start)
    if not raw:
        return None
    return _dict_to_plan(raw)


def _persist_plan(plan: PlanWeek) -> None:
    all_plans = _load_all_plans()
    athlete_plans = all_plans.setdefault(plan.athlete_id, {})
    athlete_plans[plan.week_start] = plan.to_dict()

    # Retention: keep only the last 12 weeks per athlete.
    if len(athlete_plans) > 12:
        keep = dict(sorted(athlete_plans.items(), key=lambda kv: kv[0], reverse=True)[:12])
        all_plans[plan.athlete_id] = keep

    _save_all_plans(all_plans)


def _dict_to_plan(raw: dict) -> PlanWeek:
    days = [PlanDay(**d) for d in raw.get("days", [])]
    return PlanWeek(
        athlete_id=raw["athlete_id"],
        sport=raw.get("sport", "vertical_jump"),
        week_start=raw["week_start"],
        week_end=raw.get("week_end", raw["week_start"]),
        generated_at=raw.get("generated_at", ""),
        source=raw.get("source", "deterministic"),
        summary=raw.get("summary", ""),
        adherence_pct=float(raw.get("adherence_pct", 0.0)),
        days=days,
        context=raw.get("context", {}),
    )


# ─── Helpers ────────────────────────────────────────────────────────────────


def _monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _invalidate_cache(athlete_id: str) -> None:
    progress_cache._data.pop(f"plan:{athlete_id}", None)


def _ensure_athlete(athlete_id: str) -> None:
    if athlete_id not in ATHLETE_DB:
        raise HTTPException(404, f"athlete not found: {athlete_id}")


async def _build_fresh_plan(athlete_id: str, week_start: date) -> PlanWeek:
    ctx = build_plan_context(athlete_id, days=14)
    plan = generate_plan(ctx, week_start=week_start)
    # Optional LLM narration — falls through to deterministic on failure.
    plan = narrate_plan(plan, ctx)
    plan.adherence_pct = 0.0
    return plan


# ─── Routes ─────────────────────────────────────────────────────────────────


@router.get("/{athlete_id}/weekly", dependencies=[Depends(require_athlete_owner())])
async def get_weekly_plan(
    athlete_id: str,
    week_start: str | None = Query(
        default=None,
        description="Monday of the target week (YYYY-MM-DD). Defaults to current week.",
    ),
    refresh: bool = False,
) -> dict[str, Any]:
    _ensure_athlete(athlete_id)

    if week_start:
        try:
            target_week = date.fromisoformat(week_start)
        except ValueError as exc:
            raise HTTPException(400, f"invalid week_start: {exc}") from exc
        target_week = _monday_of_week(target_week)
    else:
        target_week = _monday_of_week(_today_utc())
    week_key = target_week.isoformat()

    cache_key = f"plan:{athlete_id}:{week_key}"
    if not refresh:
        cached = progress_cache.get(cache_key)
        if cached:
            return cached

    async with _lock_for(athlete_id):
        existing = _load_plan(athlete_id, week_key)
        if existing and not refresh:
            existing.adherence_pct = compute_adherence(existing)
            payload = existing.to_dict()
            progress_cache.set(cache_key, payload)
            return payload

        plan = await _build_fresh_plan(athlete_id, target_week)
        # Carry over completion flags if refreshing an existing plan.
        if existing:
            prior_by_date = {d.date: d for d in existing.days}
            for day in plan.days:
                prior = prior_by_date.get(day.date)
                if prior and prior.completed:
                    day.completed = True
                    day.completed_at = prior.completed_at
        plan.adherence_pct = compute_adherence(plan)
        _persist_plan(plan)
        payload = plan.to_dict()
        progress_cache.set(cache_key, payload)
        log.info(
            "plan generated",
            extra={
                "athlete_id": athlete_id,
                "week_start": week_key,
                "source": plan.source,
                "volume_state": plan.context.get("volume_state"),
                "injury_risk": plan.context.get("injury_risk"),
            },
        )
        return payload


@router.post("/{athlete_id}/regenerate", dependencies=[Depends(require_athlete_owner())])
async def regenerate_plan(
    athlete_id: str,
    week_start: str | None = Query(default=None),
) -> dict[str, Any]:
    _ensure_athlete(athlete_id)

    if week_start:
        try:
            target_week = date.fromisoformat(week_start)
        except ValueError as exc:
            raise HTTPException(400, f"invalid week_start: {exc}") from exc
        target_week = _monday_of_week(target_week)
    else:
        target_week = _monday_of_week(_today_utc())

    async with _lock_for(athlete_id):
        plan = await _build_fresh_plan(athlete_id, target_week)
        plan.adherence_pct = compute_adherence(plan)
        _persist_plan(plan)
        payload = plan.to_dict()
        _invalidate_cache(athlete_id)
        progress_cache.set(f"plan:{athlete_id}:{target_week.isoformat()}", payload)
        return payload


@router.post("/{athlete_id}/day/{day_date}/complete", dependencies=[Depends(require_athlete_owner())])
async def complete_day(athlete_id: str, day_date: str) -> dict[str, Any]:
    _ensure_athlete(athlete_id)

    try:
        parsed = date.fromisoformat(day_date)
    except ValueError as exc:
        raise HTTPException(400, f"invalid date: {exc}") from exc

    week_start = _monday_of_week(parsed).isoformat()

    async with _lock_for(athlete_id):
        plan = _load_plan(athlete_id, week_start)
        if not plan:
            raise HTTPException(404, "no plan for that week — generate one first")

        updated = mark_day_complete(plan, parsed.isoformat())
        if not updated:
            raise HTTPException(404, "day not found in plan")

        plan.adherence_pct = compute_adherence(plan)
        _persist_plan(plan)
        payload = plan.to_dict()
        progress_cache.set(f"plan:{athlete_id}:{week_start}", payload)
        return {
            "athlete_id": athlete_id,
            "date": parsed.isoformat(),
            "completed": True,
            "completed_at": updated.completed_at,
            "adherence_pct": plan.adherence_pct,
        }


@router.get("/{athlete_id}/history", dependencies=[Depends(require_athlete_owner())])
async def plan_history(athlete_id: str, limit: int = Query(default=4, ge=1, le=12)) -> dict[str, Any]:
    _ensure_athlete(athlete_id)
    all_plans = _load_all_plans()
    athlete_plans = all_plans.get(athlete_id, {})
    sorted_weeks = sorted(athlete_plans.items(), key=lambda kv: kv[0], reverse=True)[:limit]
    return {
        "athlete_id": athlete_id,
        "count": len(sorted_weeks),
        "weeks": [v for _, v in sorted_weeks],
    }
