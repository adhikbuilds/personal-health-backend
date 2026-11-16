from __future__ import annotations

"""
Sports Catalog — single registry for every sport the platform supports.

Replaces the scattered `SPORT_INDEX` (model_registry.py), `REP_TRANSITIONS`
(video_pipeline.py), per-sport `DRILL_LIBRARY` (training_plan.py), and
per-sport `COACHING_RULES` (smart_coach.py) with one place where a new
sport registers itself.

Why this exists: adding a new sport used to require edits in 4+ modules,
each of which had subtly different lookup fallbacks. This catalog makes
the registration atomic — you extend one dict and the rest of the
platform picks it up through typed accessors.

The drill and coaching-rule *bodies* still live in their respective
service modules because they reference service-level helpers; the
catalog just maps sport keys to the right dict entries.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SportMeta:
    """Static metadata about a sport — index for the model, rep transitions,
    display label, ML feature normalisation hints, and a default body-height
    scaling for jump-height-style calculations.
    """

    key: str
    label: str
    sport_index: int
    rep_from: str
    rep_to: str
    has_jump_height: bool = False
    default_height_cm: float = 170.0
    # Typical ranges for the primary scoring angles (hint only — actual
    # thresholds live in pose_analyzer). Used for health checks and client
    # display hints.
    typical_knee_range: tuple[int, int] = (85, 170)
    typical_hip_range: tuple[int, int] = (60, 170)
    typical_trunk_range: tuple[int, int] = (0, 30)


_CATALOG: dict[str, SportMeta] = {
    "general": SportMeta(
        key="general",
        label="General",
        sport_index=0,
        rep_from="descent",
        rep_to="setup",
        has_jump_height=False,
    ),
    "vertical_jump": SportMeta(
        key="vertical_jump",
        label="Vertical Jump",
        sport_index=0,
        rep_from="descent",
        rep_to="takeoff",
        has_jump_height=True,
        typical_knee_range=(85, 110),
        typical_hip_range=(80, 110),
        typical_trunk_range=(8, 18),
    ),
    "snatch": SportMeta(
        key="snatch",
        label="Snatch",
        sport_index=1,
        rep_from="descent",
        rep_to="catch",
        typical_knee_range=(95, 125),
        typical_hip_range=(85, 100),
        typical_trunk_range=(18, 30),
    ),
    "sprint": SportMeta(
        key="sprint",
        label="Sprint",
        sport_index=2,
        rep_from="drive",
        rep_to="flight",
        typical_knee_range=(85, 115),
        typical_hip_range=(42, 62),
        typical_trunk_range=(12, 22),
    ),
    "javelin": SportMeta(
        key="javelin",
        label="Javelin",
        sport_index=3,
        rep_from="wind_up",
        rep_to="release",
        typical_knee_range=(140, 165),
        typical_hip_range=(105, 125),
        typical_trunk_range=(28, 45),
    ),
    "cricket_bat": SportMeta(
        key="cricket_bat",
        label="Cricket Bat",
        sport_index=4,
        rep_from="backswing",
        rep_to="contact",
        typical_knee_range=(132, 158),
        typical_hip_range=(118, 145),
        typical_trunk_range=(18, 30),
    ),
    "squat": SportMeta(
        key="squat",
        label="Squat",
        sport_index=0,
        rep_from="descent",
        rep_to="setup",
        typical_knee_range=(75, 110),
        typical_hip_range=(75, 105),
        typical_trunk_range=(0, 25),
    ),
    "push_up": SportMeta(
        key="push_up",
        label="Push-up",
        sport_index=0,
        rep_from="descent",
        rep_to="setup",
        typical_knee_range=(165, 180),
        typical_hip_range=(165, 180),
        typical_trunk_range=(0, 8),
    ),
    "pull_up": SportMeta(
        key="pull_up",
        label="Pull-up",
        sport_index=0,
        rep_from="descent",
        rep_to="setup",
        typical_knee_range=(140, 180),
        typical_hip_range=(140, 180),
        typical_trunk_range=(0, 15),
    ),
}


# ─── Public accessors ──────────────────────────────────────────────────────


def get(sport: str) -> SportMeta:
    """Return the metadata for a sport, falling back to 'general' if unknown."""
    return _CATALOG.get(sport) or _CATALOG["general"]


def all_sports() -> list[SportMeta]:
    return list(_CATALOG.values())


def is_known(sport: str) -> bool:
    return sport in _CATALOG


def sport_index(sport: str) -> int:
    return get(sport).sport_index


def rep_transition(sport: str) -> tuple[str, str]:
    meta = get(sport)
    return meta.rep_from, meta.rep_to


def keys() -> list[str]:
    return list(_CATALOG.keys())


def register(sport: SportMeta) -> None:
    """Runtime registration — primarily for tests or plugin-style sport packs."""
    _CATALOG[sport.key] = sport


# ─── Backward-compatibility shims ──────────────────────────────────────────
# Other modules can still import these names; migrate callers when touching
# the respective file, but leave the shims so the tree keeps building.


def build_sport_index_map() -> dict[str, int]:
    return {k: v.sport_index for k, v in _CATALOG.items() if k != "general"}


def build_rep_transitions_map() -> dict[str, tuple[str, str]]:
    return {k: (v.rep_from, v.rep_to) for k, v in _CATALOG.items()}
