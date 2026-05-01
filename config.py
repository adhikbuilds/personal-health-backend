from __future__ import annotations

"""
Personal Health — typed runtime configuration.

Single source of truth for env vars. Imported by every module that needs a
setting so we never sprinkle os.environ.get across the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _split_csv(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    # ── server ─────────────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8082")))
    env: str = field(default_factory=lambda: os.environ.get("ENV", "dev"))

    # ── auth ───────────────────────────────────────────────────────────────
    jwt_secret: str = field(default_factory=lambda: os.environ.get("JWT_SECRET", "dev-only-insecure-secret-change-me"))
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = field(default_factory=lambda: int(os.environ.get("JWT_TTL_MINUTES", "1440")))

    # ── cors ───────────────────────────────────────────────────────────────
    cors_origins: list[str] = field(default_factory=lambda: _split_csv(os.environ.get("CORS_ORIGINS", "*")))

    # ── rate limit ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120")))
    rate_limit_burst: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_BURST", "30")))

    # ── ai coach ───────────────────────────────────────────────────────────
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"))

    # ── paths ──────────────────────────────────────────────────────────────
    db_path: Path = field(
        default_factory=lambda: (
            Path(os.environ["DB_PATH"])
            if os.environ.get("DB_PATH")
            else Path(os.path.dirname(os.path.abspath(__file__))) / "db"
        )
    )

    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    @property
    def users_sqlite(self) -> Path:
        return self.db_path / "users.sqlite3"


settings = Settings()

if settings.is_prod:
    _errors: list[str] = []
    if settings.jwt_secret == "dev-only-insecure-secret-change-me":
        _errors.append("JWT_SECRET must be changed from the insecure default in production")
    if "*" in settings.cors_origins:
        _errors.append("CORS_ORIGINS='*' is not allowed in production — set explicit origins")
    if _errors:
        raise SystemExit("FATAL — production config errors:\n  • " + "\n  • ".join(_errors))
