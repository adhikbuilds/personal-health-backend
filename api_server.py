from __future__ import annotations

"""
Personal Health — FastAPI REST Server
Base URL: http://localhost:8082
Docs:     http://localhost:8082/docs
"""

import asyncio
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.openapi.utils import get_openapi

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn")

if FASTAPI_AVAILABLE:
    import database
    from config import settings
    from database import _load_db, _save_db
    from logging_setup import configure_logging, get_logger
    from middleware import install_middleware
    from routes.admin import router as admin_router
    from routes.analytics import router as analytics_router
    from routes.athletes import router as athletes_router
    from routes.auth import router as auth_router
    from routes.baseline import router as baseline_router
    from routes.coach import router as coach_router
    from routes.coach_morning import router as coach_morning_router
    from routes.coach_roster import router as coach_roster_router
    from routes.data_export import router as export_router
    from routes.fitness import analysis_worker, session_cleanup_worker
    from routes.fitness import router as fitness_router
    from routes.health import router as health_router
    from routes.huddle import list_router as huddle_list_router
    from routes.huddle import router as huddle_router
    from routes.intelligence_report import router as intelligence_report_router
    from routes.leaderboard import router as leaderboard_router
    from routes.load import router as load_router
    from routes.notifications import router as notifications_router
    from routes.nutrition_ai import router as nutrition_ai_router
    from routes.plan import router as plan_router
    from routes.progress import router as progress_router
    from routes.realtime import router as realtime_router
    from routes.scorecard import router as scorecard_router
    from routes.session_replay import router as session_replay_router
    from routes.share_card import router as share_card_router
    from routes.social import router as social_router
    from routes.streaks import router as streaks_router
    from routes.weekly_summary import router as summary_router
    from routes.wellness import router as wellness_router
    from routes.workouts import router as workouts_router
    from sqlite_store import init_db

    configure_logging("INFO")
    log = get_logger("api_server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        init_db()
        _load_db()
        database.ANALYSIS_QUEUE = asyncio.Queue(maxsize=200)
        task = asyncio.create_task(analysis_worker())
        cleanup_task = asyncio.create_task(session_cleanup_worker())
        log.info(
            "api startup",
            extra={
                "port": settings.port,
                "env": settings.env,
                "athletes": len(database.ATHLETE_DB),
                "sessions": len(database.SESSION_DB),
            },
        )
        yield
        # Shutdown — drain workers, then save
        task.cancel()
        cleanup_task.cancel()
        import contextlib

        with contextlib.suppress(Exception):
            await asyncio.gather(task, cleanup_task, return_exceptions=True)
        _save_db()
        log.info("api shutdown — db saved")

    app = FastAPI(
        title="Personal Health API",
        description="Sports Biomechanics REST API powering the Android app and dashboard",
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ─── Middleware ──────────────────────────────────────────────────────────
    install_middleware(app)

    _allow_credentials = False if "*" in settings.cors_origins else True
    if settings.is_prod and "*" in settings.cors_origins:
        log.warning("CORS wildcard in prod — set CORS_ORIGINS env explicitly")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "Retry-After"],
    )

    # ─── Routers ────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(auth_router)
    app.include_router(fitness_router)
    app.include_router(athletes_router)
    app.include_router(social_router)

    app.include_router(progress_router)
    app.include_router(analytics_router)
    app.include_router(coach_router)
    from routes.nutrition import nutrition_router, router

    app.include_router(router)
    app.include_router(nutrition_router)
    app.include_router(plan_router)
    app.include_router(summary_router)
    app.include_router(load_router)
    app.include_router(scorecard_router)
    app.include_router(huddle_router)
    app.include_router(huddle_list_router)
    app.include_router(export_router)
    app.include_router(nutrition_ai_router)
    app.include_router(realtime_router)
    app.include_router(workouts_router)
    app.include_router(intelligence_report_router)
    app.include_router(coach_roster_router)
    app.include_router(coach_morning_router)
    app.include_router(streaks_router)
    app.include_router(session_replay_router)
    app.include_router(share_card_router)
    app.include_router(leaderboard_router)
    app.include_router(notifications_router)
    app.include_router(baseline_router)
    app.include_router(wellness_router)

    # ─── OpenAPI: advertise bearer scheme ───────────────────────────────────
    def _custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
        app.openapi_schema = schema
        return schema

    app.openapi = _custom_openapi  # type: ignore[assignment]


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install: pip install fastapi uvicorn pydantic")
    else:
        uvicorn.run("api_server:app", host=settings.host, port=settings.port, reload=False, log_level="info")
