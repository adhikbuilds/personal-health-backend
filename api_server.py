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
    from fastapi.staticfiles import StaticFiles

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
    from routes.coach import billing_router, voice_router
    from routes.coach import router as coach_router
    from routes.compare import router as compare_router
    from routes.data_export import router as export_router
    from routes.drills import router as drills_router
    from routes.fitness import analysis_worker, session_cleanup_worker
    from routes.fitness import router as fitness_router
    from routes.goals import router as goals_router
    from routes.health import router as health_router
    from routes.huddle import list_router as huddle_list_router
    from routes.huddle import router as huddle_router
    from routes.load import router as load_router
    from routes.nutrition_ai import router as nutrition_ai_router
    from routes.personal_bests import router as personal_bests_router
    from routes.plan import router as plan_router
    from routes.progress import router as progress_router
    from routes.realtime import router as realtime_router
    from routes.recovery import router as recovery_router
    from routes.scorecard import router as scorecard_router
    from routes.session_media import router as session_media_router
    from routes.social import router as social_router
    from routes.weekly_summary import router as summary_router
    from sqlite_store import init_db

    configure_logging("INFO")
    log = get_logger("api_server")

    async def _guarded(coro, name: str):
        """Wrap a long-running worker so an unhandled exception logs + restarts."""
        while True:
            try:
                await coro()
                break
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("worker crashed — restarting in 5s", extra={"worker": name})
                await asyncio.sleep(5)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        init_db()
        _load_db()
        if len(database.ATHLETE_DB) < 10:
            import pathlib
            import subprocess
            import sys

            seed_dir = pathlib.Path(__file__).parent
            for script in ("seed_athletes.py", "seed_sessions.py"):
                p = seed_dir / script
                if p.exists():
                    subprocess.run([sys.executable, str(p)], cwd=str(seed_dir), capture_output=True)
            _load_db()
            log.info("auto-seeded db", extra={"athletes": len(database.ATHLETE_DB)})
        database.ANALYSIS_QUEUE = asyncio.Queue(maxsize=200)
        task = asyncio.create_task(_guarded(analysis_worker, "analysis"))
        cleanup_task = asyncio.create_task(_guarded(session_cleanup_worker, "cleanup"))
        # 5s save window keeps the worst-case data-loss for in-flight frame counts
        # well under a typical session length, while keeping disk pressure trivial
        # at MVP scale (~50KB write per cycle for 30 athletes, 500 sessions).
        save_task = asyncio.create_task(_guarded(lambda: database.periodic_save_worker(5), "save"))
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
        save_task.cancel()
        import contextlib

        with contextlib.suppress(Exception):
            await asyncio.gather(task, cleanup_task, save_task, return_exceptions=True)
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

    _allow_credentials = "*" not in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Idempotency-Key"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "Retry-After"],
    )

    # ─── Static Files ────────────────────────────────────────────────────────
    import os

    public_path = os.path.join(os.path.dirname(__file__), "public")
    if os.path.exists(public_path):
        app.mount("/public", StaticFiles(directory=public_path), name="public")

    # ─── Routers ────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(auth_router)
    app.include_router(drills_router)
    app.include_router(fitness_router)
    app.include_router(athletes_router)
    app.include_router(social_router)
    app.include_router(progress_router)
    app.include_router(analytics_router)
    app.include_router(coach_router)
    from routes.nutrition import router as nutrition_router

    app.include_router(nutrition_router)
    from routes.parent import router as parent_router

    app.include_router(parent_router)
    from routes.notifications import router as notif_router

    app.include_router(notif_router)
    app.include_router(plan_router)
    app.include_router(summary_router)
    app.include_router(load_router)
    app.include_router(scorecard_router)
    app.include_router(huddle_router)
    app.include_router(huddle_list_router)
    app.include_router(export_router)
    app.include_router(nutrition_ai_router)
    app.include_router(recovery_router)
    app.include_router(goals_router)
    app.include_router(compare_router)
    app.include_router(personal_bests_router)
    app.include_router(realtime_router)
    app.include_router(voice_router)
    app.include_router(billing_router)
    from routes.messaging import router as messaging_router

    app.include_router(messaging_router)
    app.include_router(session_media_router)

    # ─── Static: serve uploaded voice notes ─────────────────────────────────
    from fastapi.staticfiles import StaticFiles

    _voice_dir = database.DB_PATH / "voice_notes"
    _voice_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/voice-notes", StaticFiles(directory=str(_voice_dir)), name="voice-notes")

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
