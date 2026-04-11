from __future__ import annotations

"""
Personal Health — FastAPI REST Server

Base URL: http://localhost:8082
Docs:     http://localhost:8082/docs

Start:
    python api_server.py
    # OR: uvicorn api_server:app --host 0.0.0.0 --port 8082 --reload
"""

import asyncio
import os
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
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn")

if FASTAPI_AVAILABLE:
    import database
    from database import _load_db, _save_db
    from routes.athletes import router as athletes_router
    from routes.fitness import analysis_worker, session_cleanup_worker
    from routes.fitness import router as fitness_router
    from routes.health import router as health_router
    from routes.social import router as social_router
    from routes.nutrition import router as nutrition_router

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load_db()
        database.ANALYSIS_QUEUE = asyncio.Queue(maxsize=200)
        task = asyncio.create_task(analysis_worker())
        cleanup_task = asyncio.create_task(session_cleanup_worker())
        print("[API] Personal Health API running -> http://localhost:8082")
        print("[API] Swagger docs -> http://localhost:8082/docs")
        print("[API] Async analysis worker started")
        yield
        task.cancel()
        cleanup_task.cancel()
        _save_db()
        print("[API] DB saved. Shutting down.")

    app = FastAPI(
        title="Personal Health API",
        description="Sports Biomechanics REST API powering the Android app and dashboard",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    _cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False if "*" in _cors_origins else True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(fitness_router)
    app.include_router(athletes_router)
    app.include_router(social_router)
    app.include_router(nutrition_router)

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install: pip install fastapi uvicorn pydantic")
    else:
        _host = os.environ.get("HOST", "0.0.0.0")
        _port = int(os.environ.get("PORT", "8082"))
        print("\n Personal Health REST API")
        print(f" http://localhost:{_port}")
        print(f" http://localhost:{_port}/docs\n")
        uvicorn.run("api_server:app", host=_host, port=_port, reload=False, log_level="info")
