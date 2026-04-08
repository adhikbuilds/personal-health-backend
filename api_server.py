from __future__ import annotations
from fastapi import HTTPException
from datetime import datetime

"""
Personal Health — FastAPI REST Server
Base URL: http://localhost:8082
Docs:     http://localhost:8082/docs
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load_db()
        database.ANALYSIS_QUEUE = asyncio.Queue(maxsize=200)
        task = asyncio.create_task(analysis_worker())
        cleanup_task = asyncio.create_task(session_cleanup_worker())
        print("[API] Running -> http://localhost:8082")
        yield
        task.cancel()
        cleanup_task.cancel()
        _save_db()
        print("[API] DB saved. Shutdown.")

    app = FastAPI(
        title="Personal Health API",
        version="2.0.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(fitness_router)
    app.include_router(athletes_router)
    app.include_router(social_router)


# ================= WN-06 START ================= #

@app.post("/athlete/{athlete_id}/nutrition/goals")
def set_goals(athlete_id: str, payload: dict):
    db = _load_db()

    # Validation
    for key in ["daily_calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]:
        if key not in payload or not isinstance(payload[key], int) or payload[key] <= 0:
            raise HTTPException(status_code=400, detail="All values must be positive integers")

    if athlete_id not in db:
        db[athlete_id] = {}

    db[athlete_id]["goals"] = payload
    db[athlete_id]["goals"]["updated_at"] = datetime.utcnow().isoformat()

    _save_db(db)

    return {"status": "success", "data": db[athlete_id]["goals"]}


@app.get("/athlete/{athlete_id}/nutrition/goals")
def get_goals(athlete_id: str, sport: str = "sprint"):
    db = _load_db()

    # If goals exist
    if athlete_id in db and "goals" in db[athlete_id]:
        return {"status": "success", "data": db[athlete_id]["goals"]}

    # Default goals by sport
    if sport in ["vertical_jump", "sprint", "javelin"]:
        default = {"daily_calories": 2600, "protein_g": 130, "carbs_g": 320, "fat_g": 70}
    elif sport in ["squat", "push_up", "pull_up"]:
        default = {"daily_calories": 2800, "protein_g": 150, "carbs_g": 280, "fat_g": 80}
    elif sport == "cricket_bat":
        default = {"daily_calories": 2400, "protein_g": 110, "carbs_g": 310, "fat_g": 65}
    else:
        default = {"daily_calories": 2400, "protein_g": 120, "carbs_g": 300, "fat_g": 60}

    default["fiber_g"] = 30
    default["updated_at"] = datetime.utcnow().isoformat()

    return {"status": "default", "data": default}

# ================= WN-06 END ================= #


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        uvicorn.run("api_server:app", host="0.0.0.0", port=8082)