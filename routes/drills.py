"""
Drill library catalog and assignments.

Endpoints:
  GET  /drills/catalog                          — full drill catalog
  POST /coach/{coach_id}/drill-assignment       — assign a drill to an athlete
  GET  /athlete/{athlete_id}/drill-assignments  — get assignments for an athlete
  POST /athlete/{athlete_id}/drill-assignments/{assignment_id}/complete  — mark done
"""

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import current_user
from database import _load_json

router = APIRouter(tags=["Drills"])

_ASSIGNMENTS_FILE = "db/drill_assignments.json"


def _load_assignments() -> dict:
    try:
        with open(_ASSIGNMENTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"assignments": []}


def _save_assignments(data: dict) -> None:
    os.makedirs("db", exist_ok=True)
    with open(_ASSIGNMENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/drills/catalog")
async def get_drills_catalog():
    data = _load_json("db/drills.json")
    return {"drills": data.get("drills", [])}


class DrillAssignmentIn(BaseModel):
    drill_id: str
    drill_name: str
    athlete_id: str
    sets: int = 3
    reps: int = 10
    note: str = ""


@router.post("/coach/{coach_id}/drill-assignment", dependencies=[Depends(current_user)])
async def assign_drill(coach_id: str, body: DrillAssignmentIn):
    if not body.athlete_id or not body.drill_id:
        raise HTTPException(400, "athlete_id and drill_id are required")

    data = _load_assignments()
    assignment = {
        "id": uuid4().hex[:12],
        "coach_id": coach_id,
        "athlete_id": body.athlete_id,
        "drill_id": body.drill_id,
        "drill_name": body.drill_name,
        "sets": body.sets,
        "reps": body.reps,
        "note": body.note,
        "assigned_at": datetime.now(timezone.utc).isoformat(),
        "completed": False,
        "completed_at": None,
    }
    data["assignments"].append(assignment)
    _save_assignments(data)
    return {"ok": True, "assignment": assignment}


@router.get("/athlete/{athlete_id}/drill-assignments", dependencies=[Depends(current_user)])
async def get_drill_assignments(athlete_id: str):
    data = _load_assignments()
    assignments = [
        a for a in data.get("assignments", [])
        if a["athlete_id"] == athlete_id
    ]
    assignments.sort(key=lambda a: a.get("assigned_at", ""), reverse=True)
    return {"athlete_id": athlete_id, "assignments": assignments[:20]}


@router.post(
    "/athlete/{athlete_id}/drill-assignments/{assignment_id}/complete",
    dependencies=[Depends(current_user)],
)
async def complete_drill_assignment(athlete_id: str, assignment_id: str):
    data = _load_assignments()
    for a in data["assignments"]:
        if a["id"] == assignment_id and a["athlete_id"] == athlete_id:
            a["completed"] = True
            a["completed_at"] = datetime.now(timezone.utc).isoformat()
            _save_assignments(data)
            return {"ok": True, "assignment": a}
    raise HTTPException(404, "assignment not found")
