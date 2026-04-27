from __future__ import annotations

from fastapi import APIRouter, WebSocket

router = APIRouter(tags=["Realtime"])


@router.websocket("/realtime/{session_id}/landmarks")
async def realtime_landmarks(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await websocket.close(code=4410, reason="endpoint deprecated — use /ws/session/{session_id}/frames-jpeg")
