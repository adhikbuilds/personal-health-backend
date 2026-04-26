from __future__ import annotations

"""Tests for the Phase 2 WebSocket frame ingest endpoint
(/ws/session/{sid}/frames-jpeg). Covers connect/refuse/ingest paths
without needing an actual JPEG decoder — we only verify the protocol,
not the analyzer output (which has its own coverage)."""

import base64
import json

# A minimal 1x1 JPEG (FF D8 ... FF D9) — enough to satisfy the bytes-non-empty
# check. The pose analyzer will fail to detect anything, which is expected.
MINIMAL_JPEG_B64 = base64.b64encode(
    bytes.fromhex(
        "FFD8FFE000104A46494600010100000100010000FFDB004300080606"
        "07060805070707090908060A0C140D0C0B0B0C1912130F141D1A1F1E"
        "1D1A1C1C20242E2720222C231C1C2837292C30313434341F2739"
        "3D38323C2E333432FFC0000B080001000101011100FFC400140100"
        "01000000000000000000000000000000FFC4002510000202010"
        "303020403000000000000000000010003020405061107213141"
        "FFDA0008010100003F00FB0F88B7BD7E10C9F8AB67BBC4A28A28"
        "FFD9"
    )
).decode()


def _start_session(client) -> str:
    r = client.post("/session/start", json={"athlete_id": "athlete_01", "sport": "vertical_jump"})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_ws_refuses_unknown_session(client):
    with client.websocket_connect("/ws/session/totally_fake/frames-jpeg") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "session_not_found"


def test_ws_refuses_completed_session(client):
    sid = _start_session(client)
    client.post(f"/session/{sid}/end")
    with client.websocket_connect(f"/ws/session/{sid}/frames-jpeg") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "session_not_active"


def test_ws_accepts_valid_session_and_processes_frame(client):
    sid = _start_session(client)
    with client.websocket_connect(f"/ws/session/{sid}/frames-jpeg") as ws:
        # Send a frame; the server queues it for analysis. Even if the
        # analyzer can't detect anything in our 1x1 test image, the queue
        # accept + frame_count increment is what we're verifying here.
        ws.send_text(json.dumps({"image_b64": MINIMAL_JPEG_B64, "ts": 0}))
        # No error reply within first cycle is the contract; close cleanly
        # so the test exits in well under the 30s server-side ping window.

    # frame_count should have been bumped on the session record
    r = client.get(f"/session/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body.get("frame_count", 0) >= 1


def test_ws_drops_empty_messages(client):
    sid = _start_session(client)
    with client.websocket_connect(f"/ws/session/{sid}/frames-jpeg") as ws:
        # Empty payload → should NOT crash, NOT increment frame count.
        ws.send_text(json.dumps({"ts": 0}))
        ws.send_text(json.dumps({}))

    r = client.get(f"/session/{sid}")
    assert r.json().get("frame_count", 0) == 0
