from __future__ import annotations

"""
Session media upload — video and image snapshots tied to a session.

POST /session/{session_id}/media/upload   — multipart upload (video/image)
GET  /session/{session_id}/media          — list all media for a session
GET  /session/{session_id}/media/{file_id} — serve specific file

Files stored in local `db/sessions_media/{session_id}/` directory.
Supports: mp4, mov, webm video; jpg, png, webp images.
Max: 50 MB video, 10 MB image.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from auth import current_user
from logging_setup import get_logger

router = APIRouter(tags=["Session Media"])
log = get_logger("routes.session_media")

_MEDIA_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "db" / "sessions_media"
_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

_VIDEO_MIME = {"video/mp4", "video/quicktime", "video/webm", "video/x-m4v"}
_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}
_MAX_VIDEO_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def _session_dir(session_id: str) -> Path:
    if not _safe_id(session_id):
        raise HTTPException(400, "invalid session_id")
    d = _MEDIA_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(session_id: str) -> Path:
    return _session_dir(session_id) / "manifest.json"


def _read_manifest(session_id: str) -> list:
    path = _manifest_path(session_id)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _write_manifest(session_id: str, items: list) -> None:
    _manifest_path(session_id).write_text(json.dumps(items, indent=2))


@router.post("/session/{session_id}/media/upload", dependencies=[Depends(current_user)])
async def upload_session_media(
    session_id: str,
    file: UploadFile = File(...),
    note: str = Form(""),
):
    """
    Upload a video or image snapshot for an existing session.
    Returns the URL where the media can be played back.
    """
    content_type = (file.content_type or "").split(";")[0].strip()
    is_video = content_type in _VIDEO_MIME
    is_image = content_type in _IMAGE_MIME

    if not is_video and not is_image:
        raise HTTPException(415, f"unsupported media type: {content_type}")

    data = await file.read()
    max_bytes = _MAX_VIDEO_BYTES if is_video else _MAX_IMAGE_BYTES
    if len(data) > max_bytes:
        raise HTTPException(413, f"file exceeds {max_bytes // (1024 * 1024)} MB limit")
    if len(data) == 0:
        raise HTTPException(400, "empty file")

    file_id = str(uuid.uuid4())
    ext = _ext_for_mime(content_type)
    filename = f"{file_id}{ext}"
    dest = _session_dir(session_id) / filename
    dest.write_bytes(data)

    item = {
        "file_id": file_id,
        "filename": filename,
        "kind": "video" if is_video else "image",
        "mime": content_type,
        "bytes": len(data),
        "note": note.strip()[:500],
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "url": f"/session/{session_id}/media/{file_id}",
    }

    manifest = _read_manifest(session_id)
    manifest.append(item)
    _write_manifest(session_id, manifest)

    log.info(
        "session media stored",
        extra={"session_id": session_id, "file_id": file_id, "kind": item["kind"], "bytes": len(data)},
    )
    return item


@router.get("/session/{session_id}/media", dependencies=[Depends(current_user)])
async def list_session_media(session_id: str):
    """Return all media items uploaded for the session."""
    if not _safe_id(session_id):
        raise HTTPException(400, "invalid session_id")
    return {"session_id": session_id, "media": _read_manifest(session_id)}


@router.get("/session/{session_id}/media/{file_id}", dependencies=[Depends(current_user)])
async def serve_session_media(session_id: str, file_id: str):
    """Serve a previously uploaded media file by file_id."""
    if not _safe_id(session_id) or not _safe_id(file_id):
        raise HTTPException(400, "invalid id")

    matches = list(_session_dir(session_id).glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(404, "media not found")

    path = matches[0]
    media_type = _mime_for_ext(path.suffix) or "application/octet-stream"
    return FileResponse(str(path), media_type=media_type)


@router.delete("/session/{session_id}/media/{file_id}", dependencies=[Depends(current_user)])
async def delete_session_media(session_id: str, file_id: str):
    """Remove a media file and its manifest entry."""
    if not _safe_id(session_id) or not _safe_id(file_id):
        raise HTTPException(400, "invalid id")

    matches = list(_session_dir(session_id).glob(f"{file_id}.*"))
    for path in matches:
        path.unlink()

    manifest = _read_manifest(session_id)
    manifest = [m for m in manifest if m.get("file_id") != file_id]
    _write_manifest(session_id, manifest)

    return {"ok": True, "session_id": session_id, "file_id": file_id}


def _safe_id(value: str) -> bool:
    return bool(value) and all(c.isalnum() or c in "-_" for c in value)


def _ext_for_mime(mime: str) -> str:
    return {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-m4v": ".m4v",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime, "")


def _mime_for_ext(ext: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".m4v": "video/x-m4v",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext.lower(), "")
