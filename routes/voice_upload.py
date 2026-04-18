from __future__ import annotations

"""
Voice note upload — store and serve audio files for coach/athlete messages.

POST /voice-note/upload   — multipart upload, returns a URL
GET  /voice-note/{file_id} — serve the file

Files stored in local `db/voice_notes/` directory.
In production, swap _store_file / _serve_file for S3 presigned URLs by
setting VOICE_NOTE_STORAGE=s3 and S3_BUCKET env vars.
"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from logging_setup import get_logger

router = APIRouter(tags=["Voice Notes"])
log = get_logger("routes.voice_upload")

_VOICE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "db" / "voice_notes"
_VOICE_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_MIME = {"audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/wav", "audio/aac", "audio/m4a"}
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB — 30-second voice notes are well under this


@router.post("/voice-note/upload")
async def upload_voice_note(file: UploadFile = File(...)):
    """
    Accept a voice note from browser MediaRecorder or Android.
    Returns a `url` field that callers store in broadcast/message `voice_note_url`.

    Max 5 MB. Accepted MIME types: webm, ogg, mp4, mpeg, wav, aac, m4a.
    """
    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type and content_type not in _ALLOWED_MIME:
        raise HTTPException(415, f"unsupported audio type: {content_type}")

    data = await file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, "voice note exceeds 5 MB limit")
    if len(data) == 0:
        raise HTTPException(400, "empty file")

    ext = _ext_for_mime(content_type) or _ext_from_filename(file.filename or "")
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    dest = _VOICE_DIR / filename

    dest.write_bytes(data)

    url = f"/voice-note/{file_id}"
    log.info("voice note stored", extra={"file_id": file_id, "bytes": len(data)})
    return {"file_id": file_id, "url": url, "bytes": len(data)}


@router.get("/voice-note/{file_id}")
async def serve_voice_note(file_id: str):
    """Serve a previously uploaded voice note by file_id."""
    # sanitize — allow only uuid chars
    if not all(c in "0123456789abcdef-" for c in file_id):
        raise HTTPException(400, "invalid file_id")

    matches = list(_VOICE_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(404, "voice note not found")

    path = matches[0]
    media_type = _mime_for_ext(path.suffix) or "audio/octet-stream"
    return FileResponse(str(path), media_type=media_type)


def _ext_for_mime(mime: str) -> str:
    return {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/aac": ".aac",
        "audio/m4a": ".m4a",
    }.get(mime, "")


def _ext_from_filename(name: str) -> str:
    suffix = Path(name).suffix.lower()
    allowed = {".webm", ".ogg", ".mp4", ".mp3", ".wav", ".aac", ".m4a"}
    return suffix if suffix in allowed else ".webm"


def _mime_for_ext(ext: str) -> str:
    return {
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".mp4": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".aac": "audio/aac",
        ".m4a": "audio/m4a",
    }.get(ext.lower(), "")
