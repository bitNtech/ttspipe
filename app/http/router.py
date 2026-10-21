"""Aggregates the HTTP routes into the single router `app.main` includes.

The modules behind it split by what they are *for*, not by verb:

    clone.py   POST /v1/tts/clone      one-shot: clone this upload, right now
    voice.py   GET|POST|DELETE /v1/voice
                                        the active voice every WebSocket turn
                                        falls back to
    ui.py      GET /                    the static test page
    common.py  body reading, error shape, clip validation shared by the above
"""

from __future__ import annotations

from fastapi import APIRouter

from app.http.clone import router as clone_router
from app.http.ui import router as ui_router
from app.http.voice import router as voice_router

router = APIRouter()
router.include_router(clone_router)
router.include_router(voice_router)
router.include_router(ui_router)
