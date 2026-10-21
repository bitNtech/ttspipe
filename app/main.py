"""FastAPI app: wiring, lifespan, and the two small operational endpoints.

The WebSocket protocol lives in `app.ws.router` and the HTTP routes (voice
cloning, the active voice, the test page) in `app.http`; this module's job is
getting an engine built and
loaded in the background -- so `/health` answers, and stays cheap to poll,
while a GPU model is still coming up -- plus `/health` and `/status` for
whatever is watching the process.

`app.engine.factory` (and everything behind it: torch, bitsandbytes,
fish_speech) is imported lazily, inside the lifespan, and only when
`TTS_ENGINE != "fake"`. Importing this module must work on a machine with
none of that installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import REPO_ROOT, get_settings
from app.engine.base import EngineStatus, TTSEngine
from app.http.router import router as http_router
from app.voice import ACTIVE_VOICE_ID, VoiceStore, warm_active_voice
from app.ws.router import active_session_count
from app.ws.router import router as ws_router

logger = logging.getLogger("tts")


def _configure_logging(level_name: str) -> None:
    """Make `TTS_LOG_LEVEL` mean something under uvicorn.

    `basicConfig` is defined to do nothing once the root logger already has a
    handler, and uvicorn installs its own before the app's lifespan ever runs -
    so setting the level through it worked when `app.main` was the entry point
    and was silently ignored in the only way this server is actually started.
    `run.sh` execs uvicorn. The levels below are therefore set on the loggers
    directly.

    `TTS_LOG_LEVEL=DEBUG` raises this application's loggers and leaves the root
    at INFO on purpose: torch, huggingface and the vendored fork are all noisy
    at DEBUG, and what an operator asking for it wants is the per-segment timing
    in `app.engine.worker` and `tts.session`, not a megabyte of somebody else's.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_level = max(level, logging.INFO)
    logging.basicConfig(level=root_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(root_level)
    if not logging.getLogger().handlers:  # pragma: no cover - uvicorn always has one
        logging.getLogger().addHandler(logging.StreamHandler())
    for name in ("tts", "app"):
        logging.getLogger(name).setLevel(level)


async def _build_engine() -> TTSEngine:
    if os.environ.get("TTS_ENGINE", "").strip().lower() == "fake":
        from app.engine.fake import FakeEngine

        return FakeEngine()
    from app.engine.factory import build_engine  # lazy: pulls in torch et al.

    return build_engine(get_settings())


async def _load_engine(app: FastAPI) -> None:
    """Runs as a background task so `/health` is reachable while this is slow."""
    try:
        engine = await _build_engine()
        app.state.engine = engine
        await engine.load()
        logger.info("engine ready: %s", engine.status().as_dict())
        # A voice set before this restart is about to be used cold, so prefill
        # its prompt now rather than on whatever turn arrives first.
        await warm_active_voice(app.state)
    except Exception as exc:  # pragma: no cover - depends on the real backend
        logger.exception("engine failed to load")
        app.state.engine_load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    app.state.settings = settings
    app.state.start_time = time.monotonic()
    app.state.engine = None
    app.state.engine_load_error = None
    # Built before the engine, and independently of it: setting a voice is
    # pure I/O, so it must work (and survive a restart) while the model is
    # still loading.
    app.state.voices = VoiceStore(REPO_ROOT / "references" / ACTIVE_VOICE_ID)

    load_task = asyncio.create_task(_load_engine(app))
    try:
        yield
    finally:
        load_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await load_task
        engine: TTSEngine | None = app.state.engine
        if engine is not None:
            await engine.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="TTS streaming server", lifespan=lifespan)
    app.include_router(ws_router)
    app.include_router(http_router)

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500, content={"error": str(exc), "type": type(exc).__name__}
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        engine: TTSEngine | None = app.state.engine
        error = app.state.engine_load_error
        if error is not None:
            return JSONResponse(status_code=503, content={"status": "error", "error": error})
        if engine is None or not engine.ready:
            return JSONResponse(status_code=200, content={"status": "loading"})
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/status")
    async def status() -> JSONResponse:
        engine: TTSEngine | None = app.state.engine
        payload = engine.status().as_dict() if engine is not None else EngineStatus().as_dict()
        voices: VoiceStore | None = getattr(app.state, "voices", None)
        active_voice = voices.get() if voices is not None else None
        payload.update(
            {
                "uptime_seconds": time.monotonic() - app.state.start_time,
                "active_sessions": active_session_count(),
                "active_voice": active_voice.as_dict() if active_voice else None,
                "settings": app.state.settings.as_dict(),
            }
        )
        return JSONResponse(status_code=200, content=payload)

    return app


app = create_app()
